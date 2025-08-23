# api/src/ml/train.py
from __future__ import annotations
import json, time, hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Import all required components
from api.src.ml import config
from api.src.ml.config import (
    TrainingConfig, DEFAULTS, DevTrainConfig, DEFAULT_DEV_TRAIN_CONFIG,
    get_master_parquet_path, get_stacking_default_params
)
from api.src.ml.models.models import make_estimator
from api.src.ml.column_schema import load_schema_from_yaml, SchemaConfig
from api.src.ml.features.feature_engineering import engineer_features
from api.src.ml.preprocessing.preprocessor import fit_preprocessor, transform_preprocessor
from api.src.ml.preprocessing.feature_store.feature_store import FeatureStore
from api.src.ml.preprocessing.feature_store.spec_builder import FeatureSpec, select_model_features, build_feature_spec_from_schema_and_preprocessor

from api.src.ml.preprocessing.feature_selection import propose_feature_spec
from api.src.ml.ml_config import SelectionConfig
from api.src.ml.models.tune import optimize

import matplotlib.pyplot as plt 
from mlflow.models.signature import infer_signature
from mlflow import sklearn as _sklog

# Set matplotlib backend for compatibility
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.ion()  # Enable interactive mode if needed

# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED MAIN TRAINING FUNCTION WITH BAYESIAN SUPPORT
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: TrainingConfig = DEFAULTS) -> Path:
    """
    Enhanced training function with proper preprocessing, feature selection, and Bayesian support.
    """
    print(f"[train] Starting training with config: {cfg}")
    
    # Step 1: Load master dataset
    try:
        p = get_master_parquet_path()
        print(f"[train] Loading data from: {p}")
        df = pd.read_parquet(p)
        print(f"[train] Loaded dataset: {df.shape}")
    except Exception as e:
        print(f"[train] Master parquet not found ({e}), trying alternative path...")
        alt_path = config.FINAL_ENGINEERED_DATASET_DIR / "final_merged_with_all.parquet"
        if alt_path.exists():
            df = pd.read_parquet(alt_path)
            print(f"[train] Loaded from alternative path: {df.shape}")
        else:
            raise FileNotFoundError(f"Could not find dataset at {p} or {alt_path}")

    # Step 2: Apply feature engineering
    try:
        df, _ = engineer_features(df)
        print("✓ Applied feature engineering")
    except Exception as e:
        print(f"[train] Feature engineering failed: {e}")
        raise

    # Step 3: Choose target
    target = "AAV_PCT_CAP" if cfg.use_cap_pct_target else cfg.target
    print(f"[train] Target: {target}")

    # Step 4: Clean and filter data
    try:
        # Drop exact columns
        if cfg.drop_columns_exact:
            df_clean = df.drop(columns=cfg.drop_columns_exact, errors="ignore")
            print(f"[train] Dropped {len(cfg.drop_columns_exact)} exact columns")
        else:
            df_clean = df.copy()

        # Filter by prefix
        if cfg.feature_exclude_prefixes:
            exclude_cols = [c for c in df_clean.columns 
                          if any(c.startswith(prefix) for prefix in cfg.feature_exclude_prefixes)]
            df_clean = df_clean.drop(columns=exclude_cols)
            print(f"[train] Dropped {len(exclude_cols)} prefix-filtered columns")

        # Limit rows if specified
        if cfg.max_train_rows and len(df_clean) > cfg.max_train_rows:
            df_clean = df_clean.head(cfg.max_train_rows)
            print(f"[train] Limited to {cfg.max_train_rows} rows")

        print(f"[train] Clean dataset: {df_clean.shape}")
    except Exception as e:
        print(f"[train] Data cleaning failed: {e}")
        raise

    # Step 5: Load schema and preprocessor
    try:
        schema = load_schema_from_yaml(config.COLUMN_SCHEMA_PATH)
        print("✓ Loaded column schema")
    except Exception as e:
        print(f"[train] Schema loading failed: {e}")
        raise

    # Step 6: Create train/test split for honest evaluation
    try:
        # Use Season-based split if available, otherwise random
        if "Season" in df_clean.columns:
            # Time series split: use latest seasons for test
            seasons = sorted(df_clean["Season"].unique())
            if len(seasons) >= 2:
                test_seasons = seasons[-1:]  # Latest season for test
                train_df = df_clean[~df_clean["Season"].isin(test_seasons)].copy()
                test_df = df_clean[df_clean["Season"].isin(test_seasons)].copy()
                print(f"[train] Time series split: train={len(train_df)}, test={len(test_df)}")
            else:
                train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=cfg.random_state)
                print(f"[train] Random split: train={len(train_df)}, test={len(test_df)}")
        else:
            train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=cfg.random_state)
            print(f"[train] Random split: train={len(train_df)}, test={len(test_df)}")
    except Exception as e:
        print(f"[train] Train/test split failed: {e}")
        raise

    # Step 7: Fit preprocessor on training data
    try:
        dev_cfg = DEFAULT_DEV_TRAIN_CONFIG
        X_train_np, y_train, preprocessor = fit_preprocessor(
            train_df,
            schema=schema,
            model_type="linear",  # For initial preprocessing
            numerical_imputation=dev_cfg.numerical_imputation,
            debug=False,
            quantiles=dev_cfg.quantile_clipping,
            max_safe_rows=200000,
            apply_type_conversions=True,
            drop_unexpected_schema_columns=True,
        )
        print(f"✓ Fitted preprocessor: {X_train_np.shape}")
    except Exception as e:
        print(f"[train] Preprocessor fitting failed: {e}")
        raise

    # Step 8: Bayesian training branch (enhanced with MLflow logging and holdout eval)
    if cfg.model_family == "bayes_hier":
        # Build spec using all features (no selection for Bayes by default)
        spec = build_feature_spec_from_schema_and_preprocessor(
            df=train_df,  # use the training portion for selecting columns
            target=target,
            schema=schema,
            preprocessor=preprocessor,
            final_features=None,  # use all features flowing through preprocessor/spec
            clip_bounds=None,
        )
        fs = FeatureStore(cfg.model_family, target)
        fs.save_spec(spec, {"rows": int(len(df_clean))})
        print("✓ Saved FeatureSpec for Bayesian model")

        # --- MLflow logging & holdout eval to match sklearn families ---
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        run_name = f"bayes_hier::{target}"

        with mlflow.start_run(run_name=run_name):
            # log hyperparams
            mlflow.log_params({
                "model_family": cfg.model_family,
                "target": target,
                "bayes_draws": cfg.bayes_draws,
                "bayes_tune": cfg.bayes_tune,
                "bayes_target_accept": cfg.bayes_target_accept,
                "bayes_chains": cfg.bayes_chains,
                "bayes_cores": cfg.bayes_cores,
                "bayes_group_cols": ",".join(cfg.bayes_group_cols or ()),
            })

            # train on the training split only (to enable honest holdout)
            from api.src.ml.bayes_hier import train_bayesian, predict_bayesian
            out_dir, idata = train_bayesian(
                df=train_df.copy(),
                spec=spec,
                draws=cfg.bayes_draws,
                tune=cfg.bayes_tune,
                target_accept=cfg.bayes_target_accept,
                chains=cfg.bayes_chains,
                cores=cfg.bayes_cores,
                group_cols=cfg.bayes_group_cols,
                random_seed=cfg.random_state,
                out_dir=(config.ARTIFACTS_DIR / f"bayes_hier_{target}")
            )
            print(f"✓ Bayesian posterior saved → {out_dir}")

            # Evaluate on holdout test_df (posterior mean)
            y_true = test_df[target].astype(float).values
            y_pred = predict_bayesian(
                df=test_df.copy(),
                spec=spec,
                artifact_dir=out_dir,
                group_cols=cfg.bayes_group_cols,
            ).values

            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            print(f"Bayesian holdout: RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

            mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})

            # log key artifacts
            mlflow.log_artifact(str(out_dir / "posterior.nc"))
            mlflow.log_artifact(str(out_dir / "feature_names.txt"))
            if (out_dir / "config_groups.json").exists():
                mlflow.log_artifact(str(out_dir / "config_groups.json"))

            # Return the artifact path to match other families' behavior
            return out_dir / "posterior.nc"
    
    # Step 9: For sklearn models, run feature selection
    try:
        # Create selection config
        selection_cfg = SelectionConfig(
            perm_n_repeats=10,
            perm_max_samples=0.5,
            perm_n_jobs=2,
            perm_threshold=0.001,
            shap_nsamples=100,
            shap_threshold=0.001,
            mode="union",
            min_features=10,
            max_features=None,
            fallback_strategy="top_permutation",
            max_relative_regression=0.05,
        )

        # Run feature selection on training data
        spec = propose_feature_spec(
            df=train_df,
            target=target,
            schema=schema,
            preprocessor=preprocessor,
            selection_config=selection_cfg,
        )
        print(f"✓ Feature selection: {len(spec.final_features)} features selected")

        # Save spec to feature store
        fs = FeatureStore(cfg.model_family, target)
        fs.save_spec(spec, {"rows": int(len(df_clean))})
        print("✓ Saved FeatureSpec to feature store")

    except Exception as e:
        print(f"[train] Feature selection failed: {e}")
        raise

    # Step 10: Run optimization
    try:
        # Special handling for stacking models
        if config.is_stacking_family(cfg.model_family):
            print(f"[train] Training stacking ensemble: {cfg.model_family}")
            stacking_params = get_stacking_default_params(target)
            print(f"[train] Using stacking params: {list(stacking_params.keys())}")
        
        # Run optimization on full clean dataset
        pipe, best_params, rmse = optimize(
            df=df_clean,
            feature_spec=spec,
            model_family=cfg.model_family,
            n_splits=cfg.n_splits,
            n_trials=cfg.n_trials,
            random_state=cfg.random_state,
            experiment_name=config.MLFLOW_EXPERIMENT_NAME,
        )
        print(f"✓ Optimization completed: RMSE={rmse:.4f}")
        
    except Exception as e:
        print(f"[train] Optimization failed: {e}")
        raise

    # Step 11: Save model artifacts
    try:
        out_dir = config.ARTIFACTS_DIR / f"{cfg.model_family}_{target}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "model.joblib"
        joblib.dump(pipe, out_file)
        print(f"Saved model → {out_file} (CV RMSE ≈ {rmse:,.4f})")
        
        # Save additional metadata for stacking models
        if config.is_stacking_family(cfg.model_family):
            meta_file = out_dir / "stacking_meta.json"
            try:
                stacking_model = pipe.named_steps['model']
                from api.src.ml.models.models import get_stacking_feature_importance
                importance_info = get_stacking_feature_importance(stacking_model)
                
                meta_data = {
                    "base_estimators": importance_info.get("base_estimators", []),
                    "base_weights": importance_info.get("base_predictions_weight", {}),
                    "meta_learner": importance_info.get("meta_learner", ""),
                    "cv_rmse": float(rmse),
                    "n_features": len(spec.final_features),
                }
                meta_file.write_text(json.dumps(meta_data, indent=2))
                print(f"✓ Saved stacking metadata → {meta_file}")
            except Exception as e:
                print(f"[train] Stacking metadata save failed: {e}")

        return out_file

    except Exception as e:
        print(f"[train] Model saving failed: {e}")
        raise

if __name__ == "__main__":
    # Test the training function
    cfg = TrainingConfig(
        model_family="bayes_hier",
        target="AAV",
        bayes_draws=300,  # quick smoke test
        bayes_tune=300,
        bayes_group_cols=("position", "Season"),
    )
    result_path = train(cfg)
    print(f"Training completed: {result_path}")
