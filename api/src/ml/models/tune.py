from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
import optuna
from optuna.integration import MLflowCallback
import mlflow, mlflow.sklearn
from sklearn.base import clone as skl_clone

from api.src.ml.preprocessing.preprocessor import build_robust_preprocessor
from api.src.ml.preprocessing.feature_store.contract_selector import FeatureContractSelector
from api.src.ml.preprocessing.feature_store.spec_builder import FeatureSpec
from api.src.ml.models.models import make_estimator



def _suggest_params_enhanced(trial: optuna.Trial, model_family: str) -> Dict[str, Any]:
    """
    Enhanced hyperparameter suggestion with stacking ensemble support.

    NEW: Stacking ensemble hyperparameter optimization
    """
    if model_family == "stacking":
        return _suggest_stacking_params(trial)

    elif model_family == "linear_ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 50.0, log=True),
            "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr"]),
            "max_iter": trial.suggest_int("max_iter", 10000, 100000, step=10000),
            "tol": trial.suggest_float("tol", 1e-8, 1e-3, log=True),
        }

    elif model_family == "lasso":
        alpha = trial.suggest_float("alpha", 1e-6, 1.0, log=True)
        base_max_iter = 50000 if alpha >= 1e-4 else 100000
        return {
            "alpha": alpha,
            "max_iter": trial.suggest_int("max_iter", base_max_iter, base_max_iter * 2, step=10000),
            "tol": trial.suggest_float("tol", 1e-8, 1e-4, log=True),
            "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
        }

    elif model_family == "elasticnet":
        alpha = trial.suggest_float("alpha", 1e-6, 1.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.01, 0.99)
        base_max_iter = 50000
        if alpha < 1e-4 or l1_ratio < 0.1 or l1_ratio > 0.9:
            base_max_iter = 100000

        return {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "max_iter": trial.suggest_int("max_iter", base_max_iter, base_max_iter * 2, step=10000),
            "tol": trial.suggest_float("tol", 1e-8, 1e-4, log=True),
            "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
        }

    elif model_family == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

    elif model_family == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }

    elif model_family == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
        }

    elif model_family == "cat":
        return {
            "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "od_wait": trial.suggest_int("od_wait", 20, 100, step=10),
        }

    raise ValueError(f"Unknown model family: {model_family}")


def _suggest_stacking_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Suggest hyperparameters for stacking ensemble.

    This function optimizes:
    1. Which base estimators to include
    2. Meta-learner choice and parameters  
    3. Cross-validation strategy
    4. Base estimator parameters
    """
    # Available base estimator families
    available_base = ["linear_ridge", "lasso", "elasticnet", "rf", "xgb", "lgbm", "cat"]

    # Suggest which base estimators to use (at least 2, at most all)
    n_base = trial.suggest_int("n_base_estimators", 2, len(available_base))
    base_estimators = trial.suggest_categorical(
        "base_estimators_combo",
        _get_base_estimator_combinations(available_base, n_base)
    )

    # Meta-learner selection
    meta_learner = trial.suggest_categorical("meta_learner", ["linear_ridge", "lasso", "elasticnet"])

    # Cross-validation configuration
    cv_folds = trial.suggest_int("cv_folds", 3, 7)
    cv_strategy = trial.suggest_categorical("cv_strategy", ["kfold", "time_series"])

    # Whether to pass through original features to meta-learner
    passthrough = trial.suggest_categorical("passthrough", [False, True])

    # Meta-learner hyperparameters
    meta_params = {}
    if meta_learner == "linear_ridge":
        meta_params["alpha"] = trial.suggest_float("meta_alpha", 1e-3, 10.0, log=True)
    elif meta_learner == "lasso":
        meta_params["alpha"] = trial.suggest_float("meta_alpha", 1e-5, 1.0, log=True)
        meta_params["max_iter"] = trial.suggest_int("meta_max_iter", 10000, 50000, step=5000)
    elif meta_learner == "elasticnet":
        meta_params["alpha"] = trial.suggest_float("meta_alpha", 1e-5, 1.0, log=True)
        meta_params["l1_ratio"] = trial.suggest_float("meta_l1_ratio", 0.01, 0.99)
        meta_params["max_iter"] = trial.suggest_int("meta_max_iter", 10000, 50000, step=5000)

    # Base estimator hyperparameters (simplified to avoid explosion of search space)
    base_params = {}
    for base_family in base_estimators:
        base_prefix = f"base_{base_family}_"

        if base_family == "linear_ridge":
            base_params[base_family] = {
                "alpha": trial.suggest_float(f"{base_prefix}alpha", 1e-2, 10.0, log=True)
            }
        elif base_family == "lasso":
            base_params[base_family] = {
                "alpha": trial.suggest_float(f"{base_prefix}alpha", 1e-4, 1.0, log=True)
            }
        elif base_family == "elasticnet":
            base_params[base_family] = {
                "alpha": trial.suggest_float(f"{base_prefix}alpha", 1e-4, 1.0, log=True),
                "l1_ratio": trial.suggest_float(f"{base_prefix}l1_ratio", 0.1, 0.9)
            }
        elif base_family == "rf":
            base_params[base_family] = {
                "n_estimators": trial.suggest_int(f"{base_prefix}n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int(f"{base_prefix}max_depth", 5, 15)
            }
        elif base_family == "xgb":
            base_params[base_family] = {
                "n_estimators": trial.suggest_int(f"{base_prefix}n_estimators", 100, 600, step=50),
                "max_depth": trial.suggest_int(f"{base_prefix}max_depth", 3, 8),
                "learning_rate": trial.suggest_float(f"{base_prefix}learning_rate", 0.01, 0.3, log=True)
            }
        elif base_family == "lgbm":
            base_params[base_family] = {
                "n_estimators": trial.suggest_int(f"{base_prefix}n_estimators", 100, 800, step=50),
                "max_depth": trial.suggest_int(f"{base_prefix}max_depth", 3, 12),
                "learning_rate": trial.suggest_float(f"{base_prefix}learning_rate", 0.01, 0.3, log=True)
            }
        elif base_family == "cat":
            base_params[base_family] = {
                "iterations": trial.suggest_int(f"{base_prefix}iterations", 100, 600, step=50),
                "depth": trial.suggest_int(f"{base_prefix}depth", 4, 8),
                "learning_rate": trial.suggest_float(f"{base_prefix}learning_rate", 0.01, 0.3, log=True)
            }

    return {
        "base_estimators": base_estimators,
        "meta_learner": meta_learner,
        "cv_folds": cv_folds,
        "cv_strategy": cv_strategy,
        "passthrough": passthrough,
        "base_params": base_params,
        "meta_params": meta_params
    }


def _get_base_estimator_combinations(available_families: List[str], n_estimators: int) -> List[List[str]]:
    """
    Generate reasonable combinations of base estimators for stacking.

    Rather than trying all combinations, we use predefined sensible combinations
    to keep the search space manageable.
    """
    from itertools import combinations

    # Predefined good combinations based on diversity
    predefined_combos = [
        ["linear_ridge", "xgb", "lgbm"],  # Linear + two tree models
        ["linear_ridge", "xgb", "lgbm", "cat"],  # Linear + three tree models
        ["lasso", "rf", "xgb", "lgbm"],  # Regularized linear + tree models
        ["linear_ridge", "elasticnet", "xgb", "lgbm"],  # Two linear + two tree
        ["rf", "xgb", "lgbm", "cat"],  # All tree models
        ["linear_ridge", "lasso", "elasticnet"],  # All linear models
        ["xgb", "lgbm", "cat"],  # Gradient boosting models
        ["linear_ridge", "rf", "xgb"],  # Classic diverse combination
        ["lasso", "xgb", "cat"],  # Regularized + boosting
        ["elasticnet", "rf", "lgbm"]  # Elastic net + ensemble models
    ]

    # Filter combinations that match the requested number of estimators
    valid_combos = [combo for combo in predefined_combos if len(combo) == n_estimators]

    # If no predefined combinations match, generate them dynamically
    if not valid_combos:
        # Generate all combinations of the requested size
        all_combos = list(combinations(available_families, n_estimators))
        valid_combos = [list(combo) for combo in all_combos[:20]]  # Limit to first 20 to keep manageable

    return valid_combos


def _rmse(y_true, y_pred):
    return -np.sqrt(mean_squared_error(y_true, y_pred))



def optimize(
    df: pd.DataFrame,
    feature_spec: FeatureSpec,
    model_family: str,
    n_splits: int = 4,
    n_trials: int = 20,
    random_state: int = 42,
    experiment_name: str = "nba_player_valuation",
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Returns (fitted_pipeline, best_params, cv_rmse).

    Unified pipeline with explicit contract:
        RAW DF -> preprocessor (pandas DataFrame out) -> FeatureContractSelector(final_features) -> model

    IMPORTANT:
      - This function expects a RAW dataframe (not the encoded output).
      - Encoded feature selection uses the exact names stored in feature_spec.feature_selection["final_features"].
    """

    # ---- helpers (local) -----------------------------------------------------
    def _resolve_spec_columns(spec: FeatureSpec, frame: pd.DataFrame):
        """Return (num_cols, ord_cols, nom_cols, raw_cols_in_df).
        Uses spec.raw_features if available; falls back to union of declared groups.
        """
        num_cols = list(getattr(spec, "numerical", []) or [])
        ord_cols = list(getattr(spec, "ordinal_cats", []) or list(getattr(spec, "ordinal_map", {}).keys()))
        nom_cols = list(getattr(spec, "nominal_cats", []) or [])

        # preferred source of truth for order:
        raw_pref = list(getattr(spec, "raw_features", []) or [])
        if raw_pref:
            raw_all = raw_pref
        else:
            # stable union (preserve first occurrence)
            seen = set()
            raw_all = []
            for c in (num_cols + ord_cols + nom_cols):
                if c not in seen:
                    seen.add(c)
                    raw_all.append(c)

        # keep only columns that actually exist in df and are not the target
        tgt = getattr(spec, "target", None)
        raw_in_df = [c for c in raw_all if c in frame.columns and c != tgt]
        return num_cols, ord_cols, nom_cols, raw_all, raw_in_df

    def _looks_encoded(frame: pd.DataFrame) -> bool:
        # heuristic: any column has the transformer prefix pattern like "num__" / "ord__" / "nom__"
        return any(("__" in c) for c in frame.columns[: min(50, len(frame.columns))])

    # ---- target --------------------------------------------------------------
    target = feature_spec.target
    if target not in df.columns:
        raise KeyError(f"Target '{target}' missing from dataframe.")

    y = df[target].astype(float)

    # ---- resolve raw features from spec -------------------------------------
    num_cols, ord_cols, nom_cols, raw_all, raw_in_df = _resolve_spec_columns(feature_spec, df)

    if not raw_in_df:
        # actionable diagnostics
        sample_expected = raw_all[:10]
        sample_actual = list(df.columns[:10])
        hint = ""
        if _looks_encoded(df):
            hint = (
                " Hint: dataframe looks ALREADY ENCODED (columns contain '__'). "
                "Pass the RAW dataframe to optimize(); encoding happens inside the pipeline."
            )
        raise ValueError(
            "No raw feature columns found for this FeatureSpec. "
            f"Expected some of {sample_expected} but got columns like {sample_actual}.{hint}"
        )

    X_df = df[raw_in_df].copy()

    # ---- optional time ordering ---------------------------------------------
    if "Season" in df.columns:
        # Keep chronological order for CV when a season field exists
        order = np.argsort(df["Season"].astype(str).values)
        X_df = X_df.iloc[order]
        y = y.iloc[order]

    # ---- build preprocessor (pandas out) ------------------------------------
    pre = build_robust_preprocessor(
        numerical_cols=num_cols,
        ordinal_cols=ord_cols,
        nominal_cols=nom_cols,
        model_type=model_family,
    )
    try:
        # scikit-learn ≥ 1.2 API: make transformers return pandas DataFrames
        pre.set_output(transform="pandas")  # keeps encoded feature names stable
    except Exception:
        # older sklearn: contract selector will raise clearly if it doesn't get DataFrames
        pass  # safe to continue

    # ---- final encoded feature list from spec -------------------------------
    final_features = None
    fs_block = getattr(feature_spec, "feature_selection", None)
    if isinstance(fs_block, dict):
        final_features = fs_block.get("final_features")
    if final_features is None and hasattr(feature_spec, "final_features"):
        final_features = getattr(feature_spec, "final_features")

    contract = FeatureContractSelector(
        final_features=final_features,
        raise_on_missing=True,
        verbose=True,
        name="FeatureContractSelector",
    )

    # NEW: early cloneability probe with targeted diagnostics
    try:
        if getattr(contract, "verbose", False):
            contract.debug_contract_state()
        skl_clone(contract)
        print("[optimize] Contract cloneability check: OK")
    except Exception as e:
        print("[optimize] Contract cloneability check: FAILED")
        print(f"[optimize] Reason: {e}")
        # Surface immediately; this is the same failure CV would hit, but clearer here
        raise

    # ---- MLflow setup --------------------------------------------------------
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(
        log_models=True,
        log_input_examples=True,
        log_model_signatures=True,
    )

    # ---- CV & scorer (maximize negative RMSE) --------------------------------
    cv = TimeSeriesSplit(n_splits=n_splits)
    scorer = make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp)))

    def objective(trial: optuna.Trial) -> float:
        params: Dict[str, Any] = {"random_state": random_state}
        params.update(_suggest_params_enhanced(trial, model_family))
        est = make_estimator(model_family, params)
        pipe = Pipeline([("pre", pre), ("contract", contract), ("model", est)])
        scores = cross_val_score(pipe, X_df, y, cv=cv, scoring=scorer)
        return scores.mean()

    mlf_cb = MLflowCallback(metric_name="rmse_cv")
    study = optuna.create_study(
        direction="maximize", study_name=f"{model_family}_{feature_spec.target}"
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[mlf_cb])

    # ---- fit best pipeline on full data -------------------------------------
    best_params: Dict[str, Any] = {"random_state": random_state}
    best_params.update(study.best_params)
    best_est = make_estimator(model_family, best_params)
    best_pipe = Pipeline([("pre", pre), ("contract", contract), ("model", best_est)])
    best_pipe.fit(X_df, y)

    cv_rmse = -study.best_value  # convert back to positive RMSE
    return best_pipe, best_params, cv_rmse



# Keep the original function for backward compatibility
def _suggest_params(trial: optuna.Trial, model_family: str) -> Dict[str, Any]:
    """
    Original function kept for backward compatibility.
    Use _suggest_params_enhanced for better convergence and stacking support.
    """
    return _suggest_params_enhanced(trial, model_family)


if __name__ == "__main__":
    # Test stacking optimization
    df = pd.DataFrame({
        "Season": ["2019-20"]*100 + ["2020-21"]*100,
        "PTS_x": np.random.randint(5, 25, 200),
        "Age_x": np.random.randint(20, 35, 200),
        "pos": np.random.choice(list("GFC"), 200),
        "aav": np.random.uniform(1e6, 10e6, 200),
    })

    from api.src.ml.preprocessing.feature_store.spec_builder import build_feature_spec_from_schema_and_preprocessor
    spec = build_feature_spec_from_schema_and_preprocessor(df, target="aav")

    print("Testing stacking optimization...")
    pipe, params, rmse = optimize(df, spec, model_family="stacking", n_trials=5, n_splits=3)
    print("Stacking optimization completed!")
    print("Best params:", params)
    print("RMSE:", rmse)
    print("Base estimators:", [name for name, _ in pipe.named_steps['model'].estimators])
    print("Meta-learner:", type(pipe.named_steps['model'].final_estimator).__name__)
