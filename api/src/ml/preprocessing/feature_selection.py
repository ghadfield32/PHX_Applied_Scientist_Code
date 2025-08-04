# file: api/src/ml/preprocessing/feature_selection.py
"""
Main Purpose:
- Selecting Features for the model using Permutation and SHAP importance thresholds.
- Setting the Feature Store for the model.
- Testing the Dev/Staging/Production specs with our Feature Store
- Training the model with the selected features for example of how we would validate the feature store
Tracking RMSE:

    Validate selection didn’t degrade performance: compare RMSE before vs after selection (e.g., staging vs production gating). The “relative regression” metric in gating quantifies how much worse the new subset is; you only promote if it’s within tolerance.

    Detect drift or regression over time: if new data yields higher RMSE with the same spec, that’s a red flag.

    Choose thresholds sensibly: if reducing features (via higher perm/shap thresholds) starts increasing RMSE beyond acceptable bounds, you know you were too aggressive.

    Support model auditing: you can explain, “We dropped these features because their combined contribution was negligible and the RMSE stayed stable.”

"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error

import shap
from shapash import SmartExplainer  # optional
import shapiq                      # optional

from api.src.ml.column_schema import (
    load_schema_from_yaml,
    SchemaConfig,
)
from api.src.ml.preprocessing.preprocessor import (
    fit_preprocessor,
    transform_preprocessor,
)
from api.src.ml.features.feature_engineering import engineer_features
from api.src.ml.preprocessing.feature_store.spec_builder import (
    build_feature_spec_from_schema_and_preprocessor,
    select_model_features,
    FeatureSpec,
)
from api.src.ml.preprocessing.feature_store.feature_store import FeatureStore
from api.src.ml.ml_config import SelectionConfig
from api.src.ml import config
from api.src.ml.config import DEFAULT_DEV_TRAIN_CONFIG
from api.src.ml.ml_config import SelectionConfig

dev_cfg = DEFAULT_DEV_TRAIN_CONFIG
from api.src.ml.column_schema import hash_schema  

# --- NEW: Spec validation utilities ---
class StaleSpecError(Exception):
    """Raised when a loaded FeatureSpec is no longer compatible with the current schema or encoded data."""


def _spec_schema_is_stale(spec, schema, stage: str, debug: bool = False) -> bool:
    """
    Compare the saved schema hash in the spec metadata against the current schema.
    If they differ, the spec is considered stale.
    """
    from api.src.ml.column_schema import hash_schema

    current_hash = hash_schema(schema)
    spec_hash = None

    # Try common metadata locations; adapt if underlying FeatureSpec uses different attr names.
    if hasattr(spec, "extra_meta") and isinstance(getattr(spec, "extra_meta"), dict):
        spec_hash = spec.extra_meta.get("schema_hash")
    elif hasattr(spec, "metadata") and isinstance(getattr(spec, "metadata"), dict):
        spec_hash = spec.metadata.get("schema_hash")
    if spec_hash is None:
        # If no schema_hash was saved, we can't validate—assume not stale.
        return False
    stale = spec_hash != current_hash
    if stale and debug:
        print(f"[validate_spec] {stage} spec schema hash mismatch (spec={spec_hash} vs current={current_hash})")
    return stale


def _validate_spec_against_encoded(
    spec,
    X_encoded: pd.DataFrame,
    schema: SchemaConfig,
    stage: str,
    debug: bool = False,
):
    """
    Composite validation:
      1. Check schema drift (hash mismatch) -> stale.
      2. Ensure final_features exist in encoded matrix -> missing columns implies stale/invalid.
    Raises StaleSpecError with explanation if invalid.
    """
    # 1. Schema fingerprint drift
    if _spec_schema_is_stale(spec, schema, stage, debug=debug):
        raise StaleSpecError(f"{stage} spec schema fingerprint differs from current schema.")

    # 2. Final feature presence
    final_feats = None
    try:
        final_feats = spec.feature_selection.get("final_features", None)
    except Exception:
        final_feats = None

    if not final_feats:
        raise StaleSpecError(f"{stage} spec missing 'final_features' in its feature_selection block.")

    missing = set(final_feats) - set(X_encoded.columns)
    if missing:
        raise StaleSpecError(f"{stage} spec expected encoded features that are absent: {missing}")


def build_feature_importance_table(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    perm_n_repeats: int = 10,
    perm_max_samples: float | int | None = None,
    perm_n_jobs: int = 1,
    shap_nsamples: int = 100,
    perm_threshold: float | None = None,
    shap_threshold: float | None = None,
    final_features: Optional[list[str]] = None,
    selection_mode: str = "union",
    random_state: int = 42,
    verbose: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper: builds a temporary SelectionConfig internally and
    delegates to the reasoning-rich builder, but then drops some of the extra columns
    to approximate the previous output shape.
    """
    # Build a temporary SelectionConfig-like object; we only need the relevant attributes
    class TmpCfg:
        pass

    cfg = TmpCfg()
    cfg.perm_n_repeats = perm_n_repeats
    cfg.perm_max_samples = perm_max_samples
    cfg.perm_n_jobs = perm_n_jobs
    cfg.shap_nsamples = shap_nsamples
    cfg.perm_threshold = perm_threshold if perm_threshold is not None else 0.0
    cfg.shap_threshold = shap_threshold if shap_threshold is not None else 0.0
    cfg.fallback_strategy = "top_permutation"
    cfg.mode = selection_mode
    cfg.max_relative_regression = 0.0
    cfg.min_features = 0
    cfg.max_features = None
    # delegate to reasoning builder to get detailed table
    detailed = build_feature_importance_table_with_reasoning(
        model,
        X,
        y,
        selection_cfg=cfg,
        final_features=final_features,
        fallback_added_in_order=[],
        random_state=random_state,
        verbose=verbose,
        debug=debug,
    )
    # Now reduce to the older column set: feature, selected, importance_mean, shap_importance, combined_score, combined_rank, perm_rank, shap_rank
    cols = [
        "feature",
        "selected",
        "importance_mean",
        "shap_importance",
        "combined_score",
        "combined_rank",
        "perm_rank",
        "shap_rank",
    ]
    # safe guard: if any missing, fill
    for c in cols:
        if c not in detailed.columns:
            detailed[c] = pd.NA
    return detailed[cols]


def build_feature_importance_table_with_reasoning(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    selection_cfg: SelectionConfig,
    final_features: Optional[list[str]] = None,
    fallback_added_in_order: Optional[list[str]] = None,
    random_state: int = 42,
    verbose: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    # Compute importances
    perm_imp = compute_permutation_importance(
        model,
        X,
        y,
        n_repeats=selection_cfg.perm_n_repeats,
        n_jobs=selection_cfg.perm_n_jobs,
        max_samples=selection_cfg.perm_max_samples,
        random_state=random_state,
        verbose=verbose,
        debug=debug,
    )
    shap_imp = compute_shap_importance(model, X, nsamples=selection_cfg.shap_nsamples, debug=debug)

    merged = perm_imp.merge(shap_imp, on="feature", how="outer")
    merged["importance_mean"] = merged["importance_mean"].fillna(0.0)
    merged["importance_std"] = merged["importance_std"].fillna(0.0)
    merged["shap_importance"] = merged["shap_importance"].fillna(0.0)

    # Ranks
    merged["perm_rank"] = merged["importance_mean"].rank(method="min", ascending=False).astype(int)
    merged["shap_rank"] = merged["shap_importance"].rank(method="min", ascending=False).astype(int)

    # Normalize
    def normalize(series):
        mx = series.max()
        return series if mx == 0 else series / mx

    merged["normalized_perm"] = normalize(merged["importance_mean"])
    merged["normalized_shap"] = normalize(merged["shap_importance"])
    merged["combined_score"] = (merged["normalized_perm"] + merged["normalized_shap"]) / 2
    merged["combined_rank"] = merged["combined_score"].rank(method="min", ascending=False).astype(int)

    # Threshold passes
    merged["passes_perm"] = merged["importance_mean"] > selection_cfg.perm_threshold
    merged["passes_shap"] = merged["shap_importance"] > selection_cfg.shap_threshold
    merged["in_intersection"] = merged["passes_perm"] & merged["passes_shap"]

    # Selected
    if final_features is not None:
        merged["selected"] = merged["feature"].isin(final_features)
    else:
        merged["selected"] = False

    # Fallback tracking
    merged["fallback_added"] = False
    merged["fallback_order"] = pd.NA
    if fallback_added_in_order:
        for idx, f in enumerate(fallback_added_in_order, start=1):
            mask = merged["feature"] == f
            merged.loc[mask, "fallback_added"] = True
            merged.loc[mask, "fallback_order"] = idx

    # Reason summary
    def _reason(row):
        if row["selected"]:
            if row["in_intersection"]:
                return "passed both thresholds"
            if row["fallback_added"]:
                return f"added by fallback (strategy={selection_cfg.fallback_strategy}) order={row['fallback_order']}"
            parts = []
            if row["passes_perm"]:
                parts.append("passes perm only")
            if row["passes_shap"]:
                parts.append("passes shap only")
            if parts:
                return " & ".join(parts)
            return "selected (unknown path)"
        else:
            reasons = []
            if not row["passes_perm"]:
                reasons.append("failed perm")
            if not row["passes_shap"]:
                reasons.append("failed shap")
            return " & ".join(reasons) if reasons else "dropped"
    merged["selection_reason"] = merged.apply(_reason, axis=1)

    # Reorder columns for human consumption
    cols = [
        "feature",
        "selected",
        "selection_reason",
        "fallback_added",
        "fallback_order",
        "in_intersection",
        "passes_perm",
        "passes_shap",
        "importance_mean",
        "importance_std",
        "shap_importance",
        "perm_rank",
        "shap_rank",
        "combined_score",
        "combined_rank",
        "normalized_perm",
        "normalized_shap",
    ]
    return merged[cols]




def save_feature_importance_table(df: pd.DataFrame, path: str | Path, include_index: bool = False) -> None:
    """
    Save the importance table to CSV for external analysis.
    """
    path = Path(path)
    df.to_csv(path, index=include_index)

def print_top_feature_importances(df: pd.DataFrame, n: int = 20) -> None:
    """
    Nicely print the top-n features by combined rank, with key columns.
    """
    top = df.sort_values("combined_rank").head(n)
    display_cols = [
        "feature",
        "selected",
        "importance_mean",
        "shap_importance",
        "combined_score",
        "combined_rank",
        "perm_rank",
        "shap_rank",
    ]
    print(f"\nTop {n} features by combined importance:")
    print(top[display_cols].to_string(index=False))




# ---------- Enhanced helper functions with debugging ----------
def train_baseline_model(X: pd.DataFrame, y: pd.Series, debug: bool = False) -> RandomForestRegressor:
    """Train RandomForest with enhanced debugging"""
    if debug:
        print(f"[train_baseline_model] Training on X shape: {X.shape}")
        print(f"[train_baseline_model] X columns sample: {list(X.columns[:10])}...")
        print(f"[train_baseline_model] y shape: {y.shape}, y sample: {y.head(3).tolist()}")

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    if debug:
        print(f"[train_baseline_model] Model trained successfully")
        print(f"[train_baseline_model] Feature names stored in model: {hasattr(model, 'feature_names_in_')}")
        if hasattr(model, 'feature_names_in_'):
            print(f"[train_baseline_model] Model expects {len(model.feature_names_in_)} features")

    return model

def validate_feature_consistency(
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    model: RandomForestRegressor,
    stage: str = "",
    debug: bool = False
) -> bool:
    """Validate that feature names are consistent between train/test and model expectations"""
    if not debug:
        return True

    print(f"\n[validate_feature_consistency] {stage}")
    print(f"X_train columns: {len(X_train.columns)} features")
    print(f"X_test columns: {len(X_test.columns)} features") 
    print(f"Columns match: {list(X_train.columns) == list(X_test.columns)}")

    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
        train_features = list(X_train.columns)
        test_features = list(X_test.columns)

        train_match = model_features == train_features
        test_match = model_features == test_features

        print(f"Model expects {len(model_features)} features")
        print(f"Train features match model: {train_match}")
        print(f"Test features match model: {test_match}")

        if not train_match:
            missing_in_train = set(model_features) - set(train_features)
            extra_in_train = set(train_features) - set(model_features)
            if missing_in_train:
                print(f"  Missing in train: {list(missing_in_train)[:5]}...")
            if extra_in_train:
                print(f"  Extra in train: {list(extra_in_train)[:5]}...")

        if not test_match:
            missing_in_test = set(model_features) - set(test_features)
            extra_in_test = set(test_features) - set(model_features)
            if missing_in_test:
                print(f"  Missing in test: {list(missing_in_test)[:5]}...")
            if extra_in_test:
                print(f"  Extra in test: {list(extra_in_test)[:5]}...")

        return train_match and test_match
    else:
        print("Model has no feature_names_in_ attribute")
        return True

def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    n_jobs: int = 1,
    max_samples: float | int | None = None,
    random_state: int = 42,
    verbose: bool = True,
    debug: bool = False,
) -> pd.DataFrame:
    """Enhanced permutation importance with validation"""
    if verbose:
        print(f"⏳ Permutation importances on {X.shape[0]}×{X.shape[1]} (repeats={n_repeats}, jobs={n_jobs})")

    # Validate feature consistency first
    validate_feature_consistency(X, X, model, "Before permutation importance", debug)

    X_sel, y_sel = X, y
    if max_samples is not None:
        nsamp = int(len(X) * max_samples) if isinstance(max_samples, float) else int(max_samples)
        if verbose:
            print(f"   Subsampling to {nsamp} rows")
        X_sel, y_sel = resample(X, y, replace=False, n_samples=nsamp, random_state=random_state)

    try:
        if debug:
            print(f"[compute_permutation_importance] Testing model prediction before permutation...")
            test_pred = model.predict(X_sel.head(1))
            print(f"[compute_permutation_importance] Test prediction successful: {test_pred}")

        result = permutation_importance(
            model, X_sel, y_sel, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs,
        )
    except OSError:
        if debug:
            print("[compute_permutation_importance] OSError, retrying with n_jobs=1")
        result = permutation_importance(
            model, X_sel, y_sel, n_repeats=n_repeats, random_state=random_state, n_jobs=1,
        )

    importance_df = (
        pd.DataFrame({
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )

    if debug:
        print(f"[compute_permutation_importance] Generated importance for {len(importance_df)} features")
        print(f"[compute_permutation_importance] Top 5 features: {importance_df.head()['feature'].tolist()}")

    return importance_df

def compute_shap_importance(model, X: pd.DataFrame, nsamples: int = 100, debug: bool = False) -> pd.DataFrame:
    """Enhanced SHAP importance with validation"""
    if debug:
        print(f"[compute_shap_importance] Computing SHAP on {X.shape} with {nsamples} samples")

    # Validate before SHAP
    validate_feature_consistency(X, X, model, "Before SHAP importance", debug)

    explainer = shap.TreeExplainer(model)
    X_sample = X.sample(n=min(nsamples, len(X)), random_state=42)

    if debug:
        print(f"[compute_shap_importance] Using sample shape: {X_sample.shape}")

    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_matrix = np.mean(np.stack(shap_values, axis=0), axis=0)
    else:
        shap_matrix = shap_values

    importance_df = (
        pd.DataFrame({
            "feature": X_sample.columns,
            "shap_importance": np.abs(shap_matrix).mean(axis=0),
        })
        .sort_values("shap_importance", ascending=False)
        .reset_index(drop=True)
    )

    if debug:
        print(f"[compute_shap_importance] Generated SHAP importance for {len(importance_df)} features")
        print(f"[compute_shap_importance] Top 5 features: {importance_df.head()['feature'].tolist()}")

    return importance_df

def _select_features(perm_imp: pd.DataFrame, shap_imp: pd.DataFrame, cfg: SelectionConfig, debug: bool = False) -> Tuple[list[str], list[str]]:
    """Enhanced feature selection with fallback logic and debugging.
    Returns: (final_features, fallback_added_in_order)
    """
    perm_feats = perm_imp.loc[perm_imp["importance_mean"] > cfg.perm_threshold, "feature"].tolist()
    shap_feats = shap_imp.loc[shap_imp["shap_importance"] > cfg.shap_threshold, "feature"].tolist()

    if debug:
        print(f"[_select_features] Permutation features above {cfg.perm_threshold}: {len(perm_feats)}")
        print(f"[_select_features] SHAP features above {cfg.shap_threshold}: {len(shap_feats)}")
        print(f"[_select_features] Selection mode: {cfg.mode}")
        print(f"[_select_features] Fallback strategy: {cfg.fallback_strategy}")

    s_perm, s_shap = set(perm_feats), set(shap_feats)

    if cfg.mode == "union":
        final_set = s_perm | s_shap
        if debug:
            print(f"[_select_features] Union initial: {len(final_set)} features")
    else:  # intersection
        final_set = s_perm & s_shap
        if debug:
            print(f"[_select_features] Intersection initial: {len(final_set)} features")

    final_list = sorted(final_set)
    fallback_added_in_order: list[str] = []

    # Enforce min_features with fallback strategy
    if len(final_list) < cfg.min_features:
        if debug:
            print(f"[_select_features] Only {len(final_list)} selected; applying fallback to reach min_features={cfg.min_features}")
        if cfg.fallback_strategy == "top_permutation":
            candidates = perm_imp.sort_values("importance_mean", ascending=False)["feature"].tolist()
        elif cfg.fallback_strategy == "top_shap":
            candidates = shap_imp.sort_values("shap_importance", ascending=False)["feature"].tolist()
        else:  # "all"
            candidates = list(dict.fromkeys(perm_imp["feature"].tolist() + shap_imp["feature"].tolist()))

        for f in candidates:
            if f not in final_list:
                final_list.append(f)
                fallback_added_in_order.append(f)
            if len(final_list) >= cfg.min_features:
                break

    if cfg.max_features is not None and len(final_list) > cfg.max_features:
        if debug:
            print(f"[_select_features] Trimming from {len(final_list)} to max_features={cfg.max_features}")
        final_list = final_list[: cfg.max_features]
        # If trimming removed some fallback-added, adjust fallback_added_in_order
        fallback_added_in_order = [f for f in fallback_added_in_order if f in final_list]

    if debug:
        print(f"[_select_features] Final selected: {len(final_list)} features")
        if final_list:
            print(f"[_select_features] Sample features: {final_list[:5]}...")
        if fallback_added_in_order:
            print(f"[_select_features] Fallback additions in order: {fallback_added_in_order}")

    return final_list, fallback_added_in_order



def safe_model_predict(model, X: pd.DataFrame, stage: str = "", debug: bool = False) -> np.ndarray:
    """Safely predict with comprehensive error handling and debugging"""
    if debug:
        print(f"\n[safe_model_predict] {stage}")
        print(f"Input X shape: {X.shape}")
        print(f"Input X columns sample: {list(X.columns[:5])}...")

    # Validate feature consistency
    validate_feature_consistency(X, X, model, f"Prediction validation - {stage}", debug)

    try:
        predictions = model.predict(X)
        if debug:
            print(f"[safe_model_predict] SUCCESS: Generated {len(predictions)} predictions")
        return predictions
    except Exception as e:
        print(f"[safe_model_predict] ERROR in {stage}: {e}")
        if hasattr(model, 'feature_names_in_'):
            expected = list(model.feature_names_in_)
            actual = list(X.columns)
            missing = set(expected) - set(actual)
            extra = set(actual) - set(expected)
            print(f"Expected features: {len(expected)}")
            print(f"Actual features: {len(actual)}")
            if missing:
                print(f"Missing features: {list(missing)[:10]}...")
            if extra:
                print(f"Extra features: {list(extra)[:10]}...")
        raise

def propose_feature_spec(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: SchemaConfig,
    preprocessor,
    selection_cfg: SelectionConfig,
    model_family: str,
    fs: FeatureStore,
    debug: bool = False,
    selection_valid_frac: float = 0.25,
) -> Tuple[FeatureSpec, float]:
    """
    Build a staging FeatureSpec with clean selection:
      1) Inner split of train_df -> (sel_train, sel_valid)
      2) Fit baseline on sel_train (encoded); compute perm/SHAP on sel_valid (encoded)
      3) Select encoded features; retrain model on FULL train_df (encoded) using selected cols
      4) Evaluate RMSE on test_df (encoded) using the same selected cols
      5) Save spec (with selection metadata) to Staging
    """
    if debug:
        print(f"\n[propose_feature_spec] === STARTING FEATURE SELECTION ===")
        print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    from sklearn.model_selection import train_test_split

    raw = schema.model_features(include_target=False)
    target = schema.target()
    if target is None:
        raise ValueError("Schema has no target defined.")

    # ----- Inner split for selection -----
    sel_train_df, sel_valid_df = train_test_split(
        train_df, test_size=selection_valid_frac, random_state=42
    )
    if debug:
        print(f"[propose_feature_spec] Inner split: sel_train={len(sel_train_df)}, sel_valid={len(sel_valid_df)}")

    # Transform encodings (use the ALREADY-FITTED preprocessor)
    X_sel_tr_np, y_sel_tr = transform_preprocessor(sel_train_df, preprocessor, schema, debug=debug)
    X_sel_va_np, y_sel_va = transform_preprocessor(sel_valid_df, preprocessor, schema, debug=debug)
    X_tr_full_np, y_tr_full = transform_preprocessor(train_df, preprocessor, schema, debug=debug)
    X_te_np, y_te = transform_preprocessor(test_df, preprocessor, schema, debug=debug)

    feat_names = preprocessor.get_feature_names_out()
    X_sel_tr = pd.DataFrame(X_sel_tr_np, columns=feat_names, index=sel_train_df.index)
    X_sel_va = pd.DataFrame(X_sel_va_np, columns=feat_names, index=sel_valid_df.index)
    X_tr_full = pd.DataFrame(X_tr_full_np, columns=feat_names, index=train_df.index)
    X_te = pd.DataFrame(X_te_np, columns=feat_names, index=test_df.index)

    # ----- Baseline on sel_train; importances on sel_valid -----
    if debug:
        print(f"\n[propose_feature_spec] STEP 1: Train baseline on sel_train (ALL features)")
    model_all = train_baseline_model(X_sel_tr, y_sel_tr, debug=debug)

    if debug:
        print(f"\n[propose_feature_spec] STEP 2: Compute importances on sel_valid")
    perm_imp = compute_permutation_importance(
        model_all, X_sel_va, y_sel_va,
        n_repeats=selection_cfg.perm_n_repeats,
        n_jobs=selection_cfg.perm_n_jobs,
        max_samples=selection_cfg.perm_max_samples,
        verbose=False,
        debug=debug,
    )
    shap_imp = compute_shap_importance(model_all, X_sel_va, nsamples=selection_cfg.shap_nsamples, debug=debug)

    if debug:
        print(f"[propose_feature_spec] Top permutation: {perm_imp.head(5)['feature'].tolist()}")
        print(f"[propose_feature_spec] Top SHAP: {shap_imp.head(5)['feature'].tolist()}")

    # ----- Build initial importance table (pre-selection) -----
    if debug:
        print(f"\n[propose_feature_spec] STEP 2.5: Build initial importance diagnostics (no selected flag)")
    importance_table_initial = build_feature_importance_table(
        model_all,
        X_sel_va,
        y_sel_va,
        perm_n_repeats=selection_cfg.perm_n_repeats,
        perm_max_samples=selection_cfg.perm_max_samples,
        perm_n_jobs=selection_cfg.perm_n_jobs,
        shap_nsamples=selection_cfg.shap_nsamples,
        perm_threshold=selection_cfg.perm_threshold,
        shap_threshold=selection_cfg.shap_threshold,
        final_features=None,  # before selection
        debug=debug,
    )
    if debug:
        print_top_feature_importances(importance_table_initial, n=15)

    # ----- Select encoded features (with fallback tracking) -----
    if debug:
        print(f"\n[propose_feature_spec] STEP 3: Select features")
    final_feats, fallback_added_in_order = _select_features(perm_imp, shap_imp, selection_cfg, debug=debug)
    if not final_feats:
        if debug:
            print("[propose_feature_spec] WARNING: 0 features; fallback to ALL")
        final_feats = list(X_sel_tr.columns)
        fallback_added_in_order = []
    elif len(final_feats) < selection_cfg.min_features:
        if debug:
            print(f"[propose_feature_spec] Enforcing min_features={selection_cfg.min_features}")
        top_perm = perm_imp.sort_values("importance_mean", ascending=False)["feature"].tolist()
        for f in top_perm:
            if f not in final_feats:
                final_feats.append(f)
                fallback_added_in_order.append(f)
            if len(final_feats) >= selection_cfg.min_features:
                break

    # ----- Build initial importance diagnostics (before knowing final_feats) -----
    if debug:
        print(f"\n[propose_feature_spec] STEP 2.5: Build initial importance diagnostics (no selected flag)")
    importance_table_initial = build_feature_importance_table(
        model_all,
        X_sel_va,
        y_sel_va,
        perm_n_repeats=selection_cfg.perm_n_repeats,
        perm_max_samples=selection_cfg.perm_max_samples,
        perm_n_jobs=selection_cfg.perm_n_jobs,
        shap_nsamples=selection_cfg.shap_nsamples,
        perm_threshold=selection_cfg.perm_threshold,
        shap_threshold=selection_cfg.shap_threshold,
        final_features=None,
        selection_mode=selection_cfg.mode,
        debug=debug,
    )
    if debug:
        print_top_feature_importances(importance_table_initial, n=15)

    # ----- Build enhanced reasoning importance table -----
    if debug:
        print(f"\n[propose_feature_spec] STEP 3.5: Build importance diagnostics with reasoning")
    importance_table_with_reasoning = build_feature_importance_table_with_reasoning(
        model_all,
        X_sel_va,
        y_sel_va,
        selection_cfg=selection_cfg,
        final_features=final_feats,
        fallback_added_in_order=fallback_added_in_order,
        debug=debug,
    )
    print_top_feature_importances(importance_table_with_reasoning, n=15)

    # ----- Save diagnostics -----
    fs_dir = Path(config.FEATURE_SELECTION_DIR)
    # Initial (pre-selection)
    save_feature_importance_table(
        importance_table_initial,
        fs_dir / "feature_importance_diagnostics_initial.csv",
    )
    # Full reasoning
    save_feature_importance_table(
        importance_table_with_reasoning,
        fs_dir / "feature_importance_diagnostics_with_reasoning.csv",
    )
    # Selected only
    save_feature_importance_table(
        importance_table_with_reasoning.loc[importance_table_with_reasoning["selected"]],
        fs_dir / "feature_importance_selected_reasoning.csv",
    )
    # Unselected only
    save_feature_importance_table(
        importance_table_with_reasoning.loc[~importance_table_with_reasoning["selected"]],
        fs_dir / "feature_importance_unselected_reasoning.csv",
    )


    # ----- Retrain on FULL train using selected subset; evaluate on test -----
    if debug:
        print(f"\n[propose_feature_spec] STEP 4: Retrain on FULL train (selected={len(final_feats)})")
    X_tr_sel = X_tr_full[final_feats]
    X_te_sel = X_te[final_feats]
    model_sel = train_baseline_model(X_tr_sel, y_tr_full, debug=debug)
    preds = safe_model_predict(model_sel, X_te_sel, "Holdout RMSE (selected subset)", debug=debug)
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    if debug:
        print(f"[propose_feature_spec] Holdout RMSE: {rmse:.4f}")

    # ----- Build and save spec with selection metadata -----
    if debug:
        print(f"\n[propose_feature_spec] STEP 5: Build & save FeatureSpec (Staging)")
    # Clip bounds
    lower = getattr(preprocessor, "lower_", None)
    upper = getattr(preprocessor, "upper_", None)
    meta = getattr(preprocessor, "bounds_meta_", None)
    clip_bounds = None
    if lower is not None and upper is not None:
        if isinstance(meta, dict) and meta.get("shape") == "scalar":
            clip_bounds = {meta.get("target", target): (float(lower), float(upper))}
        elif np.isscalar(lower) and np.isscalar(upper):
            clip_bounds = {target: (float(lower), float(upper))}
        else:
            from api.src.ml.preprocessing.data_prep import resolve_clip_bounds_map
            raw_features_all = list(train_df.columns)
            numeric_features = [c for c in schema.numerical() if c in raw_features_all]
            clip_map = resolve_clip_bounds_map(
                raw_feature_names=raw_features_all,
                numeric_feature_names=numeric_features,
                lower=lower,
                upper=upper,
                target_col=target,
                debug=False,
            )
            clip_bounds = {k: v for k, v in clip_map.items() if not (v[0] is None and v[1] is None)}

    spec = build_feature_spec_from_schema_and_preprocessor(
        df=train_df, target=target, schema=schema, preprocessor=preprocessor,
        permutation_importance_df=perm_imp, shap_importance_df=shap_imp,
        final_features=final_feats, perm_thresh=selection_cfg.perm_threshold,
        shap_thresh=selection_cfg.shap_threshold, mode=selection_cfg.mode,
        clip_bounds=clip_bounds,
    )
    schema_hash_value = hash_schema(schema)
    # Save spec including schema fingerprint so future loads can detect staleness
    fs.save_spec(
        spec,
        extra_meta={"model_family": model_family, "schema_hash": schema_hash_value},
        stage="Staging",
    )

    if debug:
        print(f"[propose_feature_spec] === COMPLETE: selected={len(final_feats)}, RMSE={rmse:.4f} ===")
        print(f"[propose_feature_spec] Saved spec with schema_hash={schema_hash_value}")
    return spec, rmse


def gate_and_promote_spec(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: SchemaConfig,
    preprocessor,
    fs: FeatureStore,
    selection_cfg: SelectionConfig,
    debug: bool = False,
) -> dict:
    """
    Evaluate Production vs Staging specs with validation, promote if Staging is within
    relative regression tolerance. Handles stale/missing specs by rebuilding staging if needed.
    """
    if debug:
        print(f"\n[gate_and_promote_spec] === GATING PROCESS START ===")

    target = schema.target()
    if target is None:
        raise ValueError("Schema has no target defined for gating.")

    # Encode once (same preprocessor)
    X_tr_np, y_tr = transform_preprocessor(train_df, preprocessor, schema, debug=debug)
    X_te_np, y_te = transform_preprocessor(test_df, preprocessor, schema, debug=debug)

    feat_names = preprocessor.get_feature_names_out()
    X_tr_all = pd.DataFrame(X_tr_np, columns=feat_names, index=train_df.index)
    X_te_all = pd.DataFrame(X_te_np, columns=feat_names, index=test_df.index)

    def _eval_spec(stage_name: str) -> tuple[Optional[float], Optional[list[str]]]:
        try:
            spec = fs.load_spec(stage=stage_name)
        except Exception as e:
            if debug:
                print(f"[gate_and_promote_spec] No {stage_name} spec loaded: {e}")
            return None, None

        # Validate spec: schema drift and missing encoded features
        try:
            _validate_spec_against_encoded(spec, X_tr_all, schema, stage_name, debug=debug)
            _validate_spec_against_encoded(spec, X_te_all, schema, stage_name, debug=debug)
        except StaleSpecError as e:
            if debug:
                print(f"[gate_and_promote_spec] {stage_name} spec considered stale/invalid: {e}")
            return None, None

        # Safe to select features now
        X_tr_sel = select_model_features(X_tr_all, spec)
        X_te_sel = select_model_features(X_te_all, spec)

        if debug:
            print(f"[gate_and_promote_spec] {stage_name}: using {X_tr_sel.shape[1]} features")

        model = train_baseline_model(X_tr_sel, y_tr, debug=debug)
        preds = safe_model_predict(model, X_te_sel, f"{stage_name} evaluation", debug=debug)
        rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        return rmse, list(X_tr_sel.columns)

    # Evaluate Production spec (may be missing or stale)
    rmse_prod, prod_feats = _eval_spec("Production")

    # Evaluate Staging spec; if missing/stale, attempt to rebuild it
    rmse_stag, stag_feats = _eval_spec("Staging")
    if rmse_stag is None:
        if debug:
            print("[gate_and_promote_spec] Staging spec missing or stale; regenerating via propose_feature_spec.")
        # Derive model_family from FeatureStore if available (fallback to linear_ridge)
        model_family = getattr(fs, "model_family", None) or "linear_ridge"
        # Rebuild staging spec
        spec_new, rmse_stag = propose_feature_spec(
            train_df=train_df,
            test_df=test_df,
            schema=schema,
            preprocessor=preprocessor,
            selection_cfg=selection_cfg,
            model_family=model_family,
            fs=fs,
            debug=debug,
        )
        stag_feats = spec_new.feature_selection.get("final_features", [])

    # Decision logic
    decision = "promote"
    rel_reg = None
    if rmse_prod is not None:
        rel_reg = (rmse_stag - rmse_prod) / max(rmse_prod, 1e-8)
        if rel_reg > selection_cfg.max_relative_regression:
            decision = "hold"

    if debug:
        print(f"[gate_and_promote_spec] rmse_prod={rmse_prod}, rmse_stag={rmse_stag}, relative_regression={rel_reg}, decision={decision}")

    if decision == "promote":
        try:
            fs.promote("Staging", "Production")
            if debug:
                print("[gate_and_promote_spec] Promoted Staging -> Production")
        except Exception as e:
            if debug:
                print(f"[gate_and_promote_spec] Promotion failed: {e}")

    return {
        "rmse_prod": rmse_prod,
        "rmse_stag": rmse_stag,
        "relative_regression": rel_reg,
        "decision": decision,
    }




# ─────────────────────────────────────────────────────────────────────────────
# Enhanced smoke test with comprehensive debugging
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import hashlib
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    from api.src.ml.column_schema import load_schema_from_yaml
    from api.src.ml.preprocessing.preprocessor import fit_preprocessor
    from api.src.ml.features.feature_engineering import engineer_features
    from api.src.ml.preprocessing.feature_store.spec_builder import select_model_features
    from api.src.ml.preprocessing.feature_store.feature_store import FeatureStore
    from api.src.ml.ml_config import SelectionConfig
    from api.src.ml import config

    def spec_hash(spec) -> str:
        return hashlib.sha256(spec.to_json().encode("utf-8")).hexdigest()[:8]

    print("=== ENHANCED FEATURE SELECTION SMOKE TEST ===")

    # Enable comprehensive debugging
    DEBUG = False

    # Step 1: Load schema
    schema_path = config.COLUMN_SCHEMA_PATH
    schema = load_schema_from_yaml(str(schema_path))
    target = schema.target()
    model_family = "linear_ridge"

    # Step 2: Load and engineer data  
    data_path = config.FINAL_ENGINEERED_DATASET_DIR / "final_merged_with_all.parquet"
    from api.src.ml.features.load_data_utils import load_data_optimized

    df_raw = load_data_optimized(data_path, debug=DEBUG, use_sample=True, drop_null_rows=True)
    df_eng, _ = engineer_features(df_raw)

    if target not in df_eng.columns:
        raise RuntimeError(f"Target column '{target}' missing from engineered df.")

    # Step 3: Train/test split with debugging
    train_df, test_df = train_test_split(df_eng, test_size=0.2, random_state=42)
    print(f"[1] Split: train={len(train_df)} rows, test={len(test_df)} rows")

    if DEBUG:
        print(f"[DEBUG] Train columns: {len(train_df.columns)}")
        print(f"[DEBUG] Test columns: {len(test_df.columns)}")
        print(f"[DEBUG] Columns match: {list(train_df.columns) == list(test_df.columns)}")

    # Step 4: Fit preprocessor with debugging
    print("\n[2] Fitting preprocessor with enhanced debugging...")
    X_train_np, y_train, preproc = fit_preprocessor(
        train_df,
        schema=schema,
        model_type="linear",
        numerical_imputation=dev_cfg.numerical_imputation,
        debug=True,
        quantiles=dev_cfg.quantile_clipping,
        max_safe_rows=dev_cfg.max_safe_rows,
        apply_type_conversions=dev_cfg.apply_type_conversions,
        drop_unexpected_schema_columns=dev_cfg.drop_unexpected_columns,
    )


    feat_names = preproc.get_feature_names_out()
    print(f"[2] Preprocessor fitted: {X_train_np.shape} features generated")

    # Step 5: Feature store setup
    fs = FeatureStore(model_family=model_family, target=target)
    try:
        prod_spec_before = fs.load_spec(stage="Production")
        prod_hash_before = spec_hash(prod_spec_before)
        print(f"[3] Existing Production spec found (hash={prod_hash_before})")
    except Exception:
        prod_spec_before = None
        prod_hash_before = None
        print("[3] No existing Production spec")

    # Step 6: Run enhanced propose with debugging
    selection_cfg = SelectionConfig(
        perm_n_repeats=dev_cfg.perm_n_repeats,
        perm_max_samples=dev_cfg.perm_max_samples,
        perm_n_jobs=dev_cfg.perm_n_jobs,
        perm_threshold=dev_cfg.perm_threshold,
        shap_nsamples=dev_cfg.shap_nsamples,
        shap_threshold=dev_cfg.shap_threshold,
        mode=dev_cfg.selection_mode,
        max_relative_regression=dev_cfg.max_relative_regression,
        min_features=dev_cfg.min_features,
        max_features=dev_cfg.max_features,
        fallback_strategy=dev_cfg.fallback_strategy,
    )
    print(f"selection_cfg: {selection_cfg}")

    print(f"\n[4] Running enhanced feature selection...")
    staging_spec, staging_rmse = propose_feature_spec(
        train_df=train_df,
        test_df=test_df,
        schema=schema,
        preprocessor=preproc,
        selection_cfg=selection_cfg,
        model_family=model_family,
        fs=fs,
        debug=DEBUG,  # Enable full debugging
    )

    staging_hash = spec_hash(staging_spec)
    print(f"[4] Staging spec created (hash={staging_hash}), RMSE={staging_rmse:.4f}")

    # Step 7: Validate staging spec
    if not staging_spec.feature_selection or not staging_spec.feature_selection.get("final_features"):
        raise AssertionError("Staging spec missing final_features")

    final_features = staging_spec.feature_selection["final_features"]
    if len(final_features) == 0:
        print("WARNING: Staging spec has empty final feature list")
    else:
        print(f"[5] Staging spec contains {len(final_features)} final features")

    # Step 8: Enhanced gating with debugging
    print(f"\n[6] Running enhanced gating process...")
    gate_report = gate_and_promote_spec(
        train_df=train_df,
        test_df=test_df,
        schema=schema,
        preprocessor=preproc,
        fs=fs,
        selection_cfg=selection_cfg,
        debug=DEBUG,
    )
    print(f"[6] Gate report: {json.dumps(gate_report, indent=2)}")

    # Step 9: Final validation
    try:
        prod_spec_after = fs.load_spec(stage="Production")
        prod_hash_after = spec_hash(prod_spec_after)
    except Exception as e:
        raise RuntimeError(f"Failed to load Production spec after gating: {e}")

    decision = gate_report["decision"]
    print(f"\n[7] Final validation:")
    print(f"    Decision: {decision}")
    print(f"    Production hash before: {prod_hash_before}")
    print(f"    Production hash after: {prod_hash_after}")
    print(f"    Staging RMSE: {staging_rmse:.4f}")

    # Step 10: End-to-end inference test
    print(f"\n[8] Testing end-to-end inference...")
    try:
        X_test_tr, y_test_tr = transform_preprocessor(test_df, preproc, schema, debug=DEBUG)
        feat_names = preproc.get_feature_names_out()
        X_test_encoded = pd.DataFrame(X_test_tr, columns=feat_names, index=test_df.index)

        X_infer = select_model_features(X_test_encoded, prod_spec_after)
        print(f"[8] Inference feature selection: {X_infer.shape[1]} columns")

        # Test with baseline model
        X_train_tr, y_train_tr = transform_preprocessor(train_df, preproc, schema, debug=DEBUG) 
        X_train_encoded = pd.DataFrame(X_train_tr, columns=feat_names, index=train_df.index)
        X_train_sel = select_model_features(X_train_encoded, prod_spec_after)

        baseline_model = train_baseline_model(X_train_sel, y_train_tr, debug=DEBUG)
        preds = safe_model_predict(baseline_model, X_infer, "End-to-end test", debug=DEBUG)

        prod_rmse = float(np.sqrt(mean_squared_error(test_df[target], preds)))
        print(f"[8] End-to-end inference RMSE: {prod_rmse:.4f}")

    except Exception as e:
        print(f"[8] ERROR in end-to-end test: {e}")
        raise

    print("\n=== ENHANCED SMOKE TEST SUMMARY ===")
    print(f"✅ All steps completed successfully")
    print(f"✅ Feature selection pipeline validated")
    print(f"✅ No feature name mismatches detected")
    print(f"Final Production RMSE: {prod_rmse:.4f}")

    if DEBUG:
        print(f"\nDEBUG SUMMARY:")
        print(f"- Enhanced debugging enabled throughout")
        print(f"- Feature name consistency validated at each step") 
        print(f"- Comprehensive error handling implemented")
        print(f"- End-to-end pipeline tested successfully")
