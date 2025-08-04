"""
Enhanced preprocessing module with comprehensive debugging and robust feature name handling.
Addresses feature name mismatches and adds configurable numerical imputation.
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from api.src.ml.column_schema import (
    load_schema_from_yaml, SchemaConfig, prune_dataframe_to_schema,
    extract_feature_lists_from_schema, SchemaValidationError
)

import json
from api.src.ml.preprocessing.data_prep import compute_clip_bounds, clip_extreme_ev, filter_and_clip

# ───────────────────────────────────────────────────────────────────────
# Enhanced utility functions with better debugging
# ───────────────────────────────────────────────────────────────────────

def is_categorical_like(dtype):
    """Modern replacement for deprecated pd.api.types.is_categorical_dtype"""
    return (isinstance(dtype, pd.CategoricalDtype) or 
            pd.api.types.is_object_dtype(dtype) or
            pd.api.types.is_string_dtype(dtype))

def drop_all_null_columns(df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns that are entirely null before passing to transformers."""
    all_null = df.columns[df.isna().all()].tolist()
    if verbose and all_null:
        print(f"[drop_all_null_columns] Dropping {len(all_null)} all-null columns: {all_null}")
    return df.drop(columns=all_null), all_null

def sanitize_missingness_only(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """Minimal normalization: replace pandas pd.NA with np.nan."""
    out = df.copy()

    problematic = []
    for col in out.columns:
        dtype = out[col].dtype
        if pd.api.types.is_string_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            problematic.append((col, str(dtype)))
    if debug and problematic:
        print(f"[sanitize_missingness_only] Extension/string dtypes found: {len(problematic)} columns")

    # Replace explicit pandas NA
    out = out.replace({pd.NA: np.nan})

    if debug:
        total_cells = out.size
        nan_count = out.isna().sum().sum()
        print(f"[sanitize_missingness_only] Total NaN values: {nan_count} ({nan_count/total_cells:.2%})")

    return out

def debug_feature_names_flow(ct: ColumnTransformer, X: pd.DataFrame, stage: str = "", debug: bool = False):
    """Debug feature name generation in ColumnTransformer"""
    if not debug:
        return

    print(f"\n[debug_feature_names_flow] {stage}")
    print(f"Input X shape: {X.shape}")
    print(f"Input columns (first 10): {list(X.columns[:10])}")

    try:
        if hasattr(ct, 'transformers_'):
            # Fitted transformer
            total_features = 0
            for name, transformer, cols in ct.transformers_:
                if cols == 'drop' or not cols:
                    continue
                if isinstance(cols, list):
                    col_count = len(cols)
                else:
                    col_count = 1
                print(f"  Transformer '{name}': {col_count} input cols -> transformer type: {type(transformer).__name__}")
                total_features += col_count

            feat_names = ct.get_feature_names_out()
            print(f"Generated feature names: {len(feat_names)} total")
            print(f"First 10 feature names: {list(feat_names[:10])}")
            print(f"Last 10 feature names: {list(feat_names[-10:])}")
        else:
            print("  Transformer not yet fitted")
    except Exception as e:
        print(f"  Error getting feature names: {e}")

def debug_fit_transform(ct: ColumnTransformer, X: pd.DataFrame, y: pd.Series, debug: bool = False) -> np.ndarray:
    """Enhanced fit_transform with comprehensive debugging"""
    if not debug:
        return ct.fit_transform(X, y)

    print(f"\n[debug_fit_transform] Starting fit_transform on X shape {X.shape}")
    print(f"Input columns sample: {list(X.columns[:10])}...")

    # 1) Fit the transformer
    print("[debug_fit_transform] Fitting transformer...")
    ct.fit(X, y)

    # 2) Debug each fitted transformer
    debug_feature_names_flow(ct, X, "After fitting", debug=True)

    # 3) Test individual transforms
    transformers_fitted = getattr(ct, "transformers_", [])
    for name, fitted_transformer, cols in transformers_fitted:
        if name == "remainder" or cols == "drop" or not cols:
            continue

        col_list = list(cols) if isinstance(cols, (list, tuple)) else [cols]
        missing_cols = [c for c in col_list if c not in X.columns]
        if missing_cols:
            print(f"[debug_fit_transform] ERROR: '{name}' missing columns: {missing_cols}")
            continue

        try:
            subset = X[col_list]
            transformed = fitted_transformer.transform(subset)
            shape_info = transformed.shape if hasattr(transformed, 'shape') else 'unknown'
            print(f"[debug_fit_transform] '{name}': {len(col_list)} → {shape_info}")

            if hasattr(transformed, 'shape') and len(transformed.shape) == 2 and transformed.shape[1] == 0:
                print(f"[debug_fit_transform] WARNING: '{name}' produced zero features!")

        except Exception as e:
            print(f"[debug_fit_transform] ERROR in '{name}': {e}")
            raise

    # 4) Full transform
    try:
        result = ct.transform(X)
        print(f"[debug_fit_transform] SUCCESS: Final shape {result.shape}")
        return result
    except Exception as e:
        print(f"[debug_fit_transform] FINAL TRANSFORM FAILED: {e}")
        raise

def build_robust_preprocessor(
    numerical_cols: list[str],
    ordinal_cols: list[str], 
    nominal_cols: list[str],
    ordinal_categories: list[list[str]] = None,
    model_type: str = "linear",
    numerical_imputation: str = "median",  # NEW: configurable imputation
    debug: bool = False
) -> ColumnTransformer:
    """
    Build preprocessor with configurable imputation strategies.

    Args:
        numerical_imputation: 'mean', 'median', or 'iterative'
    """
    transformers = []

    # Enhanced numerical pipeline with configurable imputation
    if numerical_cols:
        if numerical_imputation == "mean":
            num_imputer = SimpleImputer(strategy="mean", add_indicator=True, missing_values=np.nan)
        elif numerical_imputation == "median":
            num_imputer = SimpleImputer(strategy="median", add_indicator=True, missing_values=np.nan)
        elif numerical_imputation == "iterative":
            num_imputer = IterativeImputer(random_state=0, add_indicator=True)
        else:
            # Default for linear models
            num_imputer = SimpleImputer(strategy="median", add_indicator=True, missing_values=np.nan)

        num_pipeline = Pipeline([
            ("impute", num_imputer),
            ("scale", StandardScaler())
        ])
        transformers.append(("num", num_pipeline, numerical_cols))
        if debug:
            print(f"[build_robust_preprocessor] Numerical: {len(numerical_cols)} cols, imputation={numerical_imputation}")
    else:
        if debug:
            print("[build_robust_preprocessor] WARNING: No numerical columns")

    # Ordinal pipeline
    if ordinal_cols:
        if ordinal_categories:
            ord_enc = OrdinalEncoder(
                categories=ordinal_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype="int32"
            )
        else:
            ord_enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype="int32"
            )
        ord_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING", add_indicator=True)),
            ("encode", ord_enc),
        ])
        transformers.append(("ord", ord_pipeline, ordinal_cols))
        if debug:
            print(f"[build_robust_preprocessor] Ordinal: {len(ordinal_cols)} cols")
    else:
        if debug:
            print("[build_robust_preprocessor] WARNING: No ordinal columns")

    # Nominal pipeline - force dense output
    if nominal_cols:
        nom_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("encode", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("nom", nom_pipeline, nominal_cols))
        if debug:
            print(f"[build_robust_preprocessor] Nominal: {len(nominal_cols)} cols")
    else:
        if debug:
            print("[build_robust_preprocessor] WARNING: No nominal columns")

    if not transformers:
        raise ValueError("No transformers configured; all column groups are empty.")

    # CRITICAL: Force consistent feature naming
    ct_kwargs = {
        "remainder": "drop",
        "verbose_feature_names_out": True,  # CHANGED: Force verbose names for consistency
        "sparse_threshold": 0.0  # Force dense output
    }

    if debug:
        print(f"[build_robust_preprocessor] ColumnTransformer config: {ct_kwargs}")

    ct = ColumnTransformer(transformers, **ct_kwargs)

    if debug:
        print(f"[build_robust_preprocessor] Built with {len(transformers)} transformers: {[name for name, _, _ in transformers]}")

    return ct

def fit_preprocessor(
    df: pd.DataFrame,
    schema: "SchemaConfig",
    model_type: str = "linear",
    numerical_imputation: str = "median",  # NEW parameter
    debug: bool = False,
    quantiles: tuple[float, float] = (0.01, 0.99),
    max_safe_rows: int = 200000,
    apply_type_conversions: bool = True,
    drop_unexpected_schema_columns: bool = True,
) -> tuple[np.ndarray, pd.Series, ColumnTransformer]:
    """Enhanced fit_preprocessor with debugging and configurable imputation"""

    if len(df) > max_safe_rows:
        warnings.warn(f"Dataset has {len(df)} rows (>{max_safe_rows}); ensure training data only.")

    target_col = schema.target()
    if target_col is None:
        raise ValueError("Schema has no target defined.")

    if debug:
        print(f"\n[fit_preprocessor] === STARTING PREPROCESSING ===")
        print(f"Input shape: {df.shape}, Target: {target_col}")
        print(f"Numerical imputation: {numerical_imputation}")

    if debug:
        # Validation and cleaning steps (same as before)
        try:
            schema.validate_dataframe(df, strict=False, debug=debug)
        except Exception as e:
            if debug:
                print(f"[fit_preprocessor] Schema validation warning: {e}")

    df = prune_dataframe_to_schema(df, schema, drop_unexpected=drop_unexpected_schema_columns, debug=debug)

    # Domain cleaning
    try:
        df_cleaned, (lower, upper) = filter_and_clip(df.copy(), schema=schema, quantiles=quantiles, debug=debug)
    except NameError:
        if debug:
            print("[fit_preprocessor] filter_and_clip not found; using bounds directly")
        lower, upper = compute_clip_bounds(df[target_col], method="quantile", quantiles=quantiles, debug=debug)
        df_cleaned = clip_extreme_ev(df.copy(), velo_col=target_col, lower=lower, upper=upper, debug=debug)

    # Data cleaning
    df_cleaned = sanitize_missingness_only(df_cleaned, debug=debug)
    df_converted = _apply_schema_aware_conversions(df_cleaned, schema, apply_type_conversions=apply_type_conversions, debug=debug)

    # Strict validation
    if debug:
        validation_report = schema.validate_dataframe(df_converted, strict=True, debug=debug)
        print(f"[fit_preprocessor] Schema validation: {len(validation_report.get('ok', []))} OK columns")

    # Feature extraction
    num_feats = [c for c in schema.numerical() if c != target_col and c in df_converted.columns]
    ord_feats = [c for c in schema.ordinal() if c in df_converted.columns]
    nom_feats = [c for c in schema.nominal() if c in df_converted.columns]

    if debug:
        print(f"[fit_preprocessor] Feature groups: Num={len(num_feats)}, Ord={len(ord_feats)}, Nom={len(nom_feats)}")

    # Build feature matrix
    all_feature_cols = num_feats + ord_feats + nom_feats
    X = df_converted[all_feature_cols].copy()
    y = df_converted[target_col]

    if debug:
        print(f"[fit_preprocessor] Feature matrix: {X.shape}, Target: {y.shape}")
        print(f"First 10 feature columns: {list(X.columns[:10])}")

    # Drop all-null columns
    X, dropped = drop_all_null_columns(X, verbose=debug)
    if dropped:
        # Update feature lists
        num_feats = [c for c in num_feats if c not in dropped]
        ord_feats = [c for c in ord_feats if c not in dropped]
        nom_feats = [c for c in nom_feats if c not in dropped]

    # Prepare ordinal categories
    ordinal_categories = []
    for c in ord_feats:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("MISSING").astype(str)
            uniques = sorted(set(X[c].unique()))
            if "MISSING" not in uniques:
                uniques.append("MISSING")
            ordinal_categories.append(uniques)
        else:
            ordinal_categories.append(["MISSING"])

    # Build preprocessor with enhanced debugging
    ct = build_robust_preprocessor(
        numerical_cols=num_feats,
        ordinal_cols=ord_feats,
        nominal_cols=nom_feats,
        ordinal_categories=ordinal_categories if ord_feats else None,
        model_type=model_type,
        numerical_imputation=numerical_imputation,  # Pass through new parameter
        debug=debug
    )

    # Store metadata for downstream use
    ct.lower_, ct.upper_ = lower, upper
    ct.numeric_feature_names_ = list(num_feats)
    ct.ordinal_feature_names_ = list(ord_feats)  
    ct.nominal_feature_names_ = list(nom_feats)
    ct.bounds_meta_ = {"shape": "scalar", "target": target_col}
    ct.original_feature_order_ = all_feature_cols  # NEW: Store original order

    # Normalize NA values
    X = X.replace({pd.NA: np.nan})

    # Fit and transform with enhanced debugging
    X_mat = debug_fit_transform(ct, X, y, debug=debug)

    if debug:
        print(f"[fit_preprocessor] === PREPROCESSING COMPLETE ===")
        print(f"Final output shape: {X_mat.shape}")
        feature_names = ct.get_feature_names_out()
        print(f"Feature names sample: {list(feature_names[:5])}...{list(feature_names[-5:])}")

    return X_mat, y, ct

def transform_preprocessor(
    df: pd.DataFrame,
    transformer: ColumnTransformer,
    schema: "SchemaConfig",
    debug: bool = False,
    apply_type_conversions: bool = True,
    drop_unexpected_schema_columns: bool = True,
) -> tuple[np.ndarray, pd.Series]:
    """Enhanced transform_preprocessor with better debugging"""

    target_col = schema.target()
    if target_col is None:
        raise ValueError("Schema has no target defined.")

    if debug:
        print(f"\n[transform_preprocessor] === TRANSFORMING DATA ===")
        print(f"Input shape: {df.shape}")

    # Use the same preprocessing steps as fit
    df = prune_dataframe_to_schema(df, schema, drop_unexpected=drop_unexpected_schema_columns, debug=debug)

    # Apply bounds from training
    from api.src.ml.preprocessing.data_prep import resolve_clip_bounds_map, apply_clip_bounds_map

    raw_features = list(df.columns)
    numeric_features = [c for c in schema.numerical() if c in df.columns]
    lower = getattr(transformer, "lower_", None)
    upper = getattr(transformer, "upper_", None)

    bounds_map = resolve_clip_bounds_map(
        raw_feature_names=raw_features,
        numeric_feature_names=numeric_features,
        lower=lower,
        upper=upper,
        target_col=target_col,
        debug=debug
    )
    df_filtered = apply_clip_bounds_map(df.copy(), bounds_map, debug=debug)

    # Same cleaning as fit
    df_filtered = sanitize_missingness_only(df_filtered, debug=debug)
    df_converted = _apply_schema_aware_conversions(df_filtered, schema, apply_type_conversions=apply_type_conversions, debug=debug)

    if debug:
        # Validate
        schema.validate_dataframe(df_converted, strict=True, debug=debug)

    # Extract features using the SAME ORDER as training
    original_order = getattr(transformer, 'original_feature_order_', None)
    if original_order:
        # Use the exact same feature order as training
        available_feats = [c for c in original_order if c in df_converted.columns]
        X = df_converted[available_feats].copy()
        if debug:
            print(f"[transform_preprocessor] Using stored feature order: {len(available_feats)} features")
    else:
        # Fallback to schema order
        num_feats = [c for c in schema.numerical() if c != target_col and c in df_converted.columns]
        ord_feats = [c for c in schema.ordinal() if c in df_converted.columns]
        nom_feats = [c for c in schema.nominal() if c in df_converted.columns]
        X = df_converted[num_feats + ord_feats + nom_feats].copy()
        if debug:
            print(f"[transform_preprocessor] Using schema order: Num={len(num_feats)}, Ord={len(ord_feats)}, Nom={len(nom_feats)}")

    y = df_converted[target_col]

    # Handle ordinal columns consistently
    ord_feats = getattr(transformer, 'ordinal_feature_names_', [])
    for c in ord_feats:
        if c in X.columns:
            X[c] = X[c].astype("string").fillna("MISSING").astype(str)

    # Normalize NA
    X = X.replace({pd.NA: np.nan})

    # Transform with debugging
    if debug:
        print(f"[transform_preprocessor] Pre-transform X shape: {X.shape}")
        debug_feature_names_flow(transformer, X, "Before transform", debug=True)

    X_mat = transformer.transform(X)

    if debug:
        print(f"[transform_preprocessor] Post-transform shape: {X_mat.shape}")
        print("[transform_preprocessor] === TRANSFORM COMPLETE ===")

    return X_mat, y

def _apply_schema_aware_conversions(
    df: pd.DataFrame,
    schema: "SchemaConfig",
    *,
    apply_type_conversions: bool = True,
    debug: bool = False
) -> pd.DataFrame:
    """Apply safe, schema-aware dtype conversions with enhanced debugging"""
    if not apply_type_conversions:
        if debug:
            print("[_apply_schema_aware_conversions] Skipping conversions (disabled)")
        return df

    df = df.copy()

    # Numeric columns: convert to numeric if stored as object/string
    for col in schema.numerical():
        if col not in df.columns:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            if debug:
                print(f"[_apply_schema_aware_conversions] Converting numeric '{col}' from {dtype}")
            coerced = pd.to_numeric(df[col], errors="coerce")
            num_na_before = df[col].isna().sum()
            num_na_after = coerced.isna().sum()
            df[col] = coerced
            if debug and num_na_after > num_na_before:
                print(f"  → Created {num_na_after - num_na_before} new NaN values")

    # Categorical columns: convert to string if numeric
    for col in schema.nominal() + schema.ordinal():
        if col not in df.columns:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            if debug:
                print(f"[_apply_schema_aware_conversions] Converting categorical '{col}' from {dtype} to string")
            df[col] = df[col].astype("string")

    return df

# Keep existing functions for compatibility
def inverse_transform_preprocessor(
    X_trans: np.ndarray,
    transformer: ColumnTransformer
) -> pd.DataFrame:
    """Enhanced inverse transform with better error handling"""
    import warnings
    from sklearn.impute import MissingIndicator

    # Recover original feature order
    orig_features: list[str] = []
    for name, _, cols in transformer.transformers_:
        if cols == 'drop': 
            continue
        orig_features.extend(cols)

    parts = []
    start = 0
    n_rows = X_trans.shape[0]

    # Process each transformer block
    for name, trans, cols in transformer.transformers_:
        if cols == 'drop':
            continue

        fitted = transformer.named_transformers_[name]

        # Determine output columns
        dummy = np.zeros((1, len(cols)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                out = fitted.transform(dummy)
            except Exception:
                out = dummy
        n_out = out.shape[1]

        # Extract block
        block = X_trans[:, start : start + n_out]
        start += n_out

        # Handle different transformer types
        if isinstance(fitted, MissingIndicator):
            continue  # Skip missing indicators
        elif trans == 'passthrough':
            inv = block
        elif name == 'num':
            # Handle numerical pipeline with scaling
            scaler = fitted.named_steps['scale']
            inv_full = scaler.inverse_transform(block)
            inv = inv_full[:, :len(cols)]  # Remove missing indicators
        else:
            # Ordinal or nominal pipelines
            if isinstance(fitted, Pipeline):
                last = list(fitted.named_steps.values())[-1]
                inv = last.inverse_transform(block)
            else:
                inv = fitted.inverse_transform(block)

        parts.append(pd.DataFrame(inv, columns=cols, index=range(n_rows)))

    # Concatenate and reorder
    df_orig = pd.concat(parts, axis=1)
    return df_orig[orig_features]

# Keep legacy pipeline definitions for compatibility  
numeric_linear = Pipeline([
    ('impute', SimpleImputer(strategy='median', add_indicator=True)),
    ('scale', StandardScaler()),
])

numeric_iterative = Pipeline([
    ('impute', IterativeImputer(random_state=0, add_indicator=True)),
    ('scale', StandardScaler()),
])

nominal_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('encode', OneHotEncoder(drop='first', handle_unknown='ignore')),
])

# Keep other helper functions unchanged
def prepare_for_mixed_and_hierarchical(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans data for hierarchical and mixed-effects models."""
    cols = _ColumnSchema()
    TARGET = cols.target()
    df, _ = filter_and_clip(df.copy())
    df["batter_id"] = df["batter_id"].astype("category")
    df["season_cat"] = df["season"].astype("category")
    df["season_idx"] = df["season_cat"].cat.codes
    df["pitcher_cat"] = df["pitcher_id"].astype("category")
    df["pitcher_idx"] = df["pitcher_cat"].cat.codes
    return df

def prepare_for_pymc(df: pd.DataFrame, predictor: str, target: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Prepare data for PyMC models."""
    df_clean, _bounds = filter_and_clip(df.copy(), debug=False)
    df_clean = df_clean.dropna(subset=[predictor, target]).reset_index(drop=True)
    x = df_clean[predictor].values
    y = df_clean[target].values
    return df_clean, x, y



if __name__ == "__main__":
    from pathlib import Path
    from api.src.ml.features.load_data_utils import load_data_optimized
    from api.src.ml import config
    from api.src.ml.features.feature_engineering import engineer_features
    from api.src.ml.column_schema import load_schema_from_yaml
    import json

    # Ensure you invoke this from project root so `api` is importable, or set PYTHONPATH appropriately.
    schema_path = Path("api/src/ml/column_schema.yaml")
    schema = load_schema_from_yaml(schema_path)

    # Load data
    data_path = config.FINAL_ENGINEERED_DATASET_DIR / "final_merged_with_all.parquet"
    df = load_data_optimized(data_path)
    df_eng, _ = engineer_features(df)

    # Split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df_eng, test_size=0.2, random_state=42)

    # Run fit/transform
    train_df, test_df = train_test_split(df_eng, test_size=0.2, random_state=42)
    X_train, y_train, ct = fit_preprocessor(train_df, schema=schema, model_type="linear", debug=False)
    X_test, y_test = transform_preprocessor(test_df, transformer=ct, schema=schema, debug=False)

    # ----------Add save and load here with feature store--------

    # Example of inverse transform: 
    print("==========Example of inverse transform:==========")
    df_back = inverse_transform_preprocessor(X_train, ct)
    print("\n✅ Inverse‐transformed head (should mirror your original X_train):")
    print(df_back.head())
    print("Shape:", df_back.shape, "→ original X_train shape before transform:", X_train.shape)
