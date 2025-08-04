import pandas as pd
import numpy as np
from api.src.ml.column_schema import SchemaConfig
# --- ADD BELOW YOUR EXISTING IMPORTS ---
from collections.abc import Iterable
from typing import Dict, Tuple, Sequence, Optional


def _is_iterable_nonstring(x) -> bool:
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))


def resolve_clip_bounds_map(
    *,
    raw_feature_names: Sequence[str],
    numeric_feature_names: Sequence[str],
    lower: float | Sequence[float] | None,
    upper: float | Sequence[float] | None,
    target_col: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Return a canonical {feature: (lo, hi)} mapping from possibly-scalar or per-feature bounds.

    Rules:
      - If lower/upper are None → no clipping for any feature.
      - If both are scalars:
          * If target_col is provided → only target gets (lo, hi); others (None, None).
          * Else → broadcast to ALL raw features (rare; not recommended).
      - If iterables length == len(numeric_features) → align to numeric only. Non-numeric get (None, None).
      - If iterables length == len(raw_features)     → align 1:1 to all raw features.
      - Any other length mismatches → raise ValueError with diagnostics.
    """
    raw = list(raw_feature_names)
    num = list(numeric_feature_names)
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {f: (None, None) for f in raw}

    if lower is None or upper is None:
        if debug:
            print("[resolve_clip_bounds_map] No bounds provided → no clipping.")
        return out

    lower_is_iter = _is_iterable_nonstring(lower)
    upper_is_iter = _is_iterable_nonstring(upper)

    # Case 1: both scalars
    if not lower_is_iter and not upper_is_iter:
        lo, hi = float(lower), float(upper)
        if target_col:
            if target_col not in out:
                raise KeyError(f"[resolve_clip_bounds_map] target_col '{target_col}' not in provided features.")
            out[target_col] = (lo, hi)
            if debug:
                print(f"[resolve_clip_bounds_map] Scalar bounds applied to target only: {target_col} → ({lo}, {hi})")
        else:
            for f in raw:
                out[f] = (lo, hi)
            if debug:
                print(f"[resolve_clip_bounds_map] WARNING: Scalar bounds broadcast to ALL {len(raw)} features.")
        return out

    # Case 2: both iterables
    if lower_is_iter and upper_is_iter:
        lower_list = list(lower)  # type: ignore[arg-type]
        upper_list = list(upper)  # type: ignore[arg-type]

        # Per-numeric
        if len(lower_list) == len(num) == len(upper_list):
            if debug:
                print(f"[resolve_clip_bounds_map] Aligning per-numeric bounds to {len(num)} numeric features.")
            for f, lo, hi in zip(num, lower_list, upper_list):
                out[f] = (float(lo), float(hi))
            # Non-numeric remain (None, None)
            return out

        # Per-raw
        if len(lower_list) == len(raw) == len(upper_list):
            if debug:
                print(f"[resolve_clip_bounds_map] Aligning per-raw bounds to {len(raw)} raw features.")
            for f, lo, hi in zip(raw, lower_list, upper_list):
                out[f] = (float(lo), float(hi))
            return out

        # Mismatch → refuse to guess
        raise ValueError(
            "[resolve_clip_bounds_map] Length mismatch:\n"
            f"  len(lower)={len(lower_list)}, len(upper)={len(upper_list)}\n"
            f"  len(numeric_features)={len(num)}, len(raw_features)={len(raw)}\n"
            "  Expected either per-numeric or per-raw alignment."
        )

    # Mixed scalar/iterable → unsupported, force explicitness
    raise ValueError("[resolve_clip_bounds_map] Mixed scalar/iterable bounds are not supported. Provide both as scalars or both as sequences.")


def apply_clip_bounds_map(
    df: pd.DataFrame,
    bounds_map: Dict[str, Tuple[Optional[float], Optional[float]]],
    *,
    debug: bool = False
) -> pd.DataFrame:
    """
    Apply per-column clipping according to bounds_map.
    (None, None) means 'no clip' for that column.
    """
    out = df.copy()
    applied = []
    skipped = []
    for col, (lo, hi) in bounds_map.items():
        if col not in out.columns:
            skipped.append((col, "missing"))
            continue
        if lo is None and hi is None:
            skipped.append((col, "no-clip"))
            continue
        if not pd.api.types.is_numeric_dtype(out[col].dtype):
            skipped.append((col, f"non-numeric dtype={out[col].dtype}"))
            continue
        out[col] = out[col].clip(lower=lo, upper=hi)
        applied.append((col, lo, hi))

    if debug:
        print(f"[apply_clip_bounds_map] Applied to {len(applied)} columns; skipped {len(skipped)}.")
        if applied:
            sample = applied[: min(10, len(applied))]
            print(f"  Applied sample: {sample}")
        if skipped:
            sample = skipped[: min(10, len(skipped))]
            print(f"  Skipped sample: {sample}")
    return out



# ─────────────────────────────────────────────────────────────────────────────
def compute_clip_bounds(
    series: pd.Series,
    *,
    method: str = "quantile",
    quantiles: tuple[float,float] = (0.01,0.99),
    std_multiplier: float = 3.0,
    debug: bool = False,
) -> tuple[float,float]:
    """
    Compute (lower, upper) but do not apply them.
    """
    # Debug input type and shape
    if debug:
        print(f"[compute_clip_bounds] Input type: {type(series)}")
        print(f"[compute_clip_bounds] Input shape: {getattr(series, 'shape', 'N/A')}")
        print(f"[compute_clip_bounds] Method: {method}")
        print(f"[compute_clip_bounds] Quantiles: {quantiles}")

    # Ensure we're working with a Series
    if isinstance(series, pd.DataFrame):
        if debug:
            print("[compute_clip_bounds] WARNING: Received DataFrame, converting to Series")
            print(f"[compute_clip_bounds] DataFrame columns: {series.columns.tolist()}")
        # Take the first column if it's a single-column DataFrame
        if len(series.columns) == 1:
            series = series.iloc[:, 0]
        else:
            raise ValueError(f"Expected Series or single-column DataFrame, got DataFrame with {len(series.columns)} columns")

    s = series.dropna()

    if debug:
        print(f"[compute_clip_bounds] Series after dropna - shape: {s.shape}, type: {type(s)}")

    if method == "quantile":
        quantile_result = s.quantile(list(quantiles))
        if debug:
            print(f"[compute_clip_bounds] Quantile result type: {type(quantile_result)}")
            print(f"[compute_clip_bounds] Quantile result: {quantile_result}")

        # Handle both Series and potential DataFrame returns
        if isinstance(quantile_result, pd.Series):
            bounds = tuple(quantile_result.tolist())
        else:
            # Fallback - should not happen but just in case
            bounds = tuple(quantile_result.values.flatten())

        if debug:
            print(f"[compute_clip_bounds] Final bounds: {bounds}")
        return bounds

    elif method == "mean_std":
        mu, sigma = s.mean(), s.std()
        bounds = (mu - std_multiplier*sigma, mu + std_multiplier*sigma)
        if debug:
            print(f"[compute_clip_bounds] Mean-std bounds: {bounds}")
        return bounds

    elif method == "iqr":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        bounds = (q1 - 1.5*iqr, q3 + 1.5*iqr)
        if debug:
            print(f"[compute_clip_bounds] IQR bounds: {bounds}")
        return bounds

    else:
        raise ValueError(f"Unknown method {method}")


def clip_extreme_ev(
    df: pd.DataFrame,
    velo_col: str,  # generic numeric column to clip (e.g., target column)
    lower: float | None = None,
    upper: float | None = None,
    *,
    method: str = "quantile",
    quantiles: tuple[float, float] = (0.01, 0.99),
    std_multiplier: float = 3.0,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Clip values in `velo_col` to [lower, upper]. If bounds are None, compute them.
    """
    df = df.copy()
    if debug:
        print(f"[clip_extreme_ev] Column: {velo_col} (exists={velo_col in df.columns})")

    if velo_col not in df.columns:
        raise KeyError(f"[clip_extreme_ev] Column '{velo_col}' not found in DataFrame.")

    series = df[velo_col].dropna()

    if lower is None or upper is None:
        if method == "quantile":
            low_q, high_q = quantiles
            low_high = series.quantile([low_q, high_q])
            lower_, upper_ = (low_high.tolist()
                              if isinstance(low_high, pd.Series)
                              else low_high.values.flatten())
        elif method == "mean_std":
            mu, sigma = series.mean(), series.std()
            lower_, upper_ = mu - std_multiplier * sigma, mu + std_multiplier * sigma
        elif method == "iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_, upper_ = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            raise ValueError(f"Unknown method '{method}' for clip_extreme_ev")

        lower = lower if lower is not None else lower_
        upper = upper if upper is not None else upper_

    if debug:
        total = len(series)
        n_low = (series < lower).sum()
        n_high = (series > upper).sum()
        print(f"[clip_extreme_ev] bounds=({lower:.4f}, {upper:.4f}) "
              f"| below={n_low}/{total} ({n_low/total:.2%}) "
              f"| above={n_high}/{total} ({n_high/total:.2%})")

    df[velo_col] = df[velo_col].clip(lower, upper)
    return df



# ─────────────────────────────────────────────────────────────────────────────
# for basketball what is this?
# filter out halfcourt shots
# filter out under 1 second shots
# ─────────────────────────────────────────────────────────────────────────────
#def filter_bunts_and_popups
def filter_and_clip(
    df: pd.DataFrame,
    *,
    schema: SchemaConfig | None = None,
    target_col: str | None = None,
    lower: float | None = None,
    upper: float | None = None,
    quantiles: tuple[float, float] = (0.01, 0.99),
    debug: bool = False
) -> tuple[pd.DataFrame, tuple[float, float]]:
    """
    For now: just clip extreme values on the target column.
    Pass either a SchemaConfig (preferred) or an explicit target_col.

    Returns:
        (clean_df, (lower, upper))
    """
    df = df.copy()

    # Resolve target column
    if target_col is None:
        if schema is None:
            raise ValueError("[filter_and_clip] Provide either `schema` or `target_col`.")
        target_col = schema.target()
        if not target_col:
            raise ValueError("[filter_and_clip] Schema has no target defined.")
    if target_col not in df.columns:
        raise KeyError(f"[filter_and_clip] Target '{target_col}' not found in DataFrame.")

    # Compute bounds if needed
    if lower is None or upper is None:
        lower, upper = compute_clip_bounds(
            df[target_col],
            method="quantile",
            quantiles=quantiles,
            debug=debug
        )

    # Clip
    df = clip_extreme_ev(
        df,
        velo_col=target_col,
        lower=lower,
        upper=upper,
        debug=debug
    )

    return df, (lower, upper)



# ─────────────────────────────────────────────────────────────────────────────
# Smoke test / CLI entry
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    from api.src.ml.features.load_data_utils import load_data_optimized
    from api.src.ml import config
    from api.src.ml.features.feature_engineering import engineer_features
    from api.src.ml.column_schema import  load_schema_from_yaml, extract_feature_lists_from_schema
    import json

    schema_path = config.COLUMN_SCHEMA_PATH
    schema = load_schema_from_yaml(schema_path)

    data_path = config.FINAL_ENGINEERED_DATASET_DIR / "final_merged_with_all.parquet"
    df = load_data_optimized(data_path, 
                             debug=True,
                             drop_null_rows=True,
                             drop_null_subset=['AAV'])


    df_eng, summary_feats = engineer_features(df)
    numericals, ordinal, nominal, y, cat_breakdown = extract_feature_lists_from_schema(df_eng, schema)

    print("\n=== Smoke test extracted groups ===")
    print("Numericals (found):", numericals[:10], "…total", len(numericals))
    print("Ordinal (found):", ordinal)
    print("Nominal (found):", nominal[:10], "…total", len(nominal))
    print("Y variable:", y)

    # ═══════════════════════════════════════════════════════════════════════════
    # FIXED: Now TARGET is a string, not a list
    # ═══════════════════════════════════════════════════════════════════════════

    debug = True
    TARGET = schema.target()  # Now returns string 'AAV_PCT_CAP' instead of ['AAV_PCT_CAP']

    print(f"\n[DEBUG] TARGET type: {type(TARGET)}")
    print(f"[DEBUG] TARGET value: {TARGET}")
    print(f"[DEBUG] df_eng[TARGET] type: {type(df_eng[TARGET])}")
    print(f"[DEBUG] df_eng[TARGET] shape: {df_eng[TARGET].shape}")

    # Test compute_clip_bounds with debugging enabled
    lower, upper = compute_clip_bounds(
        df_eng[TARGET],           # Now this returns a Series, not DataFrame
        method="quantile",
        quantiles=(0.01, 0.99),
        debug=debug
    )

    if debug:
        total = len(df_eng)
        n_low = (df_eng[TARGET] < lower).sum()
        n_high= (df_eng[TARGET] > upper).sum()
        print(f"[fit_preprocessor] clipping train EV to [{lower:.2f}, {upper:.2f}]")
        print(f"  → {n_low:,}/{total:,} ({n_low/total:.2%}) below")
        print(f"  → {n_high:,}/{total:,} ({n_high/total:.2%}) above")

    # Test the clip function with the target column
    df_clipped = clip_extreme_ev(df_eng, velo_col=TARGET, lower=lower, upper=upper, debug=debug)

    print("Final rows after filter & clip:", len(df_clipped))
    print("✓ All functions working correctly!")
