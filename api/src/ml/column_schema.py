"""
Schema-driven preprocessing entrypoint.

Now backed by OmegaConf YAML-driven column definitions. Allows declarative
adjustment of expected columns and enforces presence / dtype categories early.
"""

from enum import Enum
from typing import List, Optional, Dict, Any, Tuple, Set
import pandas as pd
from pandas import CategoricalDtype
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_object_dtype
from pydantic import BaseModel
import json
from omegaconf import OmegaConf
import os
from pathlib import Path
import hashlib
from collections.abc import Iterable
from omegaconf import OmegaConf
from omegaconf import DictConfig


def sanitize_cat_breakdown(cat_breakdown, df, target_col, debug=False):
    """
    Keep only features that are present in df and numerically typed.
    Never coerces/fills. Logs what's dropped for transparency.

    Input:
      cat_breakdown: dict[str, List[str]] - candidate features per category (already resolved to df column names if possible)
      df: pd.DataFrame - the DataFrame to verify against
      target_col: str or None - name of the target column (used only for a dtype warning)
      debug: bool - whether to print diagnostic output about dropped items

    Returns:
      clean: dict[str, List[str]] - filtered mapping with only existing numeric columns
    """
    clean = {}
    for cat, feats in (cat_breakdown or {}).items():
        present = []
        dropped = []
        for f in feats:
            if f not in df.columns:
                dropped.append((f, "missing"))
                continue
            if not is_numeric_dtype(df[f]):
                dropped.append((f, str(df[f].dtype)))
                continue
            present.append(f)
        if present:
            # dedupe preserving order
            seen = set()
            deduped = []
            for p in present:
                if p not in seen:
                    seen.add(p)
                    deduped.append(p)
            clean[cat] = deduped
        if debug and dropped:
            print(f"[sanitize_cat_breakdown] '{cat}': dropped {len(dropped)} non-usable features:")
            for name, why in dropped[:10]:
                print(f"    - {name} ({why})")
            if len(dropped) > 10:
                print(f"    ... {len(dropped) - 10} more")
    # Target dtype check (info only; we don't coerce or fill)
    if target_col in df.columns and not is_numeric_dtype(df[target_col]):
        print(f"[WARNING] target '{target_col}' dtype is {df[target_col].dtype}. Plots that require numeric y will be skipped.")
    return clean

# ----------------------------
# Exceptions
# ----------------------------
class SchemaValidationError(Exception):
    """Raised when a DataFrame violates the declarative schema."""



# Known numerical category canonical names we care about.
_NUMERICAL_CATEGORY_SECTIONS = [
    "general",
    "scoring",
    "advanced",
    "playmaking",
    "rebounding",
    "defense",
    "usage",
    "rankings",
    "team",
    "efficiency",
    "hustle_boxouts",
    "offensive_play_types",
    "defensive_play_types",
    "usage_related",
    "miscellaneous",
]

def _canonicalize_section_name(raw_name: str) -> str:
    """
    Normalize various top-level section names to canonical category names.
    Example: "_general_features" -> "general", "scoring" -> "scoring"
    """
    name = raw_name.lower()
    # strip leading/trailing underscores
    name = name.strip("_")
    # remove trailing "_features" if present
    if name.endswith("_features"):
        name = name[: -len("_features")]
    return name

def _flatten_preserve_order(seq: Any) -> List[Any]:
    """
    Recursively flatten a possibly nested list/sequence while preserving order
    and deduplicating in the flattened result (first occurrence wins).
    """
    seen = set()
    out = []

    def _recurse(item):
        if isinstance(item, dict):
            # Shouldn't happen for our lists, but skip dicts gracefully
            return
        if isinstance(item, str):
            if item not in seen:
                seen.add(item)
                out.append(item)
        elif isinstance(item, Iterable):
            for sub in item:
                _recurse(sub)
        else:
            # scalar non-str (unlikely), coerce to str
            s = str(item)
            if s not in seen:
                seen.add(s)
                out.append(s)

    _recurse(seq)
    return out


# ----------------------------
# Schema machinery (Pydantic)
# ----------------------------
class ColumnCategory(str, Enum):
    ID = "id"
    NUMERIC = "numeric"
    ORDINAL = "ordinal"
    NOMINAL = "nominal"
    TARGET = "target"


class ColumnDefinition(BaseModel):
    name: str
    category: ColumnCategory

    class Config:
        frozen = True  # immutable once created

class SchemaConfig(BaseModel):
    columns: List[ColumnDefinition]
    numerical_categories_raw: Optional[Dict[str, List[str]]] = None  # added field

    # Grouping helpers
    def grouped_names(self) -> Dict[ColumnCategory, List[str]]:
        d: Dict[ColumnCategory, List[str]] = {}
        for col in self.columns:
            d.setdefault(col.category, []).append(col.name)
        return d

    def id(self) -> List[str]:
        return self.grouped_names().get(ColumnCategory.ID, []).copy()

    def numerical(self) -> List[str]:
        return self.grouped_names().get(ColumnCategory.NUMERIC, []).copy()

    def ordinal(self) -> List[str]:
        return self.grouped_names().get(ColumnCategory.ORDINAL, []).copy()

    def nominal(self) -> List[str]:
        return self.grouped_names().get(ColumnCategory.NOMINAL, []).copy()

    def target(self) -> Optional[str]:
        targets = self.grouped_names().get(ColumnCategory.TARGET, [])
        return targets[0] if targets else None

    def categorical(self) -> List[str]:
        """All categorical-like columns in schema-defined order."""
        return self.ordinal() + self.nominal()

    def model_features(self, include_target: bool = False) -> List[str]:
        feats = self.numerical() + self.ordinal() + self.nominal()
        tgt = self.target()
        if not include_target and tgt in feats:
            feats = [c for c in feats if c != tgt]
        return feats

    # numerical categories
    def numerical_categories(self) -> Dict[str, List[str]]:
        """
        Returns the user-defined numerical categories mapping.
        Falls back to empty dict if none provided.
        """
        return self.numerical_categories_raw or {}

    def numerical_by_category(self, category_name: str) -> List[str]:
        return self.numerical_categories().get(category_name, []).copy()

    def all_expected(self) -> List[str]:
        return [col.name for col in self.columns]

    # Diffing
    def diff_columns(self, df: pd.DataFrame) -> Tuple[Set[str], Set[str]]:
        actual = set(df.columns.tolist())
        expected = set(self.all_expected())
        missing = expected - actual
        unexpected = actual - expected
        return missing, unexpected

    # Validation
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        strict: bool = True,
        debug: bool = False
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "missing_columns": [],
            "unexpected_columns": [],
            "dtype_mismatches": {},
            "ok": [],
        }

        missing, unexpected = self.diff_columns(df)
        report["missing_columns"] = sorted(missing)
        report["unexpected_columns"] = sorted(unexpected)

        for col_def in self.columns:
            name = col_def.name
            if name not in df.columns:
                continue
            series = df[name]
            dtype = series.dtype
            category = col_def.category

            ok, reason = self._check_dtype_compatibility(dtype, category)
            if ok:
                report["ok"].append({name: str(dtype)})
                if debug:
                    print(f"[validate_dataframe] ✅ {name} ({category}): dtype={dtype}")
                    print(f"[validate_dataframe] example of {name}: {series.head(3)}")
            else:
                report["dtype_mismatches"][name] = {
                    "expected_category": category.value,
                    "actual_dtype": str(dtype),
                    "reason": reason,
                }
                if debug:
                    print(f"[validate_dataframe] ❌ {name} ({category}): {reason} (dtype={dtype})")

        errors = []
        if strict:
            if report["missing_columns"]:
                errors.append(f"Missing expected columns: {report['missing_columns']}")
            if report["unexpected_columns"]:
                errors.append(f"Unexpected columns: {report['unexpected_columns']}")
            if report["dtype_mismatches"]:
                parts = [
                    f"{name}: expected category '{info['expected_category']}', got dtype {info['actual_dtype']}"
                    for name, info in report["dtype_mismatches"].items()
                ]
                errors.append("Dtype mismatches: " + "; ".join(parts))
        if errors:
            raise SchemaValidationError(" | ".join(errors))

        return report

    def _check_dtype_compatibility(self, dtype: Any, category: ColumnCategory) -> Tuple[bool, str]:
        if category in (ColumnCategory.NUMERIC, ColumnCategory.TARGET):
            if is_numeric_dtype(dtype):
                return True, ""
            else:
                return False, f"expected numeric dtype, got {dtype}"
        elif category in (ColumnCategory.ORDINAL, ColumnCategory.NOMINAL):
            if is_string_dtype(dtype) or is_object_dtype(dtype) or isinstance(dtype, CategoricalDtype):
                return True, ""
            else:
                return False, f"expected categorical-like dtype (object/string/categorical), got {dtype}"
        elif category == ColumnCategory.ID:
            if is_numeric_dtype(dtype) or is_string_dtype(dtype) or is_object_dtype(dtype):
                return True, ""
            else:
                return False, f"expected id-like dtype (string/int), got {dtype}"
        else:
            return False, f"unknown category {category}"

    def to_json(self) -> str:
        return self.json(indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return json.loads(self.json())




def load_schema_from_yaml(path: str) -> SchemaConfig:
    """
    Load column definitions from a YAML file via OmegaConf and build SchemaConfig.

    Supports:
      * canonical nested `numerical_categories:` mapping (including OmegaConf DictConfig).
      * alternate style where each category appears as a top-level list (with flexible naming like
        "_general_features", "scoring", etc.); these are canonicalized and aggregated into numerical_categories.

    Behavior:
      - Prefers the explicit `numerical_categories` key if provided.
      - Falls back to any top-level sections that canonicalize to known category names.
      - Flattens all lists to preserve order and dedupe.
    """
    cfg = OmegaConf.load(path)

    # Debug dump of what was loaded
    try:
        print(f"[load_schema] top-level keys: {list(cfg.keys())}")
        print(f"[load_schema] YAML content:\n{OmegaConf.to_yaml(cfg)}")
    except Exception:
        pass

    raw_id = cfg.get("id", [])
    raw_numerical = cfg.get("numerical", [])
    raw_ordinal = cfg.get("ordinal", [])
    raw_nominal = cfg.get("nominal", [])
    raw_target = cfg.get("target", [])
    raw_num_cats = cfg.get("numerical_categories", None)

    # Flatten base lists
    id_list = _flatten_preserve_order(raw_id)
    numerical_list = _flatten_preserve_order(raw_numerical)
    ordinal_list = _flatten_preserve_order(raw_ordinal)
    nominal_list = _flatten_preserve_order(raw_nominal)
    target_list = _flatten_preserve_order(raw_target)

    numerical_categories_flat: Dict[str, List[str]] = {}

    # 1. Preferred path: explicit numerical_categories key (could be DictConfig)
    if raw_num_cats is not None:
        # Convert OmegaConf container to plain dict if needed
        if isinstance(raw_num_cats, DictConfig):
            raw_num_cats_resolved = OmegaConf.to_container(raw_num_cats, resolve=True)
        else:
            raw_num_cats_resolved = raw_num_cats

        if isinstance(raw_num_cats_resolved, dict):
            for cat_name, items in raw_num_cats_resolved.items():
                flattened = _flatten_preserve_order(items)
                numerical_categories_flat[cat_name] = flattened

    # 2. Fallback: scan top-level keys and canonicalize to known sections if nothing collected yet
    if not numerical_categories_flat:
        for key in cfg.keys():
            canonical = _canonicalize_section_name(key)
            if canonical in _NUMERICAL_CATEGORY_SECTIONS:
                raw_section = cfg.get(key, None)
                if raw_section is not None:
                    flattened = _flatten_preserve_order(raw_section)
                    numerical_categories_flat[canonical] = flattened

    # Debug what ended up in numerical categories
    print(f"[load_schema] resolved numerical_categories_flat keys: {list(numerical_categories_flat.keys())}")

    # Build column definitions
    definitions: List[ColumnDefinition] = []
    for name in id_list:
        definitions.append(ColumnDefinition(name=name, category=ColumnCategory.ID))
    for name in numerical_list:
        definitions.append(ColumnDefinition(name=name, category=ColumnCategory.NUMERIC))
    for name in ordinal_list:
        definitions.append(ColumnDefinition(name=name, category=ColumnCategory.ORDINAL))
    for name in nominal_list:
        definitions.append(ColumnDefinition(name=name, category=ColumnCategory.NOMINAL))
    for name in target_list:
        definitions.append(ColumnDefinition(name=name, category=ColumnCategory.TARGET))

    schema = SchemaConfig(columns=definitions, numerical_categories_raw=numerical_categories_flat)

    # Sanity warning for mismatches between declared numerical categories and flat numerical list
    flat_numerical_set = set(schema.numerical())
    for cat, feats in numerical_categories_flat.items():
        unknown = [f for f in feats if f not in flat_numerical_set]
        if unknown:
            print(f"⚠ Warning: numerical category '{cat}' contains features not in flat numerical list: {unknown}")

    return schema


# ----------------------------
# Feature extraction utility 
# ----------------------------
def extract_feature_lists_from_schema(
    df: pd.DataFrame,
    schema: SchemaConfig,
    *,
    debug: bool = False,
    normalize: bool = True,  # allow case-insensitive fallback
) -> Tuple[List[str], List[str], List[str], Optional[str], Dict[str, List[str]]]:
    """
    Given a DataFrame and a SchemaConfig, return:
      - numericals (flat)
      - ordinal
      - nominal
      - target (resolved case-insensitively)
      - numerical_categories (only those present in df, numeric, sanitized)
    """
    from difflib import get_close_matches

    # Base lists with exact presence
    numericals = [c for c in schema.numerical() if c in df.columns]
    ordinal = [c for c in schema.ordinal() if c in df.columns]
    nominal = [c for c in schema.nominal() if c in df.columns]

    if debug:
        print(f"[extract_feature_lists_from_schema] Schema numerical_categories_raw: {schema.numerical_categories_raw}")
        print(f"[extract_feature_lists_from_schema] DataFrame columns sample (first 50): {list(df.columns)[:50]}")

    # Prepare normalized lookup if requested
    col_lower_to_original = {c.lower(): c for c in df.columns}
    df_cols_set = set(df.columns)

    def resolve_feature(candidate: str) -> Optional[str]:
        if candidate in df_cols_set:
            return candidate
        if normalize:
            low = candidate.lower()
            if low in col_lower_to_original:
                return col_lower_to_original[low]
        return None

    def fuzzy_suggestions(candidate: str) -> list[str]:
        matches = []
        low_candidate = candidate.lower()
        for c in df.columns:
            if low_candidate in c.lower() or c.lower() in low_candidate:
                matches.append(c)
        close = get_close_matches(candidate, list(df.columns), n=3, cutoff=0.7)
        for c in close:
            if c not in matches:
                matches.append(c)
        return matches

    # --- Resolve target case-insensitively
    target_raw = schema.target()
    y = resolve_feature(target_raw) if target_raw else None
    if debug and target_raw and y is None:
        print(f"[extract_feature_lists_from_schema] Target '{target_raw}' not found exactly; "
              f"tried normalized lookup and failed. Example near-misses: {fuzzy_suggestions(target_raw)}")

    # First pass: resolve declared features (with normalization) and collect fuzzy suggestions
    intermediate_breakdown: Dict[str, List[str]] = {}
    if debug and not schema.numerical_categories_raw:
        print("[extract_feature_lists_from_schema] WARNING: schema.numerical_categories_raw is empty; falling back to top-level detection or default.")

    for cat_name, cat_feats in schema.numerical_categories().items():
        resolved_candidates = []
        if debug:
            print(f"[extract_feature_lists_from_schema] Evaluating numerical category '{cat_name}' with declared features: {cat_feats}")
        for feat in cat_feats:
            resolved = resolve_feature(feat)
            if resolved:
                resolved_candidates.append(resolved)
            else:
                if debug:
                    sugg = fuzzy_suggestions(feat)
                    if sugg:
                        print(f"[extract_feature_lists_from_schema] Near-miss for '{feat}' in category '{cat_name}': suggestions {sugg}")
        if resolved_candidates:
            # preserve order, dedupe
            seen = set()
            cleaned = []
            for p in resolved_candidates:
                if p not in seen:
                    seen.add(p)
                    cleaned.append(p)
            intermediate_breakdown[cat_name] = cleaned

    # Second pass: sanitize (presence + numeric) using shared helper
    final_cat_breakdown = sanitize_cat_breakdown(intermediate_breakdown, df, y or "", debug=debug)

    # Fallback: if nothing survived and we do have numericals, expose them under a default category
    if not final_cat_breakdown and numericals:
        if debug:
            print("[extract_feature_lists_from_schema] No structured numerical categories resolved; falling back to putting all available numericals under 'general'.")
        fallback = {"general": numericals}
        final_cat_breakdown = sanitize_cat_breakdown(fallback, df, y or "", debug=debug)

    if debug and not final_cat_breakdown:
        print(
            "[extract_feature_lists_from_schema] WARNING: No numerical categories survived sanitization and fallback.\n"
            "Possible causes:\n"
            "  * The schema declares no numerical categories, and the numericals in the DataFrame are missing or non-numeric.\n"
            "Suggestions:\n"
            "  * Inspect df.columns and their dtypes.\n"
            "  * Verify the YAML schema includes `numerical_categories` or canonical top-level category lists."
        )

    return numericals, ordinal, nominal, y, final_cat_breakdown


def prune_dataframe_to_schema(
    df: pd.DataFrame,
    schema: SchemaConfig,
    *,
    drop_unexpected: bool = True,
    debug: bool = False
) -> pd.DataFrame:
    """
    Drop columns not declared in the schema (unexpected columns), optionally logging what was removed.
    Returns a new DataFrame containing only schema-approved columns plus leaving missing ones for downstream validation.
    """
    expected = set(schema.all_expected())
    actual = list(df.columns)
    unexpected = [c for c in actual if c not in expected]

    if drop_unexpected and unexpected:
        if debug:
            print(f"[prune_dataframe_to_schema] Dropping {len(unexpected)} unexpected columns: {unexpected}")
        df = df.drop(columns=unexpected)
    else:
        if debug and unexpected:
            print(f"[prune_dataframe_to_schema] Found {len(unexpected)} unexpected columns (not dropped): {unexpected}")
    return df



def hash_schema(schema: SchemaConfig) -> str:
    """
    Compute a stable fingerprint of the schema based on its declared groups.
    Used to detect drift between the schema used to build a FeatureSpec and the
    current schema in use. Returns a short 8-character hex digest.
    """
    # Compose a normalized representation
    payload = {
        "id": sorted(schema.id()),
        "numerical": sorted(schema.numerical()),
        "ordinal": sorted(schema.ordinal()),
        "nominal": sorted(schema.nominal()),
        "target": schema.target(),
    }
    # JSON serialize with sorted keys for determinism
    serialized = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest[:8]




def report_schema_dtype_violations(df: pd.DataFrame, schema: SchemaConfig, max_show: int = 30) -> dict:
    """
    Diagnostics only (no coercion/fill):
      - Which declared numericals are NOT numeric in df
      - Which declared ordinal/nominal are unexpectedly numeric
    Returns a dict; also prints a concise report.
    """
    from pandas.api.types import is_numeric_dtype, is_string_dtype, is_object_dtype
    problems = {
        "numerical_not_numeric": [],
        "categorical_numeric": []
    }

    # Declared numericals that are not numeric in df
    for c in schema.numerical():
        if c in df.columns and not is_numeric_dtype(df[c]):
            problems["numerical_not_numeric"].append((c, str(df[c].dtype)))

    # Declared categorical-like that are actually numeric
    for c in (schema.ordinal() + schema.nominal()):
        if c in df.columns and is_numeric_dtype(df[c]):
            problems["categorical_numeric"].append((c, str(df[c].dtype)))

    # Print summary (truncated)
    if problems["numerical_not_numeric"]:
        print(f"[schema dtype] ⚠ {len(problems['numerical_not_numeric'])} declared numeric columns are not numeric:")
        for name, dt in problems["numerical_not_numeric"][:max_show]:
            print(f"   • {name}: dtype={dt}")
        if len(problems["numerical_not_numeric"]) > max_show:
            print(f"   …and {len(problems['numerical_not_numeric']) - max_show} more")

    if problems["categorical_numeric"]:
        print(f"[schema dtype] ⚠ {len(problems['categorical_numeric'])} declared categorical columns are numeric:")
        for name, dt in problems["categorical_numeric"][:max_show]:
            print(f"   • {name}: dtype={dt}")
        if len(problems["categorical_numeric"]) > max_show:
            print(f"   …and {len(problems['categorical_numeric']) - max_show} more")

    if not problems["numerical_not_numeric"] and not problems["categorical_numeric"]:
        print("[schema dtype] ✅ No dtype/category violations detected vs schema.")

    return problems


# ----------------------------
# Smoke test / CLI entry (updated)
# ----------------------------
if __name__ == "__main__":
    from api.src.ml.features.load_data_utils import load_data_optimized
    from api.src.ml import config
    from api.src.ml.features.feature_engineering import engineer_features
    import traceback

    print(f"config: {config}")

    try:
        schema_path = config.COLUMN_SCHEMA_PATH
    except Exception as e:
        print(f"Failed to locate schema YAML: {e}")
        raise

    print(f"[SMOKE TEST] Loading schema from: {schema_path}")
    try:
        schema = load_schema_from_yaml(str(schema_path))
    except Exception as e:
        print(f"Failed to load schema YAML: {e}")
        raise

    # Display groups for debug
    print("Schema groups:")
    print("  ID:", schema.id())
    print("  Ordinal:", schema.ordinal())
    print("  Nominal:", schema.nominal()[:10], "...")  # truncated
    print("  Numerical:", schema.numerical()[:10], "...")
    print("  Target:", schema.target())

    # Load raw data
    FINAL_DATA_PATH = config.FINAL_ENGINEERED_DATASET_DIR / 'final_merged_with_all.parquet'    
    # TEST_DATA_PATH = 'api/src//data/merged_final_dataset/nba_player_data_final_inflated.parquet'
    # Example call: drop rows where player_id or season is null
    df = load_data_optimized(
        FINAL_DATA_PATH,
        debug=True,
        drop_null_rows=True,
        drop_null_subset=['AAV']
    )
    print(df.columns.tolist())
    print(df.head())
    print("nulls in AAV")
    print(df['AAV'].isnull().sum())


    df_eng, summary = engineer_features(df)


    # Validate schema against engineered DataFrame
    try:
        validation_report = schema.validate_dataframe(df_eng, strict=False, debug=True)
        print("Schema validation report:", json.dumps(validation_report, indent=2))
    except SchemaValidationError as e:
        print("Schema validation error:", e)
        traceback.print_exc()
        raise

    # Extract features
    numericals, ordinal, nominal, y, cat_breakdown = extract_feature_lists_from_schema(df_eng, schema, debug=True)

    print("\n=== Smoke test extracted groups ===")
    print("Numericals (found):", numericals[:10], "…total", len(numericals))
    print("Ordinal (found):", ordinal)
    print("Nominal (found):", nominal[:10], "…total", len(nominal))
    print("Y variable:", y)
    print("cat_breakdown:", cat_breakdown)

    assert y is not None, "Y variable (AAV) missing from dataset"
    assert "PLAYER_ID" in df_eng.columns and "TEAM_ID" in df_eng.columns
    print("✅ Smoke test passed. Dataset ready for downstream preprocessing.")
