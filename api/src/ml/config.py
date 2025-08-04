# api/src/ml/config.py

from pathlib import Path
from typing import Any, Union, Optional, Literal, Tuple
from dataclasses import dataclass
import os
import json


# ── Project root discovery (similar to your example) ───────────────────────────
def find_project_root(name: str = "nba_player_valuation_system") -> Path:
    """
    Walk up from this file until a directory named `name` or containing .git is found.
    Fallback to cwd.
    """
    try:
        p = Path(__file__).resolve()
    except NameError:
        p = Path.cwd()
    for parent in (p, *p.parents):
        if parent.name == name or (parent / ".git").is_dir():
            return parent
    return Path.cwd()


# ------------------------------------------------------------------
# Core static paths (existing)
# ------------------------------------------------------------------

# make project root api/
# Project root (looks for directory named or .git under it)
PROJECT_ROOT: Path = Path("api")

# Data directories
DATA_DIR: Path = PROJECT_ROOT / "src" / "ml" / "data"
AIRFLOW_DATA_DIR: Path = PROJECT_ROOT / "src" / "airflow_project" / "data"
LOG_DIR: Path = PROJECT_ROOT / "src" / "logs"

# Where model artifacts live
ARTIFACTS_DIR: Path = DATA_DIR / "ml_artifacts"

# Where registry-resolved artifacts are cached locally for prediction fast-paths
REGISTRY_LOCAL_CACHE_DIR: Path = ARTIFACTS_DIR / "registry_cache"

# Engineered datasets
FINAL_ENGINEERED_DATASET_DIR: Path = AIRFLOW_DATA_DIR / "merged_final_dataset"

# Final ML-ready dataset (fixed assignment; previously missing '=' caused NameError)
FINAL_ML_DATASET_DIR: Path = DATA_DIR / "final_ml_dataset"

# Model store
MODEL_STORE_DIR: Path = DATA_DIR / "model_store"

# Column schema path (relative to project root)
COLUMN_SCHEMA_PATH: Path = PROJECT_ROOT / "src" / "ml" / "column_schema.yaml"

# Feature store
FEATURE_STORE_DIR: Path = DATA_DIR / "feature_store"

# Feature selection docs
FEATURE_SELECTION_DIR: Path = DATA_DIR / "feature_selection"

# Add near other AIRFLOW paths
MAX_CONTRACT_VALUES_CSV: Path = AIRFLOW_DATA_DIR / "spotrac_contract_data" / "exported_csv" / "max_contract_values.csv"


# ------------------------------------------------------------------
# Dev / Training adjustable section
# ------------------------------------------------------------------
# === Stage profiles & canonical bundle paths ===
# -------------------------------
# #Modelling: environment & paths
# -------------------------------
# Environment stage for modeling flows (local/dev/stage/prod)
ML_ENV: str = os.getenv("ML_ENV", "dev")

# Default raw mlruns directory (local file store)
_DEFAULT_MLFLOW_TRACKING_URI = (ARTIFACTS_DIR / "mlruns").resolve().as_uri()

# Raw override (could be a bare path, a file:// variant, or http(s))
_raw_mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_MLFLOW_TRACKING_URI)

def _canonicalize_tracking_uri(uri: str) -> str:
    """
    Normalize a user-provided tracking URI into something MLflow accepts.
    - Converts bare or relative paths into absolute file:/// URIs.
    - Repairs Windows-style file://C:\\... to file:///C:/... using pathlib.
    """
    from urllib.parse import urlparse
    parsed = urlparse(uri)

    # HTTP/HTTPS are passed through
    if parsed.scheme in ("http", "https"):
        return uri

    # File scheme: normalize via pathlib to get canonical form (e.g., file:///C:/...)
    if parsed.scheme == "file":
        # Strip the 'file://' prefix to get the path component, then feed to pathlib
        path_part = uri[len("file://") :]
        try:
            p = Path(path_part)
            return p.resolve().as_uri()
        except Exception:
            # Fallback: if something goes wrong, return original
            return uri

    # No scheme: treat as local path
    try:
        p = Path(uri)
        return p.resolve().as_uri()
    except Exception:
        return uri  # best effort

def _parse_families_env(raw: str) -> tuple[str, ...]:
    """
    Robustly parse MODEL_FAMILIES_SMOKE from either JSON (['cat', 'xgb']) or CSV.
    Strips quotes/brackets and dedups while preserving order.
    """
    s = (raw or "").strip()
    out: list[str] = []
    if not s:
        return tuple()

    # Try JSON list first (most specific)
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            for x in arr:
                t = str(x).strip().strip("'\"")
                if t and t not in out:
                    out.append(t)
            return tuple(out)
        except Exception:
            pass  # fall through to CSV

    # CSV fallback - handle mixed formats by cleaning up first
    s = s.strip("[]")  # tolerate accidental [] wrappers
    # Split by comma and clean each token
    for tok in s.split(","):
        t = tok.strip().strip("'\"[]")  # strip quotes and brackets from individual tokens
        if t and t not in out:
            out.append(t)
    return tuple(out)

# Final normalized tracking URI used by the system
MLFLOW_TRACKING_URI: str = _canonicalize_tracking_uri(_raw_mlflow_uri)

# Experiment name override
MLFLOW_EXPERIMENT_NAME: str = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "nba_featurestore_smoke"
)


# Where to save model artifacts for training/eval (smoke & beyond)
MODEL_ARTIFACTS_DIR: Path = Path(os.getenv("MODEL_ARTIFACTS_DIR", str(ARTIFACTS_DIR)))

# Feature store directory (allow override)
FEATURE_STORE_DIR: Path = Path(os.getenv("FEATURE_STORE_DIR", str(FEATURE_STORE_DIR)))

# Feature Store behaviors
FEATURESTORE_PREFERRED_STAGE: str = os.getenv("FEATURESTORE_PREFERRED_STAGE", "Production")
FEATURESTORE_AUTO_BOOTSTRAP: bool = os.getenv("FEATURESTORE_AUTO_BOOTSTRAP", "1").lower() in ("1", "true", "yes")

# The default model_family used to name the feature store namespace
DEFAULT_FEATURESTORE_MODEL_FAMILY: str = os.getenv("FEATURESTORE_MODEL_FAMILY", "linear_ridge")

# Default families for the smoke comparison run (comma-separated env override)
DEFAULT_MODEL_FAMILIES_SMOKE: tuple[str, ...] = _parse_families_env(
    os.getenv("MODEL_FAMILIES_SMOKE", "linear_ridge,lasso,elasticnet,rf,xgb,lgbm,cat")
)

# Helpful utility to format the namespace on disk
def feature_store_namespace(model_family: str, target: str) -> str:
    # keep target stable and obvious in namespace
    return f"{model_family}_{target}"

# Model selection (which metric picks the Production model)
SELECTED_METRIC: Literal["rmse", "mae", "r2"] = os.getenv("SELECTED_METRIC", "mae").lower()

def metric_higher_is_better(metric: str) -> bool:
    return metric.lower() in {"r2"}

# MLflow Model Registry integration
USE_MODEL_REGISTRY: bool = os.getenv("USE_MODEL_REGISTRY", "1").lower() in ("1","true","yes")

# Registry alias mapping (next to registry helpers)
REGISTRY_ALIAS_DEV   = os.getenv("MLFLOW_ALIAS_DEV",   "dev")
REGISTRY_ALIAS_STAGE = os.getenv("MLFLOW_ALIAS_STAGE", "stage")
REGISTRY_ALIAS_PROD  = os.getenv("MLFLOW_ALIAS_PROD",  "prod")

def registry_alias_for_env(env: Optional[str] = None) -> str:
    """
    Map ML_ENV (dev/stage/prod and their aliases) to a registry alias string.
    Defaults are 'dev','stage','prod' but can be overridden via env vars above.
    """
    e = _normalize_stage_env(env or ML_ENV)
    return {
        "dev":  REGISTRY_ALIAS_DEV,
        "stage": REGISTRY_ALIAS_STAGE,
        "prod": REGISTRY_ALIAS_PROD,
    }.get(e, REGISTRY_ALIAS_DEV)

def registry_name_for_target(target: str) -> str:
    """
    Use a single registered model per target, regardless of family.
    Keeping families as tags/params makes it easy to compare while 
    always having ONE Production pointer.
    """
    default = f"nba_{target}_regressor"
    return os.getenv("MODEL_REGISTRY_NAME", default)


# === Stage profiles & canonical bundle paths ===
# Normalize incoming ML_ENV / stage strings so that variants like "staging"
# or "production" are mapped to the canonical internal keys ("stage", "prod").
_STAGE_ALIASES: dict[str, str] = {
    "staging": "stage",
    "production": "prod",
    "development": "dev",
    # keep canonical forms mapping to themselves implicitly
}

def _normalize_stage_env(env: Optional[str] = None) -> str:
    """
    Return the canonical internal stage key for a given input.
    e.g., "staging" -> "stage", "production" -> "prod", case-insensitive.
    """
    e = (env or ML_ENV).lower()
    return _STAGE_ALIASES.get(e, e)  # fallback to itself if already canonical

@dataclass(frozen=True)
class StageProfile:
    name: Literal["dev", "stage", "prod"]  # internal canonical names
    # Which registry stage (if used) this environment maps to
    registry_stage: Literal["Staging", "Production"]
    # Which metric to compare when promoting within this env
    selected_metric: Literal["rmse", "mae", "r2"] = SELECTED_METRIC
    # Minimal improvement needed to promote (0.0 => any improvement)
    min_improvement: float = 0.0

# NOTE: we keep the internal canonical keys here; external callers can pass
# "staging" or "production" because of normalization helpers.
STAGE_PROFILES: dict[str, StageProfile] = {
    "dev":   StageProfile("dev",   "Staging",    selected_metric=SELECTED_METRIC, min_improvement=0.0),
    "stage": StageProfile("stage", "Staging",    selected_metric=SELECTED_METRIC, min_improvement=0.0),
    "prod":  StageProfile("prod",  "Production", selected_metric=SELECTED_METRIC, min_improvement=0.0),
}

# Canonical "one bundle per stage per target"
BUNDLE_ROOT: Path = ARTIFACTS_DIR / "model_bundles"
# === Per-family bundles (exactly 3 per family: dev/stage/prod) ===
FAMILY_BUNDLE_ROOT: Path = ARTIFACTS_DIR / "family_bundles"

def family_bundle_dir_for(model_family: str, target: str, env: Optional[str] = None) -> Path:
    """
    Canonical on-disk location for a family's promoted artifacts per stage env.
    e.g., api/src/ml/data/ml_artifacts/family_bundles/<target>/<family>/<env>/
    """
    canonical = _normalize_stage_env(env or ML_ENV)
    return FAMILY_BUNDLE_ROOT / target / model_family / canonical


def bundle_dir_for(target: str, env: Optional[str] = None) -> Path:
    """
    Returns the canonical bundle directory, normalizing env aliases.
    """
    canonical = _normalize_stage_env(env or ML_ENV)
    return BUNDLE_ROOT / target / canonical

# Behavior switches
AUTOCLEAN_FAMILY_ARTIFACTS: bool = os.getenv("AUTOCLEAN_FAMILY_ARTIFACTS", "1").lower() in ("1", "true", "yes")
PREDICT_USE_BUNDLE_FIRST: bool = os.getenv("PREDICT_USE_BUNDLE_FIRST", "1").lower() in ("1", "true", "yes")

def registry_stage_for_env(env: Optional[str] = None) -> str:
    """
    Given an environment name (e.g., "dev", "staging", "prod", "production"), return
    the corresponding MLflow registry stage string (e.g., "Staging" or "Production").
    """
    canonical = _normalize_stage_env(env)
    return STAGE_PROFILES.get(canonical, STAGE_PROFILES["dev"]).registry_stage


@dataclass
class DevTrainConfig:
    """
    REPLACE your existing DevTrainConfig with this enhanced version.

    NEW FEATURES:
    - Convergence-specific settings
    - Enhanced preprocessing options
    - Better cross-validation strategies
    """
    stage: Literal["dev", "train", "prod"] = "dev"

    # Enhanced preprocessing
    numerical_imputation: Literal["mean", "median", "iterative"] = "median"
    add_missing_indicators: bool = True
    quantile_clipping: tuple[float, float] = (0.01, 0.99)
    max_safe_rows: int = 200_000
    apply_type_conversions: bool = True
    drop_unexpected_columns: bool = True

    # NEW: Feature scaling improvements
    enable_robust_scaling: bool = True
    enable_outlier_detection: bool = True
    outlier_contamination: float = 0.1

    # Feature selection (existing)
    perm_threshold: float = 0.001
    shap_threshold: float = 0.001
    selection_mode: Literal["intersection", "union"] = "union"
    min_features: int = 10
    max_features: Optional[int] = None
    fallback_strategy: Literal["top_permutation", "top_shap", "all"] = "top_permutation"
    perm_n_repeats: int = 10
    perm_max_samples: float | int | None = 0.5
    perm_n_jobs: int = 2
    shap_nsamples: int = 100
    max_relative_regression: float = 0.05

    # NEW: Model-specific convergence settings
    linear_max_iter: int = 50000  # Increased from default
    linear_tol: float = 1e-6  # Tighter tolerance
    enable_feature_selection_for_linear: bool = True

    # NEW: Cross-validation improvements
    cv_strategy: Literal["time_series", "group", "stratified"] = "time_series"
    cv_n_splits: int = 5
    cv_test_size: float = 0.2

    # --- Convenience computed properties ---
    def make_selection_kwargs(self) -> dict:
        """
        Build a dict that can be unpacked into SelectionConfig-like consumers.
        """
        return {
            "perm_n_repeats": self.perm_n_repeats,
            "perm_max_samples": self.perm_max_samples,
            "perm_n_jobs": self.perm_n_jobs,
            "perm_threshold": self.perm_threshold,
            "shap_nsamples": self.shap_nsamples,
            "shap_threshold": self.shap_threshold,
            "mode": self.selection_mode,
            "min_features": self.min_features,
            "max_features": self.max_features,
            "fallback_strategy": self.fallback_strategy,
            "max_relative_regression": self.max_relative_regression,
        }

    def validate(self):
        if not (0 <= self.quantile_clipping[0] < self.quantile_clipping[1] <= 1):
            raise ValueError("quantile_clipping must satisfy 0 <= low < high <=1")
        if self.max_safe_rows < 1_000:
            raise ValueError("max_safe_rows must be sensible (>=1000)")
        if self.min_features < 1:
            raise ValueError("min_features must be >=1")
        if self.perm_threshold < 0 or self.shap_threshold < 0:
            raise ValueError("thresholds must be non-negative")


# Default instantiation accessible for quick imports
DEFAULT_DEV_TRAIN_CONFIG = DevTrainConfig()









# --- Training/Tuning section ---
@dataclass
class TuningConfig:
    """
    Controls whether Bayesian tuning runs, how many trials / splits, and
    which families to include. Separates the tuning stage from final evaluation.
    """
    model_families: Tuple[str, ...] = DEFAULT_MODEL_FAMILIES_SMOKE
    n_trials: int = 2
    n_splits: int = 4
    use_bayesian: bool = False  # whether to run Optuna tuning before final training

DEFAULT_TUNING_CONFIG = TuningConfig()






# ------------------------------------------------------------------
# CLI / direct invocation helper (for debugging)
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("all directories and dev/train config:")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"LOG_DIR: {LOG_DIR}")
    print(f"ARTIFACTS_DIR: {ARTIFACTS_DIR}")
    print(f"FINAL_ENGINEERED_DATASET_DIR: {FINAL_ENGINEERED_DATASET_DIR}")
    print(f"FINAL_ML_DATASET_DIR: {FINAL_ML_DATASET_DIR}")
    print(f"MODEL_STORE_DIR: {MODEL_STORE_DIR}")
    print(f"FEATURE_STORE_DIR: {FEATURE_STORE_DIR}")
    print(f"COLUMN_SCHEMA_PATH: {COLUMN_SCHEMA_PATH}")
    print(f"FEATURE_SELECTION_DIR: {FEATURE_SELECTION_DIR}")
    print("\nDefault Dev/Train Config:")
    print(DEFAULT_DEV_TRAIN_CONFIG)
    print(f"MAX_CONTRACT_VALUES_CSV: {MAX_CONTRACT_VALUES_CSV}")
