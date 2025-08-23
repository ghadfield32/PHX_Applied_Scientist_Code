# FILE: api/src/ml/config.py (CORRECTED AND COMPLETE)
from pathlib import Path
from typing import Any, Union, Optional, Literal, Tuple
from dataclasses import dataclass
import os
import json

# ── Project root discovery ─────────────────────────────────────────────────────
def find_project_root(name: str = "nba_player_valuation_system") -> Path:
    """Walk up from this file until a directory named `name` or containing .git is found."""
    try:
        p = Path(__file__).resolve()
    except NameError:
        p = Path.cwd()
    for parent in (p, *p.parents):
        if parent.name == name or (parent / ".git").is_dir():
            return parent
    return Path.cwd()

# Core static paths
PROJECT_ROOT: Path = Path("api")
DATA_DIR: Path = PROJECT_ROOT / "src" / "ml" / "data"
AIRFLOW_DATA_DIR: Path = PROJECT_ROOT / "src" / "airflow_project" / "data"
LOG_DIR: Path = PROJECT_ROOT / "src" / "logs"
ARTIFACTS_DIR: Path = DATA_DIR / "ml_artifacts"
REGISTRY_LOCAL_CACHE_DIR: Path = ARTIFACTS_DIR / "registry_cache"
FINAL_ENGINEERED_DATASET_DIR: Path = AIRFLOW_DATA_DIR / "merged_final_dataset"
FINAL_ML_DATASET_DIR: Path = DATA_DIR / "final_ml_dataset"
MODEL_STORE_DIR: Path = DATA_DIR / "model_store"
COLUMN_SCHEMA_PATH: Path = PROJECT_ROOT / "src" / "ml" / "column_schema.yaml"
FEATURE_STORE_DIR: Path = DATA_DIR / "feature_store"
FEATURE_SELECTION_DIR: Path = DATA_DIR / "feature_selection"
MAX_CONTRACT_VALUES_CSV: Path = AIRFLOW_DATA_DIR / "spotrac_contract_data" / "exported_csv" / "max_contract_values.csv"

# Environment and MLflow configuration
ML_ENV: str = os.getenv("ML_ENV", "dev")
_DEFAULT_MLFLOW_TRACKING_URI = (ARTIFACTS_DIR / "mlruns").resolve().as_uri()
_raw_mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_MLFLOW_TRACKING_URI)

def _canonicalize_tracking_uri(uri: str) -> str:
    """Normalize a user-provided tracking URI into something MLflow accepts."""
    from urllib.parse import urlparse
    parsed = urlparse(uri)
    if parsed.scheme in ("http", "https"):
        return uri
    if parsed.scheme == "file":
        path_part = uri[len("file://"):]
        try:
            p = Path(path_part)
            return p.resolve().as_uri()
        except Exception:
            return uri
    try:
        p = Path(uri)
        return p.resolve().as_uri()
    except Exception:
        return uri

def _parse_families_env(raw: str) -> tuple[str, ...]:
    """Robustly parse MODEL_FAMILIES_SMOKE from either JSON or CSV."""
    s = (raw or "").strip()
    out: list[str] = []
    if not s:
        return tuple()

    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            for x in arr:
                t = str(x).strip().strip("'\"")
                if t and t not in out:
                    out.append(t)
            return tuple(out)
        except Exception:
            pass

    s = s.strip("[]")
    for tok in s.split(","):
        t = tok.strip().strip("'\"[]")
        if t and t not in out:
            out.append(t)
    return tuple(out)

# Final configurations
MLFLOW_TRACKING_URI: str = _canonicalize_tracking_uri(_raw_mlflow_uri)
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "nba_featurestore_smoke")
MODEL_ARTIFACTS_DIR: Path = Path(os.getenv("MODEL_ARTIFACTS_DIR", str(ARTIFACTS_DIR)))
FEATURE_STORE_DIR: Path = Path(os.getenv("FEATURE_STORE_DIR", str(FEATURE_STORE_DIR)))
FEATURESTORE_PREFERRED_STAGE: str = os.getenv("FEATURESTORE_PREFERRED_STAGE", "Production")
FEATURESTORE_AUTO_BOOTSTRAP: bool = os.getenv("FEATURESTORE_AUTO_BOOTSTRAP", "1").lower() in ("1", "true", "yes")
DEFAULT_FEATURESTORE_MODEL_FAMILY: str = os.getenv("FEATURESTORE_MODEL_FAMILY", "linear_ridge")

# Enhanced model families with stacking
DEFAULT_MODEL_FAMILIES_SMOKE: tuple[str, ...] = _parse_families_env(
    os.getenv("MODEL_FAMILIES_SMOKE", "linear_ridge,lasso,elasticnet,rf,xgb,lgbm,cat,stacking")
)

# Stacking-specific configuration
STACKING_DEFAULT_BASE_ESTIMATORS: tuple[str, ...] = _parse_families_env(
    os.getenv("STACKING_BASE_ESTIMATORS", "linear_ridge,xgb,lgbm,cat")
)
STACKING_DEFAULT_META_LEARNER: str = os.getenv("STACKING_META_LEARNER", "linear_ridge")
STACKING_DEFAULT_CV_FOLDS: int = int(os.getenv("STACKING_CV_FOLDS", "5"))
STACKING_DEFAULT_CV_STRATEGY: str = os.getenv("STACKING_CV_STRATEGY", "time_series")
STACKING_USE_PASSTHROUGH: bool = os.getenv("STACKING_USE_PASSTHROUGH", "false").lower() in ("1", "true", "yes")

# Utility functions
def feature_store_namespace(model_family: str, target: str) -> str:
    return f"{model_family}_{target}"

SELECTED_METRIC: Literal["rmse", "mae", "r2"] = os.getenv("SELECTED_METRIC", "mae").lower()

def metric_higher_is_better(metric: str) -> bool:
    return metric.lower() in {"r2"}

# Registry configuration
USE_MODEL_REGISTRY: bool = os.getenv("USE_MODEL_REGISTRY", "1").lower() in ("1","true","yes")
REGISTRY_ALIAS_DEV = os.getenv("MLFLOW_ALIAS_DEV", "dev")
REGISTRY_ALIAS_STAGE = os.getenv("MLFLOW_ALIAS_STAGE", "stage")
REGISTRY_ALIAS_PROD = os.getenv("MLFLOW_ALIAS_PROD", "prod")

def registry_alias_for_env(env: Optional[str] = None) -> str:
    e = _normalize_stage_env(env or ML_ENV)
    return {
        "dev": REGISTRY_ALIAS_DEV,
        "stage": REGISTRY_ALIAS_STAGE,
        "prod": REGISTRY_ALIAS_PROD,
    }.get(e, REGISTRY_ALIAS_DEV)

def registry_name_for_target(target: str) -> str:
    default = f"nba_{target}_regressor"
    return os.getenv("MODEL_REGISTRY_NAME", default)

# Stage profiles
_STAGE_ALIASES: dict[str, str] = {
    "staging": "stage",
    "production": "prod",
    "development": "dev",
}

def _normalize_stage_env(env: Optional[str] = None) -> str:
    e = (env or ML_ENV).lower()
    return _STAGE_ALIASES.get(e, e)

@dataclass(frozen=True)
class StageProfile:
    name: Literal["dev", "stage", "prod"]
    registry_stage: Literal["Staging", "Production"]
    selected_metric: Literal["rmse", "mae", "r2"] = SELECTED_METRIC
    min_improvement: float = 0.0

STAGE_PROFILES: dict[str, StageProfile] = {
    "dev": StageProfile("dev", "Staging", selected_metric=SELECTED_METRIC, min_improvement=0.0),
    "stage": StageProfile("stage", "Staging", selected_metric=SELECTED_METRIC, min_improvement=0.0),
    "prod": StageProfile("prod", "Production", selected_metric=SELECTED_METRIC, min_improvement=0.0),
}

# Bundle directories
BUNDLE_ROOT: Path = ARTIFACTS_DIR / "model_bundles"
FAMILY_BUNDLE_ROOT: Path = ARTIFACTS_DIR / "family_bundles"

def family_bundle_dir_for(model_family: str, target: str, env: Optional[str] = None) -> Path:
    canonical = _normalize_stage_env(env or ML_ENV)
    return FAMILY_BUNDLE_ROOT / target / model_family / canonical

def bundle_dir_for(target: str, env: Optional[str] = None) -> Path:
    canonical = _normalize_stage_env(env or ML_ENV)
    return BUNDLE_ROOT / target / canonical

# Behavior switches
AUTOCLEAN_FAMILY_ARTIFACTS: bool = os.getenv("AUTOCLEAN_FAMILY_ARTIFACTS", "1").lower() in ("1", "true", "yes")
PREDICT_USE_BUNDLE_FIRST: bool = os.getenv("PREDICT_USE_BUNDLE_FIRST", "1").lower() in ("1", "true", "yes")

def registry_stage_for_env(env: Optional[str] = None) -> str:
    canonical = _normalize_stage_env(env)
    return STAGE_PROFILES.get(canonical, STAGE_PROFILES["dev"]).registry_stage

# MISSING TRAINING CONFIG CLASS (ROOT CAUSE OF ERROR)
@dataclass
class TrainingConfig:
    """
    Training configuration for any model family (sklearn/stacking/bayes_hier).
    """
    model_family: str = "linear_ridge"
    target: str = "AAV"
    use_cap_pct_target: bool = False
    max_train_rows: Optional[int] = None
    n_splits: int = 4
    n_trials: int = 20
    random_state: int = 42
    drop_columns_exact: list[str] = None
    feature_exclude_prefixes: list[str] = None

    # --- Bayesian-only knobs (new) ---
    bayes_draws: int = int(os.getenv("BAYES_DRAWS", "1000"))
    bayes_tune: int = int(os.getenv("BAYES_TUNE", "1000"))
    bayes_target_accept: float = float(os.getenv("BAYES_TARGET_ACCEPT", "0.9"))
    bayes_chains: int = int(os.getenv("BAYES_CHAINS", "2"))
    bayes_cores: int = int(os.getenv("BAYES_CORES", "2"))
    # configurable grouping columns; default common pair
    bayes_group_cols: tuple[str, ...] = tuple(
        os.getenv("BAYES_GROUP_COLS", "position,Season").split(",")
    )

    def __post_init__(self):
        if self.drop_columns_exact is None:
            self.drop_columns_exact = []
        if self.feature_exclude_prefixes is None:
            self.feature_exclude_prefixes = []
        # basic sanity
        if self.bayes_draws < 100 or self.bayes_tune < 100:
            print("[TrainingConfig] Warning: very small draws/tune may lead to unstable posteriors.")

@dataclass
class DevTrainConfig:
    """Enhanced DevTrainConfig with stacking ensemble support."""
    stage: Literal["dev", "train", "prod"] = "dev"

    # Enhanced preprocessing
    numerical_imputation: Literal["mean", "median", "iterative"] = "median"
    add_missing_indicators: bool = True
    quantile_clipping: tuple[float, float] = (0.01, 0.99)
    max_safe_rows: int = 200_000
    apply_type_conversions: bool = True
    drop_unexpected_columns: bool = True

    # Feature scaling improvements
    enable_robust_scaling: bool = True
    enable_outlier_detection: bool = True
    outlier_contamination: float = 0.1

    # Feature selection
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

    # Model-specific convergence settings
    linear_max_iter: int = 50000
    linear_tol: float = 1e-6
    enable_feature_selection_for_linear: bool = True

    # Cross-validation improvements
    cv_strategy: Literal["time_series", "group", "stratified"] = "time_series"
    cv_n_splits: int = 5
    cv_test_size: float = 0.2

    # Stacking ensemble configuration
    stacking_base_estimators: tuple[str, ...] = STACKING_DEFAULT_BASE_ESTIMATORS
    stacking_meta_learner: str = STACKING_DEFAULT_META_LEARNER
    stacking_cv_folds: int = STACKING_DEFAULT_CV_FOLDS
    stacking_cv_strategy: str = STACKING_DEFAULT_CV_STRATEGY
    stacking_use_passthrough: bool = STACKING_USE_PASSTHROUGH
    stacking_enable_base_tuning: bool = True
    stacking_meta_tuning_trials: int = 20

    def make_selection_kwargs(self) -> dict:
        """Build a dict that can be unpacked into SelectionConfig-like consumers."""
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

    def make_stacking_params(self) -> dict:
        """Build default stacking parameters from configuration."""
        return {
            "base_estimators": list(self.stacking_base_estimators),
            "meta_learner": self.stacking_meta_learner,
            "cv_folds": self.stacking_cv_folds,
            "cv_strategy": self.stacking_cv_strategy,
            "passthrough": self.stacking_use_passthrough,
            "base_params": self._get_default_base_params(),
            "meta_params": self._get_default_meta_params()
        }

    def _get_default_base_params(self) -> dict:
        """Get default hyperparameters for base estimators."""
        defaults = {
            "linear_ridge": {"alpha": 1.0},
            "lasso": {"alpha": 0.01},
            "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5},
            "rf": {"n_estimators": 300, "max_depth": 10},
            "xgb": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
            "lgbm": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
            "cat": {"iterations": 300, "depth": 6, "learning_rate": 0.1}
        }
        return {family: params for family, params in defaults.items() 
                if family in self.stacking_base_estimators}

    def _get_default_meta_params(self) -> dict:
        """Get default hyperparameters for meta-learner."""
        meta_defaults = {
            "linear_ridge": {"alpha": 0.1},
            "lasso": {"alpha": 0.01, "max_iter": 10000},
            "elasticnet": {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 10000}
        }
        return meta_defaults.get(self.stacking_meta_learner, {})

    def validate(self):
        """Enhanced validation with stacking checks."""
        if not (0 <= self.quantile_clipping[0] < self.quantile_clipping[1] <= 1):
            raise ValueError("quantile_clipping must satisfy 0 <= low < high <=1")
        if self.max_safe_rows < 1_000:
            raise ValueError("max_safe_rows must be sensible (>=1000)")
        if self.min_features < 1:
            raise ValueError("min_features must be >=1")
        if self.perm_threshold < 0 or self.shap_threshold < 0:
            raise ValueError("thresholds must be non-negative")

        # Stacking validation
        if len(self.stacking_base_estimators) < 2:
            raise ValueError("stacking_base_estimators must have at least 2 estimators")
        if self.stacking_cv_folds < 2:
            raise ValueError("stacking_cv_folds must be >= 2")
        if self.stacking_cv_strategy not in ("time_series", "kfold"):
            raise ValueError("stacking_cv_strategy must be 'time_series' or 'kfold'")

@dataclass
class TuningConfig:
    """Enhanced tuning configuration with stacking ensemble support."""
    model_families: Tuple[str, ...] = DEFAULT_MODEL_FAMILIES_SMOKE
    n_trials: int = 20
    n_splits: int = 4
    use_bayesian: bool = True

    # Stacking-specific tuning configuration
    stacking_n_trials: int = 50
    stacking_enable_base_tuning: bool = False
    stacking_base_trials_per_family: int = 10
    stacking_meta_trials: int = 20
    stacking_max_base_combinations: int = 10

    def get_trials_for_family(self, model_family: str) -> int:
        """Get the appropriate number of trials for a given model family."""
        if model_family == "stacking":
            return self.stacking_n_trials
        return self.n_trials

    def should_tune_family(self, model_family: str) -> bool:
        """Determine whether to tune a specific model family."""
        if not self.use_bayesian:
            return False
        if model_family == "stacking" and not self.stacking_enable_base_tuning:
            return False
        return True

# CREATE THE MISSING DEFAULTS INSTANCE (ROOT CAUSE FIX)
DEFAULT_DEV_TRAIN_CONFIG = DevTrainConfig()
DEFAULT_TUNING_CONFIG = TuningConfig()
DEFAULTS = TrainingConfig()  # This was the missing piece causing the NameError!

# Stacking ensemble utilities
def get_stacking_default_params(target: str = "AAV") -> dict:
    """Get default stacking parameters optimized for NBA player valuation."""
    return {
        "base_estimators": list(STACKING_DEFAULT_BASE_ESTIMATORS),
        "meta_learner": STACKING_DEFAULT_META_LEARNER,
        "cv_folds": STACKING_DEFAULT_CV_FOLDS,
        "cv_strategy": STACKING_DEFAULT_CV_STRATEGY,
        "passthrough": STACKING_USE_PASSTHROUGH,
        "base_params": {
            "linear_ridge": {"alpha": 1.0},
            "xgb": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            },
            "lgbm": {
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            },
            "cat": {
                "iterations": 300,
                "depth": 6,
                "learning_rate": 0.1
            }
        },
        "meta_params": {"alpha": 0.1}
    }

def is_stacking_family(model_family: str) -> bool:
    """Check if a model family is a stacking ensemble."""
    return model_family.lower() == "stacking"

def get_stacking_dependencies(model_family: str) -> list[str]:
    """Get the base model dependencies for a stacking model."""
    if not is_stacking_family(model_family):
        return []
    return list(STACKING_DEFAULT_BASE_ESTIMATORS)

def get_training_order(model_families: list[str]) -> list[str]:
    """Return model families in the correct training order. Stacking models should be trained after their base estimators."""
    stacking_families = [f for f in model_families if is_stacking_family(f)]
    base_families = [f for f in model_families if not is_stacking_family(f)]
    return base_families + stacking_families

# MISSING HELPER FUNCTION (ANOTHER ROOT CAUSE)
def get_master_parquet_path() -> Path:
    """
    Get the path to the master dataset parquet file.
    This function was referenced but not defined, causing another potential error.
    """
    return FINAL_ENGINEERED_DATASET_DIR / "final_merged_with_all.parquet"

# Debug function to validate all required components exist
def validate_configuration():
    """Debug function to validate all configuration components are properly defined."""
    errors = []

    # Check required classes exist
    try:
        TrainingConfig()
        print("✅ TrainingConfig class defined correctly")
    except Exception as e:
        errors.append(f"❌ TrainingConfig error: {e}")

    # Check DEFAULTS exists
    try:
        assert DEFAULTS is not None
        print("✅ DEFAULTS instance defined correctly")
    except Exception as e:
        errors.append(f"❌ DEFAULTS error: {e}")

    # Check required functions exist
    try:
        path = get_master_parquet_path()
        print(f"✅ get_master_parquet_path() returns: {path}")
    except Exception as e:
        errors.append(f"❌ get_master_parquet_path() error: {e}")

    # Check stacking utilities
    try:
        params = get_stacking_default_params()
        print(f"✅ Stacking params generated: {len(params)} keys")
    except Exception as e:
        errors.append(f"❌ Stacking utilities error: {e}")

    if errors:
        print("\n=== CONFIGURATION ERRORS ===")
        for error in errors:
            print(error)
        return False
    else:
        print("\n✅ All configuration components validated successfully!")
        return True

if __name__ == "__main__":
    print("=== CONFIGURATION DEBUG VALIDATION ===")
    validate_configuration()

    print("\nTesting stacking configuration:")
    config = DevTrainConfig()
    stacking_params = config.make_stacking_params()
    print(f"Default stacking params: {stacking_params}")

    print("\nTesting training order:")
    families = ["linear_ridge", "xgb", "stacking", "lgbm", "cat"]
    training_order = get_training_order(families)
    print(f"Training order: {training_order}")
