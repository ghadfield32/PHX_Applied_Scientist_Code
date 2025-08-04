# api/src/ml/ml_config.py
"""
Updated ML configuration module that integrates with existing preprocessing pipeline.
Addresses feature selection thresholds and provides configurable preprocessing options.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence, Optional

ModelFamily = Literal["linear_ridge", "lasso", "elasticnet", "rf", "xgb", "lgbm", "cat", "bayes_hier"]

@dataclass
class TrainingConfig:
    target: str = "AAV_PCT_CAP"  # Updated to match schema
    use_cap_pct_target: bool = False
    model_family: ModelFamily = "linear_ridge"
    random_state: int = 42
    n_splits: int = 4
    n_trials: int = 15
    test_size_rows: int = 1000
    max_train_rows: int | None = None
    feature_exclude_prefixes: Sequence[str] = ("_raw_", "_uid", "_merge")
    drop_columns_exact: Sequence[str] = ("PLAYER_ID", "TEAM_ID")  # Updated to match schema
    selection_strategy: Literal["separate", "inline", "none"] = "separate"
    feature_store_stage: Literal["Production", "Staging"] = "Production"

@dataclass
class SelectionConfig:
    """Updated with more permissive defaults to fix feature selection issues"""
    perm_n_repeats: int = 10
    perm_max_samples: float | int | None = 0.5
    perm_n_jobs: int = 2
    perm_threshold: float = 0.001  # REDUCED from 0.01 - was too aggressive
    shap_nsamples: int = 100
    shap_threshold: float = 0.001  # REDUCED from 0.01 - was too aggressive  
    mode: Literal["intersection", "union"] = "union"  # CHANGED from intersection - more permissive
    max_relative_regression: float = 0.05

    # NEW: Fallback options if selection is too aggressive
    min_features: int = 10  # Ensure at least this many features are selected
    max_features: Optional[int] = None  # Cap maximum features if needed
    fallback_strategy: Literal["top_permutation", "top_shap", "all"] = "top_permutation"

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.perm_n_repeats < 1:
            raise ValueError("perm_n_repeats must be >= 1")
        if self.perm_threshold < 0:
            raise ValueError("perm_threshold must be >= 0")
        if self.shap_threshold < 0:
            raise ValueError("shap_threshold must be >= 0")
        if self.shap_nsamples < 1:
            raise ValueError("shap_nsamples must be >= 1")
        if self.max_relative_regression < 0:
            raise ValueError("max_relative_regression must be >= 0")
        if self.min_features < 1:
            raise ValueError("min_features must be >= 1")

@dataclass  
class PreprocessingConfig:
    """Configuration for data preprocessing with enhanced imputation options."""

    # Model type affects preprocessing strategy
    model_type: Literal["linear", "tree", "ensemble", "neural"] = "linear"

    # Numerical imputation strategy - NEW FEATURE as requested
    numerical_imputation: Literal["mean", "median", "iterative"] = "median"
    add_missing_indicators: bool = True

    # Scaling strategy
    numerical_scaling: Literal["standard", "minmax", "robust", "none"] = "standard"

    # Categorical encoding
    ordinal_handle_unknown: Literal["error", "use_encoded_value"] = "use_encoded_value"
    nominal_handle_unknown: Literal["error", "ignore", "infrequent_if_exist"] = "ignore"
    nominal_drop_strategy: Literal["first", "if_binary"] = "first"

    # Data filtering
    quantile_clipping: tuple[float, float] = (0.01, 0.99)
    max_safe_training_rows: int = 200000

    # Schema validation
    apply_type_conversions: bool = True
    drop_unexpected_columns: bool = True
    strict_validation: bool = True

    # Feature engineering
    create_interaction_features: bool = False
    polynomial_features_degree: Optional[int] = None

    # NEW: Debug and verbose options
    debug_preprocessing: bool = False
    verbose_feature_names: bool = True  # Force consistent naming

    def __post_init__(self):
        """Validate preprocessing configuration."""
        if not (0 <= self.quantile_clipping[0] < self.quantile_clipping[1] <= 1):
            raise ValueError("quantile_clipping must be (low, high) with 0 <= low < high <= 1")
        if self.max_safe_training_rows < 1000:
            raise ValueError("max_safe_training_rows should be >= 1000")
        if self.polynomial_features_degree is not None and self.polynomial_features_degree < 2:
            raise ValueError("polynomial_features_degree must be >= 2 if specified")

@dataclass
class ModelConfig:
    """Configuration for model training and validation."""

    # Model family
    family: Literal["linear_ridge", "xgboost", "random_forest", "pymc"] = "linear_ridge"

    # Cross-validation
    cv_folds: int = 5
    cv_scoring: str = "neg_mean_squared_error"
    cv_n_jobs: int = -1

    # Random state for reproducibility
    random_state: int = 42

    # Hyperparameter optimization
    use_hyperopt: bool = False
    hyperopt_max_evals: int = 100
    hyperopt_timeout: Optional[float] = None

    # Model-specific settings
    linear_alpha_range: tuple[float, float] = (0.1, 100.0)
    xgb_n_estimators_range: tuple[int, int] = (50, 500)
    rf_n_estimators_range: tuple[int, int] = (50, 200)

    def __post_init__(self):
        """Validate model configuration."""
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
        if self.hyperopt_max_evals < 1:
            raise ValueError("hyperopt_max_evals must be >= 1")

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation and validation."""

    # Train/validation/test splits
    test_size: float = 0.2
    validation_size: Optional[float] = 0.2
    stratify_column: Optional[str] = None

    # Evaluation metrics
    primary_metric: str = "rmse"
    additional_metrics: list[str] = None

    # Validation settings
    time_series_validation: bool = False
    n_splits_time_series: int = 5

    # Performance thresholds
    minimum_acceptable_r2: float = 0.1
    maximum_acceptable_rmse: Optional[float] = None

    def __post_init__(self):
        """Validate evaluation configuration."""
        if not (0 < self.test_size < 1):
            raise ValueError("test_size must be between 0 and 1")
        if self.validation_size is not None and not (0 < self.validation_size < 1):
            raise ValueError("validation_size must be between 0 and 1")
        if self.additional_metrics is None:
            self.additional_metrics = ["mae", "r2"]

@dataclass
class PipelineConfig:
    """Master configuration combining all pipeline components."""

    # Component configurations
    preprocessing: PreprocessingConfig = None
    selection: SelectionConfig = None
    model: ModelConfig = None
    evaluation: EvaluationConfig = None

    # Pipeline control
    enable_feature_selection: bool = True
    enable_hyperparameter_tuning: bool = False
    enable_model_comparison: bool = False

    # Logging and debugging
    debug_mode: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    save_intermediate_results: bool = True

    # Reproducibility
    global_random_state: int = 42

    def __post_init__(self):
        """Initialize component configs with defaults if not provided."""
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.selection is None:
            self.selection = SelectionConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()

        # Sync random states for reproducibility
        self.model.random_state = self.global_random_state

# Convenience factory functions for common configurations

def create_debug_config() -> PipelineConfig:
    """Create configuration optimized for debugging current issues."""
    return PipelineConfig(
        preprocessing=PreprocessingConfig(
            numerical_imputation="median",  # As requested
            debug_preprocessing=True,
            verbose_feature_names=True,
            strict_validation=False,  # More lenient for debugging
        ),
        selection=SelectionConfig(
            perm_threshold=0.0001,  # Very permissive for debugging
            shap_threshold=0.0001,  # Very permissive for debugging
            mode="union",  # More permissive
            min_features=20,  # Ensure reasonable number
            fallback_strategy="top_permutation",
        ),
        model=ModelConfig(
            cv_folds=3,  # Faster for debugging
            use_hyperopt=False,
        ),
        evaluation=EvaluationConfig(
            test_size=0.25,
            validation_size=None,  # Skip for debugging
        ),
        enable_feature_selection=True,
        enable_hyperparameter_tuning=False,
        debug_mode=True,
        log_level="DEBUG",
    )

def create_production_config() -> PipelineConfig:
    """Create configuration optimized for production deployment."""
    return PipelineConfig(
        preprocessing=PreprocessingConfig(
            numerical_imputation="iterative",  # More sophisticated
            debug_preprocessing=False,
            strict_validation=True,
        ),
        selection=SelectionConfig(
            perm_threshold=0.001,  # More permissive than original
            shap_threshold=0.001,  # More permissive than original
            mode="intersection",  # Conservative for production
            min_features=50,  # Ensure good coverage
            max_features=200,  # Cap for performance
            max_relative_regression=0.02,  # Stricter gating
        ),
        model=ModelConfig(
            cv_folds=10,
            use_hyperopt=True,
            hyperopt_max_evals=200,
        ),
        evaluation=EvaluationConfig(
            test_size=0.15,
            validation_size=0.2,
            additional_metrics=["mae", "r2", "mape"],
        ),
        enable_feature_selection=True,
        enable_hyperparameter_tuning=True,
        enable_model_comparison=True,
        debug_mode=False,
        save_intermediate_results=True,
    )

def create_rapid_prototyping_config() -> PipelineConfig:
    """Create configuration optimized for rapid prototyping and development."""
    return PipelineConfig(
        preprocessing=PreprocessingConfig(
            numerical_imputation="median",
            debug_preprocessing=True,
            max_safe_training_rows=10000,
            strict_validation=False,
        ),
        selection=SelectionConfig(
            perm_n_repeats=5,
            perm_max_samples=0.3,
            shap_nsamples=50,
            perm_threshold=0.0005,  # Very permissive
            shap_threshold=0.0005,  # Very permissive
            mode="union",  # Permissive for exploration
            min_features=15,
        ),
        model=ModelConfig(
            cv_folds=3,
            use_hyperopt=False,
        ),
        evaluation=EvaluationConfig(
            test_size=0.25,
            validation_size=None,
        ),
        enable_feature_selection=True,
        enable_hyperparameter_tuning=False,
        debug_mode=True,
    )

# Keep backwards compatibility
DEFAULTS = TrainingConfig()

# Export main configurations
__all__ = [
    "SelectionConfig",
    "PreprocessingConfig", 
    "ModelConfig",
    "EvaluationConfig",
    "PipelineConfig",
    "TrainingConfig",  # Keep for compatibility
    "create_debug_config",
    "create_production_config", 
    "create_rapid_prototyping_config",
    "DEFAULTS",
]
