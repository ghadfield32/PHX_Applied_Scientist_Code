# file: api/src//ml/models.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit
import xgboost as xgb
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import logging
import numpy as np

def make_estimator(model_family: str, params: Dict[str, Any], enable_monitoring: bool = True) -> Any:
    """
    Enhanced estimator factory with stacking ensemble support.

    NEW FEATURES:
    - Stacking ensemble models with configurable base learners and meta-learner
    - Support for both linear and tree-based meta-learners
    - Proper cross-validation strategy for stacking
    """
    logger = logging.getLogger(__name__)

    if enable_monitoring:
        logger.info(f"Creating {model_family} estimator with params: {params}")

    try:
        if model_family == "stacking":
            return _create_stacking_estimator(params, logger)

        # Existing model families (unchanged)
        elif model_family == "linear_ridge":
            base_params = {
                "alpha": params.get("alpha", 1.0),
                "random_state": params.get("random_state", 42),
                "solver": params.get("solver", "auto"),
                "max_iter": params.get("max_iter", 50000),
                "tol": params.get("tol", 1e-6),
            }
            return Ridge(**base_params)

        elif model_family == "lasso":
            alpha = params.get("alpha", 0.001)
            base_params = {
                "alpha": alpha,
                "random_state": params.get("random_state", 42),
                "max_iter": params.get("max_iter", 50000),
                "tol": params.get("tol", 1e-6),
                "selection": params.get("selection", "random"),
            }

            if alpha < 1e-5:
                base_params["max_iter"] = 100000
                logger.warning(f"Very small alpha ({alpha}) detected, increasing max_iter to {base_params['max_iter']}")

            return Lasso(**base_params)

        elif model_family == "elasticnet":
            alpha = params.get("alpha", 0.001)
            l1_ratio = params.get("l1_ratio", 0.5)

            base_params = {
                "alpha": alpha,
                "l1_ratio": l1_ratio,
                "random_state": params.get("random_state", 42),
                "max_iter": params.get("max_iter", 50000),
                "tol": params.get("tol", 1e-6),
                "selection": params.get("selection", "random"),
            }

            if l1_ratio < 0.01 or l1_ratio > 0.99:
                base_params["max_iter"] = 75000
                logger.warning(f"Extreme l1_ratio ({l1_ratio}) detected, adjusting convergence parameters")

            return ElasticNet(**base_params)

        elif model_family == "rf":
            base_params = {
                "n_estimators": params.get("n_estimators", 400),
                "max_depth": params.get("max_depth", None),
                "min_samples_leaf": params.get("min_samples_leaf", 1),
                "min_samples_split": params.get("min_samples_split", 2),
                "random_state": params.get("random_state", 42),
                "n_jobs": params.get("n_jobs", -1),
                "max_features": params.get("max_features", "sqrt"),
                "bootstrap": params.get("bootstrap", True),
            }
            return RandomForestRegressor(**base_params)

        elif model_family == "xgb":
            base_params = {
                "n_estimators": params.get("n_estimators", 600),
                "max_depth": params.get("max_depth", 6),
                "learning_rate": params.get("learning_rate", 0.05),
                "subsample": params.get("subsample", 0.8),
                "colsample_bytree": params.get("colsample_bytree", 0.8),
                "reg_lambda": params.get("reg_lambda", 1.0),
                "reg_alpha": params.get("reg_alpha", 0.0),
                "random_state": params.get("random_state", 42),
                "n_jobs": params.get("n_jobs", -1),
                "tree_method": "hist",
                "eval_metric": "rmse",
            }

            if "early_stopping_rounds" in params:
                base_params["early_stopping_rounds"] = params["early_stopping_rounds"]

            return XGBRegressor(**base_params)

        elif model_family == "lgbm":
            base_params = {
                "n_estimators": params.get("n_estimators", 1000),
                "max_depth": params.get("max_depth", -1),
                "learning_rate": params.get("learning_rate", 0.05),
                "subsample": params.get("subsample", 0.8),
                "colsample_bytree": params.get("colsample_bytree", 0.8),
                "reg_lambda": params.get("reg_lambda", 1.0),
                "reg_alpha": params.get("reg_alpha", 0.0),
                "random_state": params.get("random_state", 42),
                "n_jobs": -1,
                "min_child_samples": params.get("min_child_samples", 20),
                "min_split_gain": params.get("min_split_gain", 0.0),
                "feature_fraction": params.get("feature_fraction", 0.8),
                "bagging_fraction": params.get("bagging_fraction", 0.8),
                "bagging_freq": params.get("bagging_freq", 5),
                "verbosity": -1,
            }
            extra_kwargs = {k: v for k, v in params.items() if k not in base_params}
            base_params.update(extra_kwargs)
            return LGBMRegressor(**base_params)

        elif model_family == "cat":
            base_params = {
                "depth": params.get("depth", 6),
                "learning_rate": params.get("learning_rate", 0.05),
                "iterations": params.get("iterations", 700),
                "l2_leaf_reg": params.get("l2_leaf_reg", 3.0),
                "random_state": params.get("random_state", 42),
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "verbose": False,
                "allow_writing_files": False,
                "od_type": "Iter",
                "od_wait": params.get("od_wait", 50),
            }
            return CatBoostRegressor(**base_params)

        else:
            raise ValueError(f"Unknown model family: {model_family}")

    except Exception as e:
        logger.error(f"Failed to create {model_family} estimator: {str(e)}")
        raise


def _create_stacking_estimator(params: Dict[str, Any], logger) -> StackingRegressor:
    """
    Create a stacking ensemble with configurable base learners and meta-learner.

    Expected params structure:
    {
        "base_estimators": ["linear_ridge", "xgb", "lgbm", "cat"],  # List of base model families
        "meta_learner": "linear_ridge",  # Meta-learner family
        "cv_folds": 5,  # Cross-validation folds for stacking
        "cv_strategy": "kfold",  # "kfold" or "time_series"
        "passthrough": False,  # Whether to include original features
        "random_state": 42,
        # Base model parameters (nested)
        "base_params": {
            "linear_ridge": {"alpha": 1.0},
            "xgb": {"n_estimators": 300, "learning_rate": 0.1},
            "lgbm": {"n_estimators": 300, "learning_rate": 0.1},
            "cat": {"iterations": 300, "learning_rate": 0.1}
        },
        # Meta-learner parameters
        "meta_params": {"alpha": 0.1}
    }
    """
    random_state = params.get("random_state", 42)

    # Default configuration
    base_families = params.get("base_estimators", ["linear_ridge", "xgb", "lgbm", "cat"])
    meta_family = params.get("meta_learner", "linear_ridge")
    cv_folds = params.get("cv_folds", 5)
    cv_strategy = params.get("cv_strategy", "kfold")
    passthrough = params.get("passthrough", False)

    # Get nested parameters
    base_params_dict = params.get("base_params", {})
    meta_params = params.get("meta_params", {})

    logger.info(f"Creating stacking ensemble with base_estimators={base_families}, meta_learner={meta_family}")

    # Create base estimators
    estimators = []
    for family in base_families:
        try:
            # Get family-specific parameters or use defaults
            family_params = base_params_dict.get(family, {})
            family_params["random_state"] = random_state

            base_est = make_estimator(family, family_params, enable_monitoring=False)
            estimators.append((family, base_est))
            logger.info(f"Added base estimator: {family}")
        except Exception as e:
            logger.warning(f"Failed to create base estimator {family}: {e}")
            continue

    if not estimators:
        raise ValueError("No valid base estimators could be created for stacking")

    # Create meta-learner
    meta_params["random_state"] = random_state
    final_estimator = make_estimator(meta_family, meta_params, enable_monitoring=False)
    logger.info(f"Created meta-learner: {meta_family}")

    # Configure cross-validation strategy
    if cv_strategy == "time_series":
        cv = TimeSeriesSplit(n_splits=cv_folds)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Create stacking regressor
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=cv,
        passthrough=passthrough,
        n_jobs=-1
    )

    # Add metadata for tracking
    stacking_regressor._base_families = base_families
    stacking_regressor._meta_family = meta_family
    stacking_regressor._cv_strategy = cv_strategy
    stacking_regressor._cv_folds = cv_folds

    return stacking_regressor


def get_stacking_feature_importance(stacking_model) -> Dict[str, Any]:
    """
    Extract feature importance information from a fitted stacking model.

    Returns:
        Dictionary containing base model contributions and meta-learner importance
    """
    if not hasattr(stacking_model, 'estimators_'):
        raise ValueError("Stacking model must be fitted first")

    importance_info = {
        "base_estimators": {},
        "meta_learner": {},
        "base_predictions_weight": None
    }

    # Get base estimator information
    for i, (name, estimator) in enumerate(stacking_model.estimators_):
        est_info = {"name": name, "type": type(estimator).__name__}

        # Try to get feature importance if available
        if hasattr(estimator, 'feature_importances_'):
            est_info["feature_importances"] = estimator.feature_importances_.tolist()
        elif hasattr(estimator, 'coef_'):
            est_info["coefficients"] = estimator.coef_.tolist()

        importance_info["base_estimators"][name] = est_info

    # Get meta-learner information
    meta = stacking_model.final_estimator_
    meta_info = {"type": type(meta).__name__}

    if hasattr(meta, 'coef_'):
        # For linear meta-learners, coefficients show how much each base model contributes
        meta_info["base_model_weights"] = meta.coef_.tolist()
        importance_info["base_predictions_weight"] = dict(zip(
            [name for name, _ in stacking_model.estimators_], 
            meta.coef_.tolist()
        ))
    elif hasattr(meta, 'feature_importances_'):
        meta_info["feature_importances"] = meta.feature_importances_.tolist()

    if hasattr(meta, 'intercept_'):
        meta_info["intercept"] = float(meta.intercept_)

    importance_info["meta_learner"] = meta_info

    return importance_info


def validate_model_convergence(model, X_train, y_train, model_family: str) -> Dict[str, Any]:
    """
    Enhanced convergence validation with stacking support.
    """
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)
    convergence_info = {"converged": True, "warnings": [], "metrics": {}}

    try:
        if model_family == "stacking":
            # Validate stacking model convergence
            convergence_info["metrics"]["n_base_estimators"] = len(model.estimators_)
            convergence_info["metrics"]["base_families"] = getattr(model, '_base_families', [])
            convergence_info["metrics"]["meta_family"] = getattr(model, '_meta_family', 'unknown')

            # Check if any base models failed
            if hasattr(model, 'estimators_') and len(model.estimators_) < len(getattr(model, '_base_families', [])):
                convergence_info["warnings"].append("Some base estimators failed to train")

            # Validate meta-learner convergence
            meta_conv = validate_model_convergence(
                model.final_estimator_, X_train, y_train, 
                getattr(model, '_meta_family', 'unknown')
            )
            convergence_info["meta_learner_convergence"] = meta_conv

            if not meta_conv["converged"]:
                convergence_info["converged"] = False
                convergence_info["warnings"].append("Meta-learner convergence issues")

        elif model_family in ["lasso", "elasticnet"]:
            if hasattr(model, 'n_iter_'):
                convergence_info["metrics"]["n_iterations"] = model.n_iter_
                if model.n_iter_ >= model.max_iter - 1:
                    convergence_info["converged"] = False
                    convergence_info["warnings"].append(f"Model reached max_iter ({model.max_iter}) without convergence")

        elif model_family == "linear_ridge":
            if hasattr(model, 'n_iter_'):
                convergence_info["metrics"]["n_iterations"] = model.n_iter_

        # Check training performance as convergence indicator
        train_score = model.score(X_train, y_train)
        convergence_info["metrics"]["train_r2"] = train_score

        if train_score < 0:
            convergence_info["warnings"].append(f"Negative R² score ({train_score:.4f}) may indicate convergence issues")

        # Check for infinite or NaN coefficients (if applicable)
        if hasattr(model, 'coef_'):
            n_inf_coef = np.sum(~np.isfinite(model.coef_))
            if n_inf_coef > 0:
                convergence_info["converged"] = False
                convergence_info["warnings"].append(f"Found {n_inf_coef} non-finite coefficients")

        logger.info(f"Convergence check for {model_family}: {convergence_info}")

    except Exception as e:
        logger.warning(f"Failed to validate convergence for {model_family}: {str(e)}")
        convergence_info["warnings"].append(f"Convergence validation failed: {str(e)}")

    return convergence_info


def train_with_convergence_monitoring(model_family: str, X_train, y_train, X_test, y_test, 
                                    params: dict, spec) -> tuple:
    """
    Enhanced training with stacking support and convergence monitoring.
    """
    import time
    import logging
    import numpy as np
    from sklearn.metrics import mean_squared_error, r2_score

    logger = logging.getLogger(__name__)

    # Create and train model
    model = make_estimator(model_family, params, enable_monitoring=True)

    # Time the training
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Validate convergence
    convergence_info = validate_model_convergence(model, X_train, y_train, model_family)

    # Evaluate performance
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, train_preds)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, test_preds)),
        "train_r2": r2_score(y_train, train_preds),
        "test_r2": r2_score(y_test, test_preds),
        "training_time": training_time,
        "converged": convergence_info["converged"],
        "convergence_warnings": len(convergence_info["warnings"])
    }

    # Add stacking-specific metrics
    if model_family == "stacking":
        try:
            importance_info = get_stacking_feature_importance(model)
            metrics["base_model_weights"] = importance_info.get("base_predictions_weight", {})
            metrics["n_base_estimators"] = len(model.estimators_)
        except Exception as e:
            logger.warning(f"Failed to extract stacking importance: {e}")

    if not convergence_info["converged"]:
        logger.warning(f"Convergence issues in {model_family}: {convergence_info['warnings']}")

    return model, metrics, convergence_info


if __name__ == "__main__":
    # Test stacking model creation
    stacking_params = {
        "base_estimators": ["linear_ridge", "xgb", "lgbm"],
        "meta_learner": "linear_ridge",
        "cv_folds": 3,
        "base_params": {
            "linear_ridge": {"alpha": 1.0},
            "xgb": {"n_estimators": 100},
            "lgbm": {"n_estimators": 100}
        },
        "meta_params": {"alpha": 0.1}
    }

    stacking_model = make_estimator("stacking", stacking_params)
    print("Stacking model created successfully:", stacking_model)
    print("Base estimators:", [name for name, _ in stacking_model.estimators])
    print("Meta-learner:", type(stacking_model.final_estimator).__name__)
