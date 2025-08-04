# file: api/src//ml/models.py
from __future__ import annotations
from typing import Any, Dict
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import logging

def make_estimator(model_family: str, params: Dict[str, Any], enable_monitoring: bool = True) -> Any:
    """
    Enhanced estimator factory with convergence optimization and monitoring.

    CHANGES FROM ORIGINAL:
    - Added convergence optimization for linear models
    - Enhanced parameter handling for tree-based models
    - Added monitoring capabilities
    - Better error handling and logging
    - Fixed XGBoost early stopping issue by not setting early_stopping_rounds by default
    """
    logger = logging.getLogger(__name__)

    if enable_monitoring:
        logger.info(f"Creating {model_family} estimator with params: {params}")

    try:
        if model_family == "linear_ridge":
            base_params = {
                "alpha": params.get("alpha", 1.0),
                "random_state": params.get("random_state", 42),
                "solver": params.get("solver", "auto"),
                # Enhanced convergence parameters
                "max_iter": params.get("max_iter", 50000),  # INCREASED from default
                "tol": params.get("tol", 1e-6),  # TIGHTER tolerance
            }
            return Ridge(**base_params)

        elif model_family == "lasso":
            alpha = params.get("alpha", 0.001)
            base_params = {
                "alpha": alpha,
                "random_state": params.get("random_state", 42),
                "max_iter": params.get("max_iter", 50000),  # INCREASED
                "tol": params.get("tol", 1e-6),  # TIGHTER
                "selection": params.get("selection", "random"),  # Better for convergence
            }

            # SPECIAL HANDLING for very small alpha values
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
                "max_iter": params.get("max_iter", 50000),  # INCREASED
                "tol": params.get("tol", 1e-6),  # TIGHTER
                "selection": params.get("selection", "random"),
            }

            # SPECIAL HANDLING for extreme l1_ratio values
            if l1_ratio < 0.01 or l1_ratio > 0.99:
                base_params["max_iter"] = 75000
                logger.warning(f"Extreme l1_ratio ({l1_ratio}) detected, adjusting convergence parameters")

            return ElasticNet(**base_params)

        # Tree-based models with enhanced parameters
        elif model_family == "rf":
            base_params = {
                "n_estimators": params.get("n_estimators", 400),
                "max_depth": params.get("max_depth", None),
                "min_samples_leaf": params.get("min_samples_leaf", 1),
                "min_samples_split": params.get("min_samples_split", 2),
                "random_state": params.get("random_state", 42),
                "n_jobs": params.get("n_jobs", -1),
                "max_features": params.get("max_features", "sqrt"),  # ADDED for efficiency
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
                "reg_alpha": params.get("reg_alpha", 0.0),  # ADDED L1 reg
                "random_state": params.get("random_state", 42),
                "n_jobs": params.get("n_jobs", -1),
                "tree_method": "hist",  # ADDED for performance
                "eval_metric": "rmse",
                # CRITICAL FIX: Only set early_stopping_rounds if explicitly provided
                # This prevents the "Must have at least 1 validation dataset" error
            }

            # Only add early_stopping_rounds if explicitly provided and validation data will be available
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
                # ENHANCED stability parameters
                "min_child_samples": params.get("min_child_samples", 20),
                "min_split_gain": params.get("min_split_gain", 0.0),
                "feature_fraction": params.get("feature_fraction", 0.8),
                "bagging_fraction": params.get("bagging_fraction", 0.8),
                "bagging_freq": params.get("bagging_freq", 5),
                "verbosity": -1,
            }
            # Allow additional user parameters
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
                # ENHANCED convergence
                "od_type": "Iter",
                "od_wait": params.get("od_wait", 50),
            }
            return CatBoostRegressor(**base_params)

        else:
            raise ValueError(f"Unknown model family: {model_family}")

    except Exception as e:
        logger.error(f"Failed to create {model_family} estimator: {str(e)}")
        raise

def validate_model_convergence(model, X_train, y_train, model_family: str) -> Dict[str, Any]:
    """
    NEW FUNCTION: Validate that a fitted model has converged properly.

    ADD THIS to your models.py module.
    """
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)
    convergence_info = {"converged": True, "warnings": [], "metrics": {}}

    try:
        if model_family in ["lasso", "elasticnet"]:
            # Check if coordinate descent converged
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

        # Check for infinite or NaN coefficients
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
    NEW FUNCTION: Train model with comprehensive convergence monitoring.

    ADD THIS to your training pipeline.
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

    if not convergence_info["converged"]:
        logger.warning(f"Convergence issues in {model_family}: {convergence_info['warnings']}")

    return model, metrics, convergence_info

if __name__ == "__main__":
    print("Smoke OK:", make_estimator("linear_ridge", {"alpha": 1.0}))
