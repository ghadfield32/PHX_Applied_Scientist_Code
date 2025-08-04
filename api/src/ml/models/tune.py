# file: api/src//ml/tune.py
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
from api.src.ml.preprocessing.preprocessor import build_robust_preprocessor
from api.src.ml.preprocessing.feature_store.spec_builder import select_model_features, FeatureSpec
from api.src.ml.models.models import make_estimator


def _suggest_params_enhanced(trial: optuna.Trial, model_family: str) -> Dict[str, Any]:
    """
    REPLACE the existing _suggest_params function with this enhanced version.

    CHANGES:
    - Convergence-aware parameter ranges
    - Adaptive max_iter based on alpha values
    - Better search spaces for all models
    """
    if model_family == "linear_ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 50.0, log=True),
            "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr"]),
            "max_iter": trial.suggest_int("max_iter", 10000, 100000, step=10000),
            "tol": trial.suggest_float("tol", 1e-8, 1e-3, log=True),
        }

    if model_family == "lasso":
        alpha = trial.suggest_float("alpha", 1e-6, 1.0, log=True)
        # Adaptive max_iter based on alpha - CRITICAL IMPROVEMENT
        base_max_iter = 50000 if alpha >= 1e-4 else 100000
        return {
            "alpha": alpha,
            "max_iter": trial.suggest_int("max_iter", base_max_iter, base_max_iter * 2, step=10000),
            "tol": trial.suggest_float("tol", 1e-8, 1e-4, log=True),
            "selection": trial.suggest_categorical("selection", ["cyclic", "random"]),
        }

    if model_family == "elasticnet":
        alpha = trial.suggest_float("alpha", 1e-6, 1.0, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.01, 0.99)  # Avoid extreme values

        # Adaptive max_iter based on parameters - CRITICAL IMPROVEMENT
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

    # Enhanced tree model parameters
    if model_family == "rf":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        }

    if model_family == "xgb":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),  # ADDED L1 reg
        }

    if model_family == "lgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),  # ADDED L1 reg
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.1),
        }

    if model_family == "cat":
        return {
            "iterations": trial.suggest_int("iterations", 300, 1200, step=100),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "od_wait": trial.suggest_int("od_wait", 20, 100, step=10),
        }

    raise ValueError(f"Unknown model family: {model_family}")

# Keep the original function for backward compatibility
def _suggest_params(trial: optuna.Trial, model_family: str) -> Dict[str, Any]:
    """
    Original function kept for backward compatibility.
    Use _suggest_params_enhanced for better convergence.
    """
    return _suggest_params_enhanced(trial, model_family)

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
    Returns (fitted_pipeline, best_params, cv_rmse)
    """
    if model_family == "bayes_hier":
        raise ValueError("Use bayesian module for 'bayes_hier' training.")

    y = df[feature_spec.target].astype(float)
    X_df = select_model_features(df, feature_spec)

    # Ensure Season-ordered CV if available
    if "Season" in df.columns:
        df_sorted = df.sort_values("Season")
        X_df = select_model_features(df_sorted, feature_spec)
        y = df_sorted[feature_spec.target].astype(float)

    pre = build_robust_preprocessor(
        numerical_cols=feature_spec.numerical_features,
        ordinal_cols=feature_spec.ordinal_features,
        nominal_cols=feature_spec.nominal_features,
        model_type=model_family
    )

    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()

    cv = TimeSeriesSplit(n_splits=n_splits)
    scorer = make_scorer(_rmse)

    def objective(trial: optuna.Trial):
        params = {"random_state": random_state}
        params.update(_suggest_params_enhanced(trial, model_family))  # Use enhanced version
        est = make_estimator(model_family, params)
        pipe = Pipeline([("pre", pre), ("model", est)])
        scores = cross_val_score(pipe, X_df, y, cv=cv, scoring=scorer)
        return scores.mean()

    mlf_cb = MLflowCallback(metric_name="rmse_cv")
    study = optuna.create_study(direction="maximize", study_name=f"{model_family}_{feature_spec.target}")
    study.optimize(objective, n_trials=n_trials, callbacks=[mlf_cb])

    best_params = {"random_state": random_state}
    best_params.update(study.best_params)
    best_est = make_estimator(model_family, best_params)
    best_pipe = Pipeline([("pre", pre), ("model", best_est)])
    best_pipe.fit(X_df, y)

    return best_pipe, best_params, -study.best_value

if __name__ == "__main__":
    # Smoke test: tiny synthetic sample
    df = pd.DataFrame({
        "Season": ["2019-20"]*50 + ["2020-21"]*50,
        "PTS_x": np.random.randint(5, 25, 100),
        "Age_x": np.random.randint(20, 35, 100),
        "pos": np.random.choice(list("GFC"), 100),
        "aav": np.random.uniform(1e6, 10e6, 100),
    })
    from ml.feature_spec import build_feature_spec
    spec = build_feature_spec(df, target="aav")
    pipe, params, rmse = optimize(df, spec, model_family="linear_ridge", n_trials=2, n_splits=3)
    print("Smoke OK: best params:", params, "RMSE:", rmse) 
