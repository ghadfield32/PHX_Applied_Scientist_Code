# file: api/src/ml/preprocessing/feature_store/contract_selector.py
from __future__ import annotations
from typing import Iterable, List, Optional
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureContractSelector(BaseEstimator, TransformerMixin):
    """
    Enforce the encoded feature contract inside a sklearn Pipeline.

    - Expects a pandas.DataFrame with encoded columns (use preprocessor.set_output('pandas')).
    - Selects final_features in-order; can fail-fast on missing to surface drift.
    - Emits concise debug (missing/extra) without “fixing” anything.
    """
    def __init__(
        self,
        final_features: Optional[Iterable[str]] = None,
        raise_on_missing: bool = True,
        verbose: bool = True,
        name: str = "FeatureContractSelector",
    ):
        # IMPORTANT for sklearn clone compatibility:
        # Do NOT modify params here. Just store them as-is.
        self.final_features = final_features
        self.raise_on_missing = raise_on_missing
        self.verbose = verbose
        self.name = name

        # Learned/derived attrs (set in fit)
        self.input_feature_names_ = None
        self.final_features_ = None  # resolved list or None

    def fit(self, X, y=None):
        if not hasattr(X, "columns"):
            raise TypeError(
                f"{self.name}: expected pandas.DataFrame input. "
                f"Ensure the preprocessor uses set_output(transform='pandas')."
            )
        self.input_feature_names_ = list(X.columns)

        # Resolve final_features lazily and store learned state
        if self.final_features is None:
            self.final_features_ = None
        else:
            self.final_features_ = list(self.final_features)

        return self

    def transform(self, X):
        if not hasattr(X, "columns"):
            raise TypeError(
                f"{self.name}: expected pandas.DataFrame input at transform-time too."
            )

        # If no contract provided: pass-through (but log once)
        if not self.final_features_:
            if self.verbose:
                print(f"[{self.name}] No final_features provided; passing through {X.shape[1]} columns.")
            return X

        missing = [c for c in self.final_features_ if c not in X.columns]
        extra = [c for c in X.columns if c not in self.final_features_]

        if self.verbose:
            print(f"[{self.name}] in={X.shape[1]}  select={len(self.final_features_)}  "
                  f"missing={len(missing)}  extra={len(extra)}")

        if missing and self.raise_on_missing:
            head = ", ".join(missing[:10])
            more = "..." if len(missing) > 10 else ""
            raise KeyError(f"{self.name}: encoded features missing: [{head}{more}]")

        # If we get here with missing and lenient mode, intersect (no silent fill)
        select_cols = [c for c in self.final_features_ if c in X.columns]
        return X[select_cols]

    def get_feature_names_out(self, input_features=None) -> List[str]:
        if self.final_features_ is not None:
            return self.final_features_
        return self.input_feature_names_ or []

    # Debug helper: never called by sklearn; safe to use in your code
    def debug_contract_state(self):
        def _safe_len(obj):
            try:
                return len(obj)
            except Exception:
                return "n/a"
        print(
            f"[{self.name}] param.final_features: id={id(self.final_features)} "
            f"type={type(self.final_features).__name__} len={_safe_len(self.final_features)}"
        )
        print(
            f"[{self.name}] learned.final_features_: id={id(self.final_features_)} "
            f"type={type(self.final_features_).__name__} len={_safe_len(self.final_features_)}"
        )
