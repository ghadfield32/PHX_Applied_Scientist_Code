# file: api/src/ml/preprocessing/feature_store/spec_builder.py
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Sequence, Any
import json
from pathlib import Path
import pandas as pd

@dataclass
class FeatureSpec:
    # Core
    target: str
    numerical: List[str]
    nominal_cats: List[str]
    ordinal_map: Dict[str, List[str]]
    drops: List[str]
    time_cols: List[str]

    # Derived & fitted artifacts
    raw_features: List[str] = field(default_factory=list)
    encoded_feature_map: Dict[str, List[str]] = field(default_factory=dict)
    clip_bounds: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    feature_selection: Optional[Dict[str, Any]] = None

    # NEW: versioning & encoder state
    version: int = 2
    schema_hash: Optional[str] = None
    preprocessor_signature: Optional[Dict[str, Any]] = None

    # NEW: explicit encoder categories
    # - OHE: per raw nominal column -> list of categories (full list as seen by encoder)
    # - Ordinal: per raw ordinal column -> list of categories in encoder order
    ohe_categories: Dict[str, List[str]] = field(default_factory=dict)
    nominal_drop_strategy: str = "first"
    ordinal_categories: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def ordinal_cats(self) -> List[str]:
        return list(self.ordinal_map.keys())

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "FeatureSpec":
        d = json.loads(s)
        # Back-compat defaults
        return FeatureSpec(
            target=d["target"],
            numerical=d.get("numerical", []),
            nominal_cats=d.get("nominal_cats", []),
            ordinal_map=d.get("ordinal_map", {}),
            drops=d.get("drops", []),
            time_cols=d.get("time_cols", []),
            raw_features=d.get("raw_features", []),
            encoded_feature_map=d.get("encoded_feature_map", {}),
            clip_bounds=d.get("clip_bounds", {}),
            feature_selection=d.get("feature_selection"),
            version=d.get("version", 1),
            schema_hash=d.get("schema_hash"),
            preprocessor_signature=d.get("preprocessor_signature"),
            ohe_categories=d.get("ohe_categories", {}),
            nominal_drop_strategy=d.get("nominal_drop_strategy", "first"),
            ordinal_categories=d.get("ordinal_categories", {}),
        )


def _load_ordinal_override(path: Optional[Path]) -> Dict[str, List[str]]:
    if path and path.exists():
        text = path.read_text()
        try:
            if path.suffix.lower() == ".json":
                return json.loads(text)
            mapping: Dict[str, List[str]] = {}
            for line in text.splitlines():
                if ":" in line and "[" in line and "]" in line:
                    k, rest = line.split(":", 1)
                    vals = rest[rest.find("[")+1:rest.find("]")].split(",")
                    mapping[k.strip()] = [v.strip().strip("'\"") for v in vals if v.strip()]
            return mapping
        except Exception:
            return {}
    return {}

def _detect_time_cols(df: pd.DataFrame) -> List[str]:
    time_cols: List[str] = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            time_cols.append(c)
        else:
            low = c.lower()
            if any(tok in low for tok in ["date", "season", "year", "start", "end", "return"]):
                time_cols.append(c)
    seen = set(); uniq = []
    for c in time_cols:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return uniq

def _classify_raw_features(
    df: pd.DataFrame,
    raw_features: Sequence[str],
    ordinal_override_path: Optional[Path] = None,
):
    cols = [c for c in raw_features if c in df.columns]
    time_cols = [c for c in _detect_time_cols(df) if c in cols]
    cols_wo_time = [c for c in cols if c not in time_cols]

    ordinal_override = _load_ordinal_override(ordinal_override_path)

    numerical: List[str] = []
    nominal: List[str] = []
    ordinal_map: Dict[str, List[str]] = {}

    for c in cols_wo_time:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            numerical.append(c)
        else:
            if isinstance(s.dtype, pd.CategoricalDtype):
                cat = s.dtype
                if getattr(cat, "ordered", False) and len(list(cat.categories)) > 0:
                    ordinal_map[c] = [str(v) for v in list(cat.categories)]
                else:
                    nominal.append(c)
            else:
                nominal.append(c)

    for col, order in ordinal_override.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=order, ordered=True)
            if col in nominal:
                nominal.remove(col)
            ordinal_map[col] = [str(v) for v in order]

    numerical = sorted(set(numerical))
    nominal = sorted(set(nominal))
    ordinal_map = {k: ordinal_map[k] for k in sorted(ordinal_map.keys())}
    time_cols = [c for c in df.columns if c in time_cols]
    return numerical, nominal, ordinal_map, time_cols

def build_feature_spec_from_schema_and_preprocessor(
    df: pd.DataFrame,
    target: str,
    schema,
    preprocessor,
    permutation_importance_df: Optional[pd.DataFrame] = None,
    shap_importance_df: Optional[pd.DataFrame] = None,
    final_features: Optional[List[str]] = None,
    perm_thresh: float | None = None,
    shap_thresh: float | None = None,
    mode: str = "intersection",
    clip_bounds: Optional[Dict[str, tuple[Optional[float], Optional[float]]]] = None,
    ordinal_override_path: Optional[Path] = None,
    drop_exact: Sequence[str] = ("player_id", "team_id", "_uid", "_merge"),
    exclude_prefixes: Sequence[str] = ("_raw_",),
) -> FeatureSpec:
    """
    Build a complete FeatureSpec centered on schema + fitted preprocessor.
    Captures:
      - raw feature order from schema.model_features()
      - encoded name map (anchored on transformer prefixes)
      - OHE categories and drop strategy
      - Ordinal categories from the fitted encoder
      - numeric clip bounds
      - selection metadata (perm/SHAP thresholds and final_features)
      - schema hash + a small preprocessor signature for traceability
    """
    raw_features = schema.model_features(include_target=False)
    numerical, nominal_cats, ordinal_map, time_cols = _classify_raw_features(
        df, raw_features, ordinal_override_path
    )

    # --- Encoded name mapping (anchored) ---
    encoded_feature_map: Dict[str, List[str]] = {}
    try:
        encoded_names = list(preprocessor.get_feature_names_out())
    except AttributeError:
        encoded_names = []

    def _is_num(name: str, raw: str) -> bool:
        return name.endswith(f"__{raw}") and name.startswith("num__")
    def _is_ord(name: str, raw: str) -> bool:
        return name.endswith(f"__{raw}") and name.startswith("ord__")
    def _is_nom(name: str, raw: str) -> bool:
        return name.startswith(f"nom__{raw}_")

    for raw in raw_features:
        matches = [n for n in encoded_names if (_is_num(n, raw) or _is_ord(n, raw) or _is_nom(n, raw))]
        if matches:
            encoded_feature_map[raw] = sorted(matches)

    # --- Clip bounds ---
    clip_bounds_dict: Dict[str, Dict[str, Optional[float]]] = {}
    if clip_bounds:
        for feat, (low, high) in clip_bounds.items():
            clip_bounds_dict[feat] = {"lower": low, "upper": high}

    # --- Selection metadata ---
    fs_meta: Dict[str, Any] = {}
    if permutation_importance_df is not None:
        fs_meta["permutation_importance"] = {
            row["feature"]: float(row["importance_mean"])
            for _, row in permutation_importance_df.iterrows()
        }
    if shap_importance_df is not None:
        fs_meta["shap_importance"] = {
            row["feature"]: float(row["shap_importance"])
            for _, row in shap_importance_df.iterrows()
        }
    if final_features is not None:
        fs_meta["final_features"] = list(final_features)
        fs_meta["mode"] = mode
        if perm_thresh is not None:
            fs_meta["permutation_threshold"] = perm_thresh
        if shap_thresh is not None:
            fs_meta["shap_threshold"] = shap_thresh

    # --- OHE & Ordinal categories from fitted encoders ---
    ohe_categories: Dict[str, List[str]] = {}
    ordinal_categories: Dict[str, List[str]] = {}
    nominal_drop_strategy = "first"

    try:
        # Find declared columns per transformer
        tfms = dict((name, (trans, cols)) for name, trans, cols in preprocessor.transformers_)
        # OHE
        if "nom" in tfms:
            nom_pipe, nom_cols = tfms["nom"]
            if isinstance(nom_pipe, Pipeline) and "encode" in nom_pipe.named_steps:
                ohe: OneHotEncoder = nom_pipe.named_steps["encode"]
                nominal_drop_strategy = getattr(ohe, "drop", "first") if ohe.drop is not None else "none"
                # categories_ aligns with nom_cols order
                for col, cats in zip(nom_cols, ohe.categories_):
                    ohe_categories[col] = [str(c) for c in list(cats)]
        # Ordinal
        if "ord" in tfms:
            ord_pipe, ord_cols = tfms["ord"]
            if isinstance(ord_pipe, Pipeline) and "encode" in ord_pipe.named_steps:
                ord_enc: OrdinalEncoder = ord_pipe.named_steps["encode"]
                for col, cats in zip(ord_cols, ord_enc.categories_):
                    ordinal_categories[col] = [str(c) for c in list(cats)]
    except Exception:
        # Leave empty if not available; spec stays usable
        pass

    # --- Schema hash & preprocessor signature ---
    try:
        import hashlib
        # Hash schema feature list + observed dtypes (if present)
        dtype_map = {c: str(df[c].dtype) for c in raw_features if c in df.columns}
        payload = json.dumps(
            {"features": raw_features, "dtypes": dtype_map, "target": target},
            sort_keys=True,
        ).encode("utf-8")
        schema_hash = hashlib.sha256(payload).hexdigest()[:16]
    except Exception:
        schema_hash = None

    preprocessor_signature = {
        "verbose_feature_names_out": getattr(preprocessor, "verbose_feature_names_out", True),
        "sparse_threshold": getattr(preprocessor, "sparse_threshold", 0.0),
        "has_num": "num" in [n for n, *_ in preprocessor.transformers_],
        "has_ord": "ord" in [n for n, *_ in preprocessor.transformers_],
        "has_nom": "nom" in [n for n, *_ in preprocessor.transformers_],
    }

    return FeatureSpec(
        target=target,
        numerical=numerical,
        nominal_cats=nominal_cats,
        ordinal_map=ordinal_map,
        drops=list(drop_exact),
        time_cols=time_cols,
        raw_features=raw_features,
        encoded_feature_map=encoded_feature_map,
        clip_bounds=clip_bounds_dict,
        feature_selection=fs_meta if fs_meta else None,
        version=2,
        schema_hash=schema_hash,
        preprocessor_signature=preprocessor_signature,
        ohe_categories=ohe_categories,
        nominal_drop_strategy=str(nominal_drop_strategy),
        ordinal_categories=ordinal_categories,
    )



def select_model_features(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """
    Deterministic selection for inference/training from a preprocessed df.
    Prefers explicit final_features if present.
    """
    if spec.feature_selection and spec.feature_selection.get("final_features"):
        feature_list = spec.feature_selection["final_features"]
    else:
        feature_list = []
        for raw in spec.raw_features:
            feature_list.extend(spec.encoded_feature_map.get(raw, []))
        if not feature_list:
            feature_list = spec.raw_features

    missing = set(feature_list) - set(df.columns)
    if missing:
        raise KeyError(f"select_model_features: missing columns: {missing}")
    return df[feature_list].copy()
