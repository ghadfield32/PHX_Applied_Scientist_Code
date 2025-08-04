# file: api/src/ml/feature_store/feature_store.py
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
from typing import Optional

from api.src.ml.preprocessing.feature_store.spec_builder import FeatureSpec
from api.src.ml import config

class FeatureStore:
    """
    JSON-backed feature registry with simple staging:
      - root/model_family_target/Staging|Production/{feature_spec.json, meta.json}
    """
    def __init__(self, model_family: str, target: str, root_dir: Optional[Path] = None):
        self.root = Path(root_dir) if root_dir is not None else Path(config.FEATURE_STORE_DIR)
        self.base = self.root / f"{model_family}_{target}"
        self.base.mkdir(parents=True, exist_ok=True)

    def _stage_dir(self, stage: str) -> Path:
        p = self.base / stage
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_spec(self, spec: FeatureSpec, extra_meta: Optional[dict] = None, stage: str = "Production") -> None:
        stage_dir = self._stage_dir(stage)
        spec_path = stage_dir / "feature_spec.json"
        meta_path = stage_dir / "meta.json"

        spec_path.write_text(spec.to_json())

        meta = {
            "saved_at": datetime.utcnow().isoformat(),
            "stage": stage,
            "target": spec.target,
            "numerical": spec.numerical,
            "nominal_cats": spec.nominal_cats,
            "ordinal_map": spec.ordinal_map,
            "time_cols": spec.time_cols,
            "drops": spec.drops,
            "raw_features": getattr(spec, "raw_features", []),
            "encoded_feature_map": getattr(spec, "encoded_feature_map", {}),
            "clip_bounds": getattr(spec, "clip_bounds", {}),
            "feature_selection": getattr(spec, "feature_selection", None),
        }
        if extra_meta:
            meta.update(extra_meta)
        meta_path.write_text(json.dumps(meta, indent=2))

    def load_spec(self, stage: str = "Production") -> FeatureSpec:
        spec_path = self._stage_dir(stage) / "feature_spec.json"
        return FeatureSpec.from_json(spec_path.read_text())

    def promote(self, from_stage: str = "Staging", to_stage: str = "Production") -> None:
        src_dir = self._stage_dir(from_stage)
        dst_dir = self._stage_dir(to_stage)
        (dst_dir / "feature_spec.json").write_text((src_dir / "feature_spec.json").read_text())
        (dst_dir / "meta.json").write_text((src_dir / "meta.json").read_text())

