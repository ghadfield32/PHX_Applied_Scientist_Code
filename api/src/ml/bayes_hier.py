# file: api/src/ml/bayes_hier.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Sequence
import numpy as np
import pandas as pd
import arviz as az
import pymc as pm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from api.src.ml import config
from api.src.ml.config import TrainingConfig, get_master_parquet_path

ARTIFACTS_DIR = config.ARTIFACTS_DIR

from api.src.ml.preprocessing.feature_store.spec_builder import FeatureSpec

def _prepare_design(
    df: pd.DataFrame,
    spec: FeatureSpec,
    group_cols: Sequence[str] = (),
    verbose: bool = True,
) -> Tuple[np.ndarray, list[str], np.ndarray, dict]:
    """
    Build the design matrix for PyMC with explicit debug logging:
      - Numeric: median-impute + z-score
      - Nominal: One-hot (dense)
      - Ordinal: integer codes based on spec.ordinal_map
      - Groups: integer indices per requested group columns (for random intercepts)
    """
    target = spec.target
    if target not in df.columns:
        raise ValueError(
            f"[bayes/_prepare_design] Target '{target}' not in df. "
            f"First columns: {list(df.columns)[:20]}"
        )

    # Debug: target NA count before any filtering
    if verbose:
        na_tgt = df[target].isna().sum()
        print(f"[bayes] target='{target}' rows={len(df)}  missing_target={na_tgt}")

    y = df[target].astype(float).values

    num_cols = list(spec.numerical or [])
    nom_cols = list(spec.nominal_cats or [])
    ord_cols = list(spec.ordinal_cats or [])

    if verbose:
        print(f"[bayes] numerical({len(num_cols)}): {num_cols[:10]}")
        print(f"[bayes] nominal({len(nom_cols)}): {nom_cols[:10]}")
        print(f"[bayes] ordinal({len(ord_cols)}): {ord_cols[:10]}")

    # numeric (report missing before impute)
    X_num = np.empty((len(df), 0))
    if num_cols:
        if verbose:
            n_missing = pd.isna(df[num_cols]).sum().sum()
            print(f"[bayes] numeric missing before impute: {n_missing}")
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        X_num = sc.fit_transform(imp.fit_transform(df[num_cols]))

    # nominal
    X_nom = np.empty((len(df), 0))
    nom_feature_names: list[str] = []
    if nom_cols:
        # Modern scikit-learn: `sparse_output` controls dense/sparse. :contentReference[oaicite:7]{index=7}
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_nom = ohe.fit_transform(df[nom_cols].astype("string"))
        cats = ohe.categories_
        for c, cat_vals in zip(nom_cols, cats):
            nom_feature_names.extend([f"nom:{c}={v}" for v in cat_vals])

    # ordinal
    X_ord = np.empty((len(df), 0))
    ord_feature_names: list[str] = []
    if ord_cols:
        mats = []
        for col in ord_cols:
            order = spec.ordinal_map[col]
            s = pd.Categorical(df[col].astype("string"), categories=order, ordered=True)
            mats.append(s.codes.reshape(-1, 1))
            ord_feature_names.append(f"ord:{col}")
        X_ord = np.hstack(mats) if mats else np.empty((len(df), 0))

    X = np.hstack([Z for Z in (X_num, X_nom, X_ord) if Z.size > 0])
    feature_names = [f"num:{c}" for c in num_cols] + nom_feature_names + ord_feature_names
    if verbose:
        print(f"[bayes] X shape after concat: {X.shape}")

    # group indices
    groups: dict = {}
    for gcol in group_cols:
        if gcol in df.columns:
            levels = pd.Index(df[gcol].astype("string").unique().tolist())
            level_map = {v: i for i, v in enumerate(levels)}
            idx = df[gcol].astype("string").map(level_map).values
            groups[gcol] = {"levels": levels.tolist(), "index": idx}
            if verbose:
                print(f"[bayes] group='{gcol}' levels={len(levels)}")
        else:
            if verbose:
                print(f"[bayes] group column '{gcol}' not found; skipping")

    return X, feature_names, y, groups


def train_bayesian(
    df: pd.DataFrame,
    spec: FeatureSpec,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.9,
    chains: int = 2,
    cores: int = 2,
    group_cols: Sequence[str] = (),
    random_seed: int = 42,
    out_dir: Optional[Path] = None,
) -> Tuple[Path, az.InferenceData]:
    """
    Hierarchical linear regression with random intercepts over given group columns.
      y ~ Normal(mu, sigma)
      mu = alpha + X @ beta + sum_g a_g[g_idx]
    """
    X, feat_names, y, groups = _prepare_design(df, spec, group_cols=group_cols, verbose=True)

    coords = {"obs": np.arange(len(y)), "feat": np.arange(X.shape[1])}
    for gname, gobj in groups.items():
        coords[gname] = np.arange(len(gobj["levels"]))

    with pm.Model(coords=coords) as model:
        beta = pm.Normal("beta", 0.0, 1.0, dims=("feat",))
        alpha = pm.Normal("alpha", 0.0, 1.0)
        sigma = pm.HalfNormal("sigma", 1.0)

        mu = alpha + pm.math.dot(X, beta)
        for gname, gobj in groups.items():
            a_g = pm.Normal(f"a_{gname}", 0.0, 1.0, dims=(gname,))
            mu = mu + a_g[gobj["index"]]

        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # pm.sample returns ArviZ InferenceData; we pass hyperparams through
        idata = pm.sample(
            draws=draws, tune=tune, target_accept=target_accept,
            chains=chains, cores=cores, random_seed=random_seed, return_inferencedata=True
        )

    out = out_dir or (ARTIFACTS_DIR / f"bayes_hier_{spec.target}")
    out.mkdir(parents=True, exist_ok=True)

    # Save artifacts using ArviZ NetCDF
    az.to_netcdf(idata, out / "posterior.nc")
    (out / "feature_names.txt").write_text("\n".join(feat_names))
    (out / "config_groups.json").write_text(_json_dumps({k: v["levels"] for k, v in groups.items()}))

    return out, idata

def predict_bayesian(
    df: pd.DataFrame,
    spec: FeatureSpec,
    artifact_dir: Path,
    group_cols: Sequence[str] = (),
) -> pd.Series:
    """
    Posterior mean prediction (no noise term). Uses the same design building.
    """
    X, feat_names, _, groups = _prepare_design(df, spec, group_cols=group_cols, verbose=False)
    idata = az.from_netcdf(artifact_dir / "posterior.nc")
    post = idata.posterior  # dims: chain, draw, ...
    # stack samples: [S], [feat, S]
    beta = post["beta"].stack(sample=("chain", "draw")).values  # shape [feat, S]
    alpha = post["alpha"].stack(sample=("chain", "draw")).values  # [S]
    mu = X @ beta + alpha  # [n, S]
    for gname, gobj in groups.items():
        # a_{gname}: [levels, S] → index by obs' group index
        a_g = post[f"a_{gname}"].stack(sample=("chain", "draw")).values
        mu += a_g[gobj["index"], :]

    pred_mean = mu.mean(axis=1)
    return pd.Series(pred_mean, index=df.index, name=f"pred_{spec.target}")

def _json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2)




def debug_env_report():
    """Print versions & environment facts helpful for Bayesian runs."""
    import sys
    import sklearn
    print("[env] python:", sys.version.replace("\n", " "))
    print("[env] pymc:", pm.__version__)
    print("[env] arviz:", az.__version__)
    print("[env] numpy:", np.__version__)
    print("[env] sklearn:", sklearn.__version__)
    # Non-fatal: compiler availability (PyTensor warning may appear on import)
    try:
        import shutil
        has_gpp = shutil.which("g++") is not None
        print(f"[env] g++ available: {has_gpp}")
    except Exception as e:
        print(f"[env] g++ check failed: {e} (non-fatal)")

def run_bayes_realdata_smoke() -> dict:
    """
    Thin wrapper to exercise the REAL data pipeline via train(cfg) from this module.
    Avoids NameError by lazily importing train inside the function to prevent import cycles.
    """
    from api.src.ml.config import validate_configuration, get_master_parquet_path
    from api.src.ml import config as _cfg

    debug_env_report()

    # 0) Validate wiring
    ok = validate_configuration()
    if not ok:
        raise RuntimeError("[smoke] Configuration validation failed.")
    print("✓ Configuration validated")

    # 1) Dataset preflight
    data_path = get_master_parquet_path()
    alt_path = _cfg.FINAL_ENGINEERED_DATASET_DIR / "final_merged_with_all.parquet"
    path_to_use = data_path if data_path.exists() else (alt_path if alt_path.exists() else None)
    if path_to_use is None:
        raise FileNotFoundError(
            f"[smoke] Real dataset not found at {data_path} or {alt_path}. "
            "Generate engineered parquet before smoke."
        )
    print(f"✓ Real dataset exists → {path_to_use}")

    # 2) Group/target sanity (peek columns quickly to fail fast)
    try:
        import pandas as pd
        df_head = pd.read_parquet(path_to_use, columns=None).head(1000)  # small peek
        tgt_candidates = ["AAV", "AAV_PCT_CAP", "aav", "aav_pct_of_cap"]
        found_tgt = next((t for t in tgt_candidates if t in df_head.columns), None)
        print(f"[smoke] target candidates present: {[t for t in tgt_candidates if t in df_head.columns]}")
        if "position" in df_head.columns:
            print(f"[smoke] group 'position' unique levels (peek): {df_head['position'].astype('string').nunique()}")
        else:
            print("[smoke] group 'position' not present in peek; training can still proceed if spec omits it.")
        if "Season" in df_head.columns:
            print(f"[smoke] group 'Season' unique levels (peek): {df_head['Season'].astype('string').nunique()}")
        else:
            print("[smoke] group 'Season' not present in peek; training can still proceed if spec omits it.")
    except Exception as e:
        print(f"[smoke] peek failed (non-fatal, training will reload): {e}")

    # 3) Build fast Bayesian config
    from api.src.ml.config import TrainingConfig
    bayes_cfg = TrainingConfig(
        model_family="bayes_hier",
        target="AAV",                 # change if using AAV_PCT_CAP
        max_train_rows=2000,          # keep smoke fast
        random_state=42,
        bayes_draws=300,
        bayes_tune=300,
        bayes_target_accept=0.9,
        bayes_chains=2,
        bayes_cores=2,
        bayes_group_cols=("position", "Season"),
    )
    print(f"[smoke] using cfg: {bayes_cfg}")

    # 4) Lazily import the trainer to avoid NameError and import-time cycles
    from api.src.ml.train import train as _train

    # 5) Run training; returns path to posterior.nc (Bayesian branch) per your train()
    print("[smoke] Running Bayesian quick check…")
    bayes_artifact = _train(bayes_cfg)
    print(f"✓ Bayesian smoke ok → {bayes_artifact}")

    return {"bayes_hier_posterior": str(bayes_artifact)}

if __name__ == "__main__":
    print("=== BAYESIAN REAL-DATA SMOKE (pipeline-driven) ===")
    try:
        out = run_bayes_realdata_smoke()
        print("\n=== SMOKE SUMMARY ===")
        for k, v in out.items():
            print(f"{k}: {v}")
        print("✅ Bayesian smoke finished.")
    except Exception as e:
        import traceback
        print(f"💥 Bayesian smoke failed: {e}")
        traceback.print_exc()
        raise SystemExit(1)
