
"""
Collect NBA defensive‑metrics into parquet using TaskFlow + dynamic mapping.

**Params supported**
- seasons:        explicit ["2022‑23","2023‑24"]  (overrides everything)
- season_start:   first season if `seasons` not given
- season_end:     last  season if `seasons` not given
- force_full:     if True ignore cache & rebuild all

Runs at 04:30 ET daily.
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, List

import pandas as pd
import pendulum

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from eda.defensive_metrics import DefensiveMetricsCollector
from utils.storage import write_final_dataset

# ---- Config ----
BASE_DIR   = Path("api/src/airflow_project/data")
RAW_DIR    = BASE_DIR / "raw"
FINAL_DIR  = BASE_DIR / "final"
MIN_SEASON = "2018-19"        # hard floor in case params go wild
MAX_SEASON = "2024-25"        # keep in sync with collector

# ------------------------------------------------------------------ DAG
@dag(
    dag_id="defensive_metrics_collect",
    start_date=pendulum.datetime(2025, 7, 1, tz="America/New_York"),
    schedule="30 4 * * *",
    catchup=False,
    tags=["nba", "defense", "metrics"],
    default_args={"retries": 2},
    params={
        "seasons": None,                # explicit list
        "season_start": MIN_SEASON,
        "season_end":   MAX_SEASON,
        "force_full":   False,
    },
)
def defensive_metrics_collect():
    # ---------- helpers ----------
    @task
    def determine_seasons() -> Dict[str, List[str] | bool]:
        """
        Decide which seasons to fetch by comparing DAG params to cached parquet.
        """
        ctx   = get_current_context()
        p     = ctx["params"]

        # build requested list
        if p.get("seasons"):
            requested: List[str] = list(map(str, p["seasons"]))
        else:
            start = str(p.get("season_start", MIN_SEASON))
            end   = str(p.get("season_end",   MAX_SEASON))
            # assume season format 'YYYY-YY'
            start_year = int(start.split("-")[0])
            end_year   = int(end.split("-")[0])
            requested  = [f"{y}-{str(y+1)[-2:]}" for y in range(start_year, end_year + 1)]

        force_full: bool = bool(p.get("force_full", False))

        cache = FINAL_DIR / "def_metrics_all.parquet"
        if cache.exists() and not force_full:
            existing = pd.read_parquet(cache)["season"].unique().tolist()
        else:
            existing = []

        missing = sorted(set(requested) - set(existing))

        print("[determine_seasons] requested:", requested)
        print("[determine_seasons] existing :", existing)
        print("[determine_seasons] missing  :", missing)
        print("[determine_seasons] force_full:", force_full)

        return {
            "requested": requested,
            "missing":   missing,
            "force_full": force_full,
        }

    # ---------- dynamic fetch ----------
    @task
    def seasons_for_mapping(meta: dict) -> List[str]:
        """Return only the seasons we still need (for .expand)."""
        return list(meta["missing"])

    @task
    def fetch_one_season(season: str) -> str:
        """
        Pull a single season and write a parquet shard into RAW_DIR.
        """
        df = DefensiveMetricsCollector.collect_metrics_for_seasons([season])
        out = RAW_DIR / f"def_metrics_{season}.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        print(f"[fetch_one_season] season={season} rows={len(df)} → {out}")
        return str(out)

    # ---------- merge / overwrite ----------
    @task
    def merge_shards(shard_paths: List[str], meta: dict) -> str:
        """
        Merge all shards (new or cached) and overwrite the final parquet.
        """
        final_path = FINAL_DIR / "def_metrics_all.parquet"
        fresh_dfs  = [pd.read_parquet(p) for p in shard_paths]

        if fresh_dfs:
            merged_fresh = pd.concat(fresh_dfs, ignore_index=True)
            if final_path.exists() and not meta["force_full"]:
                cached = pd.read_parquet(final_path)
                merged = pd.concat([cached, merged_fresh], ignore_index=True).drop_duplicates(
                    subset=["PLAYER_NAME", "season"]
                )
            else:
                merged = merged_fresh
            final_path = write_final_dataset(merged, final_path)
        else:
            if final_path.exists():
                print("[merge_shards] No new seasons; using cached file", final_path)
            else:
                raise RuntimeError("No shards provided and no cache present.")

        return str(final_path)

    # ---------- final assertions ----------
    @task
    def validate_outputs(final_pq: str) -> None:
        df = pd.read_parquet(final_pq)
        must_have = {
            "PLAYER_NAME", "season", "DEF_RATING", "dbpm", "dws", "PLUSMINUS"
        }
        missing = must_have - set(df.columns)
        assert not missing, f"Missing cols: {missing}"
        print("Rows/season:\n", df.groupby("season").size())

    # ---------------- DAG orchestration ----------------
    meta          = determine_seasons()
    seasons_list  = seasons_for_mapping(meta)
    shards        = fetch_one_season.expand(season=seasons_list)
    final_path    = merge_shards(shards, meta)
    validate_outputs(final_path)

defensive_metrics_collect()
