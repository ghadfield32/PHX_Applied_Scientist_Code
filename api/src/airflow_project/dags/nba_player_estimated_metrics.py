"""
Pull NBA Player Estimated Metrics from stats.nba.com via nba_api.
"""


from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import pendulum
import pandas as pd
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from eda.nba_em_connector import (
    fetch_player_estimated_metrics,
    normalize_player_estimated_metrics,
)
from utils.storage import write_final_dataset
from utils.airflow_helpers import format_season_range, normalize_xcom_input, load_cached_seasons

# Config
SEASON_START_DEFAULT = "2015-16"
SEASON_END_DEFAULT = "2024-25"
SEASON_TYPE_DEFAULT = "Regular Season"

BASE_DIR = Path("api/src/airflow_project/data")
RAW_DIR = BASE_DIR / "raw" / "player_est_metrics"
SILVER_DIR = BASE_DIR / "silver" / "player_est_metrics"
FINAL_DIR = BASE_DIR / "final"

FINAL_PARQUET = FINAL_DIR / "player_est_metrics_all.parquet"

REQ_COLS = {
    "PLAYER_ID", "PLAYER_NAME", "E_OFF_RATING", "E_DEF_RATING", 
    "E_NET_RATING", "E_USG_PCT", "E_PACE", "E_AST_RATIO", 
    "SEASON", "SEASON_TYPE",
}


@dag(
    dag_id="nba_player_estimated_metrics",
    start_date=pendulum.datetime(2025, 7, 1, tz="America/New_York"),
    schedule="15 4 * * *",
    catchup=False,
    tags=["nba", "player_estimated_metrics", "nba_api"],
    default_args={"retries": 2},
    params={
        "seasons": None,
        "season_start": SEASON_START_DEFAULT,
        "season_end": SEASON_END_DEFAULT,
        "season_type": SEASON_TYPE_DEFAULT,
        "force_full": False,
    },
)
def nba_player_estimated_metrics_dag():

    @task
    def determine_seasons() -> Dict[str, List[str] | bool | str]:
        """Figure out which seasons to process."""
        ctx = get_current_context()
        p = ctx["params"]

        if p.get("seasons"):
            requested = list(map(str, p["seasons"]))
        else:
            start_year = int(p.get("season_start", SEASON_START_DEFAULT).split("-")[0])
            end_year = int(p.get("season_end", SEASON_END_DEFAULT).split("-")[0])
            requested = format_season_range(start_year, end_year)

        season_type = str(p.get("season_type", SEASON_TYPE_DEFAULT))
        force_full = bool(p.get("force_full", False))

        existing = load_cached_seasons(FINAL_PARQUET, force_full)
        missing = sorted(set(requested) - set(existing))

        print(f"[determine_seasons] requested={requested}, existing={existing}, missing={missing}, force_full={force_full}")

        return {
            "requested": requested,
            "missing": missing,
            "season_type": season_type,
            "force_full": force_full,
        }

    @task
    def build_fetch_payloads(season_info: Dict[str, List[str] | bool | str]) -> List[Dict[str, str]]:
        """Build payloads for expand_kwargs."""
        season_type = str(season_info["season_type"])
        payloads = [{"season": s, "season_type": season_type} for s in season_info["missing"]]
        print(f"[build_fetch_payloads] {len(payloads)} payloads")
        return payloads

    @task
    def fetch_one_season(season: str, season_type: str) -> Dict[str, str]:
        """Fetch raw data for a single season."""
        RAW_DIR.mkdir(parents=True, exist_ok=True)

        result = fetch_player_estimated_metrics(
            season=season,
            season_type=season_type,
            raw_dir=RAW_DIR
        )
        print(f"[fetch_one_season] {season} saved {list(result.keys())}")
        return {k: str(v) for k, v in result.items()}

    @task
    def collect_fetched(fetched):
        """Materialize LazyXComSequence to list."""
        return list(fetched)

    @task(multiple_outputs=True)
    def normalize_shards(saved_paths_dicts):
        """Normalize shards and combine into temp parquet."""
        saved_paths_dicts = normalize_xcom_input(saved_paths_dicts)
        if not isinstance(saved_paths_dicts, list):
            saved_paths_dicts = list(saved_paths_dicts)

        SILVER_DIR.mkdir(parents=True, exist_ok=True)

        final_paths_all = []
        df_frames = []

        for paths in saved_paths_dicts:
            norm_paths, df = normalize_player_estimated_metrics(
                {k: Path(v) for k, v in paths.items()},
                silver_dir=SILVER_DIR
            )
            final_paths_all.extend([str(p) for p in norm_paths])
            if not df.empty:
                df_frames.append(df)

        tmp = pd.concat(df_frames, ignore_index=True) if df_frames else pd.DataFrame()
        tmp_path = SILVER_DIR / "_tmp_combined_current_run.parquet"
        tmp.to_parquet(tmp_path, index=False)

        print(f"[normalize_shards] combined {len(tmp)} rows")
        return {"final_paths": final_paths_all, "combined_tmp": str(tmp_path)}

    @task
    def merge_to_final(tmp_path: str, season_info: Dict[str, List[str] | bool | str]) -> str:
        """Merge temp parquet into final dataset."""
        tmp_path = Path(tmp_path) if tmp_path else None

        if tmp_path and tmp_path.exists():
            fresh = pd.read_parquet(tmp_path)
        else:
            fresh = pd.DataFrame()

        if fresh.empty and FINAL_PARQUET.exists() and not season_info["force_full"]:
            print("[merge_to_final] No new data; using cached final.")
            return str(FINAL_PARQUET)

        if FINAL_PARQUET.exists() and not season_info["force_full"]:
            cached = pd.read_parquet(FINAL_PARQUET)
            merged = pd.concat([cached, fresh], ignore_index=True)
        else:
            merged = fresh

        if not merged.empty:
            missing_cols = {"SEASON", "SEASON_TYPE"} - set(merged.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns for deduplication: {missing_cols}")
            merged = merged.drop_duplicates(subset=["PLAYER_ID", "SEASON", "SEASON_TYPE"])

        FINAL_DIR.mkdir(parents=True, exist_ok=True)
        final_path = write_final_dataset(merged, FINAL_PARQUET)
        print(f"[merge_to_final] final {len(merged)} rows")
        return str(final_path)

    @task
    def validate_outputs(final_pq: str) -> None:
        """Validate final dataset."""
        df = pd.read_parquet(final_pq)

        missing = REQ_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        print("Rows by SEASON:")
        print(df.groupby("SEASON").size())
        print("Rows by SEASON_TYPE:")
        print(df.groupby("SEASON_TYPE").size())
        print("Top 5 E_NET_RATING by season:")
        print(df.sort_values("E_NET_RATING", ascending=False)
              .groupby("SEASON")
              .head(5)[["PLAYER_NAME", "E_NET_RATING", "SEASON"]]
              .to_string(index=False))

    # Workflow
    season_info = determine_seasons()
    payloads = build_fetch_payloads(season_info)
    fetched = fetch_one_season.expand_kwargs(payloads)
    collected = collect_fetched(fetched)
    norm_res = normalize_shards(collected)
    final_pq = merge_to_final(tmp_path=norm_res["combined_tmp"], season_info=season_info)
    validate_outputs(final_pq)

nba_player_estimated_metrics_dag()




