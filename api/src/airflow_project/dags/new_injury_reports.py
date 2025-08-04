"""
## NBA Injury Reports ETL (ESPN + others)

Daily pipeline that:
1. Determines which dates to ingest (default: yesterday -> today).
2. Fetches and stores injury reports (one task per day via dynamic mapping).
3. Runs a simple validation.

Implements TaskFlow API & dynamic task mapping. :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from airflow.sdk.execution_time.lazy_sequence import LazyXComSequence  # NEW
import datetime as dt
import pendulum

from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

# Import your module
from eda.new_injury_reports import run_daily_ingestion, clean_and_normalize, incremental_store, make_sources
from eda.combining_injury_reports import (
    build_full_dataset,  # now the new function
    injury_path,
)
from utils.config import injury_path

# -------- Config defaults (override with DAG Params) -----------------
# DEFAULT_OUTPUT = Path("api/src/airflow_project/data/injury_reports")
# Use the canonical injury_reports directory from config:
DEFAULT_OUTPUT = injury_path("")      # returns data/injury_reports
print(f"DEFAULT_OUTPUT: {DEFAULT_OUTPUT}")

@dag(
    dag_id="injury_reports_daily",
    start_date=pendulum.datetime(2025, 7, 1, tz="America/New_York"),
    schedule="0 5 * * *",             # 05:00 ET daily pull
    catchup=False,
    max_active_runs=1,
    tags=["nba", "injuries", "espn"],
    default_args={"retries": 1},
    params={
        "start_date": None,           # ISO yyyy-mm-dd
        "end_date": None,             # ISO yyyy-mm-dd
        "days_back": 1,               # if no start/end given
        "force_full": False,          # not used yet, kept for parity
        "fix_return_year": True,
        "season_start_year": 2024,    # NEW: season configuration
        "allowed_years": [2024, 2025], # computed from season_start_year
        "statsurge_root": "data/statsurge",
        "parquet_path": str(DEFAULT_OUTPUT / "injuries_primary.parquet"),
    },
)
def injury_reports_daily():

    @task
    def determine_dates() -> list[str]:
        """
        Decide which dates to fetch.
        Priority:
          1. explicit start_date/end_date params
          2. days_back param (N days ending yesterday)
        Returns list of ISO date strings.
        """
        ctx = get_current_context()
        p = ctx["params"]

        if p.get("start_date") and p.get("end_date"):
            start = dt.date.fromisoformat(p["start_date"])
            end = dt.date.fromisoformat(p["end_date"])
        else:
            days_back = int(p.get("days_back", 1))
            end = dt.date.today()
            start = end - dt.timedelta(days=days_back)

        dates = [ (start + dt.timedelta(days=i)).isoformat()
                  for i in range((end - start).days + 1) ]

        print(f"[determine_dates] start={start} end={end} -> {len(dates)} days")
        return dates

    @task
    def fetch_one_day(date_str: str) -> dict[str, str]:
        """
        Ingest one day's injuries and upsert to parquet.
        RETURNS a dict so downstream can safely access by string key:
          {"parquet_path": "<path/to/file.parquet>"}
        """
        ctx = get_current_context()
        p = ctx["params"]

        date = dt.date.fromisoformat(date_str)
        # parquet_path = Path(p["parquet_path"])
        parquet_path = injury_path("injuries_primary.parquet")

        # Season wiring ----------------------------------------------------------
        season_start = int(p.get("season_start_year", 2024))
        allowed_years = (season_start, season_start + 1)

        # Let run_daily_ingestion build sources internally to avoid duplication
        df_day = run_daily_ingestion(
            date=date,
            sources=None,  # Let it build sources internally
            cleaner=clean_and_normalize,
            store_fn=lambda d: incremental_store(d, path=parquet_path),
            drop_duplicates_on=["player_name","team","status","body_part","report_date","est_return_date"],
            fix_return_year=bool(p.get("fix_return_year", True)),
            allowed_years=allowed_years,
            statsurge_root=p.get("statsurge_root", "data/statsurge"),
        )
        print(f"[fetch_one_day] {date} rows={len(df_day)} -> {parquet_path}")

        return {"parquet_path": str(parquet_path)}

    @task
    def final_check(parquet_path: str | list[str] | None = None) -> None:
        """
        Basic validation. Handles Airflow LazyXComSequence / list / str safely.
        """
        from airflow.sdk.execution_time.lazy_sequence import LazyXComSequence  # optional import guard

        print(f"[final_check] Received parquet_path type: {type(parquet_path)}")
        print(f"[final_check] Received parquet_path value: {parquet_path}")

        if parquet_path is None:
            raise ValueError("final_check received no parquet_path")

        # Normalize lazy proxies or other iterables to a list
        if isinstance(parquet_path, LazyXComSequence) or (
            not isinstance(parquet_path, (str, Path)) and hasattr(parquet_path, "__iter__")
        ):
            parquet_path = list(parquet_path)

        # If we ended up with a list, pick the first element (or iterate/validate all)
        if isinstance(parquet_path, list):
            if not parquet_path:
                raise ValueError("final_check received an empty list for parquet_path")
            parquet_path = parquet_path[0]

        pq = Path(parquet_path)
        if not pq.exists():
            raise FileNotFoundError(f"Parquet file not found: {pq}")

        df = pd.read_parquet(pq)
        req = {"player_name","team","status","report_date","source"}
        miss = req - set(df.columns)
        assert not miss, f"Missing columns: {miss}"

        print(f"[final_check] Successfully validated {len(df)} rows")
        print("Rows by report_date:\n", df.groupby("report_date").size())

    @task
    def combine_full_dataset(parquet_path: str | list[str] | LazyXComSequence | None = None) -> str:
        # 1. Normalize lazy proxies or other iterables to a list
        if isinstance(parquet_path, LazyXComSequence) \
        or (not isinstance(parquet_path, (str, Path)) and hasattr(parquet_path, "__iter__")):
            parquet_path = list(parquet_path)

        # 2. If it’s a list, pick the first element
        if isinstance(parquet_path, list):
            if not parquet_path:
                raise ValueError("combine_full_dataset received an empty list for parquet_path")
            parquet_path = parquet_path[0]

        # 3. Now it’s safe to build a Path
        file_a = Path(parquet_path).resolve()
        file_b = injury_path("NBA Player Injury Stats(1951 - 2023).parquet")
        out_dir = injury_path("")

        full_path = build_full_dataset(
            file_a=file_a,
            file_b=file_b,
            out_dir=out_dir,
            full_out_name="historical_injuries_1951_2025_clean.parquet",
            duplicate_strategy="keep_first",
            write_intermediates=False,
        )
        return str(full_path)


    # ---- DAG flow ----
    date_list = determine_dates()
    paths = fetch_one_day.expand(date_str=date_list)

    # SAFE string-key lookup; no int indexing
    # Airflow will automatically reduce multiple identical paths to a single value
    final_check(parquet_path=paths["parquet_path"])
    combined = combine_full_dataset(parquet_path=paths["parquet_path"])

injury_reports_daily()
