# dags/nba_api_ingest.py
"""
Pulls roster + box‑score data from nba_api once per hour and writes Parquet
partitions under data/new_processed/season=<YYYY-YY>/part.parquet.

Why hourly?
• The NBA Stats endpoints update within minutes after a game ends.
• Hourly keeps your lake near‑real‑time without hammering the API.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os, sys, pathlib

# ── Use centralized config ─────────────────────────────────────────────────────
from utils.nba_basics_config import PROJECT_ROOT, DATA_DIR
from utils.incremental_utils import (
    get_latest_ingested_date, 
    get_next_date_to_pull, 
    incremental_pull,
    should_pull_incremental
)

# Allow `nba_basic_advanced_stats` imports
sys.path.insert(0, str(PROJECT_ROOT / "api" / "src"))

default_args = {
    "owner": "data_eng",
    "email": ["alerts@example.com"],
    "email_on_failure": True,
    "depends_on_past": False,      # explicit
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "sla": timedelta(hours=1),
}

with DAG(
    dag_id="nba_api_ingest",
    start_date=datetime(2025, 7, 1),
    schedule="@hourly",            # unified scheduling API (Airflow ≥ 2.4)
    catchup=False,
    default_args=default_args,
    max_active_runs=1,             # avoid overlapping pulls
    tags=["nba", "api", "ingest"],
    params={
        "start_year": 2024,  # first season to pull
        "end_year":   2025,  # last season to pull
    },
) as dag:

    def pull_incremental(**context):
        """
        Determine the next date to pull based on existing data and perform incremental pull.
        Falls back to full season pull if no existing data is found.
        """
        p = context["params"]
        sy = int(p["start_year"])
        ey = int(p["end_year"])

        # For now, focus on the current season (2024-25)
        season = f"{sy}-{str(sy+1)[-2:]}"

        print(f"[pull_incremental] Checking for incremental pull for season {season}")

        # Check if we should do incremental pull
        if should_pull_incremental(DATA_DIR, season):
            # Get the next date to pull
            next_date = get_next_date_to_pull(DATA_DIR, season)
            if next_date:
                print(f"[pull_incremental] Pulling incremental data for {next_date}")
                incremental_pull(
                    data_dir=DATA_DIR,
                    season=season,
                    date_to_pull=next_date,
                    workers=8,
                    debug=False
                )
            else:
                print(f"[pull_incremental] No next date found, skipping")
        else:
            # Fall back to full season pull
            print(f"[pull_incremental] No existing data found, doing full season pull")
            from eda.nba_basic_advanced_stats.main import main as pull_main
            pull_main(
                start_year=sy,
                end_year=ey,
                small_debug=True,
                workers=8,
                overwrite=False,
                output_base=str(DATA_DIR),
            )

    PythonOperator(
        task_id="scrape_incremental_data",
        python_callable=pull_incremental,
    ) 
