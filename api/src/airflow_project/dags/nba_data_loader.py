# dags/nba_data_loader.py
"""
Load processed NBA data into DuckDB for analysis.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
import duckdb
import pandas as pd

from pathlib import Path
import sys

# ── Use centralized config ─────────────────────────────────────────────────────
from utils.config import (
    PROJECT_ROOT, DATA_DIR, INJURY_DIR, NBA_BASE_DATA_DIR
)

# Ensure our code can import the eda package
sys.path.insert(0, str(PROJECT_ROOT / "api" / "src"))
from eda.nba_basic_advanced_stats.data_utils import validate_data

# Use centralized data root from config
DATA_ROOT = DATA_DIR

default_args = {
    "owner": "data_eng",
    "email": ["alerts@example.com"],
    "email_on_failure": True,
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "sla": timedelta(hours=3),
}

with DAG(
    dag_id="nba_data_loader",
    start_date=datetime(2025, 7, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["nba", "loader", "duckdb"],
    params={"season": "2024-25"},
) as dag:

    # ─── sensors (one per upstream DAG) ────────────────────────────────
    sensor_args = dict(
        poke_interval=300,
        mode="reschedule",   # avoids tying up a worker slot
    )
    wait_api = ExternalTaskSensor(
        task_id="wait_api_ingest",
        external_dag_id="nba_api_ingest",
        external_task_id="scrape_incremental_data",
        timeout=3600,
        **sensor_args,
    )
    wait_adv = ExternalTaskSensor(
        task_id="wait_advanced_ingest",
        external_dag_id="nba_advanced_ingest",
        external_task_id="scrape_advanced_metrics",
        timeout=3600,
        **sensor_args,
    )
    wait_injury = ExternalTaskSensor(
        task_id="wait_injury_etl",
        external_dag_id="injury_etl",
        external_task_id="process_injury_data",
        timeout=7200,
        poke_interval=600,
        mode="reschedule",
    )

    # ─── loader task ───────────────────────────────────────────────────
    def load_to_duckdb(**ctx):
        season = ctx["params"]["season"]
        # ▶ use centralized DATA_DIR for everything
        db_path = DATA_DIR / "nba_stats.duckdb"
        con = duckdb.connect(db_path)

        sources = {
            f"player_{season}": NBA_BASE_DATA_DIR / f"season={season}/part.parquet",
            f"advanced_{season}": DATA_DIR / f"new_processed/advanced_metrics/advanced_{season}.parquet",
            "injury_master": INJURY_DIR / "injury_master.parquet",
        }

        for alias, path in sources.items():
            if path.exists():
                if alias.startswith("player"):
                    df = pd.read_parquet(path)
                    validate_data(df, name=alias, save_reports=True)
                con.execute(
                    f"CREATE OR REPLACE TABLE {alias.replace('-', '_')} AS "
                    f"SELECT * FROM read_parquet('{path}')"
                )

        # materialised view – wildcard parquet scan is fine too
        con.execute(f"""
            CREATE OR REPLACE VIEW v_player_full_{season.replace('-', '_')} AS
            SELECT *
            FROM player_{season.replace('-', '_')} p
            LEFT JOIN advanced_{season.replace('-', '_')} a USING(player, season)
            LEFT JOIN injury_master i USING(player, season)
        """)
        con.close()

    loader = PythonOperator(
        task_id="validate_and_load",
        python_callable=load_to_duckdb,
    )

    [wait_api, wait_adv, wait_injury] >> loader 
