# dags/nba_advanced_ingest.py
"""
Daily scrape of Basketball‑Reference season‑level advanced metrics.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os, sys
from pathlib import Path

# ── Use centralized config ─────────────────────────────────────────────────────
from utils.nba_basics_config import PROJECT_ROOT, ADVANCED_METRICS_DIR

sys.path.insert(0, str(PROJECT_ROOT / "api" / "src"))
from eda.nba_basic_advanced_stats.scrape_utils import _season_advanced_df

default_args = {
    "owner": "data_eng",
    "email": ["alerts@example.com"],
    "email_on_failure": True,
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "sla": timedelta(hours=1),
}

with DAG(
    dag_id="nba_advanced_ingest",
    start_date=datetime(2025, 7, 1),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["nba", "advanced", "ingest"],
    params={
        "start_year": 2024,  # first season to scrape
        "end_year":   2025,  # last season to scrape
    },
) as dag:

    def scrape_adv(**ctx):
        """
        Loop through start_year..end_year, fetch each season's
        advanced stats, and write to parquet.

        - If a season fetch fails, log a big warning and skip it.
        - After looping, list all missing seasons in a single warning.
        - If _any_ missing season falls in its data-window (Nov–Jun),
        raise an error so the DAG fails (data should exist).
        """
        from datetime import datetime

        p = ctx["params"]
        start_year = int(p["start_year"])
        end_year   = int(p["end_year"])

        missing = []

        # 1️⃣ Fetch each season
        for y in range(start_year, end_year + 1):
            season = f"{y}-{str(y+1)[-2:]}"
            try:
                df = _season_advanced_df(season)
            except RuntimeError as err:
                print(f"⚠️  [WARNING] Unable to fetch advanced stats for {season}: {err}")
                missing.append(season)
                continue

            out_dir = ADVANCED_METRICS_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_dir / f"advanced_{season}.parquet", index=False)

        # 2️⃣ Summary missing seasons
        if missing:
            print("\n⚠️⚠️⚠️  Missing advanced data for seasons:", ", ".join(missing), "⚠️⚠️⚠️\n")

            # 3️⃣ If it's currently Nov–Jun for any missing season, fail
            now = datetime.now()
            should_exist = []
            for season in missing:
                sy = int(season[:4])
                if (now.year == sy and now.month >= 11) or \
                (now.year == sy + 1 and 1 <= now.month <= 6):
                    should_exist.append(season)

            if should_exist:
                raise RuntimeError(
                    f"Advanced stats pages _should_ exist for: {', '.join(should_exist)} "
                    f"(current month: {now.month}). Aborting DAG."
                )

    PythonOperator(
        task_id="scrape_advanced_metrics",
        python_callable=scrape_adv,
    ) 





