"""
Central configuration for the NBA‑Player‑Valuation project.
All magic values live here so they can be tweaked without code edits.
"""
from pathlib import Path
import os

# ── Core directories ───────────────────────────────────────────────────────────
def find_project_root(name: str = "airflow_project") -> Path:
    """
    Walk up from this file (or cwd) until a directory named `name` is found.
    Fallback to cwd if not found.
    """
    try:
        p = Path(__file__).resolve()
    except NameError:
        p = Path.cwd()
    # walk through p and its parents
    for parent in (p, *p.parents):
        if parent.name == name or (parent / ".git").is_dir():
            return parent
    # no match → fallback
    return Path.cwd()

# Allow explicit override
if find_project_root():
    PROJECT_ROOT = Path(find_project_root()).resolve() # / "api/src/airflow_project"
else:
    PROJECT_ROOT = find_project_root() # / "api/src/airflow_project"


DATA_DIR: Path = Path(PROJECT_ROOT / "data")
LOG_DIR: Path = Path(PROJECT_ROOT / "logs")
DUCKDB_FILE: Path = Path(DATA_DIR / "nba.duckdb")

# NBA stats API
NBA_API_RPM: int = int(os.getenv("NBA_API_RPM", "12"))  # requests per minute

# Spotrac scraping
SPOTRAC_BASE: str = "https://www.spotrac.com/nba"
SPOTRAC_FREE_AGENTS: str = f"{SPOTRAC_BASE}/free-agents/{{year}}/"
SPOTRAC_CAP_TRACKER: str = f"{SPOTRAC_BASE}/cap/{{year}}/"
SPOTRAC_TAX_TRACKER: str = f"{SPOTRAC_BASE}/tax/_/year/{{year}}/"
# Spotrac dedicated folder


# Injury sources
NBA_OFFICIAL_INJURY_URL: str = "https://cdn.nba.com/static/json/injury/injury_report_{{date}}.json"
ROTOWIRE_RSS: str = "https://www.rotowire.com/rss/news.php?sport=NBA"

# StatsD (optional)
STATSD_HOST: str = os.getenv("STATSD_HOST", "localhost")
STATSD_PORT: int = int(os.getenv("STATSD_PORT", "8125"))

# ── Data ranges ────────────────────────────────────────────────────────────────
SEASONS: range = range(2015, 2026)  # inclusive upper bound matches Spotrac sample

# Thread pools & concurrency
MAX_WORKERS: int = int(os.getenv("NPV_MAX_WORKERS", "8")) 


# ── Core data directories ───────────────────────────────────────────────────────────
RAW_DIR      : Path = DATA_DIR / "raw"
DEBUG_DIR    : Path = DATA_DIR / "debug"
EXPORTS_DIR  : Path = DATA_DIR / "exports"
INJURY_DIR   : Path = DATA_DIR / "injury_reports"
INJURY_DATASETS_DIR : Path = DATA_DIR / "injury_datasets"
NBA_BASIC_ADVANCED_STATS_DIR : Path = DATA_DIR / "nba_basic_advanced_stats"
ADVANCED_METRICS_DIR: Path = DATA_DIR / "new_processed" / "advanced_metrics"
NBA_BASE_DATA_DIR : Path = DATA_DIR / "nba_processed"
DEFENSE_DATA_DIR : Path = DATA_DIR / "defense_metrics"
FINAL_DATASET_DIR : Path = DATA_DIR / "merged_final_dataset"
PLAY_TYPES_DIR : Path = DATA_DIR / "synergyplay_types"
SPOTRAC_DIR  : Path = DATA_DIR / "spotrac_contract_data"
SILVER_DIR   : Path = SPOTRAC_DIR / "silver"
FINAL_DIR    : Path = SPOTRAC_DIR / "final"
SPOTRAC_DEBUG_DIR    : Path = SPOTRAC_DIR / "debug"
SPOTRAC_RAW_DIR      : Path = SPOTRAC_DIR / "raw"

# ── Helper functions (single source of truth) ────────────────────────────────
def get_injury_base_dir() -> Path:
    """
    Return the canonical injury_reports root.  
    ENV `INJURY_DATA_DIR` wins; otherwise we fall back to INJURY_DIR.
    Always creates the dir so callers can assume it exists.
    """
    base = Path(os.getenv("INJURY_DATA_DIR", INJURY_DIR)).resolve()
    return base

def injury_path(*parts: str) -> Path:
    """Shorthand for get_injury_base_dir().joinpath(*parts).resolve()."""
    return get_injury_base_dir().joinpath(*parts).resolve()

# ── One‑shot: ensure all declared dirs exist at import‑time ───────────────────
for _p in (
    DATA_DIR, RAW_DIR, DEBUG_DIR, EXPORTS_DIR, INJURY_DIR,
    SPOTRAC_DIR, SILVER_DIR, FINAL_DIR, SPOTRAC_DEBUG_DIR, SPOTRAC_RAW_DIR,
    NBA_BASIC_ADVANCED_STATS_DIR, NBA_BASE_DATA_DIR, ADVANCED_METRICS_DIR,
    DEFENSE_DATA_DIR, FINAL_DATASET_DIR,
    PLAY_TYPES_DIR   # ← ensure synergy play‑types folder exists
):
    _p.mkdir(parents=True, exist_ok=True)



print("all directories:")
print("root directory:")
print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"RAW_DIR: {RAW_DIR}")
print(f"DEBUG_DIR: {DEBUG_DIR}")
print(f"EXPORTS_DIR: {EXPORTS_DIR}")
print(f"INJURY_DIR: {INJURY_DIR}")
print(f"INJURY_DATASETS_DIR: {INJURY_DATASETS_DIR}")
print(f"NBA_BASIC_ADVANCED_STATS_DIR: {NBA_BASIC_ADVANCED_STATS_DIR}")
print(f"NBA_BASE_DATA_DIR: {NBA_BASE_DATA_DIR}")
print(f"ADVANCED_METRICS_DIR: {ADVANCED_METRICS_DIR}")
print(f"DEFENSE_DATA_DIR: {DEFENSE_DATA_DIR}")
print(f"FINAL_DATASET_DIR: {FINAL_DATASET_DIR}")
print(f"PLAY_TYPES_DIR: {PLAY_TYPES_DIR}")
print(f"SPOTRAC_DIR: {SPOTRAC_DIR}")
print(f"SILVER_DIR: {SILVER_DIR}")
print(f"FINAL_DIR: {FINAL_DIR}")
print(f"SPOTRAC_DEBUG_DIR: {SPOTRAC_DEBUG_DIR}")
print(f"SPOTRAC_RAW_DIR: {SPOTRAC_RAW_DIR}")
print("all directories:")
