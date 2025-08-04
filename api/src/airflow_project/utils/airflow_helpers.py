"""
Shared utilities for Airflow DAGs to eliminate common AI-generated patterns.
"""
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd
import re


def format_season_range(start_year: int, end_year: int) -> List[str]:
    """Generate season strings like '2015-16', '2016-17' from year range."""
    return [f"{y}-{(y+1)%100:02d}" for y in range(start_year, end_year + 1)]



def normalize_xcom_input(xcom_value) -> Union[str, List[str], None]:
    """Handle Airflow's LazyXComSequence and other XCom types consistently."""
    if xcom_value is None:
        return None

    # Handle LazyXComSequence (Airflow 3+)
    try:
        from airflow.sdk.execution_time.lazy_sequence import LazyXComSequence
        if isinstance(xcom_value, LazyXComSequence):
            xcom_value = list(xcom_value)
    except ImportError:
        pass

    # Handle other iterables
    if hasattr(xcom_value, "__iter__") and not isinstance(xcom_value, (str, Path)):
        xcom_value = list(xcom_value)

    # If it's a list, return the first element (common pattern)
    if isinstance(xcom_value, list):
        return xcom_value[0] if xcom_value else None

    return xcom_value


def load_cached_seasons(parquet_path: Path, force_full: bool) -> List[str]:
    """Load existing seasons from cache, respecting force_full flag."""
    if parquet_path.exists() and not force_full:
        return pd.read_parquet(parquet_path)["season"].unique().tolist()
    return []


def merge_parquet_shards(shard_paths: List[str], final_path: Path, 
                        force_full: bool = False, 
                        dedup_cols: Optional[List[str]] = None) -> str:
    """Merge parquet shards with caching logic."""
    clean_paths = [p for p in shard_paths if p]

    if not clean_paths:
        if final_path.exists():
            print(f"No new shards; using existing {final_path}")
            return str(final_path)
        raise RuntimeError("No shards provided and no cache present")

    fresh_dfs = [pd.read_parquet(p) for p in clean_paths]
    merged_fresh = pd.concat(fresh_dfs, ignore_index=True)

    if final_path.exists() and not force_full:
        cached = pd.read_parquet(final_path)
        merged = pd.concat([cached, merged_fresh], ignore_index=True)
        if dedup_cols:
            merged = merged.drop_duplicates(subset=dedup_cols)
    else:
        merged = merged_fresh

    final_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(final_path, index=False)
    return str(final_path)


def validate_required_columns(df: pd.DataFrame, required_cols: set, name: str = "dataframe"):
    """Validate that required columns exist in dataframe."""
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def normalize_header(header: str) -> str:
    """Normalize header for robust matching."""
    return re.sub(r'[^a-z]', '', header.lower())


def extract_year_from_path(path: Union[str, Path]) -> Optional[int]:
    """Extract year from filename like 'advanced_2024-25.parquet' -> 2024."""
    match = re.search(r'(\d{4})', Path(path).stem)
    return int(match.group(1)) if match else None 
