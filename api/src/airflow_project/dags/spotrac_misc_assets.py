
"""
Fetch Spotrac misc assets and save to final directory.
"""


from __future__ import annotations
from pathlib import Path
import pendulum, pandas as pd
import re
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context
from airflow.exceptions import AirflowSkipException
import os

from eda.spotrac_connector import (
    fetch_spotrac_salary_cap_history,
    fetch_spotrac_multi_year_tracker,
    fetch_spotrac_cash_tracker,
    fetch_spotrac_extensions,
    fetch_spotrac_tax_tracker,
    fetch_spotrac_transactions,
)
from utils.storage import write_final_dataset
from utils.config import RAW_DIR, DEBUG_DIR, FINAL_DIR, SPOTRAC_RAW_DIR, SPOTRAC_DEBUG_DIR
from utils.airflow_helpers import normalize_header, extract_year_from_path

BASE_DIR = RAW_DIR.parent


def resolve_column_aliases(df, column_map: dict[str, list[str]], 
                          extra_fixers: dict[str, callable] = None):
    """Rename columns to standard names using aliases."""
    rename_log = {}
    cols_raw = list(df.columns)
    cols_norm = {normalize_header(c): c for c in cols_raw}

    for target, aliases in column_map.items():
        if target in df.columns:
            continue
        found = None
        for alias in aliases:
            norm = normalize_header(alias)
            if norm in cols_norm:
                found = cols_norm[norm]
                break
        if found:
            df = df.rename(columns={found: target})
            rename_log[found] = target
        elif extra_fixers and target in extra_fixers:
            df = extra_fixers[target](df)

    missing = [k for k in column_map if k not in df.columns]
    return df, missing, rename_log


@dag(
    dag_id="spotrac_misc_assets",
    start_date=pendulum.datetime(2025, 7, 1, tz="America/New_York"),
    schedule="15 4 * * *",
    catchup=False,
    tags=["nba", "spotrac", "misc"],
    default_args={"retries": 2},
    params={
        "season_start": 2015,
        "season_end": 2025,
        "recent_years": 3,
        "txn_days": 7,
        "force_full": False,
    },
)
def spotrac_misc_assets():

    @task
    def determine_years() -> dict:
        """Build year lists and flags."""
        ctx = get_current_context()["params"]
        yr_all = list(range(int(ctx["season_start"]), int(ctx["season_end"]) + 1))
        recent_n = int(ctx["recent_years"])
        info = {
            "all_years": yr_all,
            "recent_years": yr_all[-recent_n:],
            "txn_days": int(ctx["txn_days"]),
            "force_full": bool(ctx["force_full"]),
        }
        print(f"[determine_years] {info}")
        return info

    @task
    def get_recent_years(info: dict) -> list[int]:
        """Extract recent years for mapping."""
        yrs = info["recent_years"]
        print(f"[get_recent_years] {yrs}")
        return yrs

    # Static assets
    @task
    def fetch_cba() -> str:
        """Fetch CBA history."""
        out_path, df, _ = fetch_spotrac_salary_cap_history(
            raw_dir=SPOTRAC_RAW_DIR,
            debug_dir=SPOTRAC_DEBUG_DIR,
        )
        abs_out = str(Path(out_path).resolve())
        print(f"[fetch_cba] rows={len(df)} cols={df.shape[1]} -> {abs_out}")
        return abs_out

    @task
    def fetch_multi_cap() -> str:
        out, df, _ = fetch_spotrac_multi_year_tracker(
            raw_dir=SPOTRAC_RAW_DIR,
            debug_dir=SPOTRAC_DEBUG_DIR,
        )
        print(f"[fetch_multi_cap] rows={len(df)} cols={df.shape[1]} -> {out}")
        return str(out)

    @task
    def fetch_multi_cash() -> str:
        out, df, _ = fetch_spotrac_cash_tracker(
            raw_dir=SPOTRAC_RAW_DIR,
            debug_dir=SPOTRAC_DEBUG_DIR,
        )
        print(f"[fetch_multi_cash] rows={len(df)} cols={df.shape[1]} -> {out}")
        return str(out)

    # Yearly extensions/tax
    @task
    def fetch_extension(year: int) -> str | None:
        """Fetch extensions for a single year."""
        try:
            out, df, _ = fetch_spotrac_extensions(
                year,
                raw_dir=SPOTRAC_RAW_DIR,
                debug_dir=SPOTRAC_DEBUG_DIR,
            )
            print(f"[fetch_extension] year={year} rows={len(df)} -> {out}")
            return str(out)
        except RuntimeError as e:
            print(f"[fetch_extension] year={year} failed: {e}")
            raise AirflowSkipException(f"No extensions table for {year}")

    @task
    def fetch_tax(year: int) -> str:
        out, df, _ = fetch_spotrac_tax_tracker(
            year,
            raw_dir=SPOTRAC_RAW_DIR,
            debug_dir=SPOTRAC_DEBUG_DIR,
        )
        print(f"[fetch_tax] year={year} rows={len(df)} -> {out}")
        return str(out)

    # Rolling transactions
    @task
    def fetch_txn(info: dict) -> str:
        """Pull transactions for rolling window."""
        from datetime import date, timedelta
        end = date.today()
        start = end - timedelta(days=info["txn_days"])
        out, df, _ = fetch_spotrac_transactions(
            start.isoformat(),
            end.isoformat(),
            raw_dir=SPOTRAC_RAW_DIR,
            debug_dir=SPOTRAC_DEBUG_DIR,
        )
        if out:
            print(f"[fetch_txn] {start}->{end} rows={len(df)} -> {out}")
            return str(out)
        print(f"[fetch_txn] {start}->{end} no data")
        return ""

    # Merge helpers
    @task
    def merge_yearly(paths: list[str | None], name: str) -> str:
        """Merge parquet shards from mapped tasks."""
        clean = [p for p in paths if p]
        final = FINAL_DIR / f"{name}_all.parquet"
        print(f"[merge_yearly] merging {len(clean)} shards of {name}")

        if clean:
            dfs = [pd.read_parquet(p) for p in clean]
            merged = pd.concat(dfs, ignore_index=True)
            write_final_dataset(merged, final)
            print(f"[merge_yearly] wrote {final}")
            return str(final)

        if final.exists():
            print(f"[merge_yearly] using existing {final}")
            return str(final)

        print(f"[merge_yearly] no data available")
        return ""

    @task
    def validate_assets(cba: str, cap: str, cash: str,
                        ext: str, tax: str, txn: str) -> None:
        """Validate all produced parquet assets."""
        import json

        input_paths = {
            "cba": cba,
            "multi_cap": cap,
            "multi_cash": cash,
            "extensions": ext,
            "tax": tax,
            "txn": txn
        }
        mandatory = {"cba", "multi_cap", "multi_cash"}

        # Check existence
        valid_paths = {}
        for label, p in input_paths.items():
            if not p or not os.path.isfile(p):
                if label in mandatory:
                    raise FileNotFoundError(f"{label} file missing: {p}")
                print(f"[validate_assets] {label} skipped")
            else:
                valid_paths[label] = p

        print(f"[validate_assets] validating: {list(valid_paths.keys())}")

        # Column mapping
        column_maps = {
            "cba": {"year": ["season", "yr"]},
            "multi_cap": {"team": ["team", "teamname"]},
            "multi_cash": {"team": ["team", "teamname"]},
            "extensions": {
                "player": ["players"],
                "total": ["total value", "value", "amount", "totalvalue"],
                "year": ["yr", "season"],
            },
            "tax": {
                "team": ["team", "teamname"],
                "tax": ["tax bill", "taxbill", "luxurytax", "taxthreshold"],
                "year": ["yr", "season"],
            },
            "txn": {
                "date": ["transactiondate"],
                "player": ["players", "player(s)"],
                "type": ["details", "move", "transactiontype"],
            },
        }

        # Setup caching
        cache_dir = DEBUG_DIR / "validate"
        cache_dir.mkdir(parents=True, exist_ok=True)
        schema_cache = cache_dir / "validate_schema_cache.json"
        stats_cache = cache_dir / "validate_stats_cache.json"

        old_schema = {}
        old_stats = {}
        if schema_cache.exists():
            old_schema = pd.read_json(schema_cache, typ="series").to_dict()
        if stats_cache.exists():
            old_stats = json.loads(stats_cache.read_text())

        drift_report = {}
        row_drift = {}
        current_stats = {}

        for label, p in valid_paths.items():
            df = pd.read_parquet(p)

            # Schema drift check
            prev_cols = set(old_schema.get(label, []))
            curr_cols = set(df.columns.tolist())
            added = sorted(curr_cols - prev_cols)
            removed = sorted(prev_cols - curr_cols)
            if added or removed:
                drift_report[label] = {"added": added, "removed": removed}

            # Resolve column aliases
            logical_req = column_maps[label]

            def add_year_if_missing(dfi: pd.DataFrame) -> pd.DataFrame:
                y = extract_year_from_path(p)
                return dfi.assign(year=y) if y is not None else dfi

            dfn, missing, renames = resolve_column_aliases(
                df, logical_req, extra_fixers={"year": add_year_if_missing}
            )
            if missing:
                raise AssertionError(f"{label} missing cols: {set(missing)}")

            # Numeric stats
            num_cols = [c for c in dfn.columns if pd.api.types.is_numeric_dtype(dfn[c])]
            means = {c: float(dfn[c].dropna().mean()) for c in num_cols}
            current_stats[label] = {
                "rows": int(len(dfn)),
                "cols": int(len(dfn.columns)),
                "means": means
            }

            # Compare with previous stats
            if label in old_stats:
                prev = old_stats[label]
                if prev["rows"] != current_stats[label]["rows"]:
                    row_drift[label] = {"old": prev["rows"], "new": current_stats[label]["rows"]}

                mean_drift = {c: (prev["means"].get(c), means.get(c)) 
                             for c in set(prev["means"]) | set(means)}
                flagged = {}
                for c, (old_m, new_m) in mean_drift.items():
                    if old_m is None or new_m is None:
                        continue
                    if old_m == 0 and new_m != 0:
                        flagged[c] = (old_m, new_m)
                    else:
                        rel = abs(new_m - old_m) / (abs(old_m) + 1e-9)
                        if rel > 0.05:
                            flagged[c] = (old_m, new_m)
                if flagged:
                    print(f"[validate_assets] {label} mean drift >5%: {flagged}")

            print(f"[validate_assets] {label}: rows={len(dfn)} OK; renames={renames}")

        # Persist snapshots
        pd.Series({
            lbl: pd.read_parquet(p).columns.tolist()
            for lbl, p in valid_paths.items()
        }).to_json(schema_cache)
        stats_cache.write_text(json.dumps(current_stats, indent=2))

        if drift_report:
            print("[validate_assets] SCHEMA DRIFT:")
            for lbl, rep in drift_report.items():
                print(f"  - {lbl}: added={rep['added']} removed={rep['removed']}")

        if row_drift:
            print("[validate_assets] ROW COUNT DRIFT:")
            for lbl, rep in row_drift.items():
                print(f"  - {lbl}: old={rep['old']} new={rep['new']}")

    # DAG graph
    info = determine_years()
    recent_list = get_recent_years(info)

    cba_p = fetch_cba()
    cap_p = fetch_multi_cap()
    cash_p = fetch_multi_cash()

    ext_paths = fetch_extension.expand(year=recent_list)
    tax_paths = fetch_tax.expand(year=recent_list)

    ext_final = merge_yearly(ext_paths, name="extensions")
    tax_final = merge_yearly(tax_paths, name="tax")

    txn_p = fetch_txn(info)

    validate_assets(cba_p, cap_p, cash_p, ext_final, tax_final, txn_p)

spotrac_misc_assets()


