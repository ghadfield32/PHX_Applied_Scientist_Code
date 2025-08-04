# file: api/src/ml/feature_engineering.py
"""
columns to work with:
['2P', '2P%', '2PA', '3P', '3P%', '3PA', '3PA_ZERO', '3PAR', 'AST',
'AST%', 'AST_PER_36', 'AGE', 'BLK', 'BLK%', 'BPM', 'DBPM', 'DRB', 'DRB%',
'DWS', 'FG', 'FG%', 'FGA', 'FT', 'FT%', 'FTA', 'FTA_ZERO', 'FTR', 'GP', 'GS',
'LOSSES', 'MP', 'OBPM', 'ORB', 'ORB%', 'OWS', 'OFFENSIVE_LOAD%', 'PER', 'PF',
'PTS', 'PTS_PER_36', 'PLAYER', 'PLAYER_ID', 'PLAYER_POSS', 'PLAYMAKING_USAGE%',
'POSITION', 'STL', 'STL%', 'SCORING_USAGE%', 'SEASON', 'TOV', 'TOV%', 'TRB',
'TRB%', 'TRB_PER_36', 'TS%', 'TEAM', 'TEAM_ID', 'TEAM_POSS', 'TM_AST', 'TM_DRB',
'TM_FG', 'TM_FGA', 'TM_FTA', 'TM_MP', 'TM_ORB', 'TM_TOV', 'TM_TRB', 'TRUE_USAGE%',
'TURNOVER_USAGE%', 'USG%', 'VORP', 'WS', 'WS/48', 'WINS', 'YEARS_OF_SERVICE',
'EFG%', 'PLAYER_NAME', 'AGE_ADV', 'AGE_HUSTLE', 'AST_DEFENSE', 'AST_RANK',
'BLK_DEFENSE', 'BLKA', 'BLKA_RANK', 'BLK_RANK', 'BOX_OUTS', 'BOX_OUT_PLAYER_REBS',
'BOX_OUT_PLAYER_TEAM_REBS', 'CHARGES_DRAWN', 'CONTESTED_SHOTS',
'CONTESTED_SHOTS_2PT', 'CONTESTED_SHOTS_3PT', 'DD2', 'DD2_RANK', 'DEFLECTIONS',
'DEF_BOXOUTS', 'DEF_LOOSE_BALLS_RECOVERED', 'DREB', 'DREB_RANK', 'D_FG_PCT',
'FG3A', 'FG3A_RANK', 'FG3M', 'FG3M_RANK', 'FG3_PCT', 'FG3_PCT_RANK', 'FGA_DEFENSE',
'FGA_RANK', 'FGM', 'FGM_RANK', 'FG_PCT', 'FG_PCT_RANK', 'FTA_DEFENSE', 'FTA_RANK',
'FTM', 'FTM_RANK', 'FT_PCT', 'FT_PCT_RANK', 'G', 'GP_DEFENSE', 'GP_RANK', 'L',
'LOOSE_BALLS_RECOVERED', 'L_RANK', 'MIN_ADV', 'MIN_HUSTLE', 'MIN_RANK',
'NBA_FANTASY_PTS', 'NBA_FANTASY_PTS_RANK', 'NICKNAME', 'OFF_BOXOUTS',
'OFF_LOOSE_BALLS_RECOVERED', 'OREB', 'OREB_RANK', 'PCT_BOX_OUTS_DEF',
'PCT_BOX_OUTS_OFF', 'PCT_BOX_OUTS_REB', 'PCT_BOX_OUTS_TEAM_REB',
'PCT_LOOSE_BALLS_RECOVERED_DEF', 'PCT_LOOSE_BALLS_RECOVERED_OFF', 'PF_DEFENSE',
'PFD', 'PFD_RANK', 'PF_RANK', 'PLAYER_NAME_DEFENSE', 'PLUS_MINUS',
'PLUS_MINUS_RANK', 'PTS_DEFENSE', 'PTS_RANK', 'REB', 'REB_RANK', 'SCREEN_ASSISTS',
'SCREEN_AST_PTS', 'SOURCE', 'STL_DEFENSE', 'STL_RANK', 'TD3', 'TD3_RANK',
'TEAM_ABBREVIATION', 'TEAM_COUNT', 'TEAM_NAME', 'TOV_DEFENSE', 'TOV_RANK', 'W',
'WNBA_FANTASY_PTS', 'WNBA_FANTASY_PTS_RANK', 'W_PCT', 'W_PCT_RANK', 'W_RANK',
'BPM_BBREF', 'VORP_BBREF', 'DWS_BBREF', 'OWS_BBREF', 'ORB%_BBREF', 'DRB%_BBREF',
'TRB%_BBREF', '_MERGE', 'NODEFSTATAVAILCHECK', 'MERGED_SEASON_MIN',
'MERGED_SEASON_MAX', 'PLAYER_NAME_EFFICIENCY', 'GP_EFFICIENCY', 'W_EFFICIENCY',
'L_EFFICIENCY', 'W_PCT_EFFICIENCY', 'MIN', 'E_OFF_RATING', 'E_DEF_RATING',
'E_NET_RATING', 'E_AST_RATIO', 'E_OREB_PCT', 'E_DREB_PCT', 'E_REB_PCT',
'E_TOV_PCT', 'E_USG_PCT', 'E_PACE', 'GP_RANK_EFFICIENCY', 'W_RANK_EFFICIENCY',
'L_RANK_EFFICIENCY', 'W_PCT_RANK_EFFICIENCY', 'MIN_RANK_EFFICIENCY',
'E_OFF_RATING_RANK', 'E_DEF_RATING_RANK', 'E_NET_RATING_RANK', 'E_AST_RATIO_RANK',
'E_OREB_PCT_RANK', 'E_DREB_PCT_RANK', 'E_REB_PCT_RANK', 'E_TOV_PCT_RANK',
'E_USG_PCT_RANK', 'E_PACE_RANK', 'SEASON_TYPE', 'TEAM_ABBREVIATION_EFFICIENCY',
'TEAM_NAME_EFFICIENCY', 'CONTRACT_SEASON_RK', 'PLAYER_CONTRACTS', 'POS', 'TEAM                     SIGNED WITH', 'AGE                     AT SIGNING', 'START_CONTRACT_YEAR', 'END_CONTRACT_YEAR', 'YRS', 'VALUE', 'AAV', '2-YEAR CASH', '3-YEAR CASH', 'SERVICE_YEARS', 'MINIMUM_CONTRACT_VALUE', 'MINIMUM_MATCHED', 'BELOW_MINIMUM_FLOOR', 'TEAM_ABBREVIATION_CONTRACTS', 'TEAM_NAME_CONTRACTS', 'PLAYER_NAME_CONTRACTS', 'TAX_THRESHOLD', 'TAX_FREE_AGENCY', 'TAX_START_DATE', 'TAX_TRADE_DEADLINE', 'TAX_END_DATE', 'CAP_MINIMUM', 'CAP_MAXIMUM', 'CAP_DOLLARS_PLUS_MINUS', 'CAP_PERCENT_PLUS_MINUS', 'CBA_FREE_AGENCY', 'CBA_START_DATE', 'CBA_TRADE_DEADLINE', 'CBA_END_DATE', 'INJURY_START_DATE', 'INJURY_END_DATE', 'TOTAL_DAYS_INJURED', 'PLAYER_NAME_INJURIES', 'TEAM_NAME_INJURIES', 'COMBINED_REASONING', 'TEAM_ABBREVIATION_PLAYTYPE', 'PLAYER_NAME_PLAYTYPE', 'OFF_EFG_PCT_CUT', 'OFF_EFG_PCT_HANDOFF', 'OFF_EFG_PCT_ISOLATION', 'OFF_EFG_PCT_MISC', 'OFF_EFG_PCT_OFFREBOUND', 'OFF_EFG_PCT_OFFSCREEN', 'OFF_EFG_PCT_PRBALLHANDLER', 'OFF_EFG_PCT_PRROLLMAN', 'OFF_EFG_PCT_POSTUP', 'OFF_EFG_PCT_SPOTUP', 'OFF_EFG_PCT_TRANSITION', 'OFF_FG_PCT_CUT', 'OFF_FG_PCT_HANDOFF', 'OFF_FG_PCT_ISOLATION', 'OFF_FG_PCT_MISC', 'OFF_FG_PCT_OFFREBOUND', 'OFF_FG_PCT_OFFSCREEN', 'OFF_FG_PCT_PRBALLHANDLER', 'OFF_FG_PCT_PRROLLMAN', 'OFF_FG_PCT_POSTUP', 'OFF_FG_PCT_SPOTUP', 'OFF_FG_PCT_TRANSITION', 'OFF_FT_POSS_PCT_CUT', 'OFF_FT_POSS_PCT_HANDOFF', 'OFF_FT_POSS_PCT_ISOLATION', 'OFF_FT_POSS_PCT_MISC', 'OFF_FT_POSS_PCT_OFFREBOUND', 'OFF_FT_POSS_PCT_OFFSCREEN', 'OFF_FT_POSS_PCT_PRBALLHANDLER', 'OFF_FT_POSS_PCT_PRROLLMAN', 'OFF_FT_POSS_PCT_POSTUP', 'OFF_FT_POSS_PCT_SPOTUP', 'OFF_FT_POSS_PCT_TRANSITION', 'OFF_POSS_CUT', 'OFF_POSS_HANDOFF', 'OFF_POSS_ISOLATION', 'OFF_POSS_MISC', 'OFF_POSS_OFFREBOUND', 'OFF_POSS_OFFSCREEN', 'OFF_POSS_PRBALLHANDLER', 'OFF_POSS_PRROLLMAN', 'OFF_POSS_POSTUP', 'OFF_POSS_SPOTUP', 'OFF_POSS_TRANSITION', 'OFF_PPP_CUT', 'OFF_PPP_HANDOFF', 'OFF_PPP_ISOLATION', 'OFF_PPP_MISC', 'OFF_PPP_OFFREBOUND', 'OFF_PPP_OFFSCREEN', 'OFF_PPP_PRBALLHANDLER', 'OFF_PPP_PRROLLMAN', 'OFF_PPP_POSTUP', 'OFF_PPP_SPOTUP', 'OFF_PPP_TRANSITION', 'DEF_EFG_PCT_HANDOFF', 'DEF_EFG_PCT_ISOLATION', 'DEF_EFG_PCT_OFFSCREEN', 'DEF_EFG_PCT_PRBALLHANDLER', 'DEF_EFG_PCT_PRROLLMAN', 'DEF_EFG_PCT_POSTUP', 'DEF_EFG_PCT_SPOTUP', 'DEF_FG_PCT_HANDOFF', 'DEF_FG_PCT_ISOLATION', 'DEF_FG_PCT_OFFSCREEN', 'DEF_FG_PCT_PRBALLHANDLER', 'DEF_FG_PCT_PRROLLMAN', 'DEF_FG_PCT_POSTUP', 'DEF_FG_PCT_SPOTUP', 'DEF_FT_POSS_PCT_HANDOFF', 'DEF_FT_POSS_PCT_ISOLATION', 'DEF_FT_POSS_PCT_OFFSCREEN', 'DEF_FT_POSS_PCT_PRBALLHANDLER', 'DEF_FT_POSS_PCT_PRROLLMAN', 'DEF_FT_POSS_PCT_POSTUP', 'DEF_FT_POSS_PCT_SPOTUP', 'DEF_POSS_HANDOFF', 'DEF_POSS_ISOLATION', 'DEF_POSS_OFFSCREEN', 'DEF_POSS_PRBALLHANDLER', 'DEF_POSS_PRROLLMAN', 'DEF_POSS_POSTUP', 'DEF_POSS_SPOTUP', 'DEF_PPP_HANDOFF', 'DEF_PPP_ISOLATION', 'DEF_PPP_OFFSCREEN', 'DEF_PPP_PRBALLHANDLER', 'DEF_PPP_PRROLLMAN', 'DEF_PPP_POSTUP', 'DEF_PPP_SPOTUP', '__MERGE_OFFDEF', 'SOURCE_PLAYTYPE']


Additions:
Summary:
To predict a free agent’s Average Annual Value (AAV) accurately, you’ll want features in six broad categories:

1. Performance Metrics


3. Health & Durability

    Availability Rate: GP divided by maximum GP per season (you already compute this) but extend to multi-season missed games trend.

    Injury Counts & Severity: number of stints on injured list, games missed per injury category.

    Career‐High Consecutive Games: longest streak of games played as proxy for reliability.

 	- length since last injury
 	- length since last mid major injury (14+ days)
 	- length since last mid major injury (30+ days)
 	- ensure we take out the off season in the injury time based on the players last game of the season/playoffs/summer league


Additions that may need base data adjusting:
4. Career Trajectory & Context
a. Age & Experience
    - Experience (look into how to add this with our dataset)
d. Draft Position & Pre-Draft Rankings
    Draft Pick (e.g., lottery vs. mid-first vs. second round): pedigree affects early career earnings and market perception.
    Age at Draft: older draftees vs. “one-and-done” differences.
b. Contract-Year Indicator
    Binary Flag for Final Contract Year: players often “overperform” in contract years.
6. Temporal & Momentum Features
    Playoff vs. Regular-Season Splits: playoff performance often carries extra weight in negotiations.



5. Contract & Market Factors

    Previous Contract AAV: last season’s AAV often sets a baseline for negotiations.

    Bird Rights Status: restricted vs. unrestricted free agent indicator.

    Team Salary Cap Space: teams’ financial flexibility can inflate/deflate offers.

    Market Size & Team Competitiveness: large markets (LAL, NYK, BOS, PHX, etc.) pay premiums—use your is_large_market flag.
    NBA

    Luxury Tax Proximity: whether taking the contract pushes team into repeater tax can temper offers.


 7. others:
Portability

    16% Scoring Efficiency
    40% Shooting Ability
    8% Defensive Ability
    5% Defensive Versatility
    25% Passing Ability
    6% Usage (penalize outliers)
⦁		pick a few from here that I can recreate easily: https://craftednba.com/glossary


"""

from __future__ import annotations
from typing import List, Tuple, Optional
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.linear_model import LinearRegression
import logging

logger = logging.getLogger(__name__)

# Try to import NBA utils for enhanced team/player matching
try:
    from utils.nba_utils import load_nba_directories, enrich_with_nba_ids, _NBA_ABBRS
    NBA_UTILS_AVAILABLE = True
except ImportError:
    NBA_UTILS_AVAILABLE = False
    print("⚠ NBA utils not available - team/player matching will be limited")


def _ci_col(df: pd.DataFrame, target: str) -> Optional[str]:
    """Return the actual column name in df matching target case-insensitively, or None."""
    for c in df.columns:
        if c.lower() == target.lower():
            return c
    return None

def _first_present_ci(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Like _first_present but case-insensitive matching of candidate names."""
    lowered = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None




def add_salary_pct_cap(df: pd.DataFrame,
                       target: str = "aav",
                       cap_col: str = "salary_cap"
                       ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create AAV_PCT_CAP if an applicable cap column is found.
    This normalizes salary values across seasons since max contracts are % of cap.
    """
    if target not in df.columns:
        raise ValueError(f"add_salary_pct_cap: target column {target} not found")

    if not cap_col:
        # error
        raise ValueError(f"add_salary_pct_cap: no cap column found: {cap_col}")

    out = df.copy()
    denom = out[cap_col].replace({0: np.nan})
    out["AAV_PCT_CAP"] = (out[target] / denom).clip(lower=0, upper=1)
    return out, ["AAV_PCT_CAP"]

def require_columns(df: pd.DataFrame, cols: List[str], context: str) -> None:
    """Raise an error if any of the required columns are missing from df."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[require_columns] Missing required columns for {context}: {missing}")


# ---------- Updated feature functions ----------

def add_season_start_year(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Extract the numeric season start year from Season-like string.
    STRICT: only accepts 'Season' or 'SEASON' exactly.
    """
    out = df.copy()
    if "Season" in out.columns:
        season_col = "Season"
    elif "SEASON" in out.columns:
        season_col = "SEASON"
    else:
        raise ValueError("add_season_start_year: neither 'Season' nor 'SEASON' column present.")

    if out[season_col].isna().all():
        raise ValueError("add_season_start_year: season column exists but all values are NaN/empty.")

    # Expect format like '2023-24'; extract first four-digit year
    out["season_start_year"] = (
        out[season_col].astype(str).str.extract(r"(\d{4})")[0]
        .astype("float64")
    )
    if out["season_start_year"].isna().all():
        raise ValueError(f"add_season_start_year: failed to parse any season start year from '{season_col}' values.")

    # change season_start_year to object
    out["season_start_year"] = out["season_start_year"].astype(str)
    return out, ["season_start_year"]



def add_experience_bucket_percentile(
    df: pd.DataFrame,
    quantiles: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    col_name: str = "experience_bucket",
    method: str = "percentile",                # <<< new
    fixed_bins: Optional[List[float]] = None,   # <<< new
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Creates an experience bucket column from years-of-service.

    Args:
        df: input DataFrame
        quantiles: fractions for percentile-based buckets
        labels: labels for buckets (percentile or fixed)
        col_name: name of new column
        method: "percentile" (default) or "fixed"
        fixed_bins: sorted edges for fixed bins (only used if method=="fixed")

    Returns:
        out: DataFrame with new column
        [col_name]: list of new columns
        used_categories: ordered list of categories
    """
    # 1) locate the YOS column
    yos_col = _first_present_ci(
        df,
        ["Years_of_Service_x", "Years_of_Service_y", "Years_of_Service", "YEARS_OF_SERVICE"],
    )
    if not yos_col:
        print(f"[debug] {col_name} skipped: no years-of-service column found")
        return df.copy(), [], []

    out = df.copy()

    # 2) FIXED‐BIN MODE
    if method == "fixed" and fixed_bins:
        edges = sorted(fixed_bins)
        # auto‐generate labels if none provided
        if labels is None:
            labels = []
            for i in range(len(edges) - 1):
                low = edges[i]
                high = edges[i + 1]
                # for all but last bin, label as "low-(high-1)"
                if i < len(edges) - 2:
                    labels.append(f"{int(low)}-{int(high-1)}")
                else:
                    labels.append(f"{int(low)}+")
        bucket_series = pd.cut(
            out[yos_col].astype("float64"),
            bins=edges,
            labels=labels,
            right=False,
            include_lowest=True,
        )

    # 3) PERCENTILE‐BIN MODE (original logic)
    else:
        # set up quantile edges
        if quantiles is None:
            q_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
        else:
            q_edges = sorted(set([0.0, 1.0] + quantiles))
        # auto labels like "0-25%"
        if labels is None:
            labels = [
                f"{int(round(low*100))}-{int(round(high*100))}%"
                for low, high in zip(q_edges[:-1], q_edges[1:])
            ]
        try:
            bucket_series = pd.qcut(
                out[yos_col].astype("float64"),
                q=q_edges,
                labels=labels,
                duplicates="drop",
                precision=3,
            )
        except ValueError:
            # fallback if too few unique values
            uniq_edges = np.unique(out[yos_col].dropna().quantile(q_edges).values)
            if len(uniq_edges) < 2:
                out[col_name] = pd.Categorical(
                    [labels[0]] * len(out),
                    categories=[labels[0]],
                    ordered=True,
                )
                return out, [col_name], [labels[0]]
            fallback_labels = [
                f"{int(round(q_edges[i]*100))}-{int(round(q_edges[i+1]*100))}%"
                for i in range(len(uniq_edges) - 1)
            ]
            bucket_series = pd.cut(
                out[yos_col].astype("float64"),
                bins=uniq_edges,
                labels=fallback_labels,
                include_lowest=True,
                ordered=True,
            )
            labels = fallback_labels

    # 4) attach and debug‐print
    out[col_name] = bucket_series
    print(f"[debug] {col_name} counts:\n{out[col_name].value_counts(dropna=False)}")

    # 5) extract the actual ordered categories
    if isinstance(out[col_name].dtype, pd.CategoricalDtype):
        used_categories = [
            str(cat) for cat in out[col_name].cat.categories if pd.notna(cat)
        ]
    else:
        used_categories = labels

    return out, [col_name], used_categories



def _resolve_stat_variant(df: pd.DataFrame, stat: str) -> Optional[str]:
    """Find the best matching column for a requested stat (case-insensitive / common variants)."""
    # exact match ignoring case
    ci = _ci_col(df, stat)
    if ci:
        return ci
    # try underscore/percent normalization heuristics
    lowered = stat.lower()
    for c in df.columns:
        if c.lower() == lowered:
            return c
        if lowered.replace("%", "") in c.lower().replace("%", ""):
            return c
    return None


def add_rolling_features(df: pd.DataFrame,
                         window: int = 3,
                         stats: List[str] = None
                        ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Rolling mean and slope for explicitly provided stat column names.
    STRICT: stats must be actual columns in df; will error if none exist.
    """
    if not stats:
        raise ValueError("add_rolling_features: must supply explicit stats list (e.g., ['WS', 'PTS_PER_36']).")
    require_columns(df, ["season_start_year", "PLAYER_ID"], "add_rolling_features")

    out = df.copy()
    created = []
    out = out.sort_values(["PLAYER_ID", "season_start_year"])
    gp = out.groupby("PLAYER_ID")

    valid_stats = [s for s in stats if s in out.columns]
    missing = [s for s in stats if s not in out.columns]
    if not valid_stats:
        raise ValueError(f"add_rolling_features: none of the requested stats are present: missing {missing}")
    if missing:
        print(f"[warning] add_rolling_features: skipping missing stats: {missing}")

    for stat in valid_stats:
        # rolling mean
        col_roll = f"{stat}_rollmean_{window}"
        out[col_roll] = gp[stat].rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        created.append(col_roll)

    # slope helper that ensures single-output
    def slope(x):
        arr = x.values
        if arr.size < 2:
            return np.nan
        X = np.arange(len(arr)).reshape(-1, 1)
        y = arr.ravel()  # ensure 1d so LinearRegression.coef_ is shape (1,)
        model = LinearRegression().fit(X, y)
        coef = model.coef_.ravel()[0]
        return float(coef)

    for stat in valid_stats:
        col_slope = f"{stat}_rollslope_{window}"
        out[col_slope] = gp[stat].rolling(window, min_periods=2).apply(slope, raw=False).reset_index(level=0, drop=True)
        created.append(col_slope)

    return out, created



def add_contract_year_flag(df: pd.DataFrame,
                           contract_end_col: str = "contract_end_season",
                           season_col: str = "Season") -> Tuple[pd.DataFrame, List[str]]:
    """
    Flag if current row is in the final year of the player's contract.
    Requires a column that denotes the last season of the contract (e.g., '2024-25').
    """
    out = df.copy()
    created = []
    if contract_end_col not in out.columns or season_col not in out.columns:
        return out, created  # missing data; skip

    def _normalize_season(season_str: str) -> Optional[int]:
        if not isinstance(season_str, str):
            return None
        m = re.match(r"(\d{4})", season_str)
        return int(m.group(1)) if m else None

    # Extract start year for both contract end and current season
    out["__season_year"] = out[season_col].astype(str).str.extract(r"(\d{4})")[0].astype("float64")
    out["__contract_end_year"] = out[contract_end_col].astype(str).str.extract(r"(\d{4})")[0].astype("float64")

    # Flag where current season equals contract end year
    out["is_contract_year"] = (out["__season_year"] == out["__contract_end_year"]).astype("int8")
    created.append("is_contract_year")

    # Clean up intermediate
    out.drop(columns=["__season_year", "__contract_end_year"], inplace=True)

    return out, created


def add_multi_season_availability_trend(df: pd.DataFrame,
                                        window: int = 3) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute rolling slope/trend in availability_rate over the past `window` seasons,
    per player. Captures escalating or declining reliability across seasons.
    """
    out = df.copy()
    created = []
    if "PLAYER_ID" not in out.columns or "availability_rate" not in out.columns:
        return out, created

    # Ensure season ordering
    if "season_start_year" not in out.columns:
        # Try to extract it if missing
        out, _ = add_season_start_year(out)

    out = out.sort_values(["PLAYER_ID", "season_start_year"])
    gp = out.groupby("PLAYER_ID")

    def slope(series):
        y = series.values.reshape(-1, 1)
        X = np.arange(len(y)).reshape(-1, 1)
        if len(y) < 2:
            return np.nan
        model = LinearRegression().fit(X, y)
        return float(model.coef_[0])

    col_name = f"availability_slope_{window}"
    out[col_name] = gp["availability_rate"].rolling(window, min_periods=2).apply(slope, raw=False).reset_index(level=0, drop=True)
    created.append(col_name)

    return out, created


def diagnose_injury_date_completeness(
    df: pd.DataFrame,
    injury_start_col: str = "INJURY_START_DATE",
    injury_end_col: str = "INJURY_END_DATE",
) -> pd.DataFrame:
    """
    Return per-row category of why TOTAL_DAYS_INJURED is missing or present:
      - both_missing: both start and end are null
      - valid_stint: both present and end >= start
      - end_before_start: both present but end < start
      - only_start_present: start present, end missing
      - only_end_present: end present, start missing
      - parse_error: parsing failed for one or both (non-datetime strings)
    """
    out = df.copy()

    def to_dt(x):
        try:
            return pd.to_datetime(x)
        except Exception:
            return pd.NaT

    start = out[injury_start_col].apply(to_dt)
    end = out[injury_end_col].apply(to_dt)

    both_missing = start.isna() & end.isna()
    valid = start.notna() & end.notna() & (end >= start)
    end_before_start = start.notna() & end.notna() & (end < start)
    only_start = start.notna() & end.isna()
    only_end = end.notna() & start.isna()
    # parse error can overlap with only_* if original strings were invalid; capture rows where original not both null
    parse_failed = (~both_missing) & (start.isna() | end.isna()) & ~(only_start | only_end)

    category = (
        np.where(both_missing, "both_missing",
        np.where(valid, "valid_stint",
        np.where(end_before_start, "end_before_start",
        np.where(only_start, "only_start_present",
        np.where(only_end, "only_end_present",
        np.where(parse_failed, "parse_error", "other"))))))
    )

    summary = (
        pd.Series(category)
        .value_counts(dropna=False)
        .rename_axis("reason")
        .reset_index(name="count")
    )
    return summary


def add_injury_reliability_features(
    df: pd.DataFrame,
    injury_start_col: str = "INJURY_START_DATE",
    injury_end_col: str = "INJURY_END_DATE",
    season_col: str = "Season",
    last_game_col: Optional[str] = None,   # optional: per-player last game date
    season_end_col: Optional[str] = None,  # optional: league season end date per season
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Derive injury-related reliability features WITHOUT over-imputing.

    Detailed diagnostics are emitted to help understand missingness sources.
    """
    out = df.copy()
    created: List[str] = []

    if injury_start_col not in out.columns or injury_end_col not in out.columns:
        print("[add_injury_reliability_features] skipped: injury date columns not present.")
        return out, created

    # Vectorized safe datetime parsing
    start_dt = pd.to_datetime(out[injury_start_col], errors="coerce")
    end_dt = pd.to_datetime(out[injury_end_col], errors="coerce")

    # Masks for categories
    both_missing_mask = start_dt.isna() & end_dt.isna()
    valid_stint_mask = start_dt.notna() & end_dt.notna() & (end_dt >= start_dt)
    end_before_start_mask = start_dt.notna() & end_dt.notna() & (end_dt < start_dt)
    only_start_present_mask = start_dt.notna() & end_dt.isna()
    only_end_present_mask = end_dt.notna() & start_dt.isna()
    ambiguous_mask = (
        (~both_missing_mask)
        & ~(valid_stint_mask)
        & ~(end_before_start_mask)
        & ~(only_start_present_mask)
        & ~(only_end_present_mask)
    )

    diag = {
        "both_missing": int(both_missing_mask.sum()),
        "valid_stint": int(valid_stint_mask.sum()),
        "end_before_start": int(end_before_start_mask.sum()),
        "only_start_present": int(only_start_present_mask.sum()),
        "only_end_present": int(only_end_present_mask.sum()),
        "ambiguous_other": int(ambiguous_mask.sum()),
    }
    print("[add_injury_reliability_features][diagnostics] injury date completeness breakdown:", diag)

    # === TOTAL_DAYS_INJURED ===
    # Start with all NaN
    total_days = pd.Series(index=out.index, dtype="Float64")

    # both missing => 0
    total_days = total_days.mask(both_missing_mask, 0)

    # valid stint => difference in days
    total_days = total_days.mask(valid_stint_mask, (end_dt - start_dt).dt.days)

    out["TOTAL_DAYS_INJURED"] = total_days
    created.append("TOTAL_DAYS_INJURED")

    # Major injury flags (only fire when value known)
    out["major_injury_14d_flag"] = (out["TOTAL_DAYS_INJURED"] >= 14).astype("Int8")
    out["major_injury_30d_flag"] = (out["TOTAL_DAYS_INJURED"] >= 30).astype("Int8")
    created += ["major_injury_14d_flag", "major_injury_30d_flag"]

    # === Consistency checks / deeper diagnostics ===
    # Check that both_missing rows have TOTAL_DAYS_INJURED == 0
    mismatched_both_missing = out.loc[both_missing_mask & (out["TOTAL_DAYS_INJURED"] != 0)]
    if not mismatched_both_missing.empty:
        print(f"[error] {len(mismatched_both_missing)} rows with both dates missing do NOT have TOTAL_DAYS_INJURED == 0.")
        print(mismatched_both_missing[[injury_start_col, injury_end_col, "TOTAL_DAYS_INJURED"]].head(5))

    # Check that valid stints have the expected difference
    computed_diff = (end_dt - start_dt).dt.days
    mismatched_valid = out.loc[valid_stint_mask & (out["TOTAL_DAYS_INJURED"] != computed_diff)]
    if not mismatched_valid.empty:
        print(f"[error] {len(mismatched_valid)} valid stint rows where TOTAL_DAYS_INJURED != end - start.")
        print(mismatched_valid[[injury_start_col, injury_end_col, "TOTAL_DAYS_INJURED"]].head(5))

    # Report summary of TOTAL_DAYS_INJURED distribution conditional on category
    def pct_zero(mask):
        if mask.sum() == 0:
            return np.nan
        return (out.loc[mask, "TOTAL_DAYS_INJURED"] == 0).mean() * 100

    summary_df = pd.DataFrame({
        "category": [
            "both_missing",
            "valid_stint",
            "end_before_start",
            "only_start_present",
            "only_end_present",
            "ambiguous_other",
        ],
        "count": [
            diag["both_missing"],
            diag["valid_stint"],
            diag["end_before_start"],
            diag["only_start_present"],
            diag["only_end_present"],
            diag["ambiguous_other"],
        ],
        "pct_total_days_zero": [
            pct_zero(both_missing_mask),
            pct_zero(valid_stint_mask),
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        "n_null_total_days": [
            out.loc[both_missing_mask, "TOTAL_DAYS_INJURED"].isna().sum() if diag["both_missing"] > 0 else 0,
            out.loc[valid_stint_mask, "TOTAL_DAYS_INJURED"].isna().sum() if diag["valid_stint"] > 0 else 0,
            out.loc[end_before_start_mask, "TOTAL_DAYS_INJURED"].isna().sum() if diag["end_before_start"] > 0 else 0,
            out.loc[only_start_present_mask, "TOTAL_DAYS_INJURED"].isna().sum() if diag["only_start_present"] > 0 else 0,
            out.loc[only_end_present_mask, "TOTAL_DAYS_INJURED"].isna().sum() if diag["only_end_present"] > 0 else 0,
            out.loc[ambiguous_mask, "TOTAL_DAYS_INJURED"].isna().sum() if diag["ambiguous_other"] > 0 else 0,
        ],
    })
    print("[add_injury_reliability_features][summary of TOTAL_DAYS_INJURED by category]\n", summary_df)

    # Warnings for partial/invalid cases
    if diag["end_before_start"] > 0:
        print(f"[warn] {diag['end_before_start']} rows have INJURY_END_DATE < INJURY_START_DATE")
    if diag["only_start_present"] > 0 or diag["only_end_present"] > 0:
        print(f"[info] Partial injury date info: only_start_present={diag['only_start_present']}, only_end_present={diag['only_end_present']}")
    if diag["ambiguous_other"] > 0:
        print(f"[warn] {diag['ambiguous_other']} rows fell into ambiguous / unexpected category in injury date logic.")

    return out, created





def add_draft_features(df: pd.DataFrame,
                       draft_year_col: str = "DRAFT_YEAR",
                       draft_pick_col: str = "DRAFT_PICK",
                       birthdate_col: str = "BIRTHDATE") -> Tuple[pd.DataFrame, List[str]]:
    """
    Add pedigree-related features:
      - age_at_draft
      - draft_bucket (lottery/mid-first/second)
    Requires birthdate, draft year, and draft pick.
    """
    out = df.copy()
    created = []

    if draft_year_col not in out.columns or draft_pick_col not in out.columns or birthdate_col not in out.columns:
        return out, created

    # Age at draft: draft year minus birth year (rough)
    def _compute_age_at_draft(row):
        try:
            birth = pd.to_datetime(row[birthdate_col])
            draft_year_raw = int(row[draft_year_col])
            # assume draft happens mid-year (June), approximate
            draft_date = pd.Timestamp(f"{draft_year_raw}-06-01")
            return (draft_date - birth).days / 365.25
        except Exception:
            return np.nan

    out["age_at_draft"] = out.apply(_compute_age_at_draft, axis=1)
    created.append("age_at_draft")

    def _bucket_pick(pick):
        try:
            pick = int(pick)
            if 1 <= pick <= 14:
                return "lottery"
            if 15 <= pick <= 30:
                return "mid_first"
            if 31 <= pick <= 60:
                return "second_round"
            return "undrafted"
        except Exception:
            return np.nan

    out["draft_bucket"] = out[draft_pick_col].apply(_bucket_pick)
    created.append("draft_bucket")

    return out, created


def compute_portability(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()
    created = []

    # Required base metrics
    for col in ["TS%", "EFG%", "USG%"]:
        if col not in out.columns:
            return out, created  # missing core piece

    def _standardize(series):
        series = series.astype("float64")
        if series.std(ddof=0) == 0 or pd.isna(series.std(ddof=0)):
            return pd.Series(0.0, index=series.index)
        return (series - series.mean()) / series.std(ddof=0)

    score = 0.0

    # 16% scoring efficiency (TS%)
    ts = out["TS%"].fillna(out["TS%"].mean())
    comp_scoring = _standardize(ts)
    score = score + 0.16 * comp_scoring

    # 40% shooting ability: EFG%
    efg = out["EFG%"].fillna(out["EFG%"].mean())
    shooting_base = _standardize(efg)
    score = score + 0.40 * shooting_base

    # 8% defensive ability: use BPM if present (upper case)
    if "BPM" in out.columns:
        def_eff = _standardize(out["BPM"].fillna(out["BPM"].mean()))
    else:
        def_eff = pd.Series(0.0, index=out.index)
    score = score + 0.08 * def_eff

    # 5% defensive versatility: PLUS_MINUS as proxy
    if "PLUS_MINUS" in out.columns:
        versatility = _standardize(out["PLUS_MINUS"].fillna(out["PLUS_MINUS"].mean()))
    else:
        versatility = pd.Series(0.0, index=out.index)
    score = score + 0.05 * versatility

    # 25% passing ability: prefer AST_PER_36, else AST
    if "AST_PER_36" in out.columns:
        passing = _standardize(out["AST_PER_36"].fillna(out["AST_PER_36"].mean()))
    elif "AST" in out.columns:
        passing = _standardize(out["AST"].fillna(out["AST"].mean()))
    else:
        passing = pd.Series(0.0, index=out.index)
    score = score + 0.25 * passing

    # 6% usage penalty: high usage gets penalized
    usage = out["USG%"].fillna(out["USG%"].mean())
    usage_std = _standardize(usage)
    usage_penalty = -usage_std.clip(lower=0)
    score = score + 0.06 * usage_penalty

    out["portability_score"] = score
    created.append("portability_score")
    return out, created

def add_market_tier(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Tag each team as 'big_market', 'small_market', or 'medium_market'.
    """
    out = df.copy()
    created = []
    team_col = _first_present_ci(out, ["TEAM_ABBREVIATION", "TEAM_ID", "TEAM"])
    if not team_col:
        return out, created

    # Define your market lists:
    big = {"NYK","BKN","LAL","LAC","CHI","PHI","DAL","TOR","GSW","ATL","HOU","WAS","BOS","MIA"}
    small = {"SAC","POR","CHA","IND","UTA","SAS","MIL","OKC","NOP","DET"}

    def _tier(x):
        if x in big:
            return "big_market"
        if x in small:
            return "small_market"
        return "medium_market"

    out["market_tier"] = out[team_col].map(_tier).fillna("medium_market")
    created.append("market_tier")
    return out, created



def add_team_cap_space(df: pd.DataFrame,
                       salary_commitment_col: str = "AAV",
                       cap_col: str = "CAP_MAXIMUM",
                       tax_threshold_col: str = "TAX_THRESHOLD",
                       team_key: str = "TEAM_ABBREVIATION",
                       season_year_col: str = "season_start_year") -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute per-team-season payroll commitments and available cap space.

    Adds:
      - team_total_commitment: sum of existing AAVs (na treated as 0) for the team-season
      - team_cap_space: CAP_MAXIMUM - team_total_commitment
      - pct_cap_used: team_total_commitment / CAP_MAXIMUM
      - is_over_cap: bool flag if current commitment exceeds CAP_MAXIMUM
      - pushes_into_tax: bool flag if commitment exceeds TAX_THRESHOLD (if available)
      - pct_to_tax: fraction of tax threshold used (if available)

    Assumptions:
      * AAV is the current season hit; if you have actual salary for the season replace salary_commitment_col accordingly.
      * CAP_MAXIMUM and TAX_THRESHOLD exist per row and are consistent within a team-season.
    """
    out = df.copy()
    created = []

    # validation
    missing = []
    for col in [salary_commitment_col, cap_col, team_key, season_year_col]:
        if col not in out.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"add_team_cap_space: missing required columns: {missing}")

    # Replace NaN AAV with 0 for aggregation (but keep original for player-level diagnostics)
    out["_salary_for_agg"] = out[salary_commitment_col].fillna(0)

    # Aggregate per team-season
    team_payroll = (
        out
        .groupby([team_key, season_year_col], dropna=False)["_salary_for_agg"]
        .sum()
        .rename("team_total_commitment")
        .reset_index()
    )

    # Merge back
    out = out.merge(team_payroll, on=[team_key, season_year_col], how="left")

    # Cap space calculation; avoid divide by zero
    out["team_cap_space"] = out[cap_col] - out["team_total_commitment"]
    out["pct_cap_used"] = np.where(
        out[cap_col].replace({0: np.nan}).notna(),
        out["team_total_commitment"] / out[cap_col],
        np.nan,
    )
    out["is_over_cap"] = (out["team_total_commitment"] > out[cap_col]).astype("int8")
    created.extend([
        "team_total_commitment",
        "team_cap_space",
        "pct_cap_used",
        "is_over_cap",
    ])

    # Luxury tax / tax proximity if available
    if tax_threshold_col in out.columns:
        out["pushes_into_tax"] = (out["team_total_commitment"] > out[tax_threshold_col]).astype("int8")
        out["pct_to_tax"] = np.where(
            out[tax_threshold_col].replace({0: np.nan}).notna(),
            out["team_total_commitment"] / out[tax_threshold_col],
            np.nan,
        )
        created.extend(["pushes_into_tax", "pct_to_tax"])

    # cleanup
    out.drop(columns=["_salary_for_agg"], inplace=True)

    return out, created


import csv
from pathlib import Path
import numpy as np
import pandas as pd

def load_max_contract_values(csv_path: str | Path) -> pd.DataFrame:
    """
    Robust loader for max-contract CSV. Enhancements over previous version:
      - Auto-detects delimiter via csv.Sniffer, with fallback to pandas inference.
      - Handles BOM by using utf-8-sig when reading.
      - If only one column is parsed (common when delimiter mismatches), attempts
        manual splitting on common delimiters to recover headers.
      - Normalizes headers, coerces types, dedupes, and sanity-checks tiers.
    Raises:
      - FileNotFoundError if path missing.
      - ValueError if expected columns are still absent or final uniqueness violated.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"[load_max_contract_values] Not found: {p}")

    # Helper to sniff delimiter
    def _detect_delimiter(sample_text: str) -> str | None:
        try:
            # candidate delimiters: comma, tab, semicolon, pipe
            dialect = csv.Sniffer().sniff(sample_text, delimiters=",\t;|")
            return dialect.delimiter
        except Exception:
            return None  # let pandas fallback

    # Read a sample to guess delimiter
    with open(p, "r", encoding="utf-8-sig", errors="ignore") as f:
        sample = f.read(4096)
    delim = _detect_delimiter(sample)

    if delim:
        raw = pd.read_csv(p, sep=delim, engine="python", dtype=str, encoding="utf-8-sig").copy()
        print(f"[load_max_contract_values] Using detected delimiter: {repr(delim)}")
    else:
        # let pandas try to infer via its own sniffing (python engine required)
        raw = pd.read_csv(p, sep=None, engine="python", dtype=str, encoding="utf-8-sig").copy()
        print("[load_max_contract_values] No explicit delimiter detected; using pandas inference (sep=None).")

    # Normalize headers: strip whitespace/BOM remnants, uppercase, replace spaces
    def _clean_col(c: str) -> str:
        if c is None:
            return ""
        c = c.strip()
        c = c.lstrip("\ufeff")  # extra safety against BOM
        return c.upper().replace(" ", "_")

    raw.columns = [_clean_col(c) for c in raw.columns]

    expected = [
        "CBA_PERIOD",
        "MAX_PERCENTAGE_OF_CAP",
        "YEAR",
        "MAXIMUM_CONTRACT_SALARY",
        "SEASON",
        "YEARS_OF_SERVICE",
    ]

    # Fallback: if expected columns missing and only one column present, try manual split
    missing = [c for c in expected if c not in raw.columns]
    if missing and len(raw.columns) == 1:
        single = raw.columns[0]
        print(f"[load_max_contract_values][fallback] Only one column detected ({single}); attempting manual split.")
        first_row = raw.iloc[0, 0] if not raw.empty else ""
        for sep in ["\t", ";", ",", "|"]:
            if sep in first_row:
                print(f"[load_max_contract_values][fallback] Splitting on {repr(sep)}")
                expanded = raw[single].str.split(sep, expand=True)
                header_vals = expanded.iloc[0].astype(str).tolist()
                new_header = [_clean_col(h) for h in header_vals]
                data = expanded.iloc[1:].copy()
                data.columns = new_header
                raw = data.reset_index(drop=True)
                break  # stop after successful heuristic
        missing = [c for c in expected if c not in raw.columns]

    if missing:
        raise ValueError(f"[load_max_contract_values] Missing columns in CSV after normalization: {missing}; saw {raw.columns.tolist()}")

    # Trim cells and coerce types
    df = raw.copy()
    for c in expected:
        df[c] = df[c].astype(str).str.strip()

    # numeric coercions
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["YEARS_OF_SERVICE"] = pd.to_numeric(df["YEARS_OF_SERVICE"], errors="coerce").astype("Int64")
    df["MAXIMUM_CONTRACT_SALARY"] = df["MAXIMUM_CONTRACT_SALARY"].astype(str).str.replace(r"[,$]", "", regex=True)
    df["MAXIMUM_CONTRACT_SALARY"] = pd.to_numeric(df["MAXIMUM_CONTRACT_SALARY"], errors="coerce").astype("Int64")
    df["MAX_PERCENTAGE_OF_CAP"] = pd.to_numeric(df["MAX_PERCENTAGE_OF_CAP"], errors="coerce").astype(float)

    # drop fully null join keys
    df = df.dropna(subset=["YEAR", "YEARS_OF_SERVICE"]).copy()

    # dedupe keep last
    before = len(df)
    df = (
        df.sort_values(["YEAR", "YEARS_OF_SERVICE"])
          .drop_duplicates(["YEAR", "YEARS_OF_SERVICE"], keep="last")
          .copy()
    )
    dups_dropped = before - len(df)
    if dups_dropped:
        print(f"[load_max_contract_values] Dropped {dups_dropped} duplicate rows on (YEAR, YEARS_OF_SERVICE).")

    # Sanity: expected percentage tiers
    yos = df["YEARS_OF_SERVICE"]
    expected_pct = pd.Series(np.nan, index=yos.index, dtype=float)
    expected_pct.loc[(yos >= 0) & (yos <= 6)] = 0.25
    expected_pct.loc[(yos >= 7) & (yos <= 9)] = 0.30
    expected_pct.loc[yos >= 10] = 0.35

    valid_mask = df["MAX_PERCENTAGE_OF_CAP"].notna()
    mismatches = df[valid_mask & (~np.isclose(df["MAX_PERCENTAGE_OF_CAP"].astype(float), expected_pct.astype(float), atol=1e-6))]
    if not mismatches.empty:
        print("[load_max_contract_values][warn] Found YOS->MAX% rows that diverge from the canonical 25/30/35 tiers (designated vet / 105% rules can explain).")
        print(mismatches.head(5))

    # Uniqueness guard
    g = df.groupby(["YEAR", "YEARS_OF_SERVICE"]).size()
    if g.max() > 1:
        raise ValueError("[load_max_contract_values] Non-unique (YEAR, YEARS_OF_SERVICE) after cleaning.")

    return df[expected].copy()

from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

def add_max_contract_salary(
    df: pd.DataFrame,
    csv_path: Optional[str | Path] = None,
    max_df: Optional[pd.DataFrame] = None,
    debug: bool = True,
    allow_fallback: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Left-join MAXIMUM_CONTRACT_SALARY and MAX_PERCENTAGE_OF_CAP from the cleaned
    max-contract table using (season_start_year == YEAR) and (YEARS_OF_SERVICE).
    Adds provenance via `max_pct_source` ('csv', 'tier_fallback', or 'missing').
    If a direct match is absent and `allow_fallback` is True, infers MAX_PERCENTAGE_OF_CAP
    from standard CBA tiers (0-6:25%, 7-9:30%, 10+:35%). Does NOT infer MAXIMUM_CONTRACT_SALARY
    unless it's present in the CSV.

    Returns:
        merged dataframe, list of created feature column names
    """
    out = df.copy()
    created: List[str] = []

    # === season_start_year ===
    if "season_start_year" not in out.columns:
        # Prefer the existing helper if available
        try:
            out, _ = add_season_start_year(out)  # reuse the existing logic for consistency
        except Exception:
            # fallback to regex extraction (mirrors prior behavior)
            if "SEASON" in out.columns:
                out["season_start_year"] = (
                    out["SEASON"].astype(str).str.extract(r"(\d{4})")[0].astype("float64")
                )
            else:
                raise ValueError("add_max_contract_salary: cannot derive season_start_year; missing 'SEASON' column.")

    # === locate years-of-service column (case-insensitive) ===
    def _first_present_ci(df_inner: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        lowered = {c.lower(): c for c in df_inner.columns}
        for cand in candidates:
            if cand.lower() in lowered:
                return lowered[cand.lower()]
        return None

    yos_col = _first_present_ci(
        out,
        ["YEARS_OF_SERVICE", "Years_of_Service", "Years_of_Service_x", "Years_of_Service_y"],
    )
    if not yos_col:
        if debug:
            print("[add_max_contract_salary] skipped: no YEARS_OF_SERVICE column in base df.")
        return out, created

    # === load / validate max-contract table ===
    if max_df is None:
        try:
            from api.src.ml import config as _cfg
            default_path = _cfg.MAX_CONTRACT_VALUES_CSV
        except Exception:
            default_path = None
        csv_path = csv_path or default_path
        if csv_path is None:
            raise ValueError("[add_max_contract_salary] No csv_path provided and config path unavailable.")
        max_df = load_max_contract_values(csv_path)

    required_max_cols = ["YEAR", "YEARS_OF_SERVICE", "MAXIMUM_CONTRACT_SALARY", "MAX_PERCENTAGE_OF_CAP"]
    missing_in_max = [c for c in required_max_cols if c not in max_df.columns]
    if missing_in_max:
        raise ValueError(f"add_max_contract_salary: max_df missing required columns: {missing_in_max}")

    # === prepare join keys ===
    out["__join_year"] = pd.to_numeric(out["season_start_year"], errors="coerce").astype("Int64")
    out["__join_yos"] = pd.to_numeric(out[yos_col], errors="coerce").astype("Int64")

    msel = max_df[required_max_cols].copy()
    msel["YEAR"] = pd.to_numeric(msel["YEAR"], errors="coerce").astype("Int64")
    msel["YEARS_OF_SERVICE"] = pd.to_numeric(msel["YEARS_OF_SERVICE"], errors="coerce").astype("Int64")

    # === merge ===
    merged = out.merge(
        msel,
        left_on=["__join_year", "__join_yos"],
        right_on=["YEAR", "YEARS_OF_SERVICE"],
        how="left",
        indicator="_merge_maxsalary",
    )

    if debug:
        print("[add_max_contract_salary] merge status:\n", merged["_merge_maxsalary"].value_counts(dropna=False))

    # === provenance / source tracking ===
    merged["max_pct_source"] = "missing"
    mask_both = merged["_merge_maxsalary"] == "both"
    merged.loc[mask_both, "max_pct_source"] = "csv"

    # === fallback logic for MAX_PERCENTAGE_OF_CAP ===
    if allow_fallback:
        mask_fallback = (merged["_merge_maxsalary"] == "left_only") & merged["MAX_PERCENTAGE_OF_CAP"].isna()

        def _infer_pct(yos_value):
            try:
                if pd.isna(yos_value):
                    return np.nan
                yos_int = int(yos_value)
                if 0 <= yos_int <= 6:
                    return 0.25
                if 7 <= yos_int <= 9:
                    return 0.30
                if yos_int >= 10:
                    return 0.35
            except Exception:
                pass
            return np.nan

        fallback_vals = merged.loc[mask_fallback, "__join_yos"].apply(_infer_pct)
        merged.loc[mask_fallback, "MAX_PERCENTAGE_OF_CAP"] = fallback_vals
        merged.loc[mask_fallback, "max_pct_source"] = "tier_fallback"
        if debug and mask_fallback.any():
            print(f"[add_max_contract_salary][info] Applied fallback MAX_PERCENTAGE_OF_CAP for {mask_fallback.sum()} rows based on YOS tiers.")

    # === ratio computation ===
    if "AAV" in merged.columns and "MAXIMUM_CONTRACT_SALARY" in merged.columns:
        denom = merged["MAXIMUM_CONTRACT_SALARY"].replace({0: np.nan})
        merged["AAV_PCT_OF_MAX"] = merged["AAV"] / denom
        created.append("AAV_PCT_OF_MAX")

    # === finalize created features ===
    created.extend(["MAXIMUM_CONTRACT_SALARY", "MAX_PERCENTAGE_OF_CAP", "max_pct_source"])

    # === cleanup column collisions / temp keys ===
    # Preserve base YEARS_OF_SERVICE if both exist
    if "YEARS_OF_SERVICE_x" in merged.columns:
        merged.rename(columns={"YEARS_OF_SERVICE_x": "YEARS_OF_SERVICE"}, inplace=True)
    if "YEARS_OF_SERVICE_y" in merged.columns:
        merged.drop(columns=["YEARS_OF_SERVICE_y"], inplace=True)

    # Drop temporary join and right-year column
    merged.drop(columns=["__join_year", "__join_yos", "YEAR"], inplace=True, errors="ignore")

    return merged, created






def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    created = {"numerical": [], "ordinal": [], "nominal": [], "time": []}
    out = df.copy()

    # === Preconditions ===
    required_base_columns = ["SEASON", "PLAYER_ID", "GP", "MP", "USG%", "TOV%", "TS%", "EFG%", "PTS_PER_36", "WS", "YEARS_OF_SERVICE"]
    missing_base = [c for c in required_base_columns if c not in out.columns]
    if missing_base:
        raise ValueError(f"engineer_features: missing required base columns: {missing_base}")

    # === Time feature ===
    out, cols = add_season_start_year(out)
    created["time"].extend(cols)
    print(f"[debug] added time features: {cols}")

    # === Market Tier Feature ===
    out, cols = add_market_tier(out)
    created["nominal"].extend(cols)
    print(f"[debug] added market tier features: {cols}")

    # === Max-Contract join & ratio ===
    try:
        out, cols = add_max_contract_salary(out)  # uses config.MAX_CONTRACT_VALUES_CSV by default
        for c in cols:
            if c == "AAV_PCT_OF_MAX":
                created["numerical"].append(c)
            elif c in ("MAXIMUM_CONTRACT_SALARY", "MAX_PERCENTAGE_OF_CAP"):
                created["numerical"].append(c)
        print(f"[debug] added max-contract features: {cols}")
    except Exception as e:
        print(f"[warn] add_max_contract_salary failed: {e}")

    # === Salary cap normalization (optional) ===
    out, cols = add_salary_pct_cap(out, target="AAV", cap_col="CAP_MAXIMUM")
    created["numerical"].extend(cols)
    if cols:
        print(f"[debug] added salary cap normalization features: {cols}")

    # === Experience bucket ===
    my_bins   = [0, 1, 3, 6, 10, 15, 1000]
    my_labels = ["0","1-2","3-5","6-9","10-14","15+"]
    out, cols, order = add_experience_bucket_percentile(
        out,
        method="fixed",
        fixed_bins=my_bins,
        labels=my_labels
    )
    if not cols:
        raise ValueError("engineer_features: experience bucket feature failed to create (missing YEARS_OF_SERVICE).")
    created["ordinal"].extend(cols)
    print(f"[debug] added experience bucket: {cols} with order {order}")

    # === Team cap space features ===
    out, caps = add_team_cap_space(out,
                                   salary_commitment_col="AAV",
                                   cap_col="CAP_MAXIMUM",
                                   tax_threshold_col="TAX_THRESHOLD",
                                   team_key="TEAM_ABBREVIATION",
                                   season_year_col="season_start_year")
    created["numerical"].extend(caps)
    print(f"[debug] added team cap space features: {caps}")

    # === Cap space bins & labels ===
    kb = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    out["cap_space_bin"] = kb.fit_transform(out[["team_cap_space"]]).astype(int)
    created["ordinal"].append("cap_space_bin")
    tier_labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    out["cap_space_tier"] = out["cap_space_bin"].map(dict(enumerate(tier_labels)))
    created["nominal"].append("cap_space_tier")
    print("[debug] added cap space bins/tiers")

    # === Rolling features ===
    rolling_stats = ["WS", "PTS_PER_36"]
    out, cols = add_rolling_features(out, stats=rolling_stats)
    created["numerical"].extend(cols)
    print(f"[debug] added rolling features: {cols}")

    # === Portability composite ===
    if "TS%" not in out.columns or "EFG%" not in out.columns:
        raise ValueError("engineer_features: cannot compute portability - missing 'TS%' or 'EFG%'")
    def pct_to_decimal(v):
        return v / 100.0 if v.median(skipna=True) > 1.5 else v
    out["TS%"] = pct_to_decimal(out["TS%"])
    out["EFG%"] = pct_to_decimal(out["EFG%"])
    if "USG%" not in out.columns:
        raise ValueError("engineer_features: cannot compute portability - USG% missing after add_usage_and_true_usage")
    out, cols = compute_portability(out)
    if not cols:
        raise ValueError("engineer_features: portability_score was not created.")
    created["numerical"].extend(cols)
    print(f"[debug] added portability features: {cols}")

    # === Contract-Year flag — if available ===
    if "END_CONTRACT_YEAR" in out.columns:
        out, cols = add_contract_year_flag(out, contract_end_col="END_CONTRACT_YEAR", season_col="SEASON")
        created["nominal"].extend(cols)
        print(f"[debug] added contract year flag: {cols}")

    # === Availability trend — if available ===
    out, cols = add_multi_season_availability_trend(out, window=3)
    if cols:
        created["numerical"].extend(cols)
        print(f"[debug] added availability trend: {cols}")

    # Diagnostics before injury reliability features (use current out, not original df)
    print("===============missingness before injury reliability features")
    print(diagnose_injury_date_completeness(out, "INJURY_START_DATE", "INJURY_END_DATE"))

    # More transparent check: how TOTAL_DAYS_INJURED was populated
    if {"INJURY_START_DATE", "INJURY_END_DATE", "TOTAL_DAYS_INJURED"}.issubset(set(out.columns)):
        start = pd.to_datetime(out["INJURY_START_DATE"], errors="coerce")
        end = pd.to_datetime(out["INJURY_END_DATE"], errors="coerce")
        both_missing_mask = start.isna() & end.isna()
        valid_stint_mask = start.notna() & end.notna() & (end >= start)
        # sanity: both_missing should be zero days
        zero_from_both_missing = out.loc[both_missing_mask, "TOTAL_DAYS_INJURED"] == 0
        print(f"[post-injury-debug] both_missing count={both_missing_mask.sum()}, zeros assigned={zero_from_both_missing.sum()}")
        # valid stints differences
        expected_diff = (end - start).dt.days
        mismatched_valid = valid_stint_mask & (out["TOTAL_DAYS_INJURED"] != expected_diff)
        if mismatched_valid.any():
            print(f"[post-injury-debug][ERROR] {mismatched_valid.sum()} valid_stint rows have inconsistent TOTAL_DAYS_INJURED (expected end-start). Sample:")
            print(out.loc[mismatched_valid, ["INJURY_START_DATE", "INJURY_END_DATE", "TOTAL_DAYS_INJURED"]].head(5))
        else:
            print(f"[post-injury-debug] All valid_stint rows have TOTAL_DAYS_INJURED matching end-start difference.")


    # Diagnostics after injury reliability features
    print("===============missingness after injury reliability features")
    print(diagnose_injury_date_completeness(out, "INJURY_START_DATE", "INJURY_END_DATE"))
    # Show a few rows to inspect that TOTAL_DAYS_INJURED aligns
    if {"INJURY_START_DATE", "INJURY_END_DATE", "TOTAL_DAYS_INJURED"}.issubset(set(out.columns)):
        print(out.loc[:, ["INJURY_START_DATE", "INJURY_END_DATE", "TOTAL_DAYS_INJURED"]].head(10))



    # === Draft pedigree — if available ===
    if {"DRAFT_YEAR", "DRAFT_PICK", "BIRTHDATE"}.issubset(set(out.columns)):
        out, cols = add_draft_features(out)
        if cols:
            created["numerical"].extend([c for c in cols if "age" in c])
            created["nominal"].extend([c for c in cols if "bucket" in c])
            print(f"[debug] added draft features: {cols}")

    # === Final validation ===
    if not created["time"]:
        raise RuntimeError("engineer_features: no time features created.")
    if not created["ordinal"]:
        raise RuntimeError("engineer_features: no ordinal features created.")
    if "portability_score" not in created["numerical"]:
        raise RuntimeError("engineer_features: portability_score missing from numerical features.")

    print("Smoke test OK - Engineered features:")
    for category, features in created.items():
        if features:
            print(f"  {category}: {features}")


    return out, created






if __name__ == "__main__":
    from api.src.ml.features.load_data_utils import load_data_optimized
    from api.src.ml import config
    from api.src.ml.column_schema import report_schema_dtype_violations, load_schema_from_yaml


    # Example call: drop rows where player_id or season is null

    FINAL_DATA_PATH = config.FINAL_ENGINEERED_DATASET_DIR / 'final_merged_with_all.parquet'


    # TEST_DATA_PATH = 'api/src//data/merged_final_dataset/nba_player_data_final_inflated.parquet'
    # Example call: drop rows where player_id or season is null
    df = load_data_optimized(
        FINAL_DATA_PATH,
        debug=False,
        # drop_null_rows=True,
        # drop_null_subset=['AAV'],
        # use_sample=True,
        # sample_size=10000
    )
    df_eng, summary = engineer_features(df)
    print(df_eng.dtypes)
    print("================")
    print(df.columns.tolist())
    print(df.head())

    from api.src.ml.column_schema import report_schema_dtype_violations, load_schema_from_yaml

    print(f"config: {config}")

    try:
        schema_path = config.COLUMN_SCHEMA_PATH
    except Exception as e:
        print(f"Failed to locate schema YAML: {e}")
        raise

    print(f"[SMOKE TEST] Loading schema from: {schema_path}")
    try:
        schema = load_schema_from_yaml(str(schema_path))
    except Exception as e:
        print(f"Failed to load schema YAML: {e}")
        raise
    _ = report_schema_dtype_violations(df_eng, schema, max_show=50)
