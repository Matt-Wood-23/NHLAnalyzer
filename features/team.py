"""
Rolling team features computed from MoneyPuck per-game stats.

Each row in the output represents one team's *pre-game* feature snapshot —
i.e., rolling averages of past games (current game excluded via shift(1)).
Rolling is bounded by season: no stats bleed across season boundaries.

Output columns include suffix _l5, _l10, _l20 for each lookback window.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"

WINDOWS: list[int] = [5, 10, 20]

# Raw per-game stats to roll
ROLL_STATS = [
    "xg_for",
    "xg_against",
    "xg_for_5v5",
    "xg_against_5v5",
    "xgf_pct",
    "xgf_pct_5v5",
    "cf_pct",
    "sf_pct",
    "goals_for",
    "goals_against",
    "goal_diff",
    "hd_chances_for",
    "hd_chances_against",
    "won",
]


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio metrics and win flag to the raw per-game data."""
    # xg_against_5v5 = the opponent's xg_for_5v5 in the same game
    xg5v5_lookup = (
        df[["game_id", "team", "xg_for_5v5"]]
        .rename(columns={"team": "opp_team", "xg_for_5v5": "xg_against_5v5"})
    )
    df = df.merge(xg5v5_lookup, on=["game_id", "opp_team"], how="left")

    eps = 1e-9  # avoid division by zero
    df["xgf_pct"]     = df["xg_for"]     / (df["xg_for"]     + df["xg_against"]     + eps)
    df["xgf_pct_5v5"] = df["xg_for_5v5"] / (df["xg_for_5v5"] + df["xg_against_5v5"] + eps)
    df["cf_pct"]      = df["corsi_for"]   / (df["corsi_for"]  + df["corsi_against"]  + eps)
    df["sf_pct"]      = df["shots_for"]   / (df["shots_for"]  + df["shots_against"]  + eps)
    df["goal_diff"]   = df["goals_for"]   - df["goals_against"]
    # won: 1.0 if this team won, 0.0 otherwise
    df["won"]         = (df["goals_for"] > df["goals_against"]).astype(float)

    return df


def _rolling_team_season(grp: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling means within a single (team, season) group.
    shift(1) ensures the current game's stats are NOT included in the window
    (strict pre-game features — no leakage).
    min_periods=1 fills partial windows at season start rather than producing NaN.
    """
    grp = grp.copy()
    for col in ROLL_STATS:
        if col not in grp.columns:
            continue
        shifted = grp[col].shift(1)
        for w in WINDOWS:
            grp[f"{col}_l{w}"] = shifted.rolling(w, min_periods=1).mean()
    # Track games played this season (useful for confidence / early-season flagging)
    grp["games_played"] = np.arange(len(grp))  # 0-indexed, first game = 0
    return grp


def load_team_features(parquet_path: Path | str | None = None) -> pd.DataFrame:
    """
    Load MoneyPuck per-game stats and compute rolling pre-game team features.

    Returns a DataFrame with one row per (game_id, team).  Rolling columns
    have suffix _l5, _l10, _l20.  Raw per-game stats are also retained so
    the assembler can extract the target (home_win) from home-team rows.

    Args:
        parquet_path: path to moneypuck_team_game_stats.parquet.
                      Defaults to data/parquet/moneypuck_team_game_stats.parquet.
    """
    if parquet_path is None:
        parquet_path = PARQUET_DIR / "moneypuck_team_game_stats.parquet"

    logger.info("Loading MoneyPuck parquet: %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d team-game rows", len(df))

    df = _add_derived_columns(df)

    # Sort key: last 4 digits of NHL game_id = sequential game number within season
    df["game_num"] = df["game_id"].str[-4:].astype(int)
    df = df.sort_values(["team", "season", "game_num"])

    logger.info("Computing rolling features (windows=%s) ...", WINDOWS)
    parts = []
    for (team, season), grp in df.groupby(["team", "season"], sort=False):
        parts.append(_rolling_team_season(grp))

    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(["season", "game_id", "team"]).reset_index(drop=True)

    n_roll = len([c for c in result.columns if any(f"_l{w}" in c for w in WINDOWS)])
    logger.info(
        "Team features ready: %d rows, %d rolling feature columns",
        len(result), n_roll,
    )
    return result
