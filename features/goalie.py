"""
Goalie rolling features: save%, GAA, and rest days.

Reads from the goalie_stats + games tables (populated by ingestion/nhl_api.py).
If Postgres is not available or the table is empty, returns an empty DataFrame
with the expected schema so the rest of the pipeline continues gracefully.

GSAx (Goals Saved Above Expected) requires per-shot xG data at the goalie level.
This is not stored in goalie_stats yet — that column is left NULL until the
MoneyPuck shot-level reprocessing is added in a future pass.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GOALIE_WINDOWS: list[int] = [5, 10]

# Columns in the final output (excluding game_id, team)
ROLL_GOALIE_COLS = [f"{s}_l{w}" for s in ["save_pct", "gaa"] for w in GOALIE_WINDOWS]


def load_goalie_features(conn=None) -> pd.DataFrame:
    """
    Compute rolling goalie features from the goalie_stats + games tables.

    Returns a DataFrame keyed by (game_id, team) representing the starting
    goalie's rolling stats *entering* that game.

    Args:
        conn: psycopg2 connection, or None to skip (returns empty DataFrame).
    """
    if conn is None:
        logger.warning("No DB connection — skipping goalie features")
        return _empty_goalie_df()

    try:
        sql = """
            SELECT
                gs.game_id,
                gs.goalie_id,
                gs.goalie_name,
                gs.team,
                gs.is_starter,
                gs.shots_against,
                gs.saves,
                gs.goals_against,
                gs.save_pct,
                gs.toi_seconds,
                g.date
            FROM goalie_stats gs
            JOIN games g USING (game_id)
            WHERE g.date IS NOT NULL
            ORDER BY gs.team, g.date, gs.game_id
        """
        df = pd.read_sql(sql, conn)
    except Exception as e:
        logger.warning("Could not query goalie_stats: %s — skipping", e)
        return _empty_goalie_df()

    if df.empty:
        logger.warning("goalie_stats table is empty — skipping goalie features")
        return _empty_goalie_df()

    df["date"] = pd.to_datetime(df["date"])
    df["save_pct"] = pd.to_numeric(df["save_pct"], errors="coerce")
    # GAA = goals against per 60 minutes
    toi_hours = (df["toi_seconds"] / 3600).replace(0, np.nan)
    df["gaa"] = df["goals_against"] / toi_hours

    # Per-goalie rolling (within their career — no season boundary needed for goalies)
    goalie_parts = []
    for goalie_id, grp in df.groupby("goalie_id"):
        grp = grp.copy().sort_values("date")
        for col in ["save_pct", "gaa"]:
            shifted = grp[col].shift(1)
            for w in GOALIE_WINDOWS:
                grp[f"{col}_l{w}"] = shifted.rolling(w, min_periods=1).mean()
        # Days since last appearance (clipped at 30 — start-of-season reset)
        grp["days_rest"] = grp["date"].diff().dt.days.clip(upper=30)
        goalie_parts.append(grp)

    goalies = pd.concat(goalie_parts, ignore_index=True)

    # Keep only the starter for each (game_id, team)
    starters = goalies[goalies["is_starter"] == True].copy()  # noqa: E712
    # If two goalies are flagged as starter (edge case), keep highest TOI
    starters = (
        starters
        .sort_values("toi_seconds", ascending=False)
        .drop_duplicates(subset=["game_id", "team"])
    )

    keep = ["game_id", "team", "goalie_id", "goalie_name", "days_rest"] + ROLL_GOALIE_COLS
    existing = [c for c in keep if c in starters.columns]
    result = starters[existing].reset_index(drop=True)

    logger.info(
        "Goalie features ready: %d game-team rows, %d games",
        len(result), result["game_id"].nunique(),
    )
    return result


def _empty_goalie_df() -> pd.DataFrame:
    cols = ["game_id", "team", "goalie_id", "goalie_name", "days_rest"] + ROLL_GOALIE_COLS
    return pd.DataFrame(columns=cols)
