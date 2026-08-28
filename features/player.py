"""
Rolling player features for shot-on-goal (SOG) prediction.

Two modes:
  Historical (training):
    Load player_game_stats.parquet → compute rolling SOG/xG rates per player.
    Merge with opponent team defensive quality from the team feature matrix.

  Live (inference):
    Fetch current-season game logs from NHL API → compute rolling rates.
    Merge with today's opponent defensive snapshot.

Rolling windows: last 10 and 20 games (5 is too noisy for individual players).
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ingestion.player_stats import (
    load_player_game_stats,
    fetch_player_game_log,
    game_log_to_dataframe,
    toi_to_seconds,
)
from config.season import current_season_api

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"

PLAYER_WINDOWS = [10, 20]

# Minimum games played to produce a reliable estimate
MIN_GAMES = 5

# Per-player stats to roll
PLAYER_ROLL_STATS = ["sog", "xg", "shot_attempts", "xg_per_attempt"]

# Team-level defensive stats to merge in as opponent context
_OPP_TEAM_COLS = [
    "sf_pct_l20",       # team shot share against (how many shots opponent allows)
    "xg_against_l20",   # xG allowed (shot quality context)
    "hd_chances_against_l20",  # high danger chances allowed
]


# ---------------------------------------------------------------------------
# Rolling feature computation
# ---------------------------------------------------------------------------

def _rolling_player(grp: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling means within a single player's history (no season boundary reset)."""
    grp = grp.copy()
    for col in PLAYER_ROLL_STATS:
        if col not in grp.columns:
            continue
        shifted = grp[col].shift(1)  # exclude current game (pre-game snapshot)
        for w in PLAYER_WINDOWS:
            grp[f"{col}_l{w}"] = shifted.rolling(w, min_periods=1).mean()
    grp["games_played"] = np.arange(len(grp))
    return grp


def build_player_rolling_features(
    player_game_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute rolling pre-game features for all players in the historical dataset.

    Returns one row per (game_id, player_id) with rolling stat columns.
    Suitable for merging onto a training feature matrix.
    """
    if player_game_stats is None:
        player_game_stats = load_player_game_stats()

    df = player_game_stats.copy()
    # Sort: chronological within each player (game_num = last 4 digits of game_id)
    df["game_num"] = df["game_id"].str[-4:].astype(int)
    df = df.sort_values(["player_id", "season", "game_num"])

    parts = []
    for player_id, grp in df.groupby("player_id", sort=False):
        parts.append(_rolling_player(grp))

    result = pd.concat(parts, ignore_index=True)
    roll_cols = [c for c in result.columns if any(f"_l{w}" in c for w in PLAYER_WINDOWS)]
    logger.info(
        "Player rolling features: %d rows, %d players, %d feature cols",
        len(result), result["player_id"].nunique(), len(roll_cols),
    )
    return result


# ---------------------------------------------------------------------------
# Live snapshot: rolling stats from current-season API game logs
# ---------------------------------------------------------------------------

def _parse_toi_log(game_log: list[dict], player_id: int, player_name: str) -> pd.DataFrame:
    """Parse NHL API game log into a rolling-ready DataFrame including TOI."""
    rows = []
    for g in game_log:
        toi_sec = toi_to_seconds(g.get("toi", "0:00"))
        rows.append({
            "game_id":     str(g["gameId"]),
            "player_id":   player_id,
            "player_name": player_name,
            "team":        g.get("teamAbbrev", ""),
            "sog":         int(g.get("shots", 0)),
            "goals":       int(g.get("goals", 0)),
            "assists":     int(g.get("assists", 0)),
            "toi_seconds": toi_sec,
            "pp_points":   int(g.get("powerPlayPoints", 0)),
            # xg / shot_attempts not available from NHL API — filled as NaN
            "xg":              np.nan,
            "shot_attempts":   np.nan,
            "xg_per_attempt":  np.nan,
        })
    return pd.DataFrame(rows)


def build_live_player_snapshot(
    players: list[dict],
    season: str | None = None,
    max_window: int = 20,
) -> pd.DataFrame:
    """
    Fetch current-season game logs from NHL API for a list of players and
    compute rolling SOG/TOI snapshot for use in today's predictions.

    Args:
        players: list of dicts with keys: id, name, team, position
        season:  NHL API season string (e.g. "20252026"); defaults to current
        max_window: how many recent games to look back

    Returns:
        DataFrame indexed by player_id with rolling stat columns.
    """
    season = season or current_season_api()

    rows = []
    for p in players:
        if p.get("position") == "G":
            continue  # skip goalies

        log = fetch_player_game_log(p["id"], season)
        if not log:
            logger.debug("No game log for %s (%d)", p["name"], p["id"])
            continue

        df = _parse_toi_log(log, p["id"], p["name"])
        if len(df) < MIN_GAMES:
            logger.debug("Skipping %s — only %d games", p["name"], len(df))
            continue

        recent = df.tail(max_window)
        row: dict = {
            "player_id":   p["id"],
            "player_name": p["name"],
            "team":        p.get("team", df["team"].iloc[-1] if len(df) else ""),
            "position":    p.get("position", ""),
            "games_played": len(df),
        }
        for col in ["sog", "toi_seconds", "xg"]:
            if col in recent.columns:
                for w in PLAYER_WINDOWS:
                    vals = recent.tail(w)[col].dropna()
                    row[f"{col}_l{w}"] = float(vals.mean()) if len(vals) > 0 else np.nan
        rows.append(row)

    return pd.DataFrame(rows).set_index("player_id") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Merge opponent defensive context
# ---------------------------------------------------------------------------

def merge_opponent_context(
    player_df: pd.DataFrame,
    team_snapshot: pd.DataFrame,
    game_home_away: dict[str, tuple[str, str]],
) -> pd.DataFrame:
    """
    Attach opponent defensive quality columns to a player DataFrame.

    Args:
        player_df:     player-level feature rows (must have 'team' column)
        team_snapshot: team rolling stats snapshot (index = team abbrev)
                       from pipeline.live._build_team_snapshot()
        game_home_away: {home_team: (home_team, away_team)} lookup for today's games

    Returns:
        player_df with opp_* columns added.
    """
    # Build team → opponent mapping for today
    team_to_opp: dict[str, str] = {}
    for home, away in game_home_away.values():
        team_to_opp[home] = away
        team_to_opp[away] = home

    opp_rows = []
    for _, row in player_df.iterrows():
        team = row.get("team", "")
        opp = team_to_opp.get(team, "")
        opp_row: dict = {}
        if opp and opp in team_snapshot.index:
            opp_stats = team_snapshot.loc[opp]
            for col in _OPP_TEAM_COLS:
                if col in opp_stats.index:
                    opp_row[f"opp_{col}"] = opp_stats[col]
        opp_rows.append(opp_row)

    opp_df = pd.DataFrame(opp_rows, index=player_df.index)
    return pd.concat([player_df, opp_df], axis=1)
