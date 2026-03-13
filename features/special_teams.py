"""
Special teams (PP/PK) features from MoneyPuck shot-level CSVs.

Classifies each shot by manpower situation (PP, PK, 5v5, other)
using homeSkatersOnIce / awaySkatersOnIce columns, then aggregates
to per-game team stats and computes rolling averages.

Output: one row per (game_id, team) with rolling PP/PK features.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.moneypuck import mp_game_id_to_nhl, MP_SEASONS

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"

ST_WINDOWS = [10, 20]  # Longer windows — PP/PK events are sparser per game

_USECOLS = [
    "game_id", "isPlayoffGame", "teamCode", "homeTeamCode", "awayTeamCode",
    "isHomeTeam", "homeSkatersOnIce", "awaySkatersOnIce",
    "goal", "xGoal", "shotWasOnGoal",
]


def _load_raw_shots(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load regular-season shots with skater count columns."""
    if raw_dir is None:
        raw_dir = RAW_DIR

    frames = []
    for season, year in MP_SEASONS.items():
        path = raw_dir / f"moneypuck_shots_{year}.csv"
        if not path.exists():
            logger.warning("Missing %s — skipping season %s", path.name, season)
            continue

        df = pd.read_csv(path, usecols=_USECOLS, low_memory=False)
        df["season"] = season
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    shots = pd.concat(frames, ignore_index=True)
    shots = shots[shots["isPlayoffGame"] == 0].copy()
    return shots


def _classify_situation(row_is_home: pd.Series, home_sk: pd.Series, away_sk: pd.Series) -> pd.Series:
    """
    Classify the manpower situation for the SHOOTING team.

    Returns a Series with values: 'PP', 'PK', '5v5', 'other'.
    """
    # Shooting team's skaters vs defending team's skaters
    shooting_sk = np.where(row_is_home, home_sk, away_sk)
    defending_sk = np.where(row_is_home, away_sk, home_sk)

    situation = pd.Series("other", index=row_is_home.index)
    situation[(shooting_sk == 5) & (defending_sk == 5)] = "5v5"
    situation[shooting_sk > defending_sk] = "PP"
    situation[shooting_sk < defending_sk] = "PK"

    return situation


def aggregate_special_teams_stats(raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Aggregate PP/PK stats per (game_id, team).

    PP stats = when this team was shooting on the power play.
    PK stats = when the opponent was shooting on their power play
               (i.e., this team was defending while shorthanded).

    Returns DataFrame with columns:
        game_id, team, season, game_num,
        pp_goals, pp_xg, pp_shots, pk_goals_against, pk_xg_against, pk_shots_against
    """
    shots = _load_raw_shots(raw_dir)
    if shots.empty:
        return pd.DataFrame()

    shots["situation"] = _classify_situation(
        shots["isHomeTeam"].astype(bool),
        shots["homeSkatersOnIce"],
        shots["awaySkatersOnIce"],
    )

    # --- PP offense stats (shooting team on PP) ---
    pp_shots = shots[shots["situation"] == "PP"]
    pp_agg = (
        pp_shots.groupby(["game_id", "season", "teamCode"])
        .agg(
            pp_goals=("goal", "sum"),
            pp_xg=("xGoal", "sum"),
            pp_shots=("shotWasOnGoal", "sum"),
        )
        .reset_index()
        .rename(columns={"teamCode": "team"})
    )

    # --- PK defense stats (opponent shooting on PP = this team on PK) ---
    # When teamCode is shooting on PP, the defending team is on PK
    pk_shots = shots[shots["situation"] == "PP"].copy()
    # Defending team = opposite of shooting team
    pk_shots["pk_team"] = np.where(
        pk_shots["isHomeTeam"] == 1,
        pk_shots["awayTeamCode"],   # home is shooting PP → away is on PK
        pk_shots["homeTeamCode"],   # away is shooting PP → home is on PK
    )
    pk_agg = (
        pk_shots.groupby(["game_id", "season", "pk_team"])
        .agg(
            pk_goals_against=("goal", "sum"),
            pk_xg_against=("xGoal", "sum"),
            pk_shots_against=("shotWasOnGoal", "sum"),
        )
        .reset_index()
        .rename(columns={"pk_team": "team"})
    )

    # --- Get all (game_id, team) combinations for complete coverage ---
    all_teams = (
        shots[["game_id", "season", "teamCode"]]
        .drop_duplicates()
        .rename(columns={"teamCode": "team"})
    )

    # Merge PP and PK onto complete game/team index (fill missing with 0)
    result = all_teams.merge(pp_agg, on=["game_id", "season", "team"], how="left")
    result = result.merge(pk_agg, on=["game_id", "season", "team"], how="left")

    fill_cols = ["pp_goals", "pp_xg", "pp_shots",
                 "pk_goals_against", "pk_xg_against", "pk_shots_against"]
    result[fill_cols] = result[fill_cols].fillna(0)

    # Convert game_id to NHL format
    result["game_id"] = result.apply(
        lambda r: mp_game_id_to_nhl(str(int(float(r["game_id"]))), r["season"]),
        axis=1,
    )

    # Game number for sorting
    result["game_num"] = result["game_id"].str[-4:].astype(int)

    logger.info(
        "Special teams stats: %d team-game rows across %d games",
        len(result), result["game_id"].nunique(),
    )
    return result


def compute_special_teams_rolling(
    st_games: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Rolling PP/PK rates within season boundaries.
    Uses shift(1) for pre-game features (no leakage).
    """
    if windows is None:
        windows = ST_WINDOWS

    roll_cols = ["pp_goals", "pp_xg", "pp_shots",
                 "pk_goals_against", "pk_xg_against", "pk_shots_against"]

    st_games = st_games.sort_values(["team", "season", "game_num"])

    parts = []
    for (team, season), grp in st_games.groupby(["team", "season"], sort=False):
        grp = grp.copy()
        for col in roll_cols:
            shifted = grp[col].shift(1)
            for w in windows:
                grp[f"{col}_l{w}"] = shifted.rolling(w, min_periods=1).mean()
        parts.append(grp)

    result = pd.concat(parts, ignore_index=True)

    # Keep only game_id, team, and rolling columns
    keep_cols = ["game_id", "team"]
    for col in roll_cols:
        for w in windows:
            keep_cols.append(f"{col}_l{w}")

    result = result[keep_cols]

    logger.info(
        "Special teams rolling features: %d rows, %d feature columns",
        len(result), len(keep_cols) - 2,
    )
    return result


def load_special_teams_features(raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Top-level: aggregate + rolling.

    Returns DataFrame keyed by (game_id, team) with rolling PP/PK columns,
    ready for merging in pipeline/backfill.py.
    """
    st_games = aggregate_special_teams_stats(raw_dir)
    if st_games.empty:
        return pd.DataFrame()

    rolling = compute_special_teams_rolling(st_games)
    return rolling


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    feats = load_special_teams_features()
    if feats.empty:
        print("No special teams features generated.")
    else:
        print(f"\nSpecial teams features shape: {feats.shape}")
        print(f"Columns: {list(feats.columns)}")
        print(f"\nSample (first 10 rows):")
        print(feats.head(10))
        print(f"\nNaN rates:")
        print(feats.isnull().mean().round(4))
        print(f"\nPP goals l20 stats:")
        print(feats["pp_goals_l20"].describe().round(4))
