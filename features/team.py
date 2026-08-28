"""
Rolling team features computed from MoneyPuck per-game stats.

Each row in the output represents one team's *pre-game* feature snapshot —
i.e., rolling averages of past games (current game excluded via shift(1)).
Rolling is bounded by season: no stats bleed across season boundaries.

Output columns include suffix _l5, _l10, _l20 for each lookback window,
plus _ewm7 for exponentially weighted means (halflife=7 games).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"

WINDOWS: list[int] = [5, 10, 20]
EWM_HALFLIFE: int = 7  # exponential decay halflife in games

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
    "regulation_win",
]

# Subset of stats worth computing EWM and home/away splits for
# (full list would explode feature count — focus on highest-signal stats)
EWM_STATS = [
    "xgf_pct", "xgf_pct_5v5", "cf_pct", "sf_pct",
    "goals_for", "goals_against", "goal_diff", "won",
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

    # regulation_win: 1.0 if won in regulation (no OT/SO).
    # Populated later by _add_regulation_wins() from shot-level period data.
    # Default to NaN — filled when raw shots are available.
    if "regulation_win" not in df.columns:
        df["regulation_win"] = np.nan

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

    # EWM (exponentially weighted mean) — recent games weighted more heavily
    for col in EWM_STATS:
        if col not in grp.columns:
            continue
        shifted = grp[col].shift(1)
        grp[f"{col}_ewm{EWM_HALFLIFE}"] = shifted.ewm(
            halflife=EWM_HALFLIFE, min_periods=1
        ).mean()

    # Home/away split rolling: stats computed only from games at the same venue
    # This captures teams that play very differently home vs away
    if "is_home" in grp.columns:
        for venue, label in [(True, "home"), (False, "away")]:
            venue_mask = grp["is_home"] == venue
            for col in EWM_STATS:
                if col not in grp.columns:
                    continue
                vals = grp[col].where(venue_mask, np.nan).shift(1)
                grp[f"{col}_{label}_l10"] = vals.rolling(10, min_periods=1).mean()

    # Track games played this season (useful for confidence / early-season flagging)
    grp["games_played"] = np.arange(len(grp))  # 0-indexed, first game = 0
    return grp


def rolling_feature_columns(df: pd.DataFrame) -> list[str]:
    """Column names produced by :func:`_rolling_team_season`.

    Single definition shared by the backfill assembler and the live pipeline,
    so the two can never disagree about which columns are model features.
    """
    ewm_suffix = f"_ewm{EWM_HALFLIFE}"
    return [
        c for c in df.columns
        if any(c.endswith(f"_l{w}") for w in WINDOWS)
        or c.endswith(ewm_suffix)
        or c.endswith("_home_l10")
        or c.endswith("_away_l10")
        or c == "games_played"
    ]


def pregame_snapshot(
    history: pd.DataFrame,
    teams: list[str] | None = None,
) -> pd.DataFrame:
    """Pre-game rolling features for each team's *next* game.

    Appends a placeholder row per team to that team's game history and runs
    the identical :func:`_rolling_team_season` used to build the training
    matrix, then returns the placeholder's rolling values.  Because the
    training and serving features come out of the same function, they cannot
    drift apart — previously the live pipeline re-implemented the rolling
    logic and disagreed with training on season boundaries and on the venue
    split window.

    Args:
        history: per-game rows for a *single season*, with the raw stat
                 columns, ``team``, ``is_home`` and ``game_num``.
        teams:   teams that must appear in the result.  Teams with no games
                 yet get an all-NaN row instead of being dropped, which is
                 what training saw for the opening night of every season.

    Returns:
        DataFrame indexed by team, one column per rolling feature.
    """
    wanted = sorted(set(teams or []) | set(history["team"].unique()))

    next_game_num = int(history["game_num"].max()) + 1 if len(history) else 1
    stat_cols = [c for c in set(ROLL_STATS) | set(EWM_STATS) if c in history.columns]

    rows = []
    for team in wanted:
        grp = history[history["team"] == team].sort_values("game_num")

        placeholder = {c: np.nan for c in stat_cols}
        placeholder.update({"team": team, "is_home": True, "game_num": next_game_num})
        with_placeholder = pd.concat(
            [grp, pd.DataFrame([placeholder])], ignore_index=True,
        )

        rolled = _rolling_team_season(with_placeholder)
        feature_cols = rolling_feature_columns(rolled)
        row = rolled.iloc[-1][feature_cols].to_dict()
        row["team"] = team
        rows.append(row)

    return pd.DataFrame(rows).set_index("team")


def _add_regulation_wins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Determine which games went to OT/SO from raw MoneyPuck shot CSVs.
    Sets regulation_win = 1 if team won AND game ended in regulation (max period <= 3),
    and went_to_ot = 1 for any game that went past the third period.

    ``went_to_ot`` is a post-game fact, so it is never rolled into a feature —
    the ELO updater uses it to discount coin-flip OT/SO results.
    """
    from ingestion.moneypuck import MP_SEASONS, mp_game_id_to_nhl

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    ot_games = set()

    for season, year in MP_SEASONS.items():
        path = raw_dir / f"moneypuck_shots_{year}.csv"
        if not path.exists():
            continue
        shots = pd.read_csv(
            path, usecols=["game_id", "period", "isPlayoffGame"], low_memory=False,
        )
        shots = shots[shots["isPlayoffGame"] == 0]
        max_period = shots.groupby("game_id")["period"].max()
        ot_mp_ids = max_period[max_period > 3].index

        for mp_id in ot_mp_ids:
            nhl_id = mp_game_id_to_nhl(str(int(float(mp_id))), season)
            ot_games.add(nhl_id)

    logger.info("Identified %d OT/SO games across all seasons", len(ot_games))

    went_to_ot = df["game_id"].isin(ot_games)
    df["went_to_ot"] = went_to_ot.astype(float)
    df["regulation_win"] = np.where(
        df["won"] == 1.0,
        np.where(went_to_ot, 0.0, 1.0),
        0.0,
    )
    return df


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
    df = _add_regulation_wins(df)

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
