"""
Goalie features computed from MoneyPuck shot-level CSVs.

Bypasses the Postgres-dependent goalie.py by extracting goalie stats
directly from the raw shot data already cached in data/raw/.

Output: one row per (game_id, team) for the starting goalie, with
rolling pre-game features (save_pct, gsax) over windows [5, 10].
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ingestion.moneypuck import mp_game_id_to_nhl, MP_SEASONS

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"

GOALIE_WINDOWS = [5, 10]

# Only load columns we need (saves memory on 300+ MB CSVs)
_USECOLS = [
    "game_id", "isPlayoffGame", "goalieIdForShot", "goalieNameForShot",
    "teamCode", "homeTeamCode", "awayTeamCode", "isHomeTeam",
    "shotWasOnGoal", "goal", "xGoal",
]


def _load_raw_shots(raw_dir: Path | None = None) -> pd.DataFrame:
    """Load all regular-season shots from cached MoneyPuck CSVs."""
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
        df["_year"] = year
        frames.append(df)

    if not frames:
        logger.warning("No MoneyPuck shot CSVs found in %s", raw_dir)
        return pd.DataFrame()

    shots = pd.concat(frames, ignore_index=True)

    # Filter to regular season only
    shots = shots[shots["isPlayoffGame"] == 0].copy()

    # Drop shots with no goalie (empty net, etc.)
    shots = shots.dropna(subset=["goalieIdForShot"])

    return shots


def aggregate_goalie_game_stats(raw_dir: Path | None = None) -> pd.DataFrame:
    """
    Aggregate shot data to one row per (game_id, goalie_id).

    The goalie is on the DEFENDING team — opposite of the shooting team.

    Returns DataFrame with columns:
        game_id, goalie_id, goalie_name, team, season,
        shots_against, goals_against, saves, save_pct,
        xg_against, gsax, is_starter
    """
    shots = _load_raw_shots(raw_dir)
    if shots.empty:
        return pd.DataFrame()

    # Goalie's team = the defending team (opposite of the shooter)
    shots["goalie_team"] = np.where(
        shots["isHomeTeam"] == 1,
        shots["awayTeamCode"],   # home team is shooting → goalie is away
        shots["homeTeamCode"],   # away team is shooting → goalie is home
    )

    # Aggregate per (game_id, goalie)
    agg = (
        shots.groupby(["game_id", "season", "_year", "goalieIdForShot",
                        "goalieNameForShot", "goalie_team"])
        .agg(
            shots_against=("shotWasOnGoal", "sum"),
            goals_against=("goal", "sum"),
            xg_against=("xGoal", "sum"),
        )
        .reset_index()
        .rename(columns={
            "goalieIdForShot": "goalie_id",
            "goalieNameForShot": "goalie_name",
            "goalie_team": "team",
        })
    )

    agg["saves"] = agg["shots_against"] - agg["goals_against"]
    agg["save_pct"] = agg["saves"] / agg["shots_against"].clip(lower=1)
    agg["gsax"] = agg["xg_against"] - agg["goals_against"]  # positive = saved more than expected

    # Identify starter: goalie with most shots_against per (game_id, team)
    # AND must have faced > 50% of team's total shots
    team_total = (
        agg.groupby(["game_id", "team"])["shots_against"]
        .transform("sum")
    )
    agg["shot_share"] = agg["shots_against"] / team_total.clip(lower=1)
    agg["is_starter"] = (
        agg.groupby(["game_id", "team"])["shots_against"]
        .transform("max") == agg["shots_against"]
    ) & (agg["shot_share"] > 0.5)

    # Convert MoneyPuck game_id to NHL format
    agg["game_id"] = agg.apply(
        lambda r: mp_game_id_to_nhl(str(int(float(r["game_id"]))), r["season"]),
        axis=1,
    )

    # Game number for sorting (last 4 digits)
    agg["game_num"] = agg["game_id"].str[-4:].astype(int)

    logger.info(
        "Goalie game stats: %d rows (%d starters) across %d games",
        len(agg), agg["is_starter"].sum(),
        agg["game_id"].nunique(),
    )

    return agg.drop(columns=["_year"])


def compute_goalie_rolling_features(
    goalie_games: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute per-goalie rolling stats. Pre-game features via shift(1).

    Goalie form carries across seasons (no season boundary reset).
    Filtered to starters only.

    Returns one row per (game_id, team) with rolling columns.
    """
    if windows is None:
        windows = GOALIE_WINDOWS

    # Only compute rolling for starters
    starters = goalie_games[goalie_games["is_starter"]].copy()
    if starters.empty:
        return pd.DataFrame()

    # Sort chronologically per goalie (across seasons)
    starters = starters.sort_values(["goalie_id", "season", "game_num"])

    roll_cols = ["save_pct", "gsax"]
    parts = []

    for goalie_id, grp in starters.groupby("goalie_id", sort=False):
        grp = grp.copy()
        for col in roll_cols:
            shifted = grp[col].shift(1)
            for w in windows:
                grp[f"{col}_l{w}"] = shifted.rolling(w, min_periods=1).mean()
        parts.append(grp)

    result = pd.concat(parts, ignore_index=True)

    # Keep only the columns needed for merging into the feature matrix
    keep_cols = ["game_id", "team"]
    for col in roll_cols:
        for w in windows:
            keep_cols.append(f"{col}_l{w}")

    result = result[keep_cols]

    logger.info(
        "Goalie rolling features: %d starter-game rows, %d feature columns",
        len(result), len(keep_cols) - 2,
    )
    return result


def load_goalie_features_from_mp(
    raw_dir: Path | None = None,
    save_intermediate: bool = True,
) -> pd.DataFrame:
    """
    Top-level: aggregate + rolling + filter to starters.

    Returns DataFrame keyed by (game_id, team) with rolling goalie columns,
    ready for merging in pipeline/backfill.py.

    Saves goalie_game_stats.parquet as intermediate (for live pipeline).
    """
    goalie_games = aggregate_goalie_game_stats(raw_dir)
    if goalie_games.empty:
        return pd.DataFrame()

    # Save intermediate for live pipeline to use
    if save_intermediate:
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        path = PARQUET_DIR / "goalie_game_stats.parquet"
        goalie_games.to_parquet(path, index=False)
        logger.info("Saved goalie game stats → %s (%d rows)", path, len(goalie_games))

    rolling = compute_goalie_rolling_features(goalie_games)
    return rolling


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    feats = load_goalie_features_from_mp()
    if feats.empty:
        print("No goalie features generated.")
    else:
        print(f"\nGoalie features shape: {feats.shape}")
        print(f"Columns: {list(feats.columns)}")
        print(f"\nSample (first 10 rows):")
        print(feats.head(10))
        print(f"\nNaN rates:")
        print(feats.isnull().mean().round(4))
        print(f"\nSave pct l10 stats:")
        print(feats["save_pct_l10"].describe().round(4))
