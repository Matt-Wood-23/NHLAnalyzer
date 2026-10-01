"""
Opponent quality-adjusted features.

Splits rolling stats by whether the opponent was "strong" or "weak"
based on their ELO at game time. This captures strength of schedule —
a team's xGF% against top-half ELO teams is more informative than
their overall xGF%.

Output: one row per (game_id, team) with rolling columns like
  xgf_pct_vs_strong_l20, won_vs_weak_l20, etc.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from features.elo import INITIAL_ELO, compute_elo_ratings, load_elo_params

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"

OQ_WINDOWS = [20]  # longer window — opponent splits are sparser
OQ_STATS = ["xgf_pct", "cf_pct", "won", "goal_diff"]

# An opponent rated at or above the league mean counts as strong.
STRONG_OPPONENT_ELO = INITIAL_ELO


def compute_opponent_quality_features(
    team_features: pd.DataFrame,
    elo_df: pd.DataFrame | None = None,
    save_intermediate: bool = False,
) -> pd.DataFrame:
    """
    Compute rolling stats split by opponent quality (above/below mean ELO).

    Args:
        team_features: output of features.team.load_team_features().
            Must contain game_id, season, team, home_team, away_team, is_home,
            won, and the raw stats in OQ_STATS.
        elo_df: per-game pre-game ELO ratings from
            :func:`features.elo.compute_elo_ratings`.  Pass the backfill's
            existing frame to avoid replaying the schedule a second time, and
            to make sure both use the same tuned parameters.
        save_intermediate: write the classified per-game rows to
            ``opponent_quality_games.parquet`` for the live pipeline.  Off by
            default so tests and ad-hoc calls do not touch the data directory.

    Returns:
        DataFrame with (game_id, team) and rolling columns:
          <stat>_vs_strong_l20, <stat>_vs_weak_l20
    """
    # 1. Per-game ELO for every team
    home_results = team_features[team_features["is_home"]][
        ["game_id", "season", "home_team", "away_team", "won"]
    ].rename(columns={"won": "home_win"})

    if elo_df is None:
        elo_df = compute_elo_ratings(home_results, **load_elo_params())

    # Build a per-game team ELO lookup
    home_elos = elo_df[["game_id", "home_elo"]].copy()
    home_elos = home_elos.merge(
        home_results[["game_id", "home_team"]],
        on="game_id",
    )
    home_elos = home_elos.rename(columns={"home_team": "team", "home_elo": "elo"})

    away_elos = elo_df[["game_id", "away_elo"]].copy()
    away_elos = away_elos.merge(
        home_results[["game_id", "away_team"]],
        on="game_id",
    )
    away_elos = away_elos.rename(columns={"away_team": "team", "away_elo": "elo"})

    team_elo_lookup = pd.concat([home_elos, away_elos], ignore_index=True)

    # 2. Attach opponent ELO to each team-game row
    df = team_features[
        ["game_id", "season", "team", "opp_team", "game_num", "is_home"] + OQ_STATS
    ].copy()

    df = df.merge(
        team_elo_lookup.rename(columns={"team": "opp_team", "elo": "opp_elo"}),
        on=["game_id", "opp_team"],
        how="left",
    )

    # 3. Classify opponent as strong/weak: above or below the league mean.
    #
    # This used to split on the *season median* of opponent ELO, computed over
    # the whole season -- so an October game was labelled using ratings from
    # April.  The live pipeline cannot do that; it only has the season to
    # date, whose median wanders early on (1494 after the first seven games
    # of 2026-27).  ELO is zero-sum and regresses toward INITIAL_ELO, so the
    # league mean is that constant in every season, known in advance, and the
    # same number in training and serving.
    df["opp_is_strong"] = (df["opp_elo"] >= STRONG_OPPONENT_ELO).astype(int)

    # Persist the classified games so the live snapshot rolls the very same
    # rows, rather than re-deriving opponent strength from a second ELO pass.
    if save_intermediate:
        PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        df[_GAME_COLS].to_parquet(OQ_GAMES_PATH, index=False)
        logger.info("Saved opponent-quality games → %s (%d rows)", OQ_GAMES_PATH, len(df))

    # 4. Rolling by opponent quality within (team, season)
    result = _roll_by_strength(df)[["game_id", "team"] + _rolling_columns()]

    logger.info(
        "Opponent quality features: %d rows, %d columns",
        len(result), len(result.columns) - 2,
    )
    return result


OQ_GAMES_PATH = PARQUET_DIR / "opponent_quality_games.parquet"

_GAME_COLS = ["game_id", "season", "team", "game_num", "opp_is_strong"] + OQ_STATS

# game_id of the placeholder row pregame_snapshot appends per team.
_NEXT_GAME = "__next__"


def _rolling_columns() -> list[str]:
    return [
        f"{col}_vs_{label}_l{w}"
        for col in OQ_STATS for label in ("strong", "weak") for w in OQ_WINDOWS
    ]


def _roll_by_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Shifted rolling means split by opponent strength, within (team, season).

    The one implementation behind both the training features and the live
    snapshot.
    """
    df = df.sort_values(["team", "season", "game_num"])

    parts = []
    for (team, season), grp in df.groupby(["team", "season"], sort=False):
        grp = grp.copy()
        for strength, label in [(1, "strong"), (0, "weak")]:
            mask = grp["opp_is_strong"] == strength
            for col in OQ_STATS:
                vals = grp[col].where(mask, np.nan).shift(1)
                for w in OQ_WINDOWS:
                    grp[f"{col}_vs_{label}_l{w}"] = vals.rolling(
                        w, min_periods=3
                    ).mean()
        parts.append(grp)

    return pd.concat(parts, ignore_index=True)


def pregame_snapshot(
    oq_games: pd.DataFrame,
    teams: list[str] | None = None,
) -> pd.DataFrame:
    """Opponent-quality features for each team's *next* game.

    Appends a placeholder game per team and runs the same
    :func:`_roll_by_strength` as training.  The placeholder's own opponent
    does not matter: every value is shifted, so only games already played
    reach it.

    The live pipeline used to read these off the feature matrix, taking each
    team's last *home* row.  That is the pre-game value for a game already
    played — stale by however many road games followed it — and a team that
    had only played away had no row at all.

    Args:
        oq_games: the classified per-game rows saved by
                  :func:`compute_opponent_quality_features`, scoped to a
                  *single season*.
        teams:    teams that must appear; one with no games gets an all-NaN row.

    Returns:
        DataFrame indexed by team, one column per rolling feature.
    """
    wanted = sorted(set(teams or []) | set(oq_games["team"].unique()))
    if not wanted:
        return pd.DataFrame()

    season = oq_games["season"].iloc[0] if len(oq_games) else "next"
    next_num = int(oq_games["game_num"].max()) + 1 if len(oq_games) else 1
    placeholders = pd.DataFrame({
        "game_id": _NEXT_GAME, "team": wanted, "season": season,
        "game_num": next_num, "opp_is_strong": 0,
    })

    rolled = _roll_by_strength(
        pd.concat([oq_games, placeholders], ignore_index=True)
    )
    out = rolled[rolled["game_id"] == _NEXT_GAME]
    return out.set_index("team")[_rolling_columns()]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from features.team import load_team_features

    team_feats = load_team_features()
    oq_feats = compute_opponent_quality_features(team_feats)
    print(f"\nShape: {oq_feats.shape}")
    print(f"Columns: {list(oq_feats.columns)}")
    print(f"\nNaN rates:")
    print(oq_feats.isnull().mean().round(4))
    print(f"\nSample:")
    print(oq_feats.dropna().head(10))
