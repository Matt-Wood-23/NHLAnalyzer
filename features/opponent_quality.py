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

from features.elo import compute_elo_ratings, load_elo_params

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"

OQ_WINDOWS = [20]  # longer window — opponent splits are sparser
OQ_STATS = ["xgf_pct", "cf_pct", "won", "goal_diff"]


def compute_opponent_quality_features(
    team_features: pd.DataFrame,
    elo_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute rolling stats split by opponent quality (above/below median ELO).

    Args:
        team_features: output of features.team.load_team_features().
            Must contain game_id, season, team, home_team, away_team, is_home,
            won, and the raw stats in OQ_STATS.
        elo_df: per-game pre-game ELO ratings from
            :func:`features.elo.compute_elo_ratings`.  Pass the backfill's
            existing frame to avoid replaying the schedule a second time, and
            to make sure both use the same tuned parameters.

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

    # 3. Classify opponent as strong/weak (above/below season median)
    season_median_elo = df.groupby("season")["opp_elo"].transform("median")
    df["opp_is_strong"] = (df["opp_elo"] >= season_median_elo).astype(int)

    # 4. Rolling by opponent quality within (team, season)
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

    result = pd.concat(parts, ignore_index=True)

    # Keep only game_id, team, and the new rolling columns
    keep = ["game_id", "team"]
    for col in OQ_STATS:
        for label in ["strong", "weak"]:
            for w in OQ_WINDOWS:
                keep.append(f"{col}_vs_{label}_l{w}")

    result = result[keep]

    logger.info(
        "Opponent quality features: %d rows, %d columns",
        len(result), len(keep) - 2,
    )
    return result


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
