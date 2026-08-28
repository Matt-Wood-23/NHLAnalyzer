"""
Historical feature backfill pipeline.

Loads MoneyPuck data, computes rolling team / goalie / context features,
joins them into a single game-level feature matrix, and saves to Parquet.

Usage:
    python -m pipeline.backfill            # Parquet-only (no DB needed)
    DATABASE_URL=... python -m pipeline.backfill   # enriches with dates + goalie features

Output: data/parquet/feature_matrix.parquet
  - One row per completed game
  - home_* and away_* rolling features
  - diff_* = home - away differential features
  - Context: back-to-back, rest advantage, season day
  - target: home_win (1.0 = home team won)
"""

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from features.team import load_team_features, rolling_feature_columns
from features.goalie import load_goalie_features
from features.goalie_mp import load_goalie_features_from_mp
from features.context import load_context_features
from features.special_teams import load_special_teams_features
from features.elo import compute_elo_ratings, load_elo_params, save_elo_snapshot
from features.opponent_quality import compute_opponent_quality_features

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"

# Raw per-game stats that should NOT be used as model features
# (they represent the current game's outcome or are metadata)
_META_COLS = {
    "game_id", "team", "is_home", "home_team", "away_team", "opp_team",
    "season", "source", "game_num",
    # raw per-game stats — include rolling versions only
    "xg_for", "xg_against", "xg_for_5v5", "xg_against_5v5",
    "xgf_pct", "xgf_pct_5v5", "cf_pct", "sf_pct",
    "goals_for", "goals_against", "goal_diff",
    "shots_for", "shots_against", "corsi_for", "corsi_against",
    "hd_chances_for", "hd_chances_against", "hd_goals_for", "hd_goals_against",
    "won",  # current game win — the TARGET, not a feature
    "regulation_win",  # raw per-game — rolling version is the feature
    "went_to_ot",  # post-game fact used only by the ELO updater
}


def build_feature_matrix(conn=None) -> pd.DataFrame:
    """
    Assemble the full pre-game feature matrix.

    Each row = one game.  Columns:
      - game_id, season, home_team, away_team
      - home_<feat>_l{5,10,20} for each rolling stat
      - away_<feat>_l{5,10,20}
      - diff_<feat>_l{5,10,20} = home - away
      - home_back_to_back, away_back_to_back, rest_advantage, season_day
      - home_g_save_pct_l5/l10, away_g_save_pct_l5/l10  (goalie features)
      - home_st_*/away_st_*/diff_st_*  (PP/PK special teams)
      - home_elo, away_elo, diff_elo  (ELO ratings)
      - target: home_win (1.0 = home won, 0.0 = away won)

    Args:
        conn: optional psycopg2 connection.  Without it the pipeline runs on
              Parquet alone — goalie features are omitted and dates are
              approximated.
    """
    # ------------------------------------------------------------------ #
    # 1. Team rolling features                                             #
    # ------------------------------------------------------------------ #
    team_feats = load_team_features()

    roll_cols = rolling_feature_columns(team_feats)

    home_feats = team_feats[team_feats["is_home"]].copy()
    away_feats = team_feats[~team_feats["is_home"]].copy()

    # Extract target from home-team row (home_win = did the home team win?)
    home_feats["target"] = home_feats["won"]

    # Rename rolling cols to home_* / away_*
    home_renamed = (
        home_feats[["game_id", "season", "home_team", "away_team", "target"] + roll_cols]
        .rename(columns={c: f"home_{c}" for c in roll_cols})
    )
    away_renamed = (
        away_feats[["game_id"] + roll_cols]
        .rename(columns={c: f"away_{c}" for c in roll_cols})
    )

    matrix = home_renamed.merge(away_renamed, on="game_id", how="inner")

    # ------------------------------------------------------------------ #
    # 2. Differential features (home - away)                               #
    # ------------------------------------------------------------------ #
    for col in roll_cols:
        h_col = f"home_{col}"
        a_col = f"away_{col}"
        if h_col in matrix.columns and a_col in matrix.columns:
            matrix[f"diff_{col}"] = matrix[h_col] - matrix[a_col]

    # ------------------------------------------------------------------ #
    # 3. Context features                                                  #
    # ------------------------------------------------------------------ #
    ctx = load_context_features(team_features=team_feats, conn=conn)
    ctx_cols = [
        "game_id", "home_back_to_back", "away_back_to_back",
        "home_rest_days", "away_rest_days", "rest_advantage", "season_day",
        "h2h_home_win_rate_l3", "same_division", "same_conference",
    ]
    ctx_keep = [c for c in ctx_cols if c in ctx.columns]
    matrix = matrix.merge(ctx[ctx_keep], on="game_id", how="left")

    # ------------------------------------------------------------------ #
    # 4. Goalie features (MoneyPuck-based, fallback to DB)                 #
    # ------------------------------------------------------------------ #
    goalie_feats = load_goalie_features_from_mp()
    if goalie_feats.empty and conn is not None:
        goalie_feats = load_goalie_features(conn=conn)
        logger.info("Using DB-based goalie features (MoneyPuck extraction empty)")

    if not goalie_feats.empty:
        goalie_roll = [c for c in goalie_feats.columns if c not in ("game_id", "team")]
        home_g = (
            goalie_feats
            .rename(columns={c: f"home_g_{c}" for c in goalie_roll} | {"team": "_g_team"})
            .merge(matrix[["game_id", "home_team"]], on="game_id", how="right")
            .query("_g_team == home_team", engine="python")
            .drop(columns=["_g_team", "home_team"])
        )
        away_g = (
            goalie_feats
            .rename(columns={c: f"away_g_{c}" for c in goalie_roll} | {"team": "_g_team"})
            .merge(matrix[["game_id", "away_team"]], on="game_id", how="right")
            .query("_g_team == away_team", engine="python")
            .drop(columns=["_g_team", "away_team"])
        )
        matrix = matrix.merge(home_g, on="game_id", how="left")
        matrix = matrix.merge(away_g, on="game_id", how="left")
        logger.info("Goalie features merged: %d columns", len(goalie_roll) * 2)

    # ------------------------------------------------------------------ #
    # 5. Special teams features (PP/PK from MoneyPuck)                     #
    # ------------------------------------------------------------------ #
    st_feats = load_special_teams_features()
    if not st_feats.empty:
        st_roll = [c for c in st_feats.columns if c not in ("game_id", "team")]
        home_st = (
            st_feats
            .rename(columns={c: f"home_{c}" for c in st_roll} | {"team": "_st_team"})
            .merge(matrix[["game_id", "home_team"]], on="game_id", how="right")
            .query("_st_team == home_team", engine="python")
            .drop(columns=["_st_team", "home_team"])
        )
        away_st = (
            st_feats
            .rename(columns={c: f"away_{c}" for c in st_roll} | {"team": "_st_team"})
            .merge(matrix[["game_id", "away_team"]], on="game_id", how="right")
            .query("_st_team == away_team", engine="python")
            .drop(columns=["_st_team", "away_team"])
        )
        matrix = matrix.merge(home_st, on="game_id", how="left")
        matrix = matrix.merge(away_st, on="game_id", how="left")

        # Differential features for special teams
        for col in st_roll:
            h_col = f"home_{col}"
            a_col = f"away_{col}"
            if h_col in matrix.columns and a_col in matrix.columns:
                matrix[f"diff_{col}"] = matrix[h_col] - matrix[a_col]

        logger.info("Special teams features merged: %d columns", len(st_roll) * 3)

    # ------------------------------------------------------------------ #
    # 6. ELO ratings                                                       #
    # ------------------------------------------------------------------ #
    elo_cols = ["game_id", "season", "home_team", "away_team", "won"]
    if "went_to_ot" in team_feats.columns:
        elo_cols.append("went_to_ot")  # lets the updater discount OT/SO results
    home_results = team_feats[team_feats["is_home"]][elo_cols].rename(
        columns={"won": "home_win"}
    )

    # Use tuned parameters when `python -m features.elo` has produced any.
    elo_params = load_elo_params()
    elo_df, final_elos = compute_elo_ratings(
        home_results, return_final=True, **elo_params,
    )
    matrix = matrix.merge(
        elo_df[["game_id", "home_elo", "away_elo", "diff_elo"]],
        on="game_id", how="left",
    )
    logger.info("ELO features merged")

    # Save current ELO state for live pipeline
    save_elo_snapshot(final_elos)

    # ------------------------------------------------------------------ #
    # 7. Opponent quality-adjusted features                                #
    # ------------------------------------------------------------------ #
    # Reuse the ELO pass above rather than replaying the whole schedule.
    oq_feats = compute_opponent_quality_features(team_feats, elo_df=elo_df)
    if not oq_feats.empty:
        oq_roll = [c for c in oq_feats.columns if c not in ("game_id", "team")]
        home_oq = (
            oq_feats
            .rename(columns={c: f"home_{c}" for c in oq_roll} | {"team": "_oq_team"})
            .merge(matrix[["game_id", "home_team"]], on="game_id", how="right")
            .query("_oq_team == home_team", engine="python")
            .drop(columns=["_oq_team", "home_team"])
        )
        away_oq = (
            oq_feats
            .rename(columns={c: f"away_{c}" for c in oq_roll} | {"team": "_oq_team"})
            .merge(matrix[["game_id", "away_team"]], on="game_id", how="right")
            .query("_oq_team == away_team", engine="python")
            .drop(columns=["_oq_team", "away_team"])
        )
        matrix = matrix.merge(home_oq, on="game_id", how="left")
        matrix = matrix.merge(away_oq, on="game_id", how="left")

        for col in oq_roll:
            h_col = f"home_{col}"
            a_col = f"away_{col}"
            if h_col in matrix.columns and a_col in matrix.columns:
                matrix[f"diff_{col}"] = matrix[h_col] - matrix[a_col]

        logger.info("Opponent quality features merged: %d columns", len(oq_roll) * 3)

    # ------------------------------------------------------------------ #
    # 8. Drop games with unknown outcome                                   #
    # ------------------------------------------------------------------ #
    before = len(matrix)
    matrix = matrix[matrix["target"].notna()].reset_index(drop=True)
    dropped = before - len(matrix)
    if dropped:
        logger.info("Dropped %d rows with unknown target", dropped)

    feat_cols = [c for c in matrix.columns
                 if c not in ("game_id", "season", "home_team", "away_team", "target")]
    logger.info(
        "Feature matrix complete: %d games × %d features",
        len(matrix), len(feat_cols),
    )
    return matrix


def save_feature_matrix(matrix: pd.DataFrame, name: str = "feature_matrix") -> Path:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    path = PARQUET_DIR / f"{name}.parquet"
    matrix.to_parquet(path, index=False)
    logger.info("Saved → %s", path)
    return path


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    conn = None
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        import psycopg2
        try:
            conn = psycopg2.connect(db_url)
            logger.info("Connected to Postgres")
        except Exception as e:
            logger.warning("DB connect failed: %s — running Parquet-only", e)

    matrix = build_feature_matrix(conn=conn)
    path = save_feature_matrix(matrix)

    print(f"\nFeature matrix saved: {path}")
    print(f"Shape: {matrix.shape}")
    print(f"\nTarget distribution:")
    print(matrix["target"].value_counts(normalize=True).round(3))
    print(f"\nSample feature columns (first 20 home_*):")
    print([c for c in matrix.columns if c.startswith("home_")][:20])
    print(f"\nNaN rates (top 10):")
    nan_rates = matrix.isnull().mean().sort_values(ascending=False)
    print(nan_rates[nan_rates > 0].head(10))

    if conn:
        conn.close()
