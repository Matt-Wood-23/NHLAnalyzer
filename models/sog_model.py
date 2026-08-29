"""
Shots-on-Goal (SOG) prediction model.

Approach:
  - Poisson regression: natural fit for count data (SOG is a count per game)
  - XGBoost regressor: captures non-linear interactions
  - Walk-forward expanding-window validation (same regime as game-line models)

Features:
  - Player rolling SOG rate (last 10, 20 games)
  - Player rolling xG (shot quality proxy)
  - Player rolling shot attempts (ice time proxy when TOI unavailable)
  - Opponent shot suppression quality (from team feature matrix)

Since we lack historical TOI data, shot_attempts_l10/l20 acts as a proxy
(more attempts ≈ more ice time). When TOI becomes available via API, it can
be added directly.

Scoring:
  Projections are scored at the half-integer lines books actually post
  (1.5 / 2.5 / 3.5), not against ``mean(y_true)``.  The old metric asked
  "did he beat ~2.0 shots", whose honest baseline is the majority class
  (72% "under" at 2.5), not the 0.500 coin flip it was reported against.
  Because this is a Poisson model, ``over_probability`` turns the projected
  rate into P(SOG > line), which is both the number a bet needs and a
  quantity Brier can score against climatology.

League drift:
  Combined shots per game fell from 63.2 in 2021-22 to 55.6 in 2025-26.
  An expanding window trained on the high-volume seasons over-projected
  every fold, and the error grew as the gap widened (+0.065 shots in
  2022-23 to +0.120 in 2024-25).  Two corrections, both validated by
  walk-forward: exponential recency weighting on training rows, and a
  multiplicative calibration fit on the most recent training season.
  Together they cut mean bias from +0.092 to +0.052 shots.

Known limitation:
  The residual bias is concentrated in low-experience players — roughly
  +0.17 shots for skaters under 20 career games, against +0.003 for those
  over 200.  Their rolling averages are noisy and the model cannot tell a
  6-game rookie averaging 2.0 from an established 2.0 shooter.  Shrinking
  toward the league mean makes it worse (fringe players are below average,
  so the prior pulls them further up); adding log(games_played) halves the
  low-experience bias but degrades every other metric.  Treat projections
  for skaters under ~20 games as biased high.

Usage:
    python -m models.sog_model               # train + evaluate
    python -m models.sog_model --save        # train + save to models/saved/
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config.season import all_seasons
from models.count_model import (
    RECENCY_HALF_LIFE_SEASONS,
    apply_calibration as _apply_calibration,
    build_pipeline,
    calibration_factor,
    evaluate_counts,
    fit_pipeline as _fit_pipeline,
    over_probability as _over_probability,
    print_line_table,
    season_sample_weights,
)

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"
SAVED_DIR   = Path(__file__).parent.parent / "models" / "saved"

SEASONS = all_seasons()

# Features used by the SOG model
SOG_FEATURE_COLS = [
    "sog_l10", "sog_l20",
    "xg_l10", "xg_l20",
    "shot_attempts_l10", "shot_attempts_l20",
    "xg_per_attempt_l10", "xg_per_attempt_l20",
    # Opponent context (added when team snapshot is available)
    "opp_sf_pct_l20",
    "opp_xg_against_l20",
    "opp_hd_chances_against_l20",
]

# Only skaters with at least this many games are included
MIN_GAMES_FILTER = 5

# Lines books actually post for shots on goal.  Half-integers, so there is
# no push and P(over) = P(SOG >= ceil(line)).
OU_LINES = [1.5, 2.5, 3.5]

# Exclude goalies and players with very rare appearances
_SKIP_POSITIONS = {"G", ""}


def load_training_data(
    player_parquet: Optional[Path] = None,
    team_parquet: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load player game stats + team rolling features, merge opponent context.

    Returns one row per (game_id, player_id) with SOG as target.
    """
    if player_parquet is None:
        player_parquet = PARQUET_DIR / "player_game_stats.parquet"
    if team_parquet is None:
        team_parquet = PARQUET_DIR / "moneypuck_team_game_stats.parquet"

    from features.player import build_player_rolling_features
    player_feats = build_player_rolling_features(pd.read_parquet(player_parquet))

    # Filter out goalies and low-game players
    player_feats = player_feats[
        ~player_feats["position"].isin(_SKIP_POSITIONS)
        & (player_feats["games_played"] >= MIN_GAMES_FILTER)
    ].copy()

    # Load team features (rolling cols computed in-memory) for opponent context
    from features.team import load_team_features
    team_stats = load_team_features(team_parquet)

    # Build game → home/away team lookup
    opp_lookup = team_stats[["game_id", "team", "is_home"]].drop_duplicates()
    home = opp_lookup[opp_lookup["is_home"]].rename(columns={"team": "home_team"})
    away = opp_lookup[~opp_lookup["is_home"]].rename(columns={"team": "away_team"})
    game_teams = home[["game_id", "home_team"]].merge(
        away[["game_id", "away_team"]], on="game_id"
    )

    # Map player's team → opponent team
    player_feats = player_feats.merge(game_teams, on="game_id", how="left")
    player_feats["opp_team"] = np.where(
        player_feats["team"] == player_feats["home_team"],
        player_feats["away_team"],
        player_feats["home_team"],
    )

    # Get opponent's defensive rolling stats (pre-game snapshot for that game)
    opp_stats = team_stats[["game_id", "team", "sf_pct_l20", "xg_against_l20",
                             "hd_chances_against_l20"]].copy()
    opp_stats = opp_stats.rename(columns={
        "team":                   "opp_team",
        "sf_pct_l20":             "opp_sf_pct_l20",
        "xg_against_l20":         "opp_xg_against_l20",
        "hd_chances_against_l20": "opp_hd_chances_against_l20",
    })

    player_feats = player_feats.merge(opp_stats, on=["game_id", "opp_team"], how="left")

    logger.info(
        "Training data: %d player-game rows, %d players, %d seasons",
        len(player_feats),
        player_feats["player_id"].nunique(),
        player_feats["season"].nunique(),
    )
    return player_feats


def walk_forward_folds(seasons: list[str]) -> list[tuple[list[str], str]]:
    return [(seasons[:i], seasons[i]) for i in range(1, len(seasons))]


def over_probability(mu: np.ndarray, line: float) -> np.ndarray:
    """P(SOG > line). Shots on goal are close enough to Poisson to leave the
    dispersion at zero; goalie saves are not, which is why
    :func:`models.count_model.over_probability` takes the parameter."""
    return _over_probability(mu, line, dispersion=0.0)


def fit_pipeline(
    train: pd.DataFrame,
    feature_cols: list[str],
    half_life: float = RECENCY_HALF_LIFE_SEASONS,
    calibrate: bool = True,
) -> Pipeline:
    """Fit a recency-weighted, drift-calibrated SOG model on ``train``."""
    return _fit_pipeline(
        train, feature_cols, target="sog",
        half_life=half_life, calibrate=calibrate,
    )


def evaluate_sog(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_sog: Optional[np.ndarray] = None,
) -> dict:
    """Score SOG projections at :data:`OU_LINES`. See
    :func:`models.count_model.evaluate_counts` for what each field means."""
    return evaluate_counts(y_true, y_pred, OU_LINES, train_target=train_sog)


def run_walk_forward(df: pd.DataFrame) -> list[dict]:
    """Run walk-forward validation. Returns list of fold result dicts."""
    available = [s for s in SEASONS if s in df["season"].unique()]
    folds = walk_forward_folds(available)
    results = []

    feature_cols = [c for c in SOG_FEATURE_COLS if c in df.columns]
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)

    for fold_idx, (train_seasons, test_season) in enumerate(folds, start=1):
        train = df[df["season"].isin(train_seasons)]
        test  = df[df["season"] == test_season]

        if len(train) < 100 or len(test) < 50:
            logger.warning("Fold %d: insufficient data, skipping", fold_idx)
            continue

        y_train = train["sog"].values.astype(float)
        y_test  = test["sog"].values.astype(float)

        pipeline = fit_pipeline(train, feature_cols)
        y_pred = pipeline.predict(test[feature_cols].values)

        metrics = evaluate_sog(y_test, y_pred, train_sog=y_train)
        metrics.update({
            "fold": fold_idx,
            "train_seasons": train_seasons,
            "test_season": test_season,
            "n_train": len(y_train),
        })
        results.append(metrics)

        logger.info(
            "Fold %d | test %s | MAE=%.3f bias=%+.3f Brier@2.5=%.4f (n=%d)",
            fold_idx, test_season, metrics["mae"], metrics["bias"],
            metrics["lines"][2.5]["brier"], metrics["n"],
        )

    return results


def train_final_model(df: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    """Train on all available data. Returns (pipeline, feature_cols).

    Recency-weighted and drift-calibrated exactly as the walk-forward folds
    are, so the saved model behaves the way validation said it would.
    """
    feature_cols = [c for c in SOG_FEATURE_COLS if c in df.columns]
    pipeline = fit_pipeline(df, feature_cols)
    logger.info("Final model trained on %d player-game rows", len(df))
    return pipeline, feature_cols


def save_sog_model(pipeline: Pipeline, feature_cols: list[str]) -> Path:
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SAVED_DIR / "sog_model.pkl"
    cols_path  = SAVED_DIR / "sog_model_feature_cols.json"
    joblib.dump(pipeline, model_path)
    cols_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    logger.info("SOG model saved → %s", model_path)
    return model_path


def load_sog_model() -> tuple:
    model_path = SAVED_DIR / "sog_model.pkl"
    cols_path  = SAVED_DIR / "sog_model_feature_cols.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved SOG model at {model_path}. Run `python -m models.sog_model --save` first."
        )
    return joblib.load(model_path), json.loads(cols_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train and evaluate the SOG model")
    parser.add_argument("--save", action="store_true",
                        help="Save the final model after evaluation")
    args = parser.parse_args()

    df = load_training_data()

    print(f"\nDataset: {len(df):,} player-game rows")
    print(f"SOG distribution: mean={df['sog'].mean():.2f}  median={df['sog'].median():.0f}  max={df['sog'].max()}")
    print(f"Seasons: {sorted(df['season'].unique())}")

    results = run_walk_forward(df)

    if results:
        print("\n=== Walk-Forward Results ===")
        print(f"{'Fold':<6} {'Test Season':<12} {'MAE':>6} {'RMSE':>7} {'Bias':>7} {'N':>7}")
        print("-" * 48)
        for r in results:
            print(
                f"{r['fold']:<6} {r['test_season']:<12} "
                f"{r['mae']:>6.3f} {r['rmse']:>7.3f} {r['bias']:>+7.3f} {r['n']:>7}"
            )
        print(
            f"\nAvg MAE: {np.mean([r['mae'] for r in results]):.3f}   "
            f"Avg bias: {np.mean([r['bias'] for r in results]):+.3f} shots"
        )

        print_line_table(results, OU_LINES)

    if args.save:
        pipeline, feature_cols = train_final_model(df)
        path = save_sog_model(pipeline, feature_cols)
        print(f"\nModel saved: {path}")
