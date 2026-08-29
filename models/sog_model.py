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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import poisson

from config.season import all_seasons

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

# Exponential decay applied to training rows by season age, in seasons.
# 0.5 means a season-old row carries 1/4 the weight of a current one.
# Tuned by walk-forward against projection bias; see module docstring.
RECENCY_HALF_LIFE_SEASONS = 0.5

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


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   PoissonRegressor(alpha=0.1, max_iter=300)),
    ])


def season_sample_weights(
    seasons: pd.Series,
    ordered_seasons: list[str],
    half_life: float = RECENCY_HALF_LIFE_SEASONS,
) -> np.ndarray:
    """Exponential recency weights for training rows, by season age.

    The most recent training season gets weight 1.0 and each season back
    halves every ``half_life`` seasons.  This is the first of the two drift
    corrections: it stops five-year-old, high-volume seasons from setting
    the intercept for a league that now shoots 12% less.
    """
    if half_life is None or half_life <= 0:
        return np.ones(len(seasons), dtype=float)
    newest = len(ordered_seasons) - 1
    age = seasons.map({s: newest - i for i, s in enumerate(ordered_seasons)})
    return (0.5 ** (age.to_numpy(dtype=float) / half_life))


def over_probability(mu: np.ndarray, line: float) -> np.ndarray:
    """P(SOG > line) for a Poisson rate ``mu`` at a half-integer ``line``.

    Lines are half-integers, so "over 2.5" means 3 or more shots and there
    is no push to handle: P = 1 - CDF(floor(line)).
    """
    return 1.0 - poisson.cdf(np.floor(line), mu)


def _calibration_factor(
    pipeline: Pipeline,
    train: pd.DataFrame,
    feature_cols: list[str],
) -> float:
    """Multiplicative correction so the fit matches the most recent season.

    The second drift correction.  A Poisson MLE reproduces the mean of the
    *whole* training set, which straddles seasons of different shot volume.
    Comparing predicted to actual on the newest training season alone
    recovers how far the fit sits above where the league now is.
    """
    newest = sorted(train["season"].unique())[-1]
    recent = train[train["season"] == newest]
    predicted = pipeline.predict(recent[feature_cols].values).mean()
    if not np.isfinite(predicted) or predicted <= 0:
        logger.warning("Calibration skipped: non-positive mean prediction")
        return 1.0
    factor = float(recent["sog"].mean() / predicted)
    logger.info("Calibration on %s: x%.4f", newest, factor)
    return factor


def _apply_calibration(pipeline: Pipeline, factor: float) -> Pipeline:
    """Fold the calibration factor into the model's intercept, in place.

    The link is log, so ``exp(b0 + log(f) + bx) == f * exp(b0 + bx)``: the
    correction is exact as an intercept shift.  Doing it here rather than at
    the call site means the saved pickle is already calibrated and every
    consumer (``pipeline.props_live``) gets it with no change.
    """
    if factor != 1.0 and np.isfinite(factor) and factor > 0:
        pipeline.named_steps["model"].intercept_ += float(np.log(factor))
    return pipeline


def fit_pipeline(
    train: pd.DataFrame,
    feature_cols: list[str],
    half_life: float = RECENCY_HALF_LIFE_SEASONS,
    calibrate: bool = True,
) -> Pipeline:
    """Fit a recency-weighted, drift-calibrated SOG model on ``train``."""
    ordered = sorted(train["season"].unique())
    weights = season_sample_weights(train["season"], ordered, half_life)

    pipeline = build_pipeline()
    pipeline.fit(
        train[feature_cols].values,
        train["sog"].values.astype(float),
        model__sample_weight=weights,
    )
    if calibrate and len(ordered) >= 1:
        _apply_calibration(pipeline, _calibration_factor(pipeline, train, feature_cols))
    return pipeline


def evaluate_sog(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_sog: Optional[np.ndarray] = None,
) -> dict:
    """Score projections at the lines books post, against honest baselines.

    For each line in :data:`OU_LINES` this reports:
      ``brier``       Brier score of the model's P(over).
      ``brier_base``  Brier score of climatology — the training base rate,
                      the same no-skill reference ``/history`` uses.
      ``accuracy``    Share of correct over/under calls.
      ``acc_base``    The majority class, which is what accuracy must beat.
                      At 2.5 that is ~0.72, not 0.500.

    Args:
        train_sog: training targets, used for the climatological base rate.
                   Falls back to ``y_true`` when absent, which flatters
                   climatology slightly.
    """
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    reference = y_true if train_sog is None else train_sog
    metrics = {
        "mae":  mae,
        "rmse": rmse,
        "bias": float(np.mean(y_pred) - np.mean(y_true)),
        "n":    len(y_true),
        "lines": {},
    }

    for line in OU_LINES:
        actual = (y_true > line).astype(float)
        prob   = over_probability(y_pred, line)
        rate   = float(np.mean(reference > line))
        metrics["lines"][line] = {
            "brier":      float(np.mean((prob - actual) ** 2)),
            "brier_base": float(np.mean((rate - actual) ** 2)),
            "accuracy":   float(np.mean((prob > 0.5) == (actual == 1))),
            "acc_base":   float(max(actual.mean(), 1 - actual.mean())),
            "over_rate":  float(actual.mean()),
        }

    return metrics


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

        print("\n=== Over/Under by Line ===")
        print("Brier scores P(over) against climatology; accuracy against the")
        print("majority class, which is the baseline a coin flip is not.")
        print()
        print(f"{'Line':>5} {'Brier':>8} {'no-skill':>9} {'skill':>7} "
              f"{'Acc':>7} {'majority':>9} {'Over%':>7}")
        print("-" * 58)
        for line in OU_LINES:
            per = [r["lines"][line] for r in results]
            brier = np.mean([x["brier"] for x in per])
            base  = np.mean([x["brier_base"] for x in per])
            acc   = np.mean([x["accuracy"] for x in per])
            amaj  = np.mean([x["acc_base"] for x in per])
            over  = np.mean([x["over_rate"] for x in per])
            skill = 1 - brier / base if base else float("nan")
            print(
                f"{line:>5.1f} {brier:>8.4f} {base:>9.4f} {skill:>6.1%} "
                f"{acc:>7.1%} {amaj:>9.1%} {over:>7.1%}"
            )
        print("\nBrier skill > 0 means the projections beat always quoting the")
        print("base rate. Accuracy below the majority column means a constant")
        print("'under' would have called more games correctly.")

    if args.save:
        pipeline, feature_cols = train_final_model(df)
        path = save_sog_model(pipeline, feature_cols)
        print(f"\nModel saved: {path}")
