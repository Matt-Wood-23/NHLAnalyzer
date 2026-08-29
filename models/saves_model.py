"""
Goalie saves prop model.

Saves are shots faced times save rate.  Save rate barely moves -- the league
sits near 0.905 and a good starter beats a bad one by roughly a point -- so
almost all of the variance is workload, which is a team property before it
is a goalie property.  The features reflect that: the goalie's own rolling
shots-against alongside how much his team concedes and how much tonight's
opponent generates.

Two things separate this from :mod:`models.sog_model`:

  Saves are overdispersed.  Variance runs ~1.9x the mean (~1.8x even after
  excluding short outings, so it is not just pulled goalies).  Poisson tails
  come out too tight and P(over) ends up overconfident, so the probability
  transform is negative binomial with a dispersion estimated from training
  residuals.  That alone is worth 1-7% Brier at every line.

  The drift is steeper.  Starter saves fell from 28.4 a game in 2021-22 to
  24.8 in 2025-26.  Recency weighting and newest-season calibration leave
  +0.68 saves of bias, because both can only reach the newest *training*
  season and the test season is a further step down -- almost exactly one
  season of drift.  ``pipeline.saves_live`` closes most of the rest with a
  serve-time factor fit on the current season's completed starts.

Usage:
    python -m models.saves_model                # train + evaluate
    python -m models.saves_model --save         # train + save to models/saved/
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
from features.goalie_props import (
    MIN_STARTS,
    attach_team_context,
    build_goalie_rolling_features,
    load_goalie_game_stats,
    rolling_feature_columns,
    team_context_columns,
)
from models.count_model import (
    estimate_dispersion,
    evaluate_counts,
    fit_pipeline,
    over_probability,
    print_line_table,
)

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"
SAVED_DIR   = Path(__file__).parent.parent / "models" / "saved"

SEASONS = all_seasons()
TARGET  = "saves"

# Lines books actually post for goalie saves. Half-integers, so no push.
SAVES_LINES = [22.5, 24.5, 27.5, 30.5]

SAVES_FEATURE_COLS = (
    rolling_feature_columns() + ["is_home"] + team_context_columns()
)


def load_training_data(
    goalie_parquet: Optional[Path] = None,
    team_parquet: Optional[Path] = None,
) -> pd.DataFrame:
    """One row per starter appearance, with rolling and team-context features."""
    if team_parquet is None:
        team_parquet = PARQUET_DIR / "moneypuck_team_game_stats.parquet"

    from features.team import load_team_features

    goalie_games = load_goalie_game_stats(goalie_parquet)
    features = build_goalie_rolling_features(goalie_games)
    features = attach_team_context(features, load_team_features(team_parquet))

    features = features[features["starts"] >= MIN_STARTS]
    features = features.dropna(subset=[TARGET])

    logger.info(
        "Training data: %d starter rows, %d goalies, %d seasons",
        len(features), features["goalie_id"].nunique(), features["season"].nunique(),
    )
    return features


def walk_forward_folds(seasons: list[str]) -> list[tuple[list[str], str]]:
    return [(seasons[:i], seasons[i]) for i in range(1, len(seasons))]


def run_walk_forward(df: pd.DataFrame) -> list[dict]:
    """Expanding-window validation, same regime as the other models."""
    available = [s for s in SEASONS if s in df["season"].unique()]
    results = []

    feature_cols = [c for c in SAVES_FEATURE_COLS if c in df.columns]
    logger.info("Using %d features", len(feature_cols))

    for fold_idx, (train_seasons, test_season) in enumerate(
        walk_forward_folds(available), start=1,
    ):
        train = df[df["season"].isin(train_seasons)]
        test  = df[df["season"] == test_season]

        if len(train) < 100 or len(test) < 50:
            logger.warning("Fold %d: insufficient data, skipping", fold_idx)
            continue

        y_train = train[TARGET].values.astype(float)
        y_test  = test[TARGET].values.astype(float)

        pipeline = fit_pipeline(train, feature_cols, target=TARGET)
        dispersion = estimate_dispersion(
            y_train, pipeline.predict(train[feature_cols].values),
        )
        y_pred = pipeline.predict(test[feature_cols].values)

        metrics = evaluate_counts(
            y_test, y_pred, SAVES_LINES,
            train_target=y_train, dispersion=dispersion,
        )
        metrics.update({
            "fold": fold_idx,
            "train_seasons": train_seasons,
            "test_season": test_season,
            "n_train": len(y_train),
        })
        results.append(metrics)

        logger.info(
            "Fold %d | test %s | MAE=%.3f bias=%+.3f alpha=%.4f Brier@24.5=%.4f (n=%d)",
            fold_idx, test_season, metrics["mae"], metrics["bias"],
            dispersion, metrics["lines"][24.5]["brier"], metrics["n"],
        )

    return results


def train_final_model(df: pd.DataFrame) -> tuple[Pipeline, list[str], float]:
    """Train on everything. Returns (pipeline, feature_cols, dispersion)."""
    feature_cols = [c for c in SAVES_FEATURE_COLS if c in df.columns]
    pipeline = fit_pipeline(df, feature_cols, target=TARGET)
    dispersion = estimate_dispersion(
        df[TARGET].values.astype(float), pipeline.predict(df[feature_cols].values),
    )
    logger.info(
        "Final model trained on %d starter rows (dispersion %.4f)",
        len(df), dispersion,
    )
    return pipeline, feature_cols, dispersion


def save_saves_model(
    pipeline: Pipeline,
    feature_cols: list[str],
    dispersion: float,
) -> Path:
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SAVED_DIR / "saves_model.pkl"
    meta_path  = SAVED_DIR / "saves_model_meta.json"
    joblib.dump(pipeline, model_path)
    meta_path.write_text(
        json.dumps(
            {
                "feature_cols": feature_cols,
                "dispersion": dispersion,
                "lines": SAVES_LINES,
                "target": TARGET,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saves model saved -> %s", model_path)
    return model_path


def load_saves_model() -> tuple[Pipeline, list[str], float]:
    """Returns (pipeline, feature_cols, dispersion).

    The dispersion travels with the model because P(over) is wrong without
    it -- a Poisson tail on an overdispersed target reads as far more
    confident than the data supports.
    """
    model_path = SAVED_DIR / "saves_model.pkl"
    meta_path  = SAVED_DIR / "saves_model_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved saves model at {model_path}. "
            "Run `python -m models.saves_model --save` first."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return joblib.load(model_path), meta["feature_cols"], meta["dispersion"]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train and evaluate the saves model")
    parser.add_argument("--save", action="store_true",
                        help="Save the final model after evaluation")
    args = parser.parse_args()

    df = load_training_data()

    print(f"\nDataset: {len(df):,} starter rows, {df['goalie_id'].nunique()} goalies")
    print(f"Saves: mean={df[TARGET].mean():.2f}  sd={df[TARGET].std():.2f}  "
          f"var/mean={df[TARGET].var() / df[TARGET].mean():.2f}")
    print(f"Seasons: {sorted(df['season'].unique())}")

    results = run_walk_forward(df)

    if results:
        print("\n=== Walk-Forward Results ===")
        print(f"{'Fold':<6} {'Test Season':<12} {'MAE':>6} {'RMSE':>7} "
              f"{'Bias':>7} {'alpha':>7} {'N':>7}")
        print("-" * 58)
        for r in results:
            print(
                f"{r['fold']:<6} {r['test_season']:<12} "
                f"{r['mae']:>6.3f} {r['rmse']:>7.3f} {r['bias']:>+7.3f} "
                f"{r['dispersion']:>7.4f} {r['n']:>7}"
            )
        print(
            f"\nAvg MAE: {np.mean([r['mae'] for r in results]):.3f}   "
            f"Avg bias: {np.mean([r['bias'] for r in results]):+.3f} saves"
        )
        print("Residual bias is roughly one season of league drift; "
              "pipeline.saves_live closes most of it in-season.")

        print_line_table(results, SAVES_LINES)

    if args.save:
        pipeline, feature_cols, dispersion = train_final_model(df)
        path = save_saves_model(pipeline, feature_cols, dispersion)
        print(f"\nModel saved: {path}")
