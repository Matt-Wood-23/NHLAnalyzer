"""
Backfill actual outcomes into prediction history and compute accuracy metrics.

Reads prediction_history.parquet, looks up actual game results from the
feature matrix, and computes rolling accuracy, Brier score, and calibration.

Usage:
    python -m pipeline.evaluate_history
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PARQUET_DIR  = Path(__file__).parent.parent / "data" / "parquet"
HISTORY_DIR  = Path(__file__).parent.parent / "data" / "predictions"


def backfill_outcomes() -> pd.DataFrame:
    """
    Join prediction history with actual outcomes from the feature matrix.

    Returns the updated history DataFrame with columns:
      - actual_home_win (1/0/NaN if game not yet played)
      - correct (bool: did we pick the right side?)
      - brier (per-game squared error)
    """
    hist_path = HISTORY_DIR / "prediction_history.parquet"
    fm_path = PARQUET_DIR / "feature_matrix.parquet"

    if not hist_path.exists():
        logger.warning("No prediction history found at %s", hist_path)
        return pd.DataFrame()

    hist = pd.read_parquet(hist_path)
    logger.info("Loaded %d predictions from history", len(hist))

    if not fm_path.exists():
        logger.warning("No feature matrix found — cannot look up outcomes")
        return hist

    # Drop previous outcome columns before re-merging (avoids _x/_y duplicates)
    hist = hist.drop(columns=["actual_home_win", "correct", "brier"], errors="ignore")

    fm = pd.read_parquet(fm_path, columns=["game_id", "target"])
    outcomes = fm.rename(columns={"target": "actual_home_win"})

    # Merge outcomes
    hist = hist.merge(outcomes, on="game_id", how="left")

    # Compute per-prediction metrics where outcome is known
    has_outcome = hist["actual_home_win"].notna()
    hist.loc[has_outcome, "correct"] = (
        (hist.loc[has_outcome, "prob_home_win"] >= 0.5)
        == (hist.loc[has_outcome, "actual_home_win"] == 1.0)
    )
    hist.loc[has_outcome, "brier"] = (
        hist.loc[has_outcome, "prob_home_win"]
        - hist.loc[has_outcome, "actual_home_win"]
    ) ** 2

    # Save updated history
    hist.to_parquet(hist_path, index=False)
    logger.info("Updated prediction history with outcomes → %s", hist_path)

    return hist


def print_accuracy_report(hist: pd.DataFrame) -> None:
    """Print a summary of prediction accuracy."""
    evaluated = hist[hist["actual_home_win"].notna()]

    if evaluated.empty:
        print("No evaluated predictions yet (games haven't been played).")
        return

    n = len(evaluated)
    accuracy = evaluated["correct"].mean()
    brier = evaluated["brier"].mean()

    print(f"\n{'='*50}")
    print(f"  Prediction Accuracy Report")
    print(f"{'='*50}")
    print(f"  Predictions evaluated:  {n}")
    print(f"  Accuracy:               {accuracy:.1%}")
    print(f"  Brier score:            {brier:.4f}")

    # By model
    if "model_name" in evaluated.columns:
        print(f"\n  By model:")
        for model, grp in evaluated.groupby("model_name"):
            print(
                f"    {model}: acc={grp['correct'].mean():.1%}, "
                f"brier={grp['brier'].mean():.4f}, n={len(grp)}"
            )

    # Calibration buckets
    print(f"\n  Calibration (predicted vs actual home win rate):")
    evaluated = evaluated.copy()
    evaluated["prob_bucket"] = pd.cut(
        evaluated["prob_home_win"],
        bins=[0, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0],
        labels=["<40%", "40-45%", "45-50%", "50-55%", "55-60%", ">60%"],
    )
    cal = evaluated.groupby("prob_bucket", observed=True).agg(
        n=("actual_home_win", "count"),
        predicted=("prob_home_win", "mean"),
        actual=("actual_home_win", "mean"),
    )
    for bucket, row in cal.iterrows():
        print(
            f"    {bucket:>6s}: predicted={row['predicted']:.1%}, "
            f"actual={row['actual']:.1%}, n={int(row['n'])}"
        )

    # Recent trend (last 50)
    if n >= 20:
        recent = evaluated.tail(50)
        print(f"\n  Last {len(recent)} predictions:")
        print(f"    Accuracy: {recent['correct'].mean():.1%}")
        print(f"    Brier:    {recent['brier'].mean():.4f}")

    print(f"{'='*50}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    hist = backfill_outcomes()
    if not hist.empty:
        print_accuracy_report(hist)
