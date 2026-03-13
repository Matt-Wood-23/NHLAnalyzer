"""
Evaluation framework for NHL win probability models.

Metrics computed per fold and in aggregate:
  - Accuracy           (threshold at 0.50)
  - Brier score        (lower is better; naive baseline ≈ 0.25)
  - Log loss           (lower is better)
  - ROC-AUC            (higher is better; 0.50 = random)
  - Beat-naive rate    (% of games where model assigned higher prob to actual winner
                        than the naive home-win-rate baseline)

CLV vs closing lines is tracked in Phase 5 once odds are loaded.
"""

import logging
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    model_name: str
    fold: int          # 1-indexed
    train_seasons: list[str]
    test_season: str
    n_train: int
    n_test: int
    accuracy: float
    brier_score: float
    log_loss: float
    roc_auc: float
    beat_naive_rate: float   # vs naive home-win-rate baseline
    naive_brier: float       # naive baseline Brier score for this fold


def evaluate_fold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    fold: int,
    train_seasons: list[str],
    test_season: str,
    n_train: int,
    naive_home_win_rate: float,
) -> FoldResult:
    """
    Compute all evaluation metrics for a single walk-forward fold.

    Args:
        y_true: ground-truth labels (1 = home win, 0 = away win)
        y_prob: model's predicted probability of home win
        naive_home_win_rate: home win rate in the training set (used as naive baseline)
    """
    n_test = len(y_true)

    # Naive baseline: always predict training-set home win rate
    naive_prob = np.full(n_test, naive_home_win_rate)

    # Beat-naive: model assigned higher prob to the actual outcome than naive did
    # For home wins: model_prob > naive_prob  (model was more confident on home win)
    # For away wins: (1 - model_prob) > (1 - naive_prob)  → model_prob < naive_prob
    home_wins = y_true == 1
    beat = np.where(
        home_wins,
        y_prob > naive_home_win_rate,
        y_prob < naive_home_win_rate,
    )
    beat_naive_rate = beat.mean()

    return FoldResult(
        model_name=model_name,
        fold=fold,
        train_seasons=train_seasons,
        test_season=test_season,
        n_train=n_train,
        n_test=n_test,
        accuracy=accuracy_score(y_true, y_prob >= 0.5),
        brier_score=brier_score_loss(y_true, y_prob),
        log_loss=log_loss(y_true, y_prob),
        roc_auc=roc_auc_score(y_true, y_prob),
        beat_naive_rate=beat_naive_rate,
        naive_brier=brier_score_loss(y_true, naive_prob),
    )


def summarize_results(results: list[FoldResult]) -> pd.DataFrame:
    """Aggregate fold results into a summary DataFrame."""
    rows = [asdict(r) for r in results]
    df = pd.DataFrame(rows)
    df["train_seasons"] = df["train_seasons"].apply(lambda x: ", ".join(x))

    # Weighted aggregate by n_test
    summary_cols = ["accuracy", "brier_score", "log_loss", "roc_auc", "beat_naive_rate"]
    weights = df["n_test"].values

    print("\n" + "=" * 70)
    for model_name, grp in df.groupby("model_name"):
        w = grp["n_test"].values
        print(f"\nModel: {model_name}")
        print(f"  {'Fold':<6} {'Train':<30} {'Test':<12} {'Acc':>6} {'Brier':>7} "
              f"{'LogLoss':>8} {'AUC':>6} {'BeatNaive':>10}")
        print("  " + "-" * 65)
        for _, row in grp.iterrows():
            print(
                f"  {int(row['fold']):<6} {row['train_seasons']:<30} {row['test_season']:<12} "
                f"{row['accuracy']:>6.3f} {row['brier_score']:>7.4f} "
                f"{row['log_loss']:>8.4f} {row['roc_auc']:>6.3f} "
                f"{row['beat_naive_rate']:>10.3f}"
            )
        wavg = {c: np.average(grp[c].values, weights=w) for c in summary_cols}
        print(f"  {'WAVG':<6} {'':30} {'':12} "
              f"{wavg['accuracy']:>6.3f} {wavg['brier_score']:>7.4f} "
              f"{wavg['log_loss']:>8.4f} {wavg['roc_auc']:>6.3f} "
              f"{wavg['beat_naive_rate']:>10.3f}")
        nbrier_avg = np.average(grp["naive_brier"].values, weights=w)
        print(f"  Naive Brier (baseline to beat): {nbrier_avg:.4f}")
    print("=" * 70)

    return df


def print_feature_importance(model, feature_names: list[str], top_n: int = 20) -> None:
    """Print top feature importances for tree models, or coefficients for LR."""
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        imp = imp.sort_values(ascending=False).head(top_n)
        print(f"\nTop {top_n} feature importances:")
        for feat, val in imp.items():
            print(f"  {feat:<45} {val:.4f}")
    elif hasattr(model, "coef_"):
        coef = pd.Series(model.coef_[0], index=feature_names)
        coef = coef.abs().sort_values(ascending=False).head(top_n)
        print(f"\nTop {top_n} |coefficient| magnitudes:")
        for feat, val in coef.items():
            print(f"  {feat:<45} {val:.4f}")
