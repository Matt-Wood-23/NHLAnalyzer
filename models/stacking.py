"""
Stacking meta-learner: trains a logistic regression on out-of-fold predictions
from RF, XGBoost, and LightGBM.

Instead of simple probability averaging (0.5 RF + 0.5 XGB), the meta-learner
learns optimal weights that can vary by confidence region. This almost always
beats naive ensembles.

Walk-forward structure:
  For each test fold, the base models produce out-of-fold predictions on the
  training data (using inner CV), then a logistic regression is trained on
  those predictions. The meta-model predicts on the held-out test fold.

Usage:
    python -m models.stacking                    # default 50 Optuna trials
    python -m models.stacking --trials 150       # more thorough XGB tuning
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from models.baseline import (
    SEASONS,
    load_feature_matrix,
    walk_forward_folds,
)
from models.evaluate import evaluate_fold, summarize_results
from models.xgboost_model import (
    get_feature_columns,
    tune_model,
    fit_imputer,
    _add_interaction_features,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _build_rf() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _build_xgb(params: dict) -> Pipeline:
    xgb_params = {
        **params,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", XGBClassifier(**xgb_params)),
    ])


def _build_lgbm() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )),
    ])


def _generate_oof_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_models: dict[str, Pipeline],
    n_inner_folds: int = 3,
) -> np.ndarray:
    """
    Generate out-of-fold predictions for the training set using inner CV.

    Returns array of shape (n_train, n_models) with OOF probabilities.
    """
    n = len(y_train)
    n_models = len(base_models)
    oof_preds = np.full((n, n_models), np.nan)

    # Simple chronological split into n_inner_folds chunks
    fold_size = n // n_inner_folds
    for inner_fold in range(n_inner_folds):
        val_start = inner_fold * fold_size
        val_end = val_start + fold_size if inner_fold < n_inner_folds - 1 else n

        inner_train_idx = list(range(0, val_start)) + list(range(val_end, n))
        inner_val_idx = list(range(val_start, val_end))

        if len(inner_train_idx) < 100:
            continue

        X_inner_train = X_train[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        X_inner_val = X_train[inner_val_idx]

        for j, (name, model) in enumerate(base_models.items()):
            # Clone the pipeline for each inner fold
            from sklearn.base import clone
            m = clone(model)
            m.fit(X_inner_train, y_inner_train)
            oof_preds[inner_val_idx, j] = m.predict_proba(X_inner_val)[:, 1]

    return oof_preds


def run_stacking(
    df: pd.DataFrame,
    feature_cols: list[str],
    xgb_params: dict,
    available_seasons: list[str],
) -> list:
    """
    Walk-forward stacking ensemble.

    For each outer fold:
      1. Generate OOF predictions from base models on training data
      2. Train a meta-learner (logistic regression) on OOF predictions
      3. Train base models on full training data, predict on test
      4. Meta-learner predicts on base model test predictions

    Returns list of FoldResult.
    """
    folds = walk_forward_folds(available_seasons)
    results = []

    for fold_idx, (train_seasons, test_season) in enumerate(folds, start=1):
        train_mask = df["season"].isin(train_seasons)
        test_mask = df["season"] == test_season

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "target"].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, "target"].values

        base_models = {
            "rf": _build_rf(),
            "xgb": _build_xgb(xgb_params),
            "lgbm": _build_lgbm(),
        }

        # Step 1: Generate OOF predictions on training data
        oof_preds = _generate_oof_predictions(X_train, y_train, base_models)

        # Step 2: Train meta-learner on OOF predictions
        # Drop rows where OOF is NaN (can happen at edges of inner folds)
        valid_mask = ~np.isnan(oof_preds).any(axis=1)
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        meta.fit(oof_preds[valid_mask], y_train[valid_mask])

        logger.info(
            "Fold %d meta-learner weights: RF=%.3f, XGB=%.3f, LGBM=%.3f (intercept=%.3f)",
            fold_idx,
            meta.coef_[0][0], meta.coef_[0][1], meta.coef_[0][2],
            meta.intercept_[0],
        )

        # Step 3: Train base models on FULL training data, predict on test
        test_preds = np.zeros((len(X_test), len(base_models)))
        for j, (name, model) in enumerate(base_models.items()):
            from sklearn.base import clone
            m = clone(model)
            m.fit(X_train, y_train)
            test_preds[:, j] = m.predict_proba(X_test)[:, 1]

        # Step 4: Meta-learner predicts on test
        y_prob_stack = meta.predict_proba(test_preds)[:, 1]

        # Also compute naive ensemble for comparison
        y_prob_naive = test_preds.mean(axis=1)

        result_stack = evaluate_fold(
            y_true=y_test,
            y_prob=y_prob_stack,
            model_name="stacking_meta",
            fold=fold_idx,
            train_seasons=train_seasons,
            test_season=test_season,
            n_train=len(y_train),
            naive_home_win_rate=y_train.mean(),
        )
        results.append(result_stack)

        result_naive = evaluate_fold(
            y_true=y_test,
            y_prob=y_prob_naive,
            model_name="naive_3way_ensemble",
            fold=fold_idx,
            train_seasons=train_seasons,
            test_season=test_season,
            n_train=len(y_train),
            naive_home_win_rate=y_train.mean(),
        )

        logger.info(
            "Fold %d (%s→%s) | stacking: brier=%.4f acc=%.3f auc=%.3f | "
            "naive_ens: brier=%.4f acc=%.3f",
            fold_idx, train_seasons[-1], test_season,
            result_stack.brier_score, result_stack.accuracy, result_stack.roc_auc,
            result_naive.brier_score, result_naive.accuracy,
        )

    return results


def run(n_trials: int = 50) -> None:
    """Full stacking pipeline: tune XGB → stack RF+XGB+LightGBM."""
    df = load_feature_matrix()
    df = _add_interaction_features(df)
    feature_cols = get_feature_columns(df)

    available_seasons = [s for s in SEASONS if s in df["season"].unique()]
    logger.info(
        "Feature matrix: %d games, %d features, seasons: %s",
        len(df), len(feature_cols), available_seasons,
    )

    # Tune XGBoost parameters
    print(f"\n{'='*60}")
    print(f"  Step 1: Tuning XGBoost ({n_trials} Optuna trials)")
    print(f"{'='*60}")
    best_params_xgb = tune_model(
        df, feature_cols, n_trials=n_trials, model_type="xgboost",
    )
    print(f"  Best XGB params: {best_params_xgb}")

    # Run stacking
    print(f"\n{'='*60}")
    print(f"  Step 2: Stacking walk-forward (RF + XGB + LightGBM -> LR)")
    print(f"{'='*60}")
    results = run_stacking(df, feature_cols, best_params_xgb, available_seasons)

    summary = summarize_results(results)
    print(f"\n{summary}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "stacking_results.csv", index=False)
    print(f"\nResults saved to {RESULTS_DIR / 'stacking_results.csv'}")


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Stacking meta-learner")
    parser.add_argument("--trials", type=int, default=50,
                        help="Optuna trials for XGBoost tuning")
    args = parser.parse_args()

    run(n_trials=args.trials)
