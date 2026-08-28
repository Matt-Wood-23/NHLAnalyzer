"""
Phase 3: Baseline Models

Walk-forward time-series validation — seasons expand forward, never shuffle.

Folds (expanding window):
  Fold 1: train 2021-22            → test 2022-23
  Fold 2: train 2021-22 to 2022-23 → test 2023-24
  Fold 3: train 2021-22 to 2023-24 → test 2024-25
  Fold 4: train 2021-22 to 2024-25 → test 2025-26

Models:
  - Logistic Regression (L2, scaled) — interpretable baseline
  - Random Forest                     — non-linear baseline

Each run is logged to MLflow: metrics per fold + aggregate, feature importances.

Usage:
    python -m models.baseline
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.evaluate import evaluate_fold, summarize_results, print_feature_importance
from config.season import all_seasons

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"
MLFLOW_DB   = Path(__file__).parent.parent / "mlruns" / "mlflow.db"

# Columns that are metadata, not features
_META = {"game_id", "season", "home_team", "away_team", "target",
         "date", "home_win"}

# Season ordering (chronological) — extends automatically each year.
SEASONS = all_seasons()


def load_feature_matrix(path: Path | None = None) -> pd.DataFrame:
    if path is None:
        path = PARQUET_DIR / "feature_matrix.parquet"
    df = pd.read_parquet(path)
    # Ensure chronological sort
    df = df.sort_values(["season", "game_id"]).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _META]


def walk_forward_folds(seasons: list[str]) -> list[tuple[list[str], str]]:
    """
    Expanding-window folds: (train_seasons, test_season).
    Requires at least 2 seasons to make one fold.
    """
    folds = []
    for i in range(1, len(seasons)):
        train = seasons[:i]
        test  = seasons[i]
        folds.append((train, test))
    return folds


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("model",   RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    seasons: list[str] | None = None,
) -> dict[str, list]:
    """
    Run walk-forward validation for all models.
    Returns a dict of model_name → list[FoldResult].
    """
    if seasons is None:
        seasons = [s for s in SEASONS if s in df["season"].unique()]

    folds = walk_forward_folds(seasons)
    logger.info("Walk-forward folds: %d", len(folds))

    models = build_models()
    all_results: dict[str, list] = {name: [] for name in models}

    for fold_idx, (train_seasons, test_season) in enumerate(folds, start=1):
        train_mask = df["season"].isin(train_seasons)
        test_mask  = df["season"] == test_season

        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, "target"].values
        X_test  = df.loc[test_mask,  feature_cols].values
        y_test  = df.loc[test_mask,  "target"].values

        naive_rate = y_train.mean()
        logger.info(
            "Fold %d | train: %s (%d games) → test: %s (%d games) | naive rate: %.3f",
            fold_idx, train_seasons, len(y_train), test_season, len(y_test), naive_rate,
        )

        for name, pipeline in models.items():
            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            result = evaluate_fold(
                y_true=y_test,
                y_prob=y_prob,
                model_name=name,
                fold=fold_idx,
                train_seasons=train_seasons,
                test_season=test_season,
                n_train=len(y_train),
                naive_home_win_rate=naive_rate,
            )
            all_results[name].append(result)

            logger.info(
                "  %s → acc=%.3f brier=%.4f logloss=%.4f auc=%.3f beat_naive=%.3f",
                name, result.accuracy, result.brier_score,
                result.log_loss, result.roc_auc, result.beat_naive_rate,
            )

    return all_results, models, folds


def log_to_mlflow(
    all_results: dict,
    models: dict,
    feature_cols: list[str],
    folds: list,
    df: pd.DataFrame,
) -> None:
    """Log fold metrics, aggregate metrics, and feature importance to MLflow."""
    # Imported here rather than at module scope: mlflow is an experiment-
    # tracking dependency, and pipeline.train only needs this module for its
    # feature-column helpers.  A missing mlflow should not block training.
    import mlflow
    import mlflow.sklearn

    MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("nhl_baseline_models")

    for name, results in all_results.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("model_type", name)
            mlflow.set_tag("validation", "walk_forward_expanding")
            mlflow.set_tag("n_folds", len(results))
            mlflow.set_tag("seasons", str([r.test_season for r in results]))

            # Per-fold metrics
            for r in results:
                prefix = f"fold{r.fold}"
                mlflow.log_metrics({
                    f"{prefix}_accuracy":       r.accuracy,
                    f"{prefix}_brier":          r.brier_score,
                    f"{prefix}_log_loss":       r.log_loss,
                    f"{prefix}_roc_auc":        r.roc_auc,
                    f"{prefix}_beat_naive":     r.beat_naive_rate,
                    f"{prefix}_naive_brier":    r.naive_brier,
                    f"{prefix}_n_test":         r.n_test,
                })

            # Weighted-average aggregate metrics
            weights = np.array([r.n_test for r in results])
            for metric in ["accuracy", "brier_score", "log_loss", "roc_auc", "beat_naive_rate"]:
                vals = np.array([getattr(r, metric) for r in results])
                mlflow.log_metric(f"wavg_{metric}", float(np.average(vals, weights=weights)))

            # Retrain on all available data, log final model + importances
            train_seasons = [s for r in results for s in r.train_seasons]
            train_seasons = sorted(set(train_seasons))
            last_test = results[-1].test_season
            all_seasons = train_seasons + [last_test]
            mask = df["season"].isin(all_seasons)
            X_all = df.loc[mask, feature_cols].values
            y_all = df.loc[mask, "target"].values

            # Use the last trained pipeline (already fit on most data)
            final_pipeline = models[name]
            final_pipeline.fit(X_all, y_all)
            mlflow.sklearn.log_model(final_pipeline, artifact_path="model")

            # Feature importance (on the inner estimator)
            inner = final_pipeline.named_steps["model"]
            print_feature_importance(inner, feature_cols, top_n=15)

            logger.info("Logged run for %s to MLflow", name)


def run(parquet_path: Path | None = None) -> pd.DataFrame:
    """Full Phase 3 baseline pipeline. Returns summary DataFrame."""
    df = load_feature_matrix(parquet_path)
    feature_cols = get_feature_columns(df)

    available_seasons = [s for s in SEASONS if s in df["season"].unique()]
    logger.info(
        "Feature matrix: %d games, %d features, seasons: %s",
        len(df), len(feature_cols), available_seasons,
    )

    all_results, models, folds = run_walk_forward(df, feature_cols, available_seasons)

    # Flatten all results for summary
    flat = [r for results in all_results.values() for r in results]
    summary_df = summarize_results(flat)

    log_to_mlflow(all_results, models, feature_cols, folds, df)

    return summary_df


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    summary = run()
    print("\nSummary saved. Launch MLflow UI with:")
    print("  mlflow ui --backend-store-uri mlruns/")
