"""
Train and serialize the best model (Random Forest) on all available seasons.
Saves the fitted pipeline and feature column list to models/saved/.

Usage:
    python -m pipeline.train                         # default: random_forest
    python -m pipeline.train --model random_forest
    python -m pipeline.train --model logistic_regression
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.baseline import load_feature_matrix, get_feature_columns, SEASONS

logger = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"

# Features identified as positional noise — drop before training
EXCLUDE_FEATURES = {"home_games_played", "away_games_played", "diff_games_played"}


def build_pipeline(model_name: str) -> Pipeline:
    if model_name == "random_forest":
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
    elif model_name == "logistic_regression":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs", random_state=42,
            )),
        ])
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


def train_and_save(
    model_name: str = "random_forest",
    parquet_path: Path | None = None,
) -> Path:
    """
    Train on all available seasons in the feature matrix and save to disk.

    Returns path to the saved model file.
    """
    df = load_feature_matrix(parquet_path)
    all_feature_cols = get_feature_columns(df)
    feature_cols = [c for c in all_feature_cols if c not in EXCLUDE_FEATURES]

    available_seasons = [s for s in SEASONS if s in df["season"].unique()]
    logger.info("Training on %d seasons: %s", len(available_seasons), available_seasons)

    mask = df["season"].isin(available_seasons)
    X = df.loc[mask, feature_cols].values
    y = df.loc[mask, "target"].values

    pipeline = build_pipeline(model_name)
    logger.info("Fitting %s on %d games, %d features ...", model_name, len(y), len(feature_cols))
    pipeline.fit(X, y)

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    model_path = SAVED_DIR / f"{model_name}.pkl"
    cols_path  = SAVED_DIR / f"{model_name}_feature_cols.json"

    joblib.dump(pipeline, model_path)
    cols_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    logger.info("Model saved       → %s", model_path)
    logger.info("Feature cols saved → %s", cols_path)
    return model_path


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train and serialize the final NHL model")
    parser.add_argument(
        "--model", default="random_forest",
        choices=["random_forest", "logistic_regression"],
        help="Which model to train (default: random_forest)",
    )
    args = parser.parse_args()

    path = train_and_save(model_name=args.model)
    print(f"\nModel serialized: {path}")
    print(f"Feature cols:     {path.parent / (path.stem + '_feature_cols.json')}")
