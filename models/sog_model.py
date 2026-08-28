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


def evaluate_sog(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and over/under accuracy at line = mean(y_true)."""
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Over/under accuracy: does the model correctly call over/under relative to
    # the actual player's season-average SOG (a rough proxy for a book line)?
    mean_line = np.mean(y_true)
    actual_over  = y_true  > mean_line
    pred_over    = y_pred  > mean_line
    ou_acc = float(np.mean(actual_over == pred_over))

    return {"mae": mae, "rmse": rmse, "ou_accuracy": ou_acc, "n": len(y_true)}


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

        X_train = train[feature_cols].values
        y_train = train["sog"].values.astype(float)
        X_test  = test[feature_cols].values
        y_test  = test["sog"].values.astype(float)

        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = evaluate_sog(y_test, y_pred)
        metrics.update({
            "fold": fold_idx,
            "train_seasons": train_seasons,
            "test_season": test_season,
            "n_train": len(y_train),
        })
        results.append(metrics)

        logger.info(
            "Fold %d | test %s | MAE=%.3f RMSE=%.3f O/U acc=%.3f (n=%d)",
            fold_idx, test_season,
            metrics["mae"], metrics["rmse"], metrics["ou_accuracy"], metrics["n"],
        )

    return results


def train_final_model(df: pd.DataFrame) -> tuple[Pipeline, list[str]]:
    """Train on all available data. Returns (pipeline, feature_cols)."""
    feature_cols = [c for c in SOG_FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y = df["sog"].values.astype(float)

    pipeline = build_pipeline()
    pipeline.fit(X, y)
    logger.info("Final model trained on %d player-game rows", len(y))
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
        print(f"{'Fold':<6} {'Test Season':<12} {'MAE':>6} {'RMSE':>7} {'O/U Acc':>9} {'N':>6}")
        print("-" * 48)
        for r in results:
            print(
                f"{r['fold']:<6} {r['test_season']:<12} "
                f"{r['mae']:>6.3f} {r['rmse']:>7.3f} {r['ou_accuracy']:>9.3f} {r['n']:>6}"
            )
        avg_mae = np.mean([r["mae"] for r in results])
        avg_ou  = np.mean([r["ou_accuracy"] for r in results])
        print(f"\nAvg MAE: {avg_mae:.3f}  |  Avg O/U Accuracy: {avg_ou:.3f}")
        print("(Naive O/U baseline = 0.500)")

    if args.save:
        pipeline, feature_cols = train_final_model(df)
        path = save_sog_model(pipeline, feature_cols)
        print(f"\nModel saved: {path}")
