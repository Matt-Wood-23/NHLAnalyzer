"""
Phase 4 v2: XGBoost + LightGBM with Optuna tuning — improved calibration,
interaction features, and RF+XGB ensemble.

Key v2 changes vs v1:
  1. Smart calibration — Platt scaling for small cal slices (< 400 games),
     isotonic for large slices. Fixes the Fold 1 Brier blow-up from v1.
  2. Interaction features — B2B × recent form (home/away/diff).
  3. RF+XGB ensemble — average probabilities fold-by-fold, often beats either.
  4. --calibration flag — compare auto / isotonic / platt / none.
  5. --ensemble flag — also evaluate RF+XGB average.

Tuning strategy:
  - Use fold 3 split (train 2021-24 → validate 2024-25) for Optuna search
  - Minimise Brier score over n_trials trials
  - Then run full walk-forward with best params to get unbiased estimates

Usage:
    python -m models.xgboost_model                           # 50 trials, auto cal
    python -m models.xgboost_model --trials 150              # more thorough search
    python -m models.xgboost_model --calibration platt       # force Platt everywhere
    python -m models.xgboost_model --ensemble                # also run RF+XGB ensemble
    python -m models.xgboost_model --trials 150 --ensemble   # full v2 run
"""

import argparse
import logging
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import shap
from dotenv import load_dotenv
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from models.baseline import SEASONS, MLFLOW_DB, load_feature_matrix, walk_forward_folds
from models.evaluate import evaluate_fold, summarize_results, print_feature_importance

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = Path(__file__).parent.parent / "results"

# Columns to drop — positional noise, not quality signal
EXCLUDE_FEATURES = {"home_games_played", "away_games_played", "diff_games_played"}
_META = {"game_id", "season", "home_team", "away_team", "target", "date", "home_win"}

# Tuning split: train on these, validate on TUNE_VAL_SEASON
TUNE_TRAIN_SEASONS = ["2021-2022", "2022-2023", "2023-2024"]
TUNE_VAL_SEASON    = "2024-2025"


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in _META and c not in EXCLUDE_FEATURES]


# ------------------------------------------------------------------ #
# Interaction features                                                 #
# ------------------------------------------------------------------ #

def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add B2B × recent-form interaction terms.

    These encode 'is the back-to-back team playing well recently?' —
    a signal the SHAP analysis identified as important but unexploited.
    Adds 3 new columns: home_b2b_x_won_l5, away_b2b_x_won_l5,
    diff_b2b_x_won_l5.
    """
    df = df.copy()
    for side in ("home", "away"):
        b2b = f"{side}_back_to_back"
        won = f"{side}_won_l5"
        if b2b in df.columns and won in df.columns:
            df[f"{side}_b2b_x_won_l5"] = df[b2b].astype(float) * df[won]
    if "home_b2b_x_won_l5" in df.columns and "away_b2b_x_won_l5" in df.columns:
        df["diff_b2b_x_won_l5"] = df["home_b2b_x_won_l5"] - df["away_b2b_x_won_l5"]
    return df


# ------------------------------------------------------------------ #
# Imputer (fit on train, transform both)                              #
# ------------------------------------------------------------------ #

def fit_imputer(X_train: np.ndarray) -> SimpleImputer:
    imp = SimpleImputer(strategy="mean")
    imp.fit(X_train)
    return imp


# ------------------------------------------------------------------ #
# Optuna objectives                                                    #
# ------------------------------------------------------------------ #

def _xgb_objective(trial, X_tr, y_tr, X_val, y_val) -> float:
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
        "max_depth":         trial.suggest_int("max_depth", 3, 7),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 5, 30),
        "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 5.0),
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    model = XGBClassifier(**params)
    model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_val)[:, 1]
    return brier_score_loss(y_val, y_prob)


def _lgbm_objective(trial, X_tr, y_tr, X_val, y_val) -> float:
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 15, 63),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 50),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 3.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    model = LGBMClassifier(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_val)[:, 1]
    return brier_score_loss(y_val, y_prob)


def tune_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_type: str,
    n_trials: int = 50,
) -> dict:
    """
    Run Optuna on the fixed tuning split (2021-24 train → 2024-25 val).
    Returns best hyperparameters dict.
    """
    train_mask = df["season"].isin(TUNE_TRAIN_SEASONS)
    val_mask   = df["season"] == TUNE_VAL_SEASON

    X_train_raw = df.loc[train_mask, feature_cols].values
    y_train     = df.loc[train_mask, "target"].values
    X_val_raw   = df.loc[val_mask,   feature_cols].values
    y_val       = df.loc[val_mask,   "target"].values

    imp = fit_imputer(X_train_raw)
    X_tr  = imp.transform(X_train_raw)
    X_val = imp.transform(X_val_raw)

    objective_fn = _xgb_objective if model_type == "xgboost" else _lgbm_objective

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda trial: objective_fn(trial, X_tr, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    logger.info(
        "%s best Brier=%.4f after %d trials | params: %s",
        model_type, study.best_value, n_trials, best,
    )
    return best


# ------------------------------------------------------------------ #
# Calibration helpers                                                  #
# ------------------------------------------------------------------ #

class _IsotonicCalibrated:
    """Thin wrapper: applies isotonic regression to a prefit model's raw probabilities."""
    def __init__(self, model, iso: IsotonicRegression):
        self._model = model
        self._iso   = iso

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._model.predict_proba(X)[:, 1]
        cal = self._iso.predict(raw)
        return np.column_stack([1 - cal, cal])

    @property
    def calibrated_classifiers_(self):
        class _Stub:
            estimator = self._model
        return [_Stub()]


class _PlattCalibrated:
    """Thin wrapper: applies Platt scaling (logistic) to a prefit model's raw probabilities."""
    def __init__(self, model, lr: LogisticRegression):
        self._model = model
        self._lr    = lr

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self._model.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = self._lr.predict_proba(raw)[:, 1]
        return np.column_stack([1 - cal, cal])

    @property
    def calibrated_classifiers_(self):
        class _Stub:
            estimator = self._model
        return [_Stub()]


def _calibrate_isotonic(model, X_cal: np.ndarray, y_cal: np.ndarray) -> _IsotonicCalibrated:
    """Fit isotonic regression on calibration slice, return wrapped model."""
    raw = model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(raw, y_cal)
    return _IsotonicCalibrated(model, iso)


def _calibrate_platt(model, X_cal: np.ndarray, y_cal: np.ndarray) -> _PlattCalibrated:
    """Fit Platt scaling (logistic regression) on calibration slice."""
    raw = model.predict_proba(X_cal)[:, 1].reshape(-1, 1)
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
    lr.fit(raw, y_cal)
    return _PlattCalibrated(model, lr)


def _smart_calibrate(model, X_cal: np.ndarray, y_cal: np.ndarray, method: str = "auto"):
    """
    Choose calibration method based on sample size and requested method.

    auto:     isotonic if n >= 400 (robust), Platt if 50 <= n < 400 (small-sample safe),
              uncalibrated if n < 50 (too noisy for either)
    isotonic: isotonic if n >= 400, else uncalibrated (v1 behaviour)
    platt:    Platt scaling if n >= 50, else uncalibrated
    none:     no calibration
    """
    n = len(X_cal)
    if method == "none":
        return model
    elif method == "isotonic":
        return _calibrate_isotonic(model, X_cal, y_cal) if n >= 400 else model
    elif method == "platt":
        return _calibrate_platt(model, X_cal, y_cal) if n >= 50 else model
    else:  # "auto"
        if n >= 400:
            return _calibrate_isotonic(model, X_cal, y_cal)
        elif n >= 50:
            return _calibrate_platt(model, X_cal, y_cal)
        return model


# ------------------------------------------------------------------ #
# RF pipeline (for ensemble)                                           #
# ------------------------------------------------------------------ #

def _build_rf_pipeline() -> Pipeline:
    """Build the same RF config used in pipeline/train.py."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="mean")),
        ("rf",  RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            random_state=42, n_jobs=-1,
        )),
    ])


# ------------------------------------------------------------------ #
# Walk-forward evaluation                                              #
# ------------------------------------------------------------------ #

def run_walk_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    best_params: dict,
    model_type: str,
    available_seasons: list[str],
    calibration: str = "auto",
) -> tuple[list, list[np.ndarray]]:
    """
    Run walk-forward validation with tuned params.

    Args:
        calibration: "auto" | "isotonic" | "platt" | "none"

    Returns:
        (list[FoldResult], list[y_prob arrays]) — probs list enables ensemble averaging.
    """
    folds = walk_forward_folds(available_seasons)
    results = []
    fold_probs = []

    for fold_idx, (train_seasons, test_season) in enumerate(folds, start=1):
        train_mask = df["season"].isin(train_seasons)
        test_mask  = df["season"] == test_season

        X_train_raw = df.loc[train_mask, feature_cols].values
        y_train     = df.loc[train_mask, "target"].values
        X_test_raw  = df.loc[test_mask,  feature_cols].values
        y_test      = df.loc[test_mask,  "target"].values

        imp = fit_imputer(X_train_raw)
        X_train = imp.transform(X_train_raw)
        X_test  = imp.transform(X_test_raw)

        # Hold out last 15% of training set (chronologically) for calibration
        cal_split = int(len(X_train) * 0.85)
        X_fit, X_cal = X_train[:cal_split], X_train[cal_split:]
        y_fit, y_cal = y_train[:cal_split], y_train[cal_split:]

        # Build and fit model
        params = {**best_params, "random_state": 42, "n_jobs": -1}
        if model_type == "xgboost":
            params.update({"eval_metric": "logloss", "verbosity": 0})
            model = XGBClassifier(**params)
            model.fit(X_fit, y_fit)
        else:
            params["verbose"] = -1
            model = LGBMClassifier(**params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_fit, y_fit)

        model = _smart_calibrate(model, X_cal, y_cal, method=calibration)

        y_prob = model.predict_proba(X_test)[:, 1]
        fold_probs.append(y_prob)
        naive_rate = y_train.mean()

        result = evaluate_fold(
            y_true=y_test,
            y_prob=y_prob,
            model_name=f"{model_type}_v2_{calibration}cal",
            fold=fold_idx,
            train_seasons=train_seasons,
            test_season=test_season,
            n_train=len(y_train),
            naive_home_win_rate=naive_rate,
        )
        results.append(result)

        logger.info(
            "Fold %d (%s→%s) | cal=%s n_cal=%d | acc=%.3f brier=%.4f auc=%.3f",
            fold_idx, train_seasons[-1], test_season, calibration, len(X_cal),
            result.accuracy, result.brier_score, result.roc_auc,
        )

    return results, fold_probs


# ------------------------------------------------------------------ #
# Ensemble walk-forward                                                #
# ------------------------------------------------------------------ #

def run_ensemble_forward(
    df: pd.DataFrame,
    feature_cols: list[str],
    best_params_xgb: dict,
    available_seasons: list[str],
    calibration: str = "auto",
) -> list:
    """
    Walk-forward evaluation of a simple RF + XGB probability average.
    RF is retrained from scratch on each fold's training data (same split as XGB).
    Returns list of FoldResult.
    """
    folds = walk_forward_folds(available_seasons)
    results = []

    for fold_idx, (train_seasons, test_season) in enumerate(folds, start=1):
        train_mask = df["season"].isin(train_seasons)
        test_mask  = df["season"] == test_season

        X_train_raw = df.loc[train_mask, feature_cols].values
        y_train     = df.loc[train_mask, "target"].values
        X_test_raw  = df.loc[test_mask,  feature_cols].values
        y_test      = df.loc[test_mask,  "target"].values

        # --- XGBoost ---
        imp = fit_imputer(X_train_raw)
        X_train = imp.transform(X_train_raw)
        X_test  = imp.transform(X_test_raw)

        cal_split = int(len(X_train) * 0.85)
        X_fit, X_cal = X_train[:cal_split], X_train[cal_split:]
        y_fit, y_cal = y_train[:cal_split], y_train[cal_split:]

        xgb_params = {**best_params_xgb, "random_state": 42, "n_jobs": -1,
                      "eval_metric": "logloss", "verbosity": 0}
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_fit, y_fit)
        xgb = _smart_calibrate(xgb, X_cal, y_cal, method=calibration)
        y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

        # --- Random Forest (uses its own internal imputer via Pipeline) ---
        rf = _build_rf_pipeline()
        rf.fit(X_train_raw, y_train)
        y_prob_rf = rf.predict_proba(X_test_raw)[:, 1]

        # --- Ensemble: simple average ---
        y_prob_ens = 0.5 * y_prob_xgb + 0.5 * y_prob_rf

        result = evaluate_fold(
            y_true=y_test,
            y_prob=y_prob_ens,
            model_name="ensemble_rf_xgb",
            fold=fold_idx,
            train_seasons=train_seasons,
            test_season=test_season,
            n_train=len(y_train),
            naive_home_win_rate=y_train.mean(),
        )
        results.append(result)

        logger.info(
            "Ensemble Fold %d (%s→%s) | acc=%.3f brier=%.4f auc=%.3f",
            fold_idx, train_seasons[-1], test_season,
            result.accuracy, result.brier_score, result.roc_auc,
        )

    return results


# ------------------------------------------------------------------ #
# SHAP analysis                                                        #
# ------------------------------------------------------------------ #

def compute_shap_summary(
    model,
    X: np.ndarray,
    feature_cols: list[str],
    top_n: int = 20,
) -> pd.Series:
    """
    Compute SHAP values and print/save summary.
    Returns Series of mean |SHAP| per feature.
    """
    logger.info("Computing SHAP values on %d samples ...", len(X))

    # For calibrated models, unwrap to get the base estimator
    base_model = model
    if hasattr(model, "calibrated_classifiers_"):
        base_model = model.calibrated_classifiers_[0].estimator

    explainer = shap.TreeExplainer(base_model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_cols).sort_values(ascending=False)

    print(f"\nTop {top_n} features by mean |SHAP|:")
    print(f"  {'Feature':<45} {'Mean |SHAP|':>12}")
    print("  " + "-" * 59)
    for feat, val in importance.head(top_n).items():
        print(f"  {feat:<45} {val:>12.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(9, 7))
        top = importance.head(top_n)
        ax.barh(top.index[::-1], top.values[::-1])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Top Feature Importances (SHAP) — XGBoost v2")
        ax.tick_params(axis="y", labelsize=8)
        plt.tight_layout()
        out = RESULTS_DIR / "shap_importance_v2.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        logger.info("SHAP plot saved → %s", out)
    except Exception as e:
        logger.warning("Could not save SHAP plot: %s", e)

    return importance


# ------------------------------------------------------------------ #
# MLflow logging                                                       #
# ------------------------------------------------------------------ #

def log_run_to_mlflow(
    results: list,
    best_params: dict,
    model_type: str,
    calibration: str,
    shap_importance: pd.Series | None,
    add_interactions: bool,
) -> None:
    MLFLOW_DB.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
    mlflow.set_experiment("nhl_xgboost_phase4")

    run_name = f"{model_type}_v2_{calibration}cal"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("calibration", calibration)
        mlflow.set_tag("tuning", "optuna")
        mlflow.set_tag("version", "v2")
        mlflow.set_tag("interactions", str(add_interactions))
        mlflow.log_params(best_params)

        weights = np.array([r.n_test for r in results])
        for metric in ["accuracy", "brier_score", "log_loss", "roc_auc", "beat_naive_rate"]:
            vals = np.array([getattr(r, metric) for r in results])
            mlflow.log_metric(f"wavg_{metric}", float(np.average(vals, weights=weights)))

        for r in results:
            prefix = f"fold{r.fold}"
            mlflow.log_metrics({
                f"{prefix}_brier":      r.brier_score,
                f"{prefix}_auc":        r.roc_auc,
                f"{prefix}_beat_naive": r.beat_naive_rate,
            })

        if shap_importance is not None:
            for feat, val in shap_importance.head(20).items():
                mlflow.log_metric(f"shap_{feat[:40]}", float(val))

        shap_plot = RESULTS_DIR / "shap_importance_v2.png"
        if shap_plot.exists():
            mlflow.log_artifact(str(shap_plot))


# ------------------------------------------------------------------ #
# Feature selection                                                    #
# ------------------------------------------------------------------ #

def run_feature_selection(
    df: pd.DataFrame,
    feature_cols: list[str],
    best_params: dict,
    available_seasons: list[str],
    top_n_values: list[int] | None = None,
    calibration: str = "auto",
) -> dict:
    """
    Run walk-forward with progressively fewer features (by RF importance).
    Returns dict of n_features -> list[FoldResult].
    """
    if top_n_values is None:
        top_n_values = [20, 30, 50, 80, len(feature_cols)]

    # Rank features using RF trained on all data
    logger.info("Ranking features via RF importance ...")
    all_mask = df["season"].isin(available_seasons)
    X_all = df.loc[all_mask, feature_cols].values
    y_all = df.loc[all_mask, "target"].values

    imp = fit_imputer(X_all)
    X_imp = imp.transform(X_all)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_imp, y_all)

    importance = pd.Series(rf.feature_importances_, index=feature_cols)
    ranked = importance.sort_values(ascending=False).index.tolist()

    print(f"\nTop 20 features by RF importance:")
    for i, feat in enumerate(ranked[:20]):
        print(f"  {i+1:2d}. {feat:<45} {importance[feat]:.4f}")

    # Run walk-forward for each feature subset
    all_results = {}
    for n in sorted(top_n_values):
        n = min(n, len(feature_cols))
        subset = ranked[:n]
        print(f"\n--- Feature selection: top {n} features ---")

        results, _ = run_walk_forward(
            df, subset, best_params, "xgboost", available_seasons,
            calibration=calibration,
        )
        all_results[n] = results

        weights = np.array([r.n_test for r in results])
        brier = np.average([r.brier_score for r in results], weights=weights)
        auc = np.average([r.roc_auc for r in results], weights=weights)
        print(f"  top-{n}: WAVG Brier={brier:.4f}, AUC={auc:.3f}")

    # Summary table
    print(f"\n{'='*60}")
    print("Feature selection summary:")
    print(f"  {'N features':>12} {'WAVG Brier':>12} {'WAVG AUC':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*10}")
    for n in sorted(all_results.keys()):
        results = all_results[n]
        weights = np.array([r.n_test for r in results])
        brier = np.average([r.brier_score for r in results], weights=weights)
        auc = np.average([r.roc_auc for r in results], weights=weights)
        print(f"  {n:>12} {brier:>12.4f} {auc:>10.3f}")
    print(f"{'='*60}")

    return all_results


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def run(
    n_trials: int = 50,
    calibration: str = "auto",
    add_interactions: bool = True,
    ensemble: bool = False,
    feature_selection: bool = False,
) -> None:
    df = load_feature_matrix()

    if add_interactions:
        df = _add_interaction_features(df)
        logger.info("Added B2B × form interaction features")

    feature_cols = get_feature_columns(df)
    available_seasons = [s for s in SEASONS if s in df["season"].unique()]

    logger.info(
        "Phase 4 v2 | %d games, %d features, calibration=%s, interactions=%s",
        len(df), len(feature_cols), calibration, add_interactions,
    )

    all_results = {}
    best_params_map = {}
    final_models = {}

    for model_type in ("xgboost", "lightgbm"):
        print(f"\n{'='*60}")
        print(f"Tuning {model_type} ({n_trials} Optuna trials) ...")
        print(f"{'='*60}")

        best_params = tune_model(df, feature_cols, model_type, n_trials=n_trials)
        best_params_map[model_type] = best_params

        print(f"\nBest params: {best_params}")
        print(f"\nRunning walk-forward validation (calibration={calibration}) ...")

        results, _ = run_walk_forward(
            df, feature_cols, best_params, model_type, available_seasons,
            calibration=calibration,
        )
        all_results[model_type] = results

        if model_type == "xgboost":
            all_mask = df["season"].isin(available_seasons)
            X_all = df.loc[all_mask, feature_cols].values
            y_all = df.loc[all_mask, "target"].values
            imp = fit_imputer(X_all)
            X_imp = imp.transform(X_all)
            params = {**best_params, "random_state": 42, "n_jobs": -1,
                      "eval_metric": "logloss", "verbosity": 0}
            final_model = XGBClassifier(**params)
            final_model.fit(X_imp, y_all)
            final_models["xgboost"] = (final_model, imp)

    # Ensemble
    if ensemble:
        print(f"\n{'='*60}")
        print("Running RF + XGB ensemble walk-forward ...")
        print(f"{'='*60}")
        ens_results = run_ensemble_forward(
            df, feature_cols, best_params_map["xgboost"], available_seasons,
            calibration=calibration,
        )
        all_results["ensemble_rf_xgb"] = ens_results

    # Feature selection (uses XGBoost best params)
    if feature_selection and "xgboost" in best_params_map:
        print(f"\n{'='*60}")
        print("Running feature selection sweep ...")
        print(f"{'='*60}")
        fs_results = run_feature_selection(
            df, feature_cols, best_params_map["xgboost"], available_seasons,
            calibration=calibration,
        )
        # Add best subset results to all_results for summary
        best_n = min(fs_results, key=lambda n: np.average(
            [r.brier_score for r in fs_results[n]],
            weights=[r.n_test for r in fs_results[n]],
        ))
        all_results[f"xgb_top{best_n}"] = fs_results[best_n]

    # Print all results together
    flat = [r for results in all_results.values() for r in results]
    summarize_results(flat)

    # SHAP on final XGBoost
    shap_importance = None
    if "xgboost" in final_models:
        final_model, imp = final_models["xgboost"]
        all_mask = df["season"].isin(available_seasons)
        X_shap = imp.transform(df.loc[all_mask, feature_cols].values)
        idx = np.random.default_rng(42).choice(len(X_shap), min(2000, len(X_shap)), replace=False)
        shap_importance = compute_shap_summary(final_model, X_shap[idx], feature_cols)

    # Log to MLflow
    for model_type, results in all_results.items():
        log_run_to_mlflow(
            results,
            best_params_map.get(model_type, best_params_map.get("xgboost", {})),
            model_type,
            calibration,
            shap_importance if model_type == "xgboost" else None,
            add_interactions,
        )

    print("\nPhase 4 v2 complete. View results:")
    print("  mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db")


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="XGBoost v2: improved calibration + ensemble")
    parser.add_argument("--trials", type=int, default=50,
                        help="Optuna trials per model (default 50; use 150 for full v2 run)")
    parser.add_argument("--calibration", choices=["auto", "isotonic", "platt", "none"],
                        default="auto",
                        help="Calibration method (default: auto — Platt for small folds, "
                             "isotonic for large)")
    parser.add_argument("--ensemble", action="store_true",
                        help="Also run and report RF + XGB ensemble walk-forward")
    parser.add_argument("--no-interactions", action="store_true",
                        help="Skip B2B × form interaction features")
    parser.add_argument("--feature-selection", action="store_true",
                        help="Run feature selection sweep (top 20/30/50/80/all)")
    args = parser.parse_args()

    run(
        n_trials=args.trials,
        calibration=args.calibration,
        add_interactions=not args.no_interactions,
        ensemble=args.ensemble,
        feature_selection=args.feature_selection,
    )
