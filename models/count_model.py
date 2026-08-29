"""
Shared machinery for the Poisson-family props models (SOG, goalie saves).

Everything here exists because of one finding: NHL counting rates are
drifting down hard.  Combined shots per game fell from 63.2 in 2021-22 to
55.6 in 2025-26, and starter saves fell 28.4 to 24.8 over the same span.
Any model trained on an expanding window inherits the high-volume seasons
and over-projects, with the error growing as the gap widens.

Three corrections live here, applied in this order:

  1. ``season_sample_weights`` -- exponential recency weighting, so a
     five-year-old season does not set the intercept for today's league.
  2. ``calibration_factor`` -- a multiplicative correction fit on the newest
     training season, folded into the log-link intercept by
     ``apply_calibration`` where it is exact.
  3. ``in_season_factor`` -- an optional serve-time correction against the
     current season's completed games.  Weighting and calibration can only
     reach as far as the newest *training* season; the test season is a
     further step down, which is why saves still ran +0.68 high with both
     applied.  Recalibrating on games already played closes most of the
     rest and leaks nothing.

Scoring lives here too, and deliberately does not report accuracy against
a coin flip.  At a 2.5 shots line ~72% of player-games go under, so the
majority class is the bar; the earlier SOG model reported 63.6% against
0.500 and overstated itself roughly threefold.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Exponential decay applied to training rows by season age, in seasons.
# 0.5 means a season-old row carries 1/4 the weight of a current one.
RECENCY_HALF_LIFE_SEASONS = 0.5

# Below this many completed starts, an in-season factor is too noisy to trust.
MIN_IN_SEASON_SAMPLE = 100


def build_pipeline(alpha: float = 0.1, max_iter: int = 300) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler",  StandardScaler()),
        ("model",   PoissonRegressor(alpha=alpha, max_iter=max_iter)),
    ])


# ---------------------------------------------------------------------------
# Drift corrections
# ---------------------------------------------------------------------------

def season_sample_weights(
    seasons: pd.Series,
    ordered_seasons: list[str],
    half_life: float = RECENCY_HALF_LIFE_SEASONS,
) -> np.ndarray:
    """Exponential recency weights for training rows, by season age.

    The most recent training season gets weight 1.0 and each season back
    halves every ``half_life`` seasons.
    """
    if half_life is None or half_life <= 0:
        return np.ones(len(seasons), dtype=float)
    newest = len(ordered_seasons) - 1
    age = seasons.map({s: newest - i for i, s in enumerate(ordered_seasons)})
    return 0.5 ** (age.to_numpy(dtype=float) / half_life)


def calibration_factor(
    pipeline: Pipeline,
    train: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> float:
    """Multiplicative correction so the fit matches the newest training season.

    A Poisson MLE reproduces the mean of the *whole* training set, which
    straddles seasons of different volume.  Comparing predicted to actual on
    the newest training season alone recovers how far the fit sits above
    where the league now is.
    """
    newest = sorted(train["season"].unique())[-1]
    recent = train[train["season"] == newest]
    predicted = pipeline.predict(recent[feature_cols].values).mean()
    if not np.isfinite(predicted) or predicted <= 0:
        logger.warning("Calibration skipped: non-positive mean prediction")
        return 1.0
    factor = float(recent[target].mean() / predicted)
    logger.info("Calibration on %s: x%.4f", newest, factor)
    return factor


def apply_calibration(pipeline: Pipeline, factor: float) -> Pipeline:
    """Fold a multiplicative factor into the model's intercept, in place.

    The link is log, so ``exp(b0 + log(f) + bx) == f * exp(b0 + bx)``: the
    correction is exact as an intercept shift.  Doing it here rather than at
    the call site means a saved pickle is already calibrated and every
    consumer gets it with no change.
    """
    if factor != 1.0 and np.isfinite(factor) and factor > 0:
        pipeline.named_steps["model"].intercept_ += float(np.log(factor))
    return pipeline


def in_season_factor(
    pipeline: Pipeline,
    played: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    min_sample: int = MIN_IN_SEASON_SAMPLE,
) -> float:
    """Serve-time correction against the current season's completed games.

    Uses only games that have already finished, so there is nothing to leak.
    Returns 1.0 when the sample is too small to be worth trusting, which is
    the case for roughly the first three weeks of a season.
    """
    if played is None or len(played) < min_sample:
        logger.info(
            "In-season calibration skipped: %d completed rows (need %d)",
            0 if played is None else len(played), min_sample,
        )
        return 1.0
    predicted = pipeline.predict(played[feature_cols].values).mean()
    if not np.isfinite(predicted) or predicted <= 0:
        return 1.0
    factor = float(played[target].mean() / predicted)
    logger.info("In-season calibration on %d rows: x%.4f", len(played), factor)
    return factor


def fit_pipeline(
    train: pd.DataFrame,
    feature_cols: list[str],
    target: str = "sog",
    half_life: float = RECENCY_HALF_LIFE_SEASONS,
    calibrate: bool = True,
    alpha: float = 0.1,
) -> Pipeline:
    """Fit a recency-weighted, drift-calibrated count model on ``train``."""
    ordered = sorted(train["season"].unique())
    weights = season_sample_weights(train["season"], ordered, half_life)

    pipeline = build_pipeline(alpha=alpha)
    pipeline.fit(
        train[feature_cols].values,
        train[target].values.astype(float),
        model__sample_weight=weights,
    )
    if calibrate:
        apply_calibration(
            pipeline, calibration_factor(pipeline, train, feature_cols, target),
        )
    return pipeline


# ---------------------------------------------------------------------------
# Turning a rate into the number a bet needs
# ---------------------------------------------------------------------------

def estimate_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Method-of-moments dispersion for ``var = mu + alpha * mu**2``.

    Poisson assumes alpha = 0.  Shots on goal are close enough; goalie saves
    are not -- their variance runs ~1.9x the mean, so Poisson tails come out
    too tight and P(over) ends up overconfident.
    """
    mu = np.asarray(mu, dtype=float)
    y = np.asarray(y, dtype=float)
    denominator = np.mean(mu ** 2)
    if denominator <= 0:
        return 0.0
    return max(float(np.mean((y - mu) ** 2 - mu) / denominator), 0.0)


def over_probability(mu: np.ndarray, line: float, dispersion: float = 0.0) -> np.ndarray:
    """P(count > line) at a half-integer ``line``.

    Half-integers mean no push to handle: P = 1 - CDF(floor(line)).  With
    ``dispersion`` above zero this is negative binomial rather than Poisson,
    matching a target whose variance exceeds its mean.
    """
    mu = np.asarray(mu, dtype=float)
    k = np.floor(line)
    if dispersion and dispersion > 0:
        n = 1.0 / dispersion
        return 1.0 - nbinom.cdf(k, n, n / (n + mu))
    return 1.0 - poisson.cdf(k, mu)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def evaluate_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lines: list[float],
    train_target: Optional[np.ndarray] = None,
    dispersion: float = 0.0,
) -> dict:
    """Score projections at the lines books post, against honest baselines.

    For each line this reports:
      ``brier``       Brier score of the model's P(over).
      ``brier_base``  Brier score of climatology -- the training base rate,
                      the same no-skill reference ``/history`` uses.
      ``accuracy``    Share of correct over/under calls.
      ``acc_base``    The majority class, which is what accuracy must beat.
                      At a 2.5 shots line that is ~0.72, not 0.500.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    reference = y_true if train_target is None else np.asarray(train_target, dtype=float)

    metrics = {
        "mae":  float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "bias": float(np.mean(y_pred) - np.mean(y_true)),
        "n":    len(y_true),
        "dispersion": float(dispersion),
        "lines": {},
    }

    for line in lines:
        actual = (y_true > line).astype(float)
        prob   = over_probability(y_pred, line, dispersion)
        rate   = float(np.mean(reference > line))
        metrics["lines"][line] = {
            "brier":      float(np.mean((prob - actual) ** 2)),
            "brier_base": float(np.mean((rate - actual) ** 2)),
            "accuracy":   float(np.mean((prob > 0.5) == (actual == 1))),
            "acc_base":   float(max(actual.mean(), 1 - actual.mean())),
            "over_rate":  float(actual.mean()),
        }

    return metrics


def print_line_table(results: list[dict], lines: list[float]) -> None:
    """Render the per-line scoring table shared by both props models."""
    print("\n=== Over/Under by Line ===")
    print("Brier scores P(over) against climatology; accuracy against the")
    print("majority class, which is the baseline a coin flip is not.")
    print()
    print(f"{'Line':>6} {'Brier':>8} {'no-skill':>9} {'skill':>7} "
          f"{'Acc':>7} {'majority':>9} {'Over%':>7}")
    print("-" * 59)
    for line in lines:
        per   = [r["lines"][line] for r in results]
        brier = np.mean([x["brier"] for x in per])
        base  = np.mean([x["brier_base"] for x in per])
        acc   = np.mean([x["accuracy"] for x in per])
        major = np.mean([x["acc_base"] for x in per])
        over  = np.mean([x["over_rate"] for x in per])
        skill = 1 - brier / base if base else float("nan")
        print(f"{line:>6.1f} {brier:>8.4f} {base:>9.4f} {skill:>6.1%} "
              f"{acc:>7.1%} {major:>9.1%} {over:>7.1%}")
    print("\nBrier skill > 0 means the projections beat always quoting the")
    print("base rate. Accuracy below the majority column means a constant")
    print("'under' would have called more games correctly.")
