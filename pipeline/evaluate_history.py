"""
Backfill actual outcomes into prediction history and compute accuracy metrics.

Reads prediction_history.parquet, looks up actual game results from the
feature matrix, and computes accuracy, Brier score, calibration, and a
breakdown by pick strength.

Everything here is shared by the terminal report and the Discord ``/history``
command, so the two cannot report different numbers for the same data.

Usage:
    python -m pipeline.evaluate_history                  # current season
    python -m pipeline.evaluate_history --season all     # every season
    python -m pipeline.evaluate_history --season 2025-2026
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.season import current_season, season_label

logger = logging.getLogger(__name__)

PARQUET_DIR  = Path(__file__).parent.parent / "data" / "parquet"
HISTORY_DIR  = Path(__file__).parent.parent / "data" / "predictions"
HISTORY_PATH = HISTORY_DIR / "prediction_history.parquet"

# Pick-strength tiers, highest threshold first.  Applied to the *favoured*
# side's probability.  The Discord bot labels live picks with these same
# tiers, so the history breakdown below is a direct check on whether a
# "Strong Pick" really is stronger.
CONFIDENCE_TIERS: list[tuple[float, str]] = [
    (0.60, "Strong Pick"),
    (0.55, "Lean"),
    (0.00, "Toss-Up"),
]

# Calibration buckets over the home-win probability.
CALIBRATION_BINS = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 1.0]
CALIBRATION_LABELS = ["<40%", "40-45%", "45-50%", "50-55%", "55-60%", ">60%"]


def confidence_label(prob_favourite: float) -> str:
    """Pick-strength label for the favoured side's probability."""
    for threshold, label in CONFIDENCE_TIERS:
        if prob_favourite >= threshold:
            return label
    return CONFIDENCE_TIERS[-1][1]


# ---------------------------------------------------------------------------
# Loading and scoring
# ---------------------------------------------------------------------------

def add_season(hist: pd.DataFrame) -> pd.DataFrame:
    """Tag each prediction with the season it belongs to.

    The first four digits of an NHL game_id are the season's start year, so
    this works retroactively on rows logged before the column existed.
    Without it, a lifetime accuracy number silently pools seasons together
    and a good year hides a bad one.
    """
    hist = hist.copy()
    years = pd.to_numeric(hist["game_id"].astype(str).str[:4], errors="coerce")
    hist["season"] = years.map(
        lambda y: season_label(int(y)) if pd.notna(y) else None
    )
    return hist


def attach_outcomes(
    hist: pd.DataFrame,
    outcomes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join predictions to actual results and score them.  Pure — writes nothing.

    Args:
        hist: prediction history.
        outcomes: game_id -> target frame.  Read from the feature matrix when
            omitted.

    Returns the history with actual_home_win, correct, brier and season.

    Outcomes already stored in the log are kept.  A played game's result does
    not change, and the feature matrix is not always present — it is gitignored
    and rebuilt by the daily pipeline — so re-scoring must never be able to
    erase results that were recorded earlier.
    """
    hist = hist.copy()
    if hist.empty:
        for col, dtype in (
            ("actual_home_win", "float64"), ("brier", "float64"),
            ("correct", "boolean"), ("season", "object"),
        ):
            if col not in hist.columns:
                hist[col] = pd.Series(dtype=dtype)
        return hist

    stored = pd.to_numeric(
        hist.get("actual_home_win", pd.Series(np.nan, index=hist.index)),
        errors="coerce",
    )
    # `season` is dropped alongside the scores and re-derived below, so the
    # derived columns always land in the same order.  Otherwise re-scoring
    # would rewrite the parquet with a reshuffled schema every time.
    hist = hist.drop(
        columns=["actual_home_win", "correct", "brier", "season"], errors="ignore",
    )

    if outcomes is None:
        fm_path = PARQUET_DIR / "feature_matrix.parquet"
        if fm_path.exists():
            outcomes = pd.read_parquet(fm_path, columns=["game_id", "target"])
        else:
            logger.warning(
                "No feature matrix at %s — keeping %d previously scored outcome(s)",
                fm_path, int(stored.notna().sum()),
            )

    if outcomes is not None:
        looked_up = (
            outcomes.rename(columns={"target": "actual_home_win"})
            [["game_id", "actual_home_win"]]
            .drop_duplicates("game_id")
        )
        merged = hist[["game_id"]].merge(looked_up, on="game_id", how="left")
        fresh = pd.to_numeric(merged["actual_home_win"], errors="coerce")
        fresh.index = hist.index
        hist["actual_home_win"] = fresh.fillna(stored)
    else:
        hist["actual_home_win"] = stored

    actual = pd.to_numeric(hist["actual_home_win"], errors="coerce")
    predicted_home = hist["prob_home_win"] >= 0.5

    # Typed columns throughout: the old code assigned into a slice, which left
    # `correct` as an object column that some pandas operations choke on.
    hist["correct"] = pd.Series(
        np.where(actual.notna(), predicted_home == (actual == 1.0), pd.NA),
        index=hist.index,
    ).astype("boolean")
    hist["brier"] = (hist["prob_home_win"] - actual) ** 2

    return add_season(hist)


def load_history(path: Path | None = None) -> pd.DataFrame:
    """Read the prediction log and score it in memory, writing nothing.

    Used by the Discord bot: a read-only command should not rewrite a data
    file that the daily pipeline may be writing at the same moment.
    """
    path = path or HISTORY_PATH
    if not path.exists():
        return pd.DataFrame()

    hist = pd.read_parquet(path)
    if hist.empty:
        return hist
    return attach_outcomes(hist)


def backfill_outcomes(path: Path | None = None) -> pd.DataFrame:
    """Score the prediction log and persist the results."""
    path = path or HISTORY_PATH
    if not path.exists():
        logger.warning("No prediction history found at %s", path)
        return pd.DataFrame()

    hist = pd.read_parquet(path)
    logger.info("Loaded %d predictions from history", len(hist))
    if hist.empty:
        return hist

    hist = attach_outcomes(hist)
    hist.to_parquet(path, index=False)
    logger.info("Updated prediction history with outcomes → %s", path)
    return hist


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_season(hist: pd.DataFrame, season: str | None) -> pd.DataFrame:
    """Restrict to one season.  ``None`` or ``"all"`` keeps everything."""
    if hist.empty or season is None or str(season).lower() == "all":
        return hist
    if "season" not in hist.columns:
        hist = add_season(hist)
    return hist[hist["season"] == season]


def evaluated_only(hist: pd.DataFrame) -> pd.DataFrame:
    """Predictions whose game has been played, oldest first."""
    if hist.empty or "actual_home_win" not in hist.columns:
        return hist.iloc[0:0]
    done = hist[hist["actual_home_win"].notna()].copy()
    if "predicted_at" in done.columns:
        # tail(n) is only "most recent" if the rows are actually in time order;
        # re-predicting a game moves its row to the end of the file.
        done = done.sort_values("predicted_at", kind="stable")
    return done


# ---------------------------------------------------------------------------
# Market prices and closing-line value
# ---------------------------------------------------------------------------

def backfill_closing_odds(hist: pd.DataFrame, max_dates: int = 60) -> pd.DataFrame:
    """Fill in the closing market price for games that have been played.

    Two prices matter and they answer different questions.  ``market_prob_home``
    is recorded when the prediction is made and says whether the model
    disagreed with the market.  ``closing_prob_home`` is the last price before
    puck drop and says whether the market later moved toward the model's side —
    the standard leading indicator, because prices are far less noisy than
    results.

    Only games already played are fetched, one request per date, skipping dates
    already complete.
    """
    if hist.empty or "predicted_at" not in hist.columns:
        return hist

    from ingestion.action_network import consensus_index

    hist = hist.copy()
    for col in ("closing_prob_home", "closing_n_books"):
        if col not in hist.columns:
            hist[col] = np.nan

    played = hist["actual_home_win"].notna() if "actual_home_win" in hist.columns else True
    todo = hist[played & hist["closing_prob_home"].isna()]
    if todo.empty:
        logger.info("Closing prices already complete")
        return hist

    dates = sorted(todo["predicted_at"].astype(str).str[:10].unique())
    if len(dates) > max_dates:
        logger.warning("%d dates need closing prices; fetching the %d most recent",
                       len(dates), max_dates)
        dates = dates[-max_dates:]

    filled = 0
    for day in dates:
        index = consensus_index(pd.Timestamp(day).date())
        if not index:
            continue
        rows = hist.index[
            (hist["predicted_at"].astype(str).str[:10] == day)
            & hist["closing_prob_home"].isna()
        ]
        for i in rows:
            match = index.get((hist.at[i, "home_team"], hist.at[i, "away_team"]))
            if match:
                hist.at[i, "closing_prob_home"] = match["market_prob_home"]
                hist.at[i, "closing_n_books"] = match["market_n_books"]
                filled += 1

    logger.info("Closing prices filled for %d predictions across %d dates",
                filled, len(dates))
    return hist


def _market_frame(hist: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Scored predictions that also carry a usable market price."""
    done = evaluated_only(hist)
    if done.empty or price_col not in done.columns:
        return done.iloc[0:0]
    return done[pd.to_numeric(done[price_col], errors="coerce").notna()].copy()


def market_comparison(hist: pd.DataFrame, price_col: str = "closing_prob_home") -> dict:
    """Score the model against the market on identical games.

    The market is the benchmark that matters — always-pick-home only says the
    model beat a naive rule.  Beating the closing price is what "has an edge"
    actually means.
    """
    df = _market_frame(hist, price_col)
    if df.empty:
        return {"n": 0}

    y = (pd.to_numeric(df["actual_home_win"], errors="coerce") == 1).astype(float)
    model = pd.to_numeric(df["prob_home_win"], errors="coerce")
    market = pd.to_numeric(df[price_col], errors="coerce")

    return {
        "n": len(df),
        "model_brier": float(((model - y) ** 2).mean()),
        "market_brier": float(((market - y) ** 2).mean()),
        "model_accuracy": float(((model >= 0.5) == (y == 1)).mean()),
        "market_accuracy": float(((market >= 0.5) == (y == 1)).mean()),
        "mean_abs_edge": float((model - market).abs().mean()),
    }


def clv_summary(hist: pd.DataFrame) -> dict:
    """Did the market move toward the side the model picked?

    Positive closing-line value means the model was early to information the
    market later priced in — the clearest evidence of a real edge, and it shows
    up in weeks rather than the season that win/loss records need.
    """
    df = _market_frame(hist, "closing_prob_home")
    if df.empty or "market_prob_home" not in df.columns:
        return {"n": 0}
    df = df[pd.to_numeric(df["market_prob_home"], errors="coerce").notna()]
    if df.empty:
        return {"n": 0}

    opening = pd.to_numeric(df["market_prob_home"], errors="coerce")
    closing = pd.to_numeric(df["closing_prob_home"], errors="coerce")
    picked_home = pd.to_numeric(df["prob_home_win"], errors="coerce") >= 0.5

    # Movement in the direction of our pick, in probability points.
    move = np.where(picked_home, closing - opening, opening - closing)
    return {
        "n": len(df),
        "mean_clv": float(np.mean(move)),
        "beat_close_rate": float(np.mean(move > 0)),
    }


def edge_realization(hist: pd.DataFrame, price_col: str = "closing_prob_home") -> pd.DataFrame:
    """Group predictions by claimed edge and check whether it materialised.

    The question the whole exercise exists to answer: when the model said it
    had five points on the market, did those games actually win five points
    more often?
    """
    df = _market_frame(hist, price_col)
    if df.empty:
        return pd.DataFrame()

    model = pd.to_numeric(df["prob_home_win"], errors="coerce")
    market = pd.to_numeric(df[price_col], errors="coerce")
    y = (pd.to_numeric(df["actual_home_win"], errors="coerce") == 1).astype(float)

    # Always from the perspective of the side the model favours.
    picked_home = model >= 0.5
    df = df.assign(
        edge=np.where(picked_home, model - market, (1 - model) - (1 - market)),
        won=np.where(picked_home, y, 1 - y),
        model_p=np.where(picked_home, model, 1 - model),
        market_p=np.where(picked_home, market, 1 - market),
    )
    df["bucket"] = pd.cut(
        df["edge"], bins=[-1, -0.05, -0.02, 0.02, 0.05, 1],
        labels=["< -5%", "-5..-2%", "-2..+2%", "+2..+5%", "> +5%"],
    )
    table = df.groupby("bucket", observed=True).agg(
        n=("won", "count"),
        model_said=("model_p", "mean"),
        market_said=("market_p", "mean"),
        actual=("won", "mean"),
    )
    return table[table["n"] > 0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def summarize(hist: pd.DataFrame) -> dict:
    """Headline metrics plus the baselines needed to interpret them.

    An accuracy figure on its own is not readable: home teams win about 54%
    of NHL games, so a model at 55% has found almost nothing.  Both baselines
    are computed from the same games being scored.
    """
    done = evaluated_only(hist)
    if done.empty:
        return {"n": 0, "n_logged": len(hist)}

    actual = pd.to_numeric(done["actual_home_win"], errors="coerce")
    correct = done["correct"].astype("boolean")
    home_rate = float(actual.mean())

    return {
        "n": len(done),
        "n_logged": len(hist),
        "accuracy": float(correct.mean()),
        "brier": float(done["brier"].mean()),
        "wins": int(correct.sum()),
        "losses": int((~correct).sum()),
        # Always picking the home team.
        "baseline_accuracy": max(home_rate, 1.0 - home_rate),
        "home_win_rate": home_rate,
        # Always predicting the base rate — the no-skill Brier score.
        "baseline_brier": home_rate * (1.0 - home_rate),
        "coin_flip_brier": 0.25,
    }


def calibration_table(hist: pd.DataFrame) -> pd.DataFrame:
    """Predicted vs actual home-win rate per probability bucket."""
    done = evaluated_only(hist)
    if done.empty:
        return pd.DataFrame()

    done = done.copy()
    done["bucket"] = pd.cut(
        done["prob_home_win"], bins=CALIBRATION_BINS,
        labels=CALIBRATION_LABELS, include_lowest=True,
    )
    table = done.groupby("bucket", observed=True).agg(
        n=("actual_home_win", "count"),
        predicted=("prob_home_win", "mean"),
        actual=("actual_home_win", "mean"),
    )
    return table[table["n"] > 0]


def confidence_breakdown(hist: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and Brier per pick-strength tier.

    Answers the question the ``/predictions`` labels invite: does a game
    flagged "Strong Pick" actually win more often than a "Lean"?
    """
    done = evaluated_only(hist)
    if done.empty:
        return pd.DataFrame()

    done = done.copy()
    p_fav = done["prob_home_win"].combine(1 - done["prob_home_win"], max)
    done["tier"] = p_fav.map(confidence_label)
    done["correct_f"] = done["correct"].astype("boolean").astype(float)

    order = [label for _, label in CONFIDENCE_TIERS]
    table = done.groupby("tier", observed=True).agg(
        n=("correct_f", "count"),
        accuracy=("correct_f", "mean"),
        brier=("brier", "mean"),
    )
    return table.reindex([t for t in order if t in table.index])


def recent_form(hist: pd.DataFrame, last_n: int = 25) -> dict:
    """Metrics over the most recent N evaluated predictions."""
    done = evaluated_only(hist).tail(last_n)
    if done.empty:
        return {"n": 0}
    correct = done["correct"].astype("boolean")
    return {
        "n": len(done),
        "accuracy": float(correct.mean()),
        "brier": float(done["brier"].mean()),
        "wins": int(correct.sum()),
        "losses": int((~correct).sum()),
    }


def coverage_warning(hist: pd.DataFrame, threshold: float = 0.95) -> str | None:
    """Flag predictions made on partially-populated features, if recorded."""
    if hist.empty or "feature_coverage" not in hist.columns:
        return None
    coverage = pd.to_numeric(hist["feature_coverage"], errors="coerce").dropna()
    if coverage.empty:
        return None
    degraded = int((coverage < threshold).sum())
    if not degraded:
        return None
    return (
        f"{degraded} of {len(coverage)} predictions were made with incomplete "
        f"features (min {coverage.min():.0%})"
    )


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def print_accuracy_report(hist: pd.DataFrame, season: str | None = None) -> None:
    """Print a summary of prediction accuracy."""
    scoped = filter_season(hist, season)
    stats = summarize(scoped)

    if not stats["n"]:
        print("No evaluated predictions yet (games haven't been played).")
        return

    title = f"Prediction Accuracy Report — {season}" if season else "Prediction Accuracy Report"
    print(f"\n{'='*56}")
    print(f"  {title}")
    print(f"{'='*56}")
    print(f"  Predictions evaluated:  {stats['n']} of {stats['n_logged']} logged")
    print(f"  Record:                 {stats['wins']}-{stats['losses']}")
    print(
        f"  Accuracy:               {stats['accuracy']:.1%}   "
        f"(always-home baseline: {stats['baseline_accuracy']:.1%})"
    )
    print(
        f"  Brier score:            {stats['brier']:.4f}   "
        f"(no-skill baseline: {stats['baseline_brier']:.4f})"
    )

    if "model_name" in scoped.columns:
        print(f"\n  By model:")
        for model, grp in evaluated_only(scoped).groupby("model_name"):
            m = summarize(grp)
            print(f"    {model}: acc={m['accuracy']:.1%}, brier={m['brier']:.4f}, n={m['n']}")

    tiers = confidence_breakdown(scoped)
    if not tiers.empty:
        print(f"\n  By pick strength:")
        for tier, row in tiers.iterrows():
            print(
                f"    {tier:>12s}: acc={row['accuracy']:.1%}, "
                f"brier={row['brier']:.4f}, n={int(row['n'])}"
            )

    cal = calibration_table(scoped)
    if not cal.empty:
        print(f"\n  Calibration (predicted vs actual home win rate):")
        for bucket, row in cal.iterrows():
            print(
                f"    {str(bucket):>6s}: predicted={row['predicted']:.1%}, "
                f"actual={row['actual']:.1%}, n={int(row['n'])}"
            )

    mkt = market_comparison(scoped)
    if mkt["n"]:
        print(f"\nvs the market ({mkt['n']} games with a closing price):")
        print(f"    Brier      model {mkt['model_brier']:.4f}   market {mkt['market_brier']:.4f}"
              f"   ({mkt['model_brier'] - mkt['market_brier']:+.4f})")
        print(f"    Accuracy   model {mkt['model_accuracy']:.1%}     market {mkt['market_accuracy']:.1%}")
        print(f"    mean |edge| claimed: {mkt['mean_abs_edge']:.1%}")

        er = edge_realization(scoped)
        if not er.empty:
            print(f"\ndid the claimed edge show up?")
            for b, r in er.iterrows():
                print(f"    {str(b):>9}: n={int(r['n']):>3}  model {r['model_said']:.1%}"
                      f"  market {r['market_said']:.1%}  actual {r['actual']:.1%}")

        clv = clv_summary(scoped)
        if clv["n"]:
            print(f"\nclosing-line value ({clv['n']} games): "
                  f"{clv['mean_clv']:+.2%} mean move toward our side, "
                  f"beat the close {clv['beat_close_rate']:.0%} of the time")

    recent = recent_form(scoped, last_n=25)
    if recent["n"] >= 10:
        print(f"\n  Last {recent['n']} predictions:")
        print(f"    Accuracy: {recent['accuracy']:.1%}")
        print(f"    Brier:    {recent['brier']:.4f}")

    warning = coverage_warning(scoped)
    if warning:
        print(f"\n  Note: {warning}")

    print(f"{'='*56}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Score logged NHL predictions")
    parser.add_argument(
        "--season", default=current_season(),
        help='Season to report on, or "all" (default: current season)',
    )
    args = parser.parse_args()

    hist = backfill_outcomes()
    if not hist.empty:
        hist = backfill_closing_odds(hist)
        hist.to_parquet(HISTORY_PATH, index=False)
    if hist.empty:
        print("No prediction history to evaluate yet.")
    else:
        season = None if args.season.lower() == "all" else args.season
        if season and not len(filter_season(hist, season)):
            available = sorted(hist["season"].dropna().unique())
            print(f"No predictions for {season}. Available: {', '.join(available) or 'none'}")
        else:
            print_accuracy_report(hist, season=season)
