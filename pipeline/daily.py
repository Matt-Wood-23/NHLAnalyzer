"""
Daily NHL prediction pipeline — single command for the full workflow.

Steps:
  1. Refresh MoneyPuck data (delete stale cache, re-download current season)
  2. Re-aggregate all seasons into team game stats parquet
  3. Rebuild the feature matrix (rolling stats, ELO, goalie, ST, etc.)
  4. Backfill outcomes for any previous predictions
  5. Print accuracy report
  6. Run today's live predictions and save to history

Usage:
    python -m pipeline.daily                  # full pipeline
    python -m pipeline.daily --predict-only   # skip data refresh, just predict today
    python -m pipeline.daily --refresh-only   # refresh data + score old predictions, no new predictions
    python -m pipeline.daily --date 2026-03-21  # predict for a specific date
"""

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# MoneyPuck cache file for the current season
CURRENT_SEASON_YEAR = "2025"
CACHE_FILE = RAW_DIR / f"moneypuck_shots_{CURRENT_SEASON_YEAR}.csv"


def refresh_data(conn=None):
    """Delete stale MoneyPuck cache, re-download, and rebuild feature matrix."""
    from ingestion.moneypuck import ingest_all_seasons

    # Step 1: Delete stale cache so MoneyPuck re-downloads with latest games
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        logger.info("Deleted stale cache: %s", CACHE_FILE.name)
    else:
        logger.info("No cache to delete — will download fresh")

    # Step 2: Re-aggregate all seasons (only current season re-downloads; others are cached)
    logger.info("Re-aggregating MoneyPuck data for all seasons...")
    ingest_all_seasons(conn=None)

    # Step 3: Rebuild the full feature matrix
    logger.info("Rebuilding feature matrix...")
    from pipeline.backfill import build_feature_matrix, save_feature_matrix
    matrix = build_feature_matrix(conn=conn)
    save_feature_matrix(matrix)
    logger.info("Feature matrix rebuilt: %d games x %d cols", *matrix.shape)


def score_predictions():
    """Backfill outcomes and print accuracy report."""
    from pipeline.evaluate_history import backfill_outcomes, print_accuracy_report

    hist = backfill_outcomes()
    if not hist.empty:
        print_accuracy_report(hist)
    else:
        print("No prediction history to evaluate yet.")
    return hist


def run_predictions(target_date, model_name="random_forest", conn=None):
    """Run live predictions for the given date."""
    from pipeline.live import run as live_run

    preds = live_run(
        target_date=target_date,
        model_name=model_name,
        dry_run=True,  # don't save to DB, history is saved inside run()
        conn=conn,
    )

    if preds.empty:
        print(f"\nNo games scheduled for {target_date}.")
    else:
        print(f"\nPredictions for {target_date}:")
        print(preds.to_string(index=False))

    return preds


def main():
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Daily NHL prediction pipeline")
    parser.add_argument("--date", default=None,
                        help="Date to predict (YYYY-MM-DD). Default: today.")
    parser.add_argument("--model", default="random_forest",
                        help="Model to use for predictions")
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip data refresh, just run predictions for today")
    parser.add_argument("--refresh-only", action="store_true",
                        help="Refresh data and score old predictions, skip new predictions")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()

    # Connect to DB if available (for context features like H2H, rest days)
    conn = None
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        try:
            import psycopg2
            conn = psycopg2.connect(db_url)
            logger.info("Connected to Postgres")
        except Exception as e:
            logger.warning("DB connect failed: %s — running without DB", e)

    try:
        if not args.predict_only:
            print("=" * 50)
            print("  Step 1: Refreshing data")
            print("=" * 50)
            refresh_data(conn=conn)

            print("\n" + "=" * 50)
            print("  Step 2: Scoring previous predictions")
            print("=" * 50)
            score_predictions()

        if not args.refresh_only:
            print("\n" + "=" * 50)
            print(f"  Step 3: Predictions for {target}")
            print("=" * 50)
            run_predictions(target, model_name=args.model, conn=conn)

    finally:
        if conn:
            conn.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
