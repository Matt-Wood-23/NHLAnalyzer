"""
Daily NHL prediction pipeline — single command for the full workflow.

Steps:
  1. Refresh MoneyPuck data (delete stale cache, re-download current season)
  2. Re-aggregate all seasons into team game stats parquet
  2b. Sync finished games into Postgres, so rest days have real dates
  2c. Refresh official results from the boxscores (the win label)
  3. Rebuild the feature matrix (rolling stats, ELO, goalie, ST, etc.)
  4. Backfill outcomes for any previous predictions
  5. Print accuracy report
  6. Run today's live predictions and save to history

Run it before the first puck drop.  Games already under way are predicted and
shown but not written to the history: by then the price is a live line and the
refreshed features may contain the game itself.

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

from config.season import current_season, current_season_year, season_start

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def current_season_cache_file() -> Path:
    """MoneyPuck shot cache for the season in progress (resolved at call time)."""
    return RAW_DIR / f"moneypuck_shots_{current_season_year()}.csv"


def sync_game_dates(conn=None) -> int:
    """Bring the Postgres ``games`` table up to last night.

    A failure here must not stop the run, but it must not pass quietly either:
    stale dates do not break anything visibly, they just make every team look
    rested.  Without a database the pipeline falls back to estimating dates
    from game numbers, which is good to a few days — not enough to trust a
    back-to-back flag.
    """
    if conn is None:
        logger.warning(
            "No database — game dates will be estimated from game numbers; "
            "rest days and back-to-backs are approximate",
        )
        return 0

    from ingestion.nhl_api import sync_recent_games

    try:
        return sync_recent_games(conn, season_start(current_season()))
    except Exception as e:
        conn.rollback()
        logger.error(
            "Could not sync game dates (%s) — rest days and back-to-backs "
            "will be stale for this run", e,
        )
        return 0


def refresh_data(conn=None):
    """Delete stale MoneyPuck cache, re-download, and rebuild feature matrix."""
    from ingestion.moneypuck import ingest_all_seasons

    # Step 1: Delete stale cache so MoneyPuck re-downloads with latest games.
    # Only the current season's file is dropped — prior seasons never change.
    cache_file = current_season_cache_file()
    logger.info("Current season: %s", current_season())
    if cache_file.exists():
        cache_file.unlink()
        logger.info("Deleted stale cache: %s", cache_file.name)
    else:
        logger.info("No cache to delete — will download fresh")

    # Step 2: Re-aggregate all seasons (only current season re-downloads; others are cached)
    logger.info("Re-aggregating MoneyPuck data for all seasons...")
    ingest_all_seasons(conn=None)

    # Step 2b: Game dates.  Must precede the matrix rebuild, which reads them
    # for rest / back-to-back / season-day, and the live run reads them again.
    sync_game_dates(conn)

    # Step 3: Rebuild the full feature matrix
    logger.info("Rebuilding feature matrix...")
    from pipeline.backfill import (
        build_boxscore_spine, build_feature_matrix, build_game_results,
        build_player_game_stats, save_feature_matrix,
    )

    # Official results first: the matrix takes its win label from them, and
    # last night's games are not in the file until this runs.
    logger.info("Refreshing official game results...")
    build_game_results()
    matrix = build_feature_matrix(conn=conn)
    save_feature_matrix(matrix)
    logger.info("Feature matrix rebuilt: %d games x %d cols", *matrix.shape)

    # Step 4: Player-level stats — the SOG props model both trains and
    # predicts from these, so they must refresh with everything else.
    logger.info("Rebuilding player game stats...")
    build_player_game_stats()

    # Step 5: The dressed-lineup spine, which is what the props model actually
    # reads.  Step 4 alone left it frozen while reporting a successful refresh.
    logger.info("Refreshing boxscore spine...")
    build_boxscore_spine()


def score_predictions():
    """Backfill outcomes and print accuracy report."""
    from pipeline.evaluate_history import (
        backfill_closing_odds, backfill_outcomes, print_accuracy_report, HISTORY_PATH,
    )

    hist = backfill_outcomes()
    if not hist.empty:
        # Capture the closing price for games that have now been played.  This
        # is the half of the record that expires: lines move, and historical
        # odds are not freely available after the fact.
        hist = backfill_closing_odds(hist)
        hist.to_parquet(HISTORY_PATH, index=False)
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
        # Skip the Postgres write, but this is the real daily run, so the
        # prediction history must still be logged -- it is what /yesterday and
        # the CLV scoring read back.  These were one flag until dry_run was
        # made to mean "write nothing", which silently would have stopped the
        # daily loop recording anything.
        dry_run=True,
        save_history=True,
        conn=conn,
    )

    if preds.empty:
        # Empty means either an empty slate or every game refused; the log
        # above says which.  This used to print "No games scheduled" for both,
        # directly beneath three lines naming the games it had just dropped.
        print(f"\nNo predictions for {target_date} — see the log above for why.")
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
