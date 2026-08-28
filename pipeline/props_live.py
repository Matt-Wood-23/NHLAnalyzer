"""
Live props pipeline — expected shots on goal (SOG) for today's NHL skaters.

For each game on today's slate:
  1. Fetch both teams' current rosters from NHL API
  2. For each skater, fetch their current-season game log (SOG, TOI, xG)
  3. Build rolling feature snapshot (last 10, 20 games)
  4. Load saved SOG model → predict expected SOG
  5. Display ranked table: player, team, opponent, expected SOG

Game logs are cached to data/cache/players/ with a same-day TTL so repeated
runs on the same day (e.g. Discord bot called twice) are near-instant.
Player fetches run concurrently via ThreadPoolExecutor for a fast first run.

Usage:
    python -m pipeline.props_live                        # today, top 5
    python -m pipeline.props_live --date 2026-03-10      # specific date
    python -m pipeline.props_live --min-sog 2.0          # filter low projections
    python -m pipeline.props_live --top 10               # show more players
    python -m pipeline.props_live --no-cache             # force fresh API fetch
    python -m pipeline.props_live --max-workers 5        # reduce concurrency
"""

import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ingestion.nhl_api import fetch_schedule
from ingestion.player_stats import (
    fetch_team_roster, fetch_player_game_log, toi_to_seconds,
)
from features.player import MIN_GAMES, pregame_player_snapshot
from models.sog_model import load_sog_model
from config.season import current_season_api

logger = logging.getLogger(__name__)

PARQUET_DIR  = Path(__file__).parent.parent / "data" / "parquet"
CURRENT_SEASON_API = current_season_api()
PLAYER_WINDOWS = [10, 20]

# Same rationale as pipeline.live.MIN_FEATURE_COVERAGE: the SOG pipeline
# mean-imputes missing values, so a projection built mostly from imputed
# features is the training-set average wearing a player's name.
MIN_FEATURE_COVERAGE = 0.70


# ---------------------------------------------------------------------------
# Build rolling snapshot from NHL API game logs (live)
# ---------------------------------------------------------------------------

def _rolling_mean(series: pd.Series, w: int) -> float:
    vals = series.dropna().tail(w)
    return float(vals.mean()) if len(vals) > 0 else np.nan


def fetch_player_toi(
    player_id: int,
    *,
    refresh: bool = False,
) -> dict:
    """Fetch a player's recent ice time from the NHL API game log.

    Ice time is display-only — it is not one of the SOG model's features —
    so a failure here costs a column in the embed, never a projection.
    Everything the model consumes comes from player_game_stats.parquet, the
    same source it was trained on.
    """
    log = fetch_player_game_log(player_id, CURRENT_SEASON_API, refresh=refresh)
    if not log:
        return {}

    toi = pd.Series([toi_to_seconds(g.get("toi", "0:00")) for g in log], dtype=float)
    pp_points = pd.Series(
        [float(g.get("powerPlayPoints", 0) or 0) for g in log], dtype=float
    )
    return {
        f"toi_seconds_l{w}": _rolling_mean(toi, w) for w in PLAYER_WINDOWS
    } | {"pp_points_l10": _rolling_mean(pp_points, 10)}


# ---------------------------------------------------------------------------
# Concurrent worker
# ---------------------------------------------------------------------------

def _fetch_toi_worker(
    player_id: int,
    player_name: str,
    results: dict,
    lock: threading.Lock,
    refresh: bool = False,
) -> None:
    """ThreadPoolExecutor worker: fetch one player's ice time."""
    toi = fetch_player_toi(player_id, refresh=refresh)
    if toi:
        with lock:
            results[player_id] = toi


# ---------------------------------------------------------------------------
# Opponent defensive context
# ---------------------------------------------------------------------------

def _load_opponent_snapshot() -> pd.DataFrame:
    """Opponent defensive stats, computed exactly as the training matrix does.

    Training reads the opponent's pre-game ``sf_pct_l20`` / ``xg_against_l20``
    / ``hd_chances_against_l20`` out of the team feature matrix.  This used to
    re-derive them here as a flat mean of each team's last 20 games across
    every season, which is neither season-bounded nor the same statistic.
    Reusing the live team snapshot keeps the two definitions identical.
    """
    from pipeline.live import _build_team_snapshot

    snapshot = _build_team_snapshot()
    keep = {
        "sf_pct_l20": "opp_sf_pct_l20",
        "xg_against_l20": "opp_xg_against_l20",
        "hd_chances_against_l20": "opp_hd_chances_against_l20",
    }
    available = {src: dst for src, dst in keep.items() if src in snapshot.columns}
    if not available:
        logger.warning("Team snapshot has no defensive columns — opponent context will be NaN")
        return pd.DataFrame()
    return snapshot[list(available)].rename(columns=available)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(
    target_date: Optional[date] = None,
    min_sog: float = 1.5,
    api_delay: float = 0.15,    # kept for backward compat; no longer used in inner loop
    use_cache: bool = True,
    max_workers: int = 10,
    top: int = 10,
    min_coverage: float = MIN_FEATURE_COVERAGE,
) -> pd.DataFrame:
    """
    Generate expected SOG projections for all skaters in today's games.

    Args:
        target_date: date to predict for (default: today)
        min_sog:     filter out players with projected SOG below this threshold
        api_delay:   deprecated — kept for backward compat; pool size is the
                     rate limiter now
        use_cache:   if False, bypass disk cache and re-fetch all game logs
        max_workers: concurrent HTTP threads for ice-time fetching
        top:         return only the top N players by expected SOG (0 = no limit)
        min_coverage: drop players with fewer than this fraction of the SOG
                     model's features populated

    Returns:
        DataFrame sorted by expected_sog desc, with columns:
        player_name, team, opponent, position, games_played,
        sog_l10, sog_l20, toi_min_l10, expected_sog
    """
    if target_date is None:
        target_date = date.today()
    date_str = target_date.strftime("%Y-%m-%d")

    # 1. Get today's games
    games_today = fetch_schedule(date_str)
    reg_games = [
        g for g in games_today
        if g.get("game_type") == "2" and g["date"] == date_str
    ]
    if not reg_games:
        logger.info("No regular-season games on %s", date_str)
        return pd.DataFrame()

    logger.info("Found %d games on %s", len(reg_games), date_str)

    team_to_opp: dict[str, str] = {}
    for g in reg_games:
        team_to_opp[g["home_team"]] = g["away_team"]
        team_to_opp[g["away_team"]] = g["home_team"]

    # 2. Load SOG model
    try:
        pipeline, feature_cols = load_sog_model()
    except FileNotFoundError as e:
        logger.error("%s", e)
        return pd.DataFrame()

    # 3. Opponent defensive context, from the same snapshot training used
    opp_snap = _load_opponent_snapshot()

    # 4. Rosters (who is on each team today) — one cheap call per team
    teams_today = list(team_to_opp.keys())
    roster_by_player: dict[int, dict] = {}

    for team in teams_today:
        logger.info("Fetching roster: %s", team)
        for player in fetch_team_roster(team, CURRENT_SEASON_API):
            if player.get("position") == "G":
                continue
            roster_by_player[int(player["id"])] = {
                # The roster is authoritative on who plays for whom today;
                # a traded player's history still carries his old team.
                "player_name": player["name"],
                "team": team,
                "opponent": team_to_opp.get(team, ""),
                "position": player.get("position", ""),
            }
        time.sleep(0.05)

    if not roster_by_player:
        logger.warning("No rosters fetched — check API connectivity")
        return pd.DataFrame()

    # 5. Model features from player_game_stats.parquet — the source the SOG
    #    model was trained on.  The NHL API game log does not expose xGoal, so
    #    building features from it left xg / shot_attempts / xg_per_attempt
    #    permanently NaN: six of the model's eleven inputs silently replaced
    #    by training-set means on every projection.
    stats_path = PARQUET_DIR / "player_game_stats.parquet"
    if not stats_path.exists():
        logger.error(
            "No %s — run `python -m pipeline.backfill` to build it. "
            "Without it the SOG model has no player features.", stats_path.name,
        )
        return pd.DataFrame()

    history = pd.read_parquet(stats_path)
    snapshot = pregame_player_snapshot(history, player_ids=list(roster_by_player))
    snapshot = snapshot[snapshot["games_played"] >= MIN_GAMES]
    if snapshot.empty:
        logger.warning("No skater has %d+ games yet — no projections", MIN_GAMES)
        return pd.DataFrame()

    # Roster identity overrides the history's (possibly stale) team.
    identity = pd.DataFrame.from_dict(roster_by_player, orient="index")
    player_df = snapshot.drop(
        columns=["player_name", "team", "position"], errors="ignore",
    ).join(identity, how="inner")

    # Attach the opponent's defensive stats.
    if not opp_snap.empty:
        player_df = player_df.join(opp_snap, on="opponent")

    # 6. Ice time for display only — not a model feature, so failures here
    #    cost a column, never a projection.
    logger.info(
        "Fetching ice time for %d skaters (max_workers=%d, cache=%s)",
        len(player_df), max_workers, use_cache,
    )
    toi_by_player: dict[int, dict] = {}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_toi_worker, int(pid), row["player_name"],
                toi_by_player, lock, not use_cache,
            ): row["player_name"]
            for pid, row in player_df.iterrows()
        }
        for future in as_completed(futures):
            if (exc := future.exception()):
                logger.warning("Ice-time fetch failed for %s: %s", futures[future], exc)

    if toi_by_player:
        player_df = player_df.join(pd.DataFrame.from_dict(toi_by_player, orient="index"))

    # 7. Predict, refusing to publish projections built on absent features
    player_df = player_df.reset_index(names="player_id")
    missing = [c for c in feature_cols if c not in player_df.columns]
    if missing:
        logger.warning(
            "%d/%d SOG feature(s) absent: %s", len(missing), len(feature_cols), missing,
        )
        for col in missing:
            player_df[col] = np.nan

    X = player_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    player_df["feature_coverage"] = X.notna().mean(axis=1)

    usable = player_df["feature_coverage"] >= min_coverage
    if not usable.all():
        worst = player_df.loc[~usable, "feature_coverage"]
        logger.error(
            "Dropping %d projection(s) below %.0f%% feature coverage (min %.0f%%)",
            int((~usable).sum()), min_coverage * 100, worst.min() * 100,
        )
        player_df, X = player_df[usable].copy(), X[usable]

    if player_df.empty:
        logger.error(
            "No skater had enough usable features — check that "
            "player_game_stats.parquet is current (python -m pipeline.backfill)",
        )
        return pd.DataFrame()

    logger.info(
        "Feature coverage: min %.0f%%, mean %.0f%%",
        player_df["feature_coverage"].min() * 100,
        player_df["feature_coverage"].mean() * 100,
    )
    player_df["expected_sog"] = pipeline.predict(X.values)

    # 8. Format output
    player_df["toi_min_l10"] = (player_df.get("toi_seconds_l10", np.nan) / 60).round(1)
    player_df["expected_sog"] = player_df["expected_sog"].round(2)

    out_cols = [
        "player_name", "team", "opponent", "position", "games_played",
        "sog_l10", "sog_l20", "toi_min_l10", "expected_sog",
    ]
    out = player_df[[c for c in out_cols if c in player_df.columns]].copy()
    out = out[out["expected_sog"] >= min_sog]
    out = out.sort_values("expected_sog", ascending=False).reset_index(drop=True)

    if top > 0:
        out = out.head(top)

    logger.info("Generated projections for %d skaters (min_sog=%.1f, top=%d)", len(out), min_sog, top)
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Live NHL SOG projections")
    parser.add_argument("--date", default=None,
                        help="Date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--min-sog", type=float, default=1.5,
                        help="Minimum projected SOG to show (default: 1.5)")
    parser.add_argument("--top", type=int, default=10,
                        help="Show top N players (default: 10, 0 = no limit)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass disk cache and re-fetch all player game logs from API")
    parser.add_argument("--min-coverage", type=float, default=MIN_FEATURE_COVERAGE,
                        help="Drop players with fewer than this fraction of "
                             "SOG model features populated (0-1)")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Concurrent HTTP threads for player fetching (default: 10)")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    preds = run(
        target_date=target,
        min_sog=args.min_sog,
        use_cache=not args.no_cache,
        max_workers=args.max_workers,
        top=args.top,
        min_coverage=args.min_coverage,
    )

    if preds.empty:
        print("No projections generated.")
    else:
        print(f"\nSOG Projections -- {target}  (top {len(preds)}, min_sog={args.min_sog})")
        print("=" * 80)
        print(preds.to_string(index=False))
        print(f"\nAvg projected SOG: {preds['expected_sog'].mean():.2f}")
