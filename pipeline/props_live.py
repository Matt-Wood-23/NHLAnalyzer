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
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ingestion.nhl_api import fetch_schedule, TEAM_ABBREVS
from ingestion.player_stats import (
    fetch_team_roster, fetch_player_game_log, toi_to_seconds,
)
from models.sog_model import load_sog_model, SOG_FEATURE_COLS
from config.season import current_season_api

logger = logging.getLogger(__name__)

PARQUET_DIR  = Path(__file__).parent.parent / "data" / "parquet"
CURRENT_SEASON_API = current_season_api()
PLAYER_WINDOWS = [10, 20]


# ---------------------------------------------------------------------------
# Build rolling snapshot from NHL API game logs (live)
# ---------------------------------------------------------------------------

def _rolling_mean(series: pd.Series, w: int) -> float:
    vals = series.dropna().tail(w)
    return float(vals.mean()) if len(vals) > 0 else np.nan


def build_player_features_from_log(
    player_id: int,
    player_name: str,
    team: str,
    position: str,
    *,
    refresh: bool = False,
) -> Optional[dict]:
    """
    Fetch current-season game log for one player and compute rolling features.
    Returns a feature dict or None if insufficient data.
    """
    log = fetch_player_game_log(player_id, CURRENT_SEASON_API, refresh=refresh)
    if not log:
        return None

    rows = []
    for g in log:
        rows.append({
            "sog":           int(g.get("shots", 0)),
            "toi_seconds":   toi_to_seconds(g.get("toi", "0:00")),
            "goals":         int(g.get("goals", 0)),
            "assists":       int(g.get("assists", 0)),
            "pp_points":     int(g.get("powerPlayPoints", 0)),
        })
    df = pd.DataFrame(rows)

    if len(df) < 5:
        logger.debug("Skipping %s — only %d games", player_name, len(df))
        return None

    feats: dict = {
        "player_id":    player_id,
        "player_name":  player_name,
        "team":         team,
        "position":     position,
        "games_played": len(df),
    }

    for col in ["sog", "toi_seconds"]:
        for w in PLAYER_WINDOWS:
            feats[f"{col}_l{w}"] = _rolling_mean(df[col], w)

    # Proxies for xG features (not in NHL API — set NaN; model imputes with mean)
    for col in ["xg", "shot_attempts", "xg_per_attempt"]:
        for w in PLAYER_WINDOWS:
            feats[f"{col}_l{w}"] = np.nan

    # PP usage proxy: PP points rate (signal for PP ice time)
    feats["pp_points_l10"] = _rolling_mean(df["pp_points"], 10)

    return feats


# ---------------------------------------------------------------------------
# Concurrent worker
# ---------------------------------------------------------------------------

def _fetch_player_worker(
    p: dict,
    team: str,
    opp: str,
    def_snap: pd.DataFrame,
    results: list,
    lock: threading.Lock,
    refresh: bool = False,
) -> None:
    """
    ThreadPoolExecutor worker: fetch one player's features and append to results.
    Thread-safe via lock on the shared results list.
    """
    feats = build_player_features_from_log(
        player_id=p["id"],
        player_name=p["name"],
        team=team,
        position=p.get("position", ""),
        refresh=refresh,
    )
    if feats is None:
        return

    feats["opponent"] = opp
    if opp and opp in def_snap.index:
        opp_row = def_snap.loc[opp]
        feats["opp_sf_pct_l20"]             = opp_row.get("opp_sf_pct_l20", np.nan)
        feats["opp_xg_against_l20"]         = opp_row.get("opp_xg_against_l20", np.nan)
        feats["opp_hd_chances_against_l20"] = opp_row.get("opp_hd_chances_against_l20", np.nan)
    else:
        feats["opp_sf_pct_l20"]             = np.nan
        feats["opp_xg_against_l20"]         = np.nan
        feats["opp_hd_chances_against_l20"] = np.nan

    with lock:
        results.append(feats)


# ---------------------------------------------------------------------------
# Opponent defensive context
# ---------------------------------------------------------------------------

def _load_team_defensive_snapshot() -> pd.DataFrame:
    """Load the team feature parquet and extract current defensive stats per team."""
    parquet_path = PARQUET_DIR / "moneypuck_team_game_stats.parquet"
    from features.team import _add_derived_columns
    df = pd.read_parquet(parquet_path)
    df = _add_derived_columns(df)
    df["game_num"] = df["game_id"].str[-4:].astype(int)
    df = df.sort_values(["team", "season", "game_num"])

    snap_rows = []
    for team, grp in df.groupby("team"):
        recent = grp.tail(20)
        snap_rows.append({
            "team":                    team,
            "opp_sf_pct_l20":          recent["sf_pct"].mean() if "sf_pct" in recent else np.nan,
            "opp_xg_against_l20":      recent["xg_against"].mean() if "xg_against" in recent else np.nan,
            "opp_hd_chances_against_l20": recent["hd_chances_against"].mean() if "hd_chances_against" in recent else np.nan,
        })
    return pd.DataFrame(snap_rows).set_index("team")


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
) -> pd.DataFrame:
    """
    Generate expected SOG projections for all skaters in today's games.

    Args:
        target_date: date to predict for (default: today)
        min_sog:     filter out players with projected SOG below this threshold
        api_delay:   deprecated — kept for backward compat; pool size is the
                     rate limiter now
        use_cache:   if False, bypass disk cache and re-fetch all game logs
        max_workers: concurrent HTTP threads for player game-log fetching
        top:         return only the top N players by expected SOG (0 = no limit)

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

    # 3. Load opponent defensive snapshot
    def_snap = _load_team_defensive_snapshot()

    # 4. Fetch rosters sequentially (cheap — one call per team)
    teams_today = list(team_to_opp.keys())
    all_skaters: list[tuple[dict, str, str]] = []

    for team in teams_today:
        logger.info("Fetching roster: %s", team)
        roster = fetch_team_roster(team, CURRENT_SEASON_API)
        opp = team_to_opp.get(team, "")
        skaters = [p for p in roster if p.get("position") != "G"]
        for p in skaters:
            all_skaters.append((p, team, opp))
        time.sleep(0.05)    # brief inter-team pause

    logger.info(
        "Fetching game logs for %d skaters (max_workers=%d, cache=%s)",
        len(all_skaters), max_workers, use_cache,
    )

    # 5. Concurrent player game-log fetches
    all_player_feats: list[dict] = []
    lock = threading.Lock()
    refresh = not use_cache

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_player_worker, p, team, opp, def_snap,
                all_player_feats, lock, refresh,
            ): p["name"]
            for (p, team, opp) in all_skaters
        }
        for future in as_completed(futures):
            if (exc := future.exception()):
                logger.warning("Worker error for %s: %s", futures[future], exc)

    if not all_player_feats:
        logger.warning("No player features built — check API connectivity")
        return pd.DataFrame()

    player_df = pd.DataFrame(all_player_feats)

    # 6. Predict expected SOG
    missing = [c for c in feature_cols if c not in player_df.columns]
    for col in missing:
        player_df[col] = np.nan

    X = player_df[feature_cols].values
    player_df["expected_sog"] = pipeline.predict(X)

    # 7. Format output
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
    )

    if preds.empty:
        print("No projections generated.")
    else:
        print(f"\nSOG Projections -- {target}  (top {len(preds)}, min_sog={args.min_sog})")
        print("=" * 80)
        print(preds.to_string(index=False))
        print(f"\nAvg projected SOG: {preds['expected_sog'].mean():.2f}")
