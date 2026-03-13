"""
Player statistics ingestion — two data sources:

1. MoneyPuck shot-level CSVs (already downloaded):
   Aggregated to per-player per-game: SOG, xGoal, shot attempts.
   Covers 5 seasons of history — the primary training data source.

2. NHL Stats API player game logs (fetched on demand):
   Per-game TOI, PP TOI, goals, assists for specific players.
   Used in the live pipeline for today's roster.

Usage:
    # Build and save player game stats parquet from shots CSVs
    python -m ingestion.player_stats

    # Fetch a single player's game log (by NHL player ID)
    python -m ingestion.player_stats --player-id 8478402
"""

import argparse
import json
import logging
import time
from datetime import date as _date
from pathlib import Path
from typing import Optional

import pandas as pd
import httpx

logger = logging.getLogger(__name__)

RAW_DIR    = Path(__file__).parent.parent / "data" / "raw"
PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"
CACHE_DIR   = Path(__file__).parent.parent / "data" / "cache" / "players"

BASE_WEB = "https://api-web.nhle.com/v1"

# MoneyPuck shots CSV year → season label
SHOT_SEASONS: dict[str, str] = {
    "2021": "2021-2022",
    "2022": "2022-2023",
    "2023": "2023-2024",
    "2024": "2024-2025",
    "2025": "2025-2026",
}

# Columns we need from the shots CSV
_SHOT_COLS = [
    "game_id", "season", "isPlayoffGame",
    "shooterPlayerId", "shooterName",
    "teamCode", "playerPositionThatDidEvent",
    "shotWasOnGoal", "xGoal",
]


# ---------------------------------------------------------------------------
# MoneyPuck shots → per-player per-game aggregation
# ---------------------------------------------------------------------------

def aggregate_shots_to_player_games(
    raw_dir: Optional[Path] = None,
    seasons: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Read MoneyPuck shot-level CSVs and aggregate to one row per
    (game_id, player_id, season).

    Output columns:
        game_id          — full NHL format, e.g. "2024020001"
        season           — "2024-2025"
        player_id        — NHL player ID (int)
        player_name      — display name
        team             — team abbreviation
        position         — F/D/G
        sog              — shots on goal this game
        xg               — expected goals (sum of xGoal)
        shot_attempts    — total shot attempts (corsi-ish)
        xg_per_attempt   — average shot quality
    """
    if raw_dir is None:
        raw_dir = RAW_DIR

    target_seasons = SHOT_SEASONS if seasons is None else {
        k: v for k, v in SHOT_SEASONS.items() if v in seasons
    }

    dfs = []
    for year_str, season_label in target_seasons.items():
        path = raw_dir / f"moneypuck_shots_{year_str}.csv"
        if not path.exists():
            logger.warning("Shots file not found: %s", path)
            continue

        logger.info("Reading %s ...", path.name)
        raw = pd.read_csv(path, usecols=_SHOT_COLS, low_memory=False)

        # Regular season only
        raw = raw[raw["isPlayoffGame"] == 0].copy()

        # Reconstruct full NHL game_id.
        # MoneyPuck game_id (e.g. 20001) = game_type_digit(2) + game_number(0001).
        # Full NHL format: "{season_year}0{moneypuck_game_id}"
        # e.g. season=2024, game_id=20001 → "2024020001" (10 chars, matches NHL API)
        raw["full_game_id"] = (
            raw["season"].astype(str) + "0" + raw["game_id"].astype(int).astype(str)
        )
        raw["season_label"] = season_label

        dfs.append(raw)

    if not dfs:
        raise FileNotFoundError(f"No shots CSVs found in {raw_dir}")

    shots = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d shots total across %d seasons", len(shots), len(dfs))

    # Aggregate per (game, player)
    grouped = shots.groupby(
        ["full_game_id", "season_label", "shooterPlayerId", "shooterName", "teamCode",
         "playerPositionThatDidEvent"],
        dropna=False,
    )
    agg = grouped.agg(
        sog=("shotWasOnGoal", "sum"),
        xg=("xGoal", "sum"),
        shot_attempts=("xGoal", "count"),
    ).reset_index()

    agg["xg_per_attempt"] = (agg["xg"] / agg["shot_attempts"].clip(lower=1)).round(5)
    agg = agg.rename(columns={
        "full_game_id":               "game_id",
        "season_label":               "season",
        "shooterPlayerId":            "player_id",
        "shooterName":                "player_name",
        "teamCode":                   "team",
        "playerPositionThatDidEvent": "position",
    })

    # Sort chronologically: game_id last 4 digits = game number within season
    agg["game_num"] = agg["game_id"].str[-4:].astype(int)
    agg = agg.sort_values(["player_id", "season", "game_num"]).reset_index(drop=True)
    agg["sog"] = agg["sog"].fillna(0).astype(int)

    logger.info(
        "Aggregated: %d player-game rows, %d unique players",
        len(agg), agg["player_id"].nunique(),
    )
    return agg


def save_player_game_stats(df: pd.DataFrame, name: str = "player_game_stats") -> Path:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    path = PARQUET_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved → %s (%d rows)", path, len(df))
    return path


def load_player_game_stats(path: Optional[Path] = None) -> pd.DataFrame:
    if path is None:
        path = PARQUET_DIR / "player_game_stats.parquet"
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Player game-log disk cache (same-day TTL)
# ---------------------------------------------------------------------------

def _cache_path(player_id: int, season: str) -> Path:
    return CACHE_DIR / f"{player_id}_{season}.json"


def _load_cache(player_id: int, season: str) -> list[dict] | None:
    """Return today's cached game log, or None if missing/stale."""
    path = _cache_path(player_id, season)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("date") == _date.today().isoformat():
            return data["games"]
    except Exception as e:
        logger.debug("Cache read error for player %d: %s", player_id, e)
    return None


def _save_cache(player_id: int, season: str, games: list[dict]) -> None:
    """Persist game log to disk with today's date stamp."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(player_id, season)
    try:
        path.write_text(
            json.dumps({"date": _date.today().isoformat(), "games": games}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.warning("Cache write error for player %d: %s", player_id, e)


# ---------------------------------------------------------------------------
# NHL API — player game log (for TOI, PP TOI)
# ---------------------------------------------------------------------------

def fetch_player_game_log(
    player_id: int,
    season: str = "20252026",
    game_type: int = 2,
    *,
    refresh: bool = False,
) -> list[dict]:
    """
    Fetch per-game stats for a single player from the NHL API.

    Results are cached to data/cache/players/{player_id}_{season}.json with a
    same-day TTL. Pass refresh=True to bypass the cache and force an API call.

    Args:
        player_id: NHL player ID (e.g. 8478402 for McDavid)
        season:    season in API format (e.g. "20252026")
        game_type: 2 = regular season, 3 = playoffs
        refresh:   if True, skip cache and always hit the API

    Returns:
        List of game dicts with keys: gameId, gameDate, teamAbbrev,
        opponentAbbrev, homeRoadFlag, goals, assists, shots,
        points, toi (MM:SS string), powerPlayGoals, powerPlayPoints.
    """
    if not refresh:
        cached = _load_cache(player_id, season)
        if cached is not None:
            logger.debug("Cache hit: player %d season %s", player_id, season)
            return cached

    url = f"{BASE_WEB}/player/{player_id}/game-log/{season}/{game_type}"
    try:
        resp = httpx.get(url, timeout=15)
        resp.raise_for_status()
        games = resp.json().get("gameLog", [])
    except Exception as e:
        logger.warning("Failed to fetch game log for player %d: %s", player_id, e)
        return []

    if games:   # don't cache network errors / empty responses
        _save_cache(player_id, season, games)
    return games


def toi_to_seconds(toi_str: str) -> int:
    """Convert 'MM:SS' TOI string to total seconds."""
    try:
        parts = str(toi_str).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0


def game_log_to_dataframe(game_log: list[dict], player_id: int, player_name: str) -> pd.DataFrame:
    """Convert NHL API game log list to a clean DataFrame."""
    rows = []
    for g in game_log:
        rows.append({
            "game_id":        str(g["gameId"]),
            "game_date":      g.get("gameDate", ""),
            "player_id":      player_id,
            "player_name":    player_name,
            "team":           g.get("teamAbbrev", ""),
            "opponent":       g.get("opponentAbbrev", ""),
            "home_road":      g.get("homeRoadFlag", ""),
            "goals":          int(g.get("goals", 0)),
            "assists":        int(g.get("assists", 0)),
            "points":         int(g.get("points", 0)),
            "sog":            int(g.get("shots", 0)),
            "toi_seconds":    toi_to_seconds(g.get("toi", "0:00")),
            "pp_goals":       int(g.get("powerPlayGoals", 0)),
            "pp_points":      int(g.get("powerPlayPoints", 0)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NHL API — team roster
# ---------------------------------------------------------------------------

def fetch_team_roster(team_abbrev: str, season: str = "20252026") -> list[dict]:
    """
    Fetch current roster for a team. Returns list of player dicts with
    keys: id, firstName, lastName, positionCode.
    """
    url = f"{BASE_WEB}/roster/{team_abbrev}/{season}"
    try:
        resp = httpx.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch roster for %s: %s", team_abbrev, e)
        return []

    players = []
    for group in ("forwards", "defensemen", "goalies"):
        for p in data.get(group, []):
            players.append({
                "id":       p["id"],
                "name":     f"{p['firstName']['default']} {p['lastName']['default']}",
                "position": p.get("positionCode", ""),
            })
    return players


def fetch_roster_with_logs(
    team_abbrev: str,
    season: str = "20252026",
    delay: float = 0.1,
) -> pd.DataFrame:
    """
    Fetch roster + current-season game log for each skater (not goalies).
    Returns combined DataFrame of per-player per-game stats.
    """
    roster = fetch_team_roster(team_abbrev, season)
    skaters = [p for p in roster if p["position"] != "G"]

    all_logs = []
    for player in skaters:
        log = fetch_player_game_log(player["id"], season)
        if log:
            df = game_log_to_dataframe(log, player["id"], player["name"])
            all_logs.append(df)
        time.sleep(delay)

    if not all_logs:
        return pd.DataFrame()
    return pd.concat(all_logs, ignore_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Player statistics ingestion")
    parser.add_argument("--player-id", type=int, default=None,
                        help="Fetch a single player's game log by NHL ID")
    parser.add_argument("--team", default=None,
                        help="Fetch full roster + game logs for a team (e.g. EDM)")
    args = parser.parse_args()

    if args.player_id:
        log = fetch_player_game_log(args.player_id)
        df = game_log_to_dataframe(log, args.player_id, "player")
        print(df.to_string(index=False))

    elif args.team:
        df = fetch_roster_with_logs(args.team)
        print(df.to_string(index=False))

    else:
        # Default: aggregate all shots CSVs and save parquet
        df = aggregate_shots_to_player_games()
        path = save_player_game_stats(df)
        print(f"\nSaved: {path}")
        print(f"Shape: {df.shape}")
        print(f"\nSample (top SOG games):")
        print(df.nlargest(10, "sog")[["game_id", "season", "player_name", "team", "sog", "xg"]].to_string(index=False))
