"""
NHL Stats API ingestion client.
Base URL: https://api.nhle.com/stats/rest/en
Schedule/game detail: https://api-web.nhle.com/v1/

No API key required.
"""

import time
import logging
from datetime import date, timedelta
from typing import Optional

import httpx

from config.season import all_seasons, label_to_api

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_WEB = "https://api-web.nhle.com/v1"
BASE_STATS = "https://api.nhle.com/stats/rest/en"

GAME_TYPE_REGULAR = "2"
GAME_TYPE_PLAYOFF = "3"

TEAM_ABBREVS = [
    "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
    "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
    "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
    "WPG", "WSH",
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------
def _get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    """GET with simple retry logic and rate-limit courtesy sleep."""
    for attempt in range(retries):
        try:
            resp = httpx.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Rate limited — sleeping %ss", wait)
                time.sleep(wait)
            else:
                raise
        except httpx.RequestError as e:
            logger.warning("Request error (attempt %d): %s", attempt + 1, e)
            time.sleep(1)
    raise RuntimeError(f"Failed to GET {url} after {retries} attempts")


# ---------------------------------------------------------------------------
# Schedule / game discovery
# ---------------------------------------------------------------------------
def fetch_schedule(date_str: str) -> list[dict]:
    """
    Return list of games for a given date (YYYY-MM-DD).
    Each dict contains basic game metadata.
    """
    data = _get(f"{BASE_WEB}/schedule/{date_str}")
    games = []
    for week in data.get("gameWeek", []):
        for game in week.get("games", []):
            games.append({
                "game_id":    str(game["id"]),
                "date":       week["date"],
                "season":     str(game["season"]),
                "game_type":  str(game["gameType"]),
                "home_team":  game["homeTeam"]["abbrev"],
                "away_team":  game["awayTeam"]["abbrev"],
                "home_score": game.get("homeTeam", {}).get("score"),
                "away_score": game.get("awayTeam", {}).get("score"),
            })
    return games


def fetch_season_schedule(season: str) -> list[dict]:
    """
    Return all regular-season games for a season string like "20232024".
    Iterates dates from the season start through today.
    """
    data = _get(f"{BASE_WEB}/standings-season")
    seasons = data.get("seasons", [])
    season_info = next(
        (s for s in seasons if str(s.get("id")) == season), None
    )
    if not season_info:
        raise ValueError(f"Season {season} not found in standings-season response")

    start = date.fromisoformat(season_info["standingsStart"][:10])
    end_str = season_info.get("standingsEnd", date.today().isoformat())
    end = min(date.fromisoformat(end_str[:10]), date.today())

    all_games: list[dict] = []
    seen_game_ids: set[str] = set()
    current = start
    while current <= end:
        day_games = fetch_schedule(current.isoformat())
        regular = [g for g in day_games if g["game_type"] == GAME_TYPE_REGULAR]
        for g in regular:
            if g["game_id"] not in seen_game_ids:
                seen_game_ids.add(g["game_id"])
                all_games.append(g)
        current += timedelta(days=7)  # API returns a full week per call
        time.sleep(0.1)

    logger.info("Fetched %d regular-season games for %s", len(all_games), season)
    return all_games


# ---------------------------------------------------------------------------
# Game boxscore / landing detail
# ---------------------------------------------------------------------------
def fetch_game_landing(game_id: str) -> dict:
    """
    Fetch the full game landing page (boxscore + play-by-play summary).
    Returns raw JSON dict.
    """
    return _get(f"{BASE_WEB}/gamecenter/{game_id}/landing")


def parse_game_result(landing: dict) -> dict:
    """
    Extract win/loss, OT, SO flags from a game landing response.
    Returns a dict suitable for updating the games table.
    """
    game = landing.get("summary", {}).get("scoring", [])
    period_descriptor = landing.get("periodDescriptor", {})

    home_score = landing.get("homeTeam", {}).get("score")
    away_score = landing.get("awayTeam", {}).get("score")
    home_win = None
    if home_score is not None and away_score is not None:
        home_win = home_score > away_score

    max_period = landing.get("period", 0)
    went_to_ot = max_period >= 4
    went_to_so = max_period >= 5 or period_descriptor.get("periodType") == "SO"

    return {
        "home_score": home_score,
        "away_score": away_score,
        "home_win": home_win,
        "went_to_ot": went_to_ot,
        "went_to_so": went_to_so,
    }


# ---------------------------------------------------------------------------
# Goalie stats from boxscore
# ---------------------------------------------------------------------------
def parse_goalie_stats(landing: dict) -> list[dict]:
    """
    Extract per-goalie stats from a game landing response.
    Returns list of dicts, one per goalie.
    """
    game_id = str(landing.get("id", ""))
    goalies = []

    for side in ("homeTeam", "awayTeam"):
        team_data = landing.get(side, {})
        team_abbrev = team_data.get("abbrev", "")
        box = landing.get("summary", {}).get("goalieStats", {}).get(side, [])
        for g in box:
            toi_str = g.get("toi", "0:00")
            minutes, seconds = (toi_str.split(":") + ["0"])[:2]
            toi_seconds = int(minutes) * 60 + int(seconds)

            shots = g.get("shotsAgainst", 0)
            saves = g.get("saves", 0)
            goals_against = shots - saves

            goalies.append({
                "game_id":       game_id,
                "goalie_id":     g.get("playerId"),
                "goalie_name":   g.get("name", {}).get("default", ""),
                "team":          team_abbrev,
                "is_starter":    g.get("starter", False),
                "shots_against": shots,
                "saves":         saves,
                "goals_against": goals_against,
                "save_pct":      round(saves / shots, 4) if shots else None,
                "toi_seconds":   toi_seconds,
            })

    return goalies


# ---------------------------------------------------------------------------
# Team stats from boxscore
# ---------------------------------------------------------------------------
def parse_team_stats(landing: dict) -> list[dict]:
    """
    Extract team-level stats from a game landing response.
    Returns list of 2 dicts (home and away).
    """
    game_id = str(landing.get("id", ""))
    rows = []

    team_game_stats = landing.get("summary", {}).get("teamGameStats", [])
    # Build a lookup: category -> {homeValue, awayValue}
    stat_map: dict[str, dict] = {}
    for item in team_game_stats:
        stat_map[item.get("category", "")] = {
            "home": item.get("homeValue"),
            "away": item.get("awayValue"),
        }

    def _get_stat(category: str, side: str):
        return stat_map.get(category, {}).get(side)

    for side, is_home in (("homeTeam", True), ("awayTeam", False)):
        team_abbrev = landing.get(side, {}).get("abbrev", "")
        key = "home" if is_home else "away"

        rows.append({
            "game_id":          game_id,
            "team":             team_abbrev,
            "is_home":          is_home,
            "goals_for":        landing.get(side, {}).get("score"),
            "shots_for":        _get_stat("sog", key),
            "pp_opportunities": _get_stat("powerPlayPctg", key),   # pct, not raw — TODO replace
            "pp_goals":         _get_stat("powerPlayGoals", key),
            "pim":              _get_stat("pim", key),
            "hits":             _get_stat("hits", key),
            "blocked_shots":    _get_stat("blockedShots", key),
            "faceoff_pct":      _get_stat("faceoffWinningPctg", key),
            "source":           "nhl_api",
        })

    return rows


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def upsert_game(conn, game: dict) -> None:
    sql = """
        INSERT INTO games (game_id, date, season, game_type, home_team, away_team,
                           home_score, away_score, home_win, went_to_ot, went_to_so)
        VALUES (%(game_id)s, %(date)s, %(season)s, %(game_type)s, %(home_team)s,
                %(away_team)s, %(home_score)s, %(away_score)s, %(home_win)s,
                %(went_to_ot)s, %(went_to_so)s)
        ON CONFLICT (game_id) DO UPDATE SET
            date        = COALESCE(EXCLUDED.date, games.date),
            home_score  = COALESCE(EXCLUDED.home_score, games.home_score),
            away_score  = COALESCE(EXCLUDED.away_score, games.away_score),
            home_win    = COALESCE(EXCLUDED.home_win, games.home_win),
            went_to_ot  = EXCLUDED.went_to_ot,
            went_to_so  = EXCLUDED.went_to_so,
            ingested_at = NOW()
    """
    game.setdefault("home_win", None)
    game.setdefault("went_to_ot", False)
    game.setdefault("went_to_so", False)
    with conn.cursor() as cur:
        cur.execute(sql, game)


def upsert_goalie_stats(conn, rows: list[dict]) -> None:
    # Postgres is optional — the whole pipeline runs on Parquet alone.
    # Imported here so a machine without psycopg2 can still ingest data.
    import psycopg2.extras

    if not rows:
        return
    sql = """
        INSERT INTO goalie_stats (game_id, goalie_id, goalie_name, team, is_starter,
                                  shots_against, saves, goals_against, save_pct, toi_seconds)
        VALUES (%(game_id)s, %(goalie_id)s, %(goalie_name)s, %(team)s, %(is_starter)s,
                %(shots_against)s, %(saves)s, %(goals_against)s, %(save_pct)s, %(toi_seconds)s)
        ON CONFLICT (game_id, goalie_id) DO UPDATE SET
            shots_against = EXCLUDED.shots_against,
            saves         = EXCLUDED.saves,
            goals_against = EXCLUDED.goals_against,
            save_pct      = EXCLUDED.save_pct,
            toi_seconds   = EXCLUDED.toi_seconds,
            ingested_at   = NOW()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows)


def upsert_team_stats(conn, rows: list[dict]) -> None:
    # Postgres is optional — the whole pipeline runs on Parquet alone.
    # Imported here so a machine without psycopg2 can still ingest data.
    import psycopg2.extras

    if not rows:
        return
    sql = """
        INSERT INTO team_stats (game_id, team, is_home, goals_for, shots_for,
                                pp_goals, source)
        VALUES (%(game_id)s, %(team)s, %(is_home)s, %(goals_for)s, %(shots_for)s,
                %(pp_goals)s, %(source)s)
        ON CONFLICT (game_id, team) DO UPDATE SET
            goals_for   = EXCLUDED.goals_for,
            shots_for   = EXCLUDED.shots_for,
            pp_goals    = EXCLUDED.pp_goals,
            ingested_at = NOW()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows)


# ---------------------------------------------------------------------------
# Main backfill runner
# ---------------------------------------------------------------------------
def backfill_season(conn, season: str) -> None:
    """
    Fetch all regular-season games for a season and write them to Postgres.
    Also fetches boxscore detail for completed games.
    """
    logger.info("Starting backfill for season %s", season)
    games = fetch_season_schedule(season)

    for game in games:
        upsert_game(conn, game)

    conn.commit()
    logger.info("Committed %d game stubs for %s", len(games), season)

    # Fetch boxscore detail for games that have a score
    completed = [g for g in games if g.get("home_score") is not None]
    logger.info("Fetching boxscores for %d completed games", len(completed))

    for i, game in enumerate(completed):
        try:
            landing = fetch_game_landing(game["game_id"])
            result = parse_game_result(landing)
            game.update(result)
            upsert_game(conn, game)

            goalie_rows = parse_goalie_stats(landing)
            upsert_goalie_stats(conn, goalie_rows)

            team_rows = parse_team_stats(landing)
            upsert_team_stats(conn, team_rows)

            if (i + 1) % 50 == 0:
                conn.commit()
                logger.info("Progress: %d / %d", i + 1, len(completed))

            time.sleep(0.2)
        except Exception as e:
            logger.error("Error on game %s: %s", game["game_id"], e)

    conn.commit()
    logger.info("Backfill complete for season %s", season)


if __name__ == "__main__":
    import psycopg2
    import os
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    db_url = os.environ.get("DATABASE_URL", "postgresql://localhost/nhl_ml")
    conn = psycopg2.connect(db_url)

    # Backfill all available seasons (list extends automatically each year)
    for season in all_seasons():
        backfill_season(conn, label_to_api(season))

    conn.close()
