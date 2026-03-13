"""
The Odds API ingestion client.
Docs: https://the-odds-api.com/liveapi/guides/v4/

Requires an API key in the environment variable ODDS_API_KEY.
Free tier: 500 requests/month. Historical odds require a paid plan.

Usage:
    ODDS_API_KEY=your_key python -m ingestion.odds_api
"""

import logging
import os
import time
from datetime import date, datetime, timezone

import httpx
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
NHL_SPORT_KEY = "icehockey_nhl"

# Books to track (add/remove as desired)
BOOKMAKERS = [
    "draftkings",
    "fanduel",
    "betmgm",
    "caesars",
    "pointsbet_us",
    "betrivers",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get(endpoint: str, params: dict) -> dict | list:
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise EnvironmentError("ODDS_API_KEY environment variable not set")

    params["apiKey"] = api_key
    url = f"{BASE_URL}/{endpoint}"
    resp = httpx.get(url, params=params, timeout=30)

    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    logger.debug("Odds API usage: %s used, %s remaining", used, remaining)

    resp.raise_for_status()
    return resp.json()


def american_to_prob(ml: int | None) -> float | None:
    """Convert American moneyline to raw (vig-included) implied probability."""
    if ml is None:
        return None
    if ml > 0:
        return round(100 / (ml + 100), 4)
    else:
        return round(abs(ml) / (abs(ml) + 100), 4)


def remove_vig(home_prob: float, away_prob: float) -> tuple[float, float]:
    """Strip the bookmaker margin to get fair implied probabilities."""
    total = home_prob + away_prob
    return round(home_prob / total, 4), round(away_prob / total, 4)


def parse_american(odds_str) -> int | None:
    """Safely parse an American odds value to int."""
    try:
        return int(odds_str)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Game matching
# ---------------------------------------------------------------------------
def match_game_id(conn, home_team: str, away_team: str, commence_time: str) -> str | None:
    """
    Try to find the NHL API game_id for an odds record by matching
    team abbreviations and date.
    """
    game_date = commence_time[:10]  # "2024-01-15"

    # The Odds API uses full city names; we map common ones to abbrevs
    team_name_map = {
        "Toronto Maple Leafs": "TOR", "Boston Bruins": "BOS",
        "Tampa Bay Lightning": "TBL", "Florida Panthers": "FLA",
        "Buffalo Sabres": "BUF", "Ottawa Senators": "OTT",
        "Montreal Canadiens": "MTL", "Detroit Red Wings": "DET",
        "Pittsburgh Penguins": "PIT", "Philadelphia Flyers": "PHI",
        "New Jersey Devils": "NJD", "New York Rangers": "NYR",
        "New York Islanders": "NYI", "Carolina Hurricanes": "CAR",
        "Columbus Blue Jackets": "CBJ", "Washington Capitals": "WSH",
        "Chicago Blackhawks": "CHI", "Nashville Predators": "NSH",
        "Minnesota Wild": "MIN", "St. Louis Blues": "STL",
        "Colorado Avalanche": "COL", "Winnipeg Jets": "WPG",
        "Dallas Stars": "DAL", "Arizona Coyotes": "ARI",
        "Utah Hockey Club": "UTA", "Vegas Golden Knights": "VGK",
        "Calgary Flames": "CGY", "Edmonton Oilers": "EDM",
        "Vancouver Canucks": "VAN", "Seattle Kraken": "SEA",
        "Anaheim Ducks": "ANA", "Los Angeles Kings": "LAK",
        "San Jose Sharks": "SJS",
    }

    home_abbrev = team_name_map.get(home_team)
    away_abbrev = team_name_map.get(away_team)

    if not home_abbrev or not away_abbrev:
        logger.warning("No abbrev mapping for: '%s' or '%s'", home_team, away_team)
        return None

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT game_id FROM games
            WHERE home_team = %s AND away_team = %s AND date = %s
            LIMIT 1
            """,
            (home_abbrev, away_abbrev, game_date),
        )
        row = cur.fetchone()

    return row[0] if row else None


# ---------------------------------------------------------------------------
# Fetch and parse
# ---------------------------------------------------------------------------
def fetch_live_odds(regions: str = "us") -> list[dict]:
    """
    Fetch current (live/upcoming) NHL moneyline odds across bookmakers.
    Returns the raw list of game objects from the API.
    """
    data = _get(
        f"sports/{NHL_SPORT_KEY}/odds",
        {
            "regions": regions,
            "markets": "h2h",
            "oddsFormat": "american",
            "bookmakers": ",".join(BOOKMAKERS),
        },
    )
    logger.info("Fetched live odds for %d games", len(data))
    return data


def parse_odds_records(game: dict) -> list[dict]:
    """
    Parse a single Odds API game object into a list of rows
    (one per bookmaker) suitable for the odds table.
    """
    rows = []
    commence_time = game.get("commence_time", "")

    for bookmaker in game.get("bookmakers", []):
        book = bookmaker.get("key", "")
        markets = {m["key"]: m for m in bookmaker.get("markets", [])}
        h2h = markets.get("h2h", {})
        outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}

        home_name = game.get("home_team", "")
        away_name = game.get("away_team", "")

        open_home_ml = parse_american(outcomes.get(home_name))
        open_away_ml = parse_american(outcomes.get(away_name))

        home_prob_raw = american_to_prob(open_home_ml)
        away_prob_raw = american_to_prob(open_away_ml)

        home_prob, away_prob = (None, None)
        if home_prob_raw and away_prob_raw:
            home_prob, away_prob = remove_vig(home_prob_raw, away_prob_raw)

        rows.append({
            "game_id":          None,   # filled in after matching
            "home_team_name":   home_name,
            "away_team_name":   away_name,
            "commence_time":    commence_time,
            "book":             book,
            "open_home_ml":     open_home_ml,
            "open_away_ml":     open_away_ml,
            "open_home_prob":   home_prob,
            "open_away_prob":   away_prob,
            # close values populated post-game via a separate update pass
            "close_home_ml":    None,
            "close_away_ml":    None,
            "close_home_prob":  None,
            "close_away_prob":  None,
        })

    return rows


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------
def upsert_odds(conn, rows: list[dict]) -> None:
    if not rows:
        return
    sql = """
        INSERT INTO odds (
            game_id, book,
            open_home_ml, open_away_ml,
            open_home_prob, open_away_prob,
            close_home_ml, close_away_ml,
            close_home_prob, close_away_prob
        ) VALUES (
            %(game_id)s, %(book)s,
            %(open_home_ml)s, %(open_away_ml)s,
            %(open_home_prob)s, %(open_away_prob)s,
            %(close_home_ml)s, %(close_away_ml)s,
            %(close_home_prob)s, %(close_away_prob)s
        )
        ON CONFLICT (game_id, book) DO UPDATE SET
            open_home_ml   = COALESCE(EXCLUDED.open_home_ml,   odds.open_home_ml),
            open_away_ml   = COALESCE(EXCLUDED.open_away_ml,   odds.open_away_ml),
            open_home_prob = COALESCE(EXCLUDED.open_home_prob, odds.open_home_prob),
            open_away_prob = COALESCE(EXCLUDED.open_away_prob, odds.open_away_prob),
            close_home_ml  = COALESCE(EXCLUDED.close_home_ml,  odds.close_home_ml),
            close_away_ml  = COALESCE(EXCLUDED.close_away_ml,  odds.close_away_ml),
            close_home_prob= COALESCE(EXCLUDED.close_home_prob,odds.close_home_prob),
            close_away_prob= COALESCE(EXCLUDED.close_away_prob,odds.close_away_prob),
            ingested_at    = NOW()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=100)
    logger.info("Upserted %d odds rows", len(rows))


def update_closing_odds(conn, game_id: str, book: str, close_home_ml: int, close_away_ml: int) -> None:
    """
    Update the closing line for a specific game/book after the game starts.
    Also computes and stores no-vig closing probabilities.
    """
    home_raw = american_to_prob(close_home_ml)
    away_raw = american_to_prob(close_away_ml)
    close_home_prob, close_away_prob = (None, None)
    if home_raw and away_raw:
        close_home_prob, close_away_prob = remove_vig(home_raw, away_raw)

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE odds SET
                close_home_ml   = %s,
                close_away_ml   = %s,
                close_home_prob = %s,
                close_away_prob = %s,
                ingested_at     = NOW()
            WHERE game_id = %s AND book = %s
            """,
            (close_home_ml, close_away_ml, close_home_prob, close_away_prob, game_id, book),
        )


# ---------------------------------------------------------------------------
# Main runner — ingest today's odds
# ---------------------------------------------------------------------------
def ingest_todays_odds(conn) -> None:
    """
    Fetch current odds, match to game_ids, and upsert into Postgres.
    Run pre-game. Run again just before puck drop to capture closing lines.
    """
    games_data = fetch_live_odds()
    all_rows: list[dict] = []

    for game in games_data:
        rows = parse_odds_records(game)
        for row in rows:
            game_id = match_game_id(
                conn,
                row["home_team_name"],
                row["away_team_name"],
                row["commence_time"],
            )
            row["game_id"] = game_id

        # Only keep rows where we matched a game_id
        matched = [r for r in rows if r["game_id"]]
        if not matched:
            logger.warning(
                "No game_id match for %s @ %s on %s",
                game.get("away_team"), game.get("home_team"),
                game.get("commence_time", "")[:10],
            )
        all_rows.extend(matched)
        time.sleep(0.05)

    upsert_odds(conn, all_rows)
    conn.commit()
    logger.info("Odds ingestion complete. %d rows written.", len(all_rows))


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    db_url = os.environ.get("DATABASE_URL", "postgresql://localhost/nhl_ml")
    conn = psycopg2.connect(db_url)

    ingest_todays_odds(conn)
    conn.close()
