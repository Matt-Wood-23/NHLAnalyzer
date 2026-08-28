"""
Action Network odds scraper.

Fetches NHL moneylines, spreads, and totals from the Action Network
internal API — no API key required.

Usage:
    python -m ingestion.action_network                  # today
    python -m ingestion.action_network --date 2026-04-08
    python -m ingestion.action_network --date 2026-04-08 --json
"""

import argparse
import json
import logging
from datetime import date, datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://api.actionnetwork.com/web/v2/scoreboard/nhl"

# Book IDs on Action Network that are widely available
# 15=DraftKings, 30=FanDuel, 68=BetMGM, 69=Caesars, 76=PointsBet, 123=BetRivers
_BOOK_IDS = "15,30,68,69,123"

_BOOK_NAMES = {
    "15":  "DraftKings",
    "30":  "FanDuel",
    "68":  "BetMGM",
    "69":  "Caesars",
    "123": "BetRivers",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Action Network uses team abbreviations that differ from NHL API in a few cases
_ABBR_MAP = {
    "TB":  "TBL",
    "NJ":  "NJD",
    "SJ":  "SJS",
    "LA":  "LAK",
    "CLB": "CBJ",
}


def _norm_abbr(abbr: str) -> str:
    return _ABBR_MAP.get(abbr.upper(), abbr.upper())


def _remove_vig(home_ml: int, away_ml: int) -> tuple[float, float]:
    """Convert American moneylines to no-vig probabilities."""
    def ml_to_prob(ml: int) -> float:
        if ml > 0:
            return 100 / (ml + 100)
        else:
            return abs(ml) / (abs(ml) + 100)

    p_home = ml_to_prob(home_ml)
    p_away = ml_to_prob(away_ml)
    total  = p_home + p_away
    return p_home / total, p_away / total


def fetch_odds(target_date: Optional[date] = None) -> list[dict]:
    """
    Fetch NHL odds from Action Network for a given date.

    Returns a list of game dicts, one per game:
        {
          "game_id":       str | None,   # NHL API game ID if derivable, else None
          "home_team":     str,          # NHL-normalised abbreviation
          "away_team":     str,
          "start_time":    datetime,     # UTC-aware
          "status":        str,          # "scheduled", "in_progress", "complete"
          "books": {
            "DraftKings": {
              "home_ml": int, "away_ml": int,
              "total": float | None, "total_over_ml": int | None, "total_under_ml": int | None,
              "home_spread": float | None, "home_spread_ml": int | None,
              "away_spread": float | None, "away_spread_ml": int | None,
              "prob_home_win": float,    # no-vig
              "prob_away_win": float,
            },
            ...
          },
          "consensus": { ... }           # averaged no-vig probs across books
        }
    """
    if target_date is None:
        target_date = date.today()

    date_str = target_date.strftime("%Y%m%d")
    url      = f"{_BASE}?period=game&bookIds={_BOOK_IDS}&date={date_str}"

    try:
        resp = httpx.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("Action Network fetch failed: %s", e)
        return []

    raw_games = resp.json().get("games", [])
    results   = []

    for g in raw_games:
        # Skip non-regular-season
        if g.get("type") not in ("reg", "post"):
            continue

        # Identify home / away teams
        home_id = g.get("home_team_id")
        away_id = g.get("away_team_id")
        teams   = {t["id"]: t for t in g.get("teams", [])}
        home_t  = teams.get(home_id, {})
        away_t  = teams.get(away_id, {})

        home_abbr = _norm_abbr(home_t.get("abbr", ""))
        away_abbr = _norm_abbr(away_t.get("abbr", ""))

        start_time = datetime.fromisoformat(
            g["start_time"].replace("Z", "+00:00")
        )

        status_raw = g.get("status", "")
        if status_raw in ("scheduled", "created"):
            status = "scheduled"
        elif status_raw in ("complete", "closed", "final"):
            status = "complete"
        else:
            status = "in_progress"

        books = {}
        for book_id, book_data in g.get("markets", {}).items():
            book_name = _BOOK_NAMES.get(book_id)
            if not book_name:
                continue
            event = book_data.get("event", {})

            # --- Moneyline (pregame only) ---
            home_ml = away_ml = None
            for entry in event.get("moneyline", []):
                if entry.get("is_live"):
                    continue
                if entry.get("team_id") == home_id:
                    home_ml = entry.get("odds")
                elif entry.get("team_id") == away_id:
                    away_ml = entry.get("odds")

            if home_ml is None or away_ml is None:
                continue  # skip book if no moneyline

            p_home, p_away = _remove_vig(home_ml, away_ml)

            # --- Total ---
            total = total_over_ml = total_under_ml = None
            for entry in event.get("total", []):
                if entry.get("side") == "over":
                    total        = entry.get("value")
                    total_over_ml = entry.get("odds")
                elif entry.get("side") == "under":
                    total_under_ml = entry.get("odds")

            # --- Spread ---
            home_spread = home_spread_ml = away_spread = away_spread_ml = None
            for entry in event.get("spread", []):
                if entry.get("team_id") == home_id:
                    home_spread    = entry.get("value")
                    home_spread_ml = entry.get("odds")
                elif entry.get("team_id") == away_id:
                    away_spread    = entry.get("value")
                    away_spread_ml = entry.get("odds")

            books[book_name] = {
                "home_ml":        home_ml,
                "away_ml":        away_ml,
                "prob_home_win":  round(p_home, 4),
                "prob_away_win":  round(p_away, 4),
                "total":          total,
                "total_over_ml":  total_over_ml,
                "total_under_ml": total_under_ml,
                "home_spread":    home_spread,
                "home_spread_ml": home_spread_ml,
                "away_spread":    away_spread,
                "away_spread_ml": away_spread_ml,
            }

        # Consensus: average no-vig prob across available books
        consensus = {}
        if books:
            probs_home = [b["prob_home_win"] for b in books.values()]
            probs_away = [b["prob_away_win"] for b in books.values()]
            totals     = [b["total"] for b in books.values() if b["total"] is not None]
            consensus = {
                "prob_home_win": round(sum(probs_home) / len(probs_home), 4),
                "prob_away_win": round(sum(probs_away) / len(probs_away), 4),
                "total":         round(sum(totals) / len(totals), 2) if totals else None,
                "n_books":       len(books),
            }

        results.append({
            "game_id":   None,   # could be joined to NHL API game ID if needed
            "an_game_id": g["id"],
            "home_team": home_abbr,
            "away_team": away_abbr,
            "start_time": start_time,
            "status":    status,
            "books":     books,
            "consensus": consensus,
        })

    logger.info("Fetched %d games for %s", len(results), target_date)
    return results


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Fetch NHL odds from Action Network")
    parser.add_argument("--date", default=None, help="Date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Print raw JSON")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    games  = fetch_odds(target)

    if not games:
        print("No games found.")
        sys.exit(0)

    if args.as_json:
        # Make datetimes serialisable
        for g in games:
            g["start_time"] = g["start_time"].isoformat()
        print(json.dumps(games, indent=2))
    else:
        print(f"\nNHL Odds — {target}  ({len(games)} games)\n")
        for g in games:
            home, away = g["home_team"], g["away_team"]
            con = g["consensus"]
            if not con:
                print(f"  {away} @ {home}  — no lines")
                continue
            total_str = f"  O/U {con['total']}" if con["total"] else ""
            print(
                f"  {away} @ {home}"
                f"  |  {home} {con['prob_home_win']:.1%} / {away} {con['prob_away_win']:.1%}"
                f"{total_str}"
                f"  ({con['n_books']} books)"
            )
            for book, b in g["books"].items():
                print(
                    f"    {book:<12} ML {b['home_ml']:+d}/{b['away_ml']:+d}"
                    + (f"  Total {b['total']}" if b["total"] else "")
                    + (f"  Spread {b['home_spread']:+g}" if b["home_spread"] is not None else "")
                )
        print()


# ---------------------------------------------------------------------------
# Market snapshot helpers
# ---------------------------------------------------------------------------

def consensus_index(target_date: "date | None" = None) -> dict[tuple[str, str], dict]:
    """No-vig consensus probabilities keyed by ``(home_team, away_team)``.

    Action Network does not expose NHL game IDs, so matchups are the join key.
    Returns an empty mapping rather than raising when odds are unavailable —
    a missing market price should never take down a prediction run.
    """
    try:
        games = fetch_odds(target_date)
    except Exception as e:
        logger.warning("Odds fetch failed for %s: %s", target_date, e)
        return {}

    index: dict[tuple[str, str], dict] = {}
    for g in games:
        con = g.get("consensus")
        if not con or con.get("prob_home_win") is None:
            continue
        index[(g["home_team"], g["away_team"])] = {
            "market_prob_home": float(con["prob_home_win"]),
            "market_n_books": int(con.get("n_books") or 0),
            "market_status": g.get("status"),
        }
    logger.info("Market consensus available for %d games on %s", len(index), target_date)
    return index
