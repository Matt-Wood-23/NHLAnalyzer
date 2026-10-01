"""
NHL API boxscore ingestion — the dressed lineup, not just the shooters.

Why this exists
---------------
``player_game_stats.parquet`` is aggregated from MoneyPuck's shot-level CSVs,
which only contain players who generated a shot event.  A skater held to zero
attempts does not record a 0 — he vanishes from the game entirely.  Measured:
30.7 rows per game against 36 dressed skaters, and ``min(shot_attempts) == 1``
across all 200k rows.

Every rolling average built on that file is therefore computed over a survivor
set, biased upward, in the same direction as the league drift correction in
``models.count_model`` and on the same prop.

``/gamecenter/{game-id}/boxscore`` returns all 36 dressed skaters with ``sog``
present and zero where earned.  It also carries ``points``, ``assists``,
``toi``, ``shifts``, ``hits`` and ``blockedShots`` per skater, and a
``starter`` flag on each goalie — ground truth the saves model has never had.

What it does not carry is xG.  So the boxscore is the *spine* and MoneyPuck is
left-joined onto it: a skater absent from the shot data took no attempts, and
his xG and attempt count are genuinely 0, not missing.

Caching
-------
Completed games are immutable, so boxscores cache to disk permanently
(``data/cache/boxscores/{game_id}.json``) rather than with the same-day TTL
``ingestion.player_stats`` uses for in-progress season game logs.  Only games
the API reports as final are persisted.

Usage:
    python -m ingestion.boxscore --backfill        # fetch + cache every game
    python -m ingestion.boxscore --build           # parse cache -> parquet
    python -m ingestion.boxscore --backfill --build
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

import httpx
import numpy as np
import pandas as pd

from config.season import game_number

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"
CACHE_DIR   = Path(__file__).parent.parent / "data" / "cache" / "boxscores"

BASE_WEB = "https://api-web.nhle.com/v1"

# Game states the API uses for a finished game.  Anything else (LIVE, PRE,
# FUT) is fetched but never cached, so a game pulled mid-flight is re-fetched.
FINAL_STATES = {"OFF", "FINAL", "OVER"}

# Concurrency for the backfill.  Eight keeps a full five-season pull under ten
# minutes at ~0.6s a request without hammering a free public API.
DEFAULT_WORKERS = 8

SKATER_GROUPS = ("forwards", "defense")

# Per-skater fields worth carrying forward.  goals/assists/points unlock the
# points props; hits/blockedShots are their own markets; toi and shifts are the
# ice-time signal the SOG model has been proxying with shot attempts.
SKATER_FIELDS = {
    "sog":            "sog",
    "goals":          "goals",
    "assists":        "assists",
    "points":         "points",
    "pim":            "pim",
    "hits":           "hits",
    "blockedShots":   "blocked_shots",
    "giveaways":      "giveaways",
    "takeaways":      "takeaways",
    "shifts":         "shifts",
    "plusMinus":      "plus_minus",
    "powerPlayGoals": "pp_goals",
}

GOALIE_FIELDS = {
    "saves":        "saves",
    "shotsAgainst": "shots_against",
    "goalsAgainst": "goals_against",
    "pim":          "pim",
}


# ---------------------------------------------------------------------------
# Fetch + cache
# ---------------------------------------------------------------------------

def _cache_path(game_id: str) -> Path:
    return CACHE_DIR / f"{game_id}.json"


def fetch_boxscore(
    game_id: str,
    *,
    refresh: bool = False,
    client: Optional[httpx.Client] = None,
    timeout: float = 20.0,
) -> dict | None:
    """Return one game's boxscore, from disk if it has been fetched before.

    Final games are cached permanently — their boxscore cannot change.  A game
    that is not yet final is returned but not written, so calling this during a
    game does not poison the cache with a partial line.
    """
    game_id = str(game_id)
    path = _cache_path(game_id)
    if not refresh and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug("Cache read error for game %s: %s", game_id, e)

    url = f"{BASE_WEB}/gamecenter/{game_id}/boxscore"
    try:
        resp = (client or httpx).get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Failed to fetch boxscore %s: %s", game_id, e)
        return None

    if data.get("gameState") in FINAL_STATES:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning("Cache write error for game %s: %s", game_id, e)
    else:
        logger.debug("Game %s is %s — not caching", game_id, data.get("gameState"))
    return data


def backfill_boxscores(
    game_ids: Iterable[str],
    *,
    workers: int = DEFAULT_WORKERS,
    refresh: bool = False,
) -> dict[str, int]:
    """Fetch and cache every game not already on disk.

    Returns a {cached, fetched, failed} tally.  Safe to interrupt and re-run:
    anything already written is skipped without a request.
    """
    game_ids = [str(g) for g in game_ids]
    todo = game_ids if refresh else [
        g for g in game_ids if not _cache_path(g).exists()
    ]
    tally = {"cached": len(game_ids) - len(todo), "fetched": 0, "failed": 0}
    if not todo:
        logger.info("All %d boxscores already cached", len(game_ids))
        return tally

    logger.info(
        "Fetching %d boxscores (%d already cached), %d workers",
        len(todo), tally["cached"], workers,
    )
    started = time.time()
    with httpx.Client(timeout=20.0) as client:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fetch_boxscore, g, refresh=refresh, client=client): g
                for g in todo
            }
            for i, fut in enumerate(as_completed(futures), 1):
                if fut.result() is None:
                    tally["failed"] += 1
                else:
                    tally["fetched"] += 1
                if i % 250 == 0 or i == len(todo):
                    rate = i / max(time.time() - started, 1e-9)
                    logger.info(
                        "  %d/%d  (%.1f games/s, %d failed)",
                        i, len(todo), rate, tally["failed"],
                    )
    logger.info(
        "Backfill done in %.1fs: %d fetched, %d already cached, %d failed",
        time.time() - started, tally["fetched"], tally["cached"], tally["failed"],
    )
    return tally


def cached_game_ids() -> list[str]:
    if not CACHE_DIR.exists():
        return []
    return sorted(p.stem for p in CACHE_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def toi_to_seconds(toi: str | None) -> float:
    """Convert an 'MM:SS' time-on-ice string to seconds.

    Blank or malformed values return 0.
    """
    if not toi or ":" not in str(toi):
        return 0.0
    try:
        mins, secs = str(toi).split(":")[:2]
        return float(int(mins) * 60 + int(secs))
    except (TypeError, ValueError):
        return 0.0


def _player_name(player: dict) -> str:
    name = player.get("name")
    if isinstance(name, dict):
        return name.get("default", "")
    return str(name or "")


def parse_skaters(box: dict) -> list[dict]:
    """Every dressed skater in one game, zero-shot rows included."""
    game_id = str(box.get("id", ""))
    season = box.get("season")
    rows: list[dict] = []
    stats = box.get("playerByGameStats") or {}
    for side, home_flag in (("homeTeam", True), ("awayTeam", False)):
        team = (box.get(side) or {}).get("abbrev", "")
        for group in SKATER_GROUPS:
            for p in (stats.get(side) or {}).get(group, []):
                row = {
                    "game_id":     game_id,
                    "season_api":  season,
                    "player_id":   p.get("playerId"),
                    "player_name": _player_name(p),
                    "team":        team,
                    "is_home":     home_flag,
                    "position":    p.get("position", ""),
                    "toi_seconds": toi_to_seconds(p.get("toi")),
                }
                for src, dest in SKATER_FIELDS.items():
                    row[dest] = p.get(src, 0) or 0
                rows.append(row)
    return rows


def parse_goalies(box: dict) -> list[dict]:
    """Goalie lines, carrying the ``starter`` flag.

    This is the only place in the project with ground truth on who actually
    started.  ``pipeline.saves_live`` infers the starter from each team's most
    recent start; with this, that inference can be scored instead of trusted.
    """
    game_id = str(box.get("id", ""))
    rows: list[dict] = []
    stats = box.get("playerByGameStats") or {}
    for side, home_flag in (("homeTeam", True), ("awayTeam", False)):
        team = (box.get(side) or {}).get("abbrev", "")
        for p in (stats.get(side) or {}).get("goalies", []):
            row = {
                "game_id":     game_id,
                "season_api":  box.get("season"),
                "player_id":   p.get("playerId"),
                "player_name": _player_name(p),
                "team":        team,
                "is_home":     home_flag,
                "starter":     bool(p.get("starter", False)),
                "toi_seconds": toi_to_seconds(p.get("toi")),
            }
            for src, dest in GOALIE_FIELDS.items():
                row[dest] = p.get(src, 0) or 0
            rows.append(row)
    return rows


def _season_label(season_api) -> str:
    """Convert the API's 20252026 to the parquet convention '2025-2026'."""
    s = str(season_api or "")
    return f"{s[:4]}-{s[4:]}" if len(s) == 8 else s


def _from_cache(game_ids: Optional[Iterable[str]], parser) -> pd.DataFrame:
    ids = [str(g) for g in game_ids] if game_ids is not None else cached_game_ids()
    rows: list[dict] = []
    missing = 0
    for gid in ids:
        path = _cache_path(gid)
        if not path.exists():
            missing += 1
            continue
        try:
            rows.extend(parser(json.loads(path.read_text(encoding="utf-8"))))
        except Exception as e:
            logger.warning("Parse error for game %s: %s", gid, e)
    if missing:
        logger.warning("%d requested game(s) not in cache — run --backfill", missing)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["season"] = df["season_api"].map(_season_label)
    df["game_num"] = game_number(df["game_id"])
    return df.drop(columns=["season_api"])


def build_skater_game_stats(game_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Parse cached boxscores into one row per dressed skater per game."""
    df = _from_cache(game_ids, parse_skaters)
    if df.empty:
        return df
    logger.info(
        "Skater game stats: %d rows, %d games, %d players (%.1f rows/game)",
        len(df), df["game_id"].nunique(), df["player_id"].nunique(),
        len(df) / max(df["game_id"].nunique(), 1),
    )
    return df


def build_goalie_starts(game_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Parse cached boxscores into one row per goalie appearance."""
    df = _from_cache(game_ids, parse_goalies)
    if df.empty:
        return df
    logger.info(
        "Goalie appearances: %d rows, %d starts, %d games",
        len(df), int(df["starter"].sum()), df["game_id"].nunique(),
    )
    return df


def parse_game_result(box: dict) -> list[dict]:
    """The official result of one finished game, as a one-row list.

    The score here includes the goal awarded for a shootout, which no shot
    file does.  Returns nothing for a game that is not final or has no winner
    yet, so an in-progress boxscore can never be read as a result.
    """
    home, away = box.get("homeTeam") or {}, box.get("awayTeam") or {}
    home_score, away_score = home.get("score"), away.get("score")
    if box.get("gameState") not in FINAL_STATES:
        return []
    if home_score is None or away_score is None or home_score == away_score:
        return []
    last_period = (
        (box.get("gameOutcome") or {}).get("lastPeriodType")
        or (box.get("periodDescriptor") or {}).get("periodType")
    )
    return [{
        "game_id":     str(box.get("id", "")),
        "season_api":  box.get("season"),
        "game_date":   box.get("gameDate"),
        "game_type":   str(box.get("gameType", "")),
        "home_team":   home.get("abbrev", ""),
        "away_team":   away.get("abbrev", ""),
        "home_score":  int(home_score),
        "away_score":  int(away_score),
        "home_win":    bool(home_score > away_score),
        "last_period_type": last_period,
        "went_to_ot":  last_period in ("OT", "SO"),
        "went_to_so":  last_period == "SO",
    }]


def build_game_results(game_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Parse cached boxscores into one row per finished game: who won, and how.

    This is the project's source of truth for results.  Wins used to be
    derived from MoneyPuck shot totals, which contain no shootout: a shootout
    game is tied there, so both teams were recorded as losing it and every
    home shootout win became a home loss in the training target.
    """
    df = _from_cache(game_ids, parse_game_result)
    if df.empty:
        return df
    logger.info(
        "Game results: %d games, %.1f%% home wins, %d to overtime, %d to a shootout",
        len(df), 100 * df["home_win"].mean(),
        int(df["went_to_ot"].sum()), int(df["went_to_so"].sum()),
    )
    return df


def load_game_results(path: Optional[Path] = None) -> pd.DataFrame:
    return pd.read_parquet(path or PARQUET_DIR / "game_results.parquet")


# ---------------------------------------------------------------------------
# Merge: boxscore spine + MoneyPuck shot quality
# ---------------------------------------------------------------------------

def merge_shot_data(
    skaters: Optional[pd.DataFrame] = None,
    shot_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Left-join MoneyPuck's xG onto the full dressed lineup.

    The direction of the join is the whole point.  MoneyPuck was the spine
    before, so a skater with no shot event had no row; now the boxscore is the
    spine and MoneyPuck only contributes shot quality.  A skater who does not
    match took no attempts, so ``xg`` and ``shot_attempts`` are filled with 0 —
    a real measurement, not an imputation.

    ``xg_per_attempt`` is deliberately left NaN for those rows.  It is a *rate*,
    undefined with no attempts, and filling it with 0 would claim the skater
    generated attempts of zero quality.  The rolling means in
    ``features.player`` skip NaN, so a no-attempt game contributes nothing to
    the shot-quality features while still contributing a 0 to the SOG rate.

    ``sog`` comes from the boxscore, which is the official line.
    """
    if skaters is None:
        skaters = load_skater_game_stats()
    if shot_stats is None:
        shot_stats = pd.read_parquet(PARQUET_DIR / "player_game_stats.parquet")

    skaters = skaters.copy()
    skaters["player_id"] = skaters["player_id"].astype("int64")
    skaters["game_id"] = skaters["game_id"].astype(str)

    mp = shot_stats.copy()
    mp["game_id"] = mp["game_id"].astype(str)
    # A handful of MoneyPuck rows carry player_id 0 (unattributed shot events).
    mp = mp[mp["player_id"].notna() & (mp["player_id"] > 0)]
    mp["player_id"] = mp["player_id"].astype("int64")
    mp = mp[["game_id", "player_id", "xg", "shot_attempts", "sog"]].rename(
        columns={"sog": "sog_mp"}
    )

    mp = mp.drop_duplicates(subset=["game_id", "player_id"])

    merged = skaters.merge(mp, on=["game_id", "player_id"], how="left")

    # The boxscore abbreviates first names ("S. Bennett"), which reads poorly
    # in the /props embed.  MoneyPuck carries the full name, so build one
    # id -> name map across the whole dataset and prefer it; a skater who has
    # never once taken a shot attempt in five seasons keeps the abbreviation.
    full_names = (
        shot_stats.loc[shot_stats["player_name"].notna(), ["player_id", "player_name"]]
        .dropna()
        .astype({"player_id": "int64"})
        .drop_duplicates(subset=["player_id"])
        .set_index("player_id")["player_name"]
    )
    resolved = merged["player_id"].map(full_names)
    logger.info(
        "Full names resolved for %d/%d rows (%d skaters keep the abbreviation)",
        int(resolved.notna().sum()), len(merged),
        int(merged.loc[resolved.isna(), "player_id"].nunique()),
    )
    merged["player_name"] = resolved.fillna(merged["player_name"])

    matched = merged["shot_attempts"].notna()
    merged["xg"] = merged["xg"].fillna(0.0)
    merged["shot_attempts"] = merged["shot_attempts"].fillna(0).astype("int64")
    attempts = merged["shot_attempts"].astype(float).replace(0.0, np.nan)
    merged["xg_per_attempt"] = merged["xg"] / attempts

    disagree = matched & (merged["sog_mp"] != merged["sog"])
    logger.info(
        "Merged: %d skater-games, %d matched to shot data (%.1f%%), "
        "%d with zero attempts, %d SOG disagreements with MoneyPuck (%.2f%%)",
        len(merged), int(matched.sum()), 100 * matched.mean(),
        int((merged["shot_attempts"] == 0).sum()),
        int(disagree.sum()), 100 * disagree.mean(),
    )
    return merged.drop(columns=["sog_mp"])


# ---------------------------------------------------------------------------
# Scratches (live path — who is out tonight)
# ---------------------------------------------------------------------------

def fetch_scratches(
    game_id: str, *, client: Optional[httpx.Client] = None,
) -> dict[str, list[int]]:
    """Scratched player IDs per team from ``/gamecenter/{id}/right-rail``.

    Not part of the backfill: the training spine already excludes scratches by
    construction, since a scratched player is not in the boxscore.  This is for
    ``pipeline.props_live``, which currently projects any skater with 5+ games
    whether or not he is dressed.
    """
    url = f"{BASE_WEB}/gamecenter/{game_id}/right-rail"
    try:
        resp = (client or httpx).get(url, timeout=15.0)
        resp.raise_for_status()
        info = resp.json().get("gameInfo") or {}
    except Exception as e:
        logger.warning("Failed to fetch scratches for %s: %s", game_id, e)
        return {}
    out: dict[str, list[int]] = {}
    for side in ("homeTeam", "awayTeam"):
        entries = (info.get(side) or {}).get("scratches") or []
        out[side] = [e.get("id") for e in entries if e.get("id")]
    return out


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, name: str) -> Path:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    path = PARQUET_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved %s (%d rows)", path, len(df))
    return path


def load_skater_game_stats(path: Optional[Path] = None) -> pd.DataFrame:
    return pd.read_parquet(path or PARQUET_DIR / "skater_game_stats.parquet")


def load_goalie_starts(path: Optional[Path] = None) -> pd.DataFrame:
    return pd.read_parquet(path or PARQUET_DIR / "goalie_starts.parquet")


def game_universe() -> list[str]:
    """Every game the project already has team-level rows for."""
    team = pd.read_parquet(PARQUET_DIR / "moneypuck_team_game_stats.parquet")
    return sorted(team["game_id"].astype(str).unique())


def main() -> None:
    parser = argparse.ArgumentParser(description="NHL boxscore ingestion")
    parser.add_argument("--backfill", action="store_true",
                        help="Fetch and cache boxscores for every known game")
    parser.add_argument("--build", action="store_true",
                        help="Parse the cache into skater/goalie parquets")
    parser.add_argument("--merge", action="store_true",
                        help="Join MoneyPuck xG onto the lineup spine")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--refresh", action="store_true",
                        help="Re-fetch even games already cached")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N games (for a smoke test)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    # One INFO line per request is 6,877 lines of noise on a full backfill.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not (args.backfill or args.build or args.merge):
        parser.error("nothing to do — pass --backfill, --build and/or --merge")

    games = game_universe()
    if args.limit:
        games = games[: args.limit]
    logger.info("Game universe: %d games", len(games))

    if args.backfill:
        backfill_boxscores(games, workers=args.workers, refresh=args.refresh)

    if args.build:
        skaters = build_skater_game_stats(games)
        if not skaters.empty:
            save_parquet(skaters, "skater_game_stats")
        goalies = build_goalie_starts(games)
        if not goalies.empty:
            save_parquet(goalies, "goalie_starts")

    if args.merge:
        merged = merge_shot_data()
        if not merged.empty:
            save_parquet(merged, "player_game_stats_full")


if __name__ == "__main__":
    main()
