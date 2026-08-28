"""
Single source of truth for NHL season configuration.

Before this module existed, the season list, the current season, and the
season start dates were duplicated across ten files in three different string
formats.  Every rollover meant hunting them all down, and one of them
(``features/context.py``) was silently a season behind — which blanked out
rest / back-to-back / season_day for every game of the missing season.

Everything season-related now derives from :data:`FIRST_SEASON_YEAR` and
:data:`SEASON_STARTS`.  Rolling to a new season is a one-line change: add the
opener date to :data:`SEASON_STARTS`.  If you forget, the season list still
extends automatically and the opener falls back to an estimate, so the
pipeline keeps running (with a warning) rather than producing NaNs.

Season formats used around the codebase
---------------------------------------
=================  ==============  =========================================
Format             Example         Used by
=================  ==============  =========================================
label              ``2025-2026``   MoneyPuck parquet, feature matrix, models
api                ``20252026``    NHL API endpoints (rosters, game logs)
year               ``2025``        MoneyPuck shot-ZIP filenames
=================  ==============  =========================================
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta

logger = logging.getLogger(__name__)

# First season with MoneyPuck coverage in this project.
FIRST_SEASON_YEAR = 2021

# Known regular-season opening dates, keyed by season start year.
# Add the real date here each year — it is the only annual edit required.
SEASON_STARTS: dict[int, str] = {
    2021: "2021-10-12",
    2022: "2022-10-07",
    2023: "2023-10-10",
    2024: "2024-10-08",
    2025: "2025-10-08",
}

# A new season is considered "current" from this month onward, so pre-season
# runs in September already resolve to the upcoming season.
SEASON_ROLLOVER_MONTH = 9

# Structural constants for the game_id -> approximate-date fallback.
SEASON_DAYS = 185      # regular season spans roughly 185 days
TOTAL_GAMES = 1312     # 32 teams x 82 games / 2

# Fallback opener when a season's real date has not been added yet:
# the NHL almost always opens in the first full week of October.
_FALLBACK_OPENER_MONTH = 10
_FALLBACK_OPENER_DAY = 8


# ---------------------------------------------------------------------------
# Season year <-> string formats
# ---------------------------------------------------------------------------

def season_label(year: int) -> str:
    """2025 -> "2025-2026" (MoneyPuck / feature-matrix format)."""
    return f"{year}-{year + 1}"


def season_api(year: int) -> str:
    """2025 -> "20252026" (NHL API format)."""
    return f"{year}{year + 1}"


def season_year(season: str | int) -> int:
    """Accept any season format and return the start year as an int.

    >>> season_year("2025-2026"), season_year("20252026"), season_year(2025)
    (2025, 2025, 2025)
    """
    if isinstance(season, int):
        return season
    s = str(season).strip()
    if "-" in s:
        return int(s.split("-")[0])
    if len(s) == 8:          # "20252026"
        return int(s[:4])
    if len(s) == 4:          # "2025"
        return int(s)
    raise ValueError(f"Unrecognized season format: {season!r}")


def label_to_api(season: str) -> str:
    """"2025-2026" -> "20252026"."""
    return season_api(season_year(season))


def api_to_label(season: str) -> str:
    """"20252026" -> "2025-2026"."""
    return season_label(season_year(season))


# ---------------------------------------------------------------------------
# Current season
# ---------------------------------------------------------------------------

def current_season_year(today: date | None = None) -> int:
    """Start year of the season in progress (or about to start).

    Rolls over on :data:`SEASON_ROLLOVER_MONTH` so that pre-season work in
    September targets the upcoming season.  Override with the
    ``NHL_CURRENT_SEASON`` environment variable (any season format) to pin a
    season for backfills or replay.
    """
    override = os.environ.get("NHL_CURRENT_SEASON")
    if override:
        return season_year(override)

    if today is None:
        today = date.today()
    return today.year if today.month >= SEASON_ROLLOVER_MONTH else today.year - 1


def current_season(today: date | None = None) -> str:
    """Current season as a label, e.g. ``"2026-2027"``."""
    return season_label(current_season_year(today))


def current_season_api(today: date | None = None) -> str:
    """Current season in NHL API format, e.g. ``"20262027"``."""
    return season_api(current_season_year(today))


# ---------------------------------------------------------------------------
# Season lists
# ---------------------------------------------------------------------------

def season_years(today: date | None = None) -> list[int]:
    """All season start years from :data:`FIRST_SEASON_YEAR` to the current one."""
    return list(range(FIRST_SEASON_YEAR, current_season_year(today) + 1))


def all_seasons(today: date | None = None) -> list[str]:
    """All season labels, chronological. Extends automatically each year."""
    return [season_label(y) for y in season_years(today)]


def moneypuck_seasons(today: date | None = None) -> dict[str, str]:
    """``{"2025-2026": "2025", ...}`` — season label to shot-ZIP year."""
    return {season_label(y): str(y) for y in season_years(today)}


# ---------------------------------------------------------------------------
# Season start dates
# ---------------------------------------------------------------------------

def season_start(season: str | int) -> date:
    """Opening date of a season's regular season.

    Falls back to an estimate (Oct 8) with a warning when the real date has
    not been added to :data:`SEASON_STARTS` yet, so a forgotten rollover
    degrades to a few days of date error instead of NaT.
    """
    year = season_year(season)
    known = SEASON_STARTS.get(year)
    if known:
        return date.fromisoformat(known)

    estimate = date(year, _FALLBACK_OPENER_MONTH, _FALLBACK_OPENER_DAY)
    logger.warning(
        "No opening date for %s — estimating %s. "
        "Add the real date to config.season.SEASON_STARTS.",
        season_label(year), estimate,
    )
    return estimate


def season_starts_by_label(today: date | None = None) -> dict[str, str]:
    """``{"2025-2026": "2025-10-08", ...}`` for every known season."""
    return {season_label(y): season_start(y).isoformat() for y in season_years(today)}


def approximate_game_date(season: str | int, game_num: int) -> date:
    """Estimate a game's date from its sequence number within the season.

    Used when Postgres game dates are unavailable.  Accurate to a few days —
    good enough for back-to-back detection in most cases.
    """
    offset = int(game_num / TOTAL_GAMES * SEASON_DAYS)
    return season_start(season) + timedelta(days=offset)
