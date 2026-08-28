"""
Situational context features: back-to-back, rest advantage, season day,
head-to-head recent record, and division/conference flags.

Requires game dates to compute rest correctly.  This module tries Postgres
first; if unavailable it approximates dates from the sequential game_id
(last 4 digits = game number within the season, spread over ~185 days).

Output is one row per game (not per team) with home/away perspective columns.
"""

import logging
from typing import Optional

import pandas as pd

from config.season import approximate_game_date

logger = logging.getLogger(__name__)

# NHL division/conference structure (2021-22 onwards, post-Seattle expansion)
# UTA (Utah Hockey Club) replaced ARI in 2024-25
_DIVISIONS: dict[str, str] = {
    # Atlantic
    "BOS": "Atlantic", "BUF": "Atlantic", "DET": "Atlantic", "FLA": "Atlantic",
    "MTL": "Atlantic", "OTT": "Atlantic", "TBL": "Atlantic", "TOR": "Atlantic",
    # Metropolitan
    "CAR": "Metropolitan", "CBJ": "Metropolitan", "NJD": "Metropolitan",
    "NYI": "Metropolitan", "NYR": "Metropolitan", "PHI": "Metropolitan",
    "PIT": "Metropolitan", "WSH": "Metropolitan",
    # Central
    "ARI": "Central", "UTA": "Central", "CHI": "Central", "COL": "Central",
    "DAL": "Central", "MIN": "Central", "NSH": "Central", "STL": "Central",
    "WPG": "Central",
    # Pacific
    "ANA": "Pacific", "CGY": "Pacific", "EDM": "Pacific", "LAK": "Pacific",
    "SJS": "Pacific", "SEA": "Pacific", "VAN": "Pacific", "VGK": "Pacific",
}

_CONFERENCES: dict[str, str] = {
    team: ("Eastern" if div in ("Atlantic", "Metropolitan") else "Western")
    for team, div in _DIVISIONS.items()
}

# Season start dates and the game_id -> date estimator now live in
# config.season, so they cannot drift out of sync with the rest of the
# pipeline (this table used to silently lag a season behind, which blanked
# out rest / back-to-back / season_day for every game of the newest season).


def _load_game_dates_from_db(conn) -> pd.DataFrame:
    """Query game dates, teams, and result from Postgres."""
    sql = """
        SELECT game_id, date, home_team, away_team, home_win
        FROM games
        WHERE date IS NOT NULL
        ORDER BY date, game_id
    """
    df = pd.read_sql(sql, conn)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _approximate_dates(team_features: pd.DataFrame) -> pd.DataFrame:
    """
    Build an approximate game-date DataFrame from game_id sequence numbers.
    Accuracy is ±3 days — good enough for back-to-back detection in most cases.
    """
    games = (
        team_features[["game_id", "season", "game_num", "home_team", "away_team"]]
        .drop_duplicates("game_id")
        .copy()
    )

    def _estimate(row: pd.Series) -> pd.Timestamp:
        try:
            return pd.Timestamp(approximate_game_date(row["season"], row["game_num"]))
        except (ValueError, TypeError):
            logger.warning("Cannot estimate date for season %r", row["season"])
            return pd.NaT

    games["date"] = games.apply(_estimate, axis=1)
    games["home_win"] = None  # unknown without DB
    return games[["game_id", "date", "home_team", "away_team", "home_win"]]


def load_context_features(
    team_features: pd.DataFrame,
    conn=None,
) -> pd.DataFrame:
    """
    Build game-level contextual features.

    Args:
        team_features: output of features.team.load_team_features().
                       Must contain game_id, season, game_num, home_team, away_team.
        conn: optional psycopg2 connection for real dates.

    Returns:
        DataFrame with one row per game_id containing:
        - home_back_to_back, away_back_to_back  (1/0)
        - home_rest_days, away_rest_days         (days since last game; NaN for first game)
        - rest_advantage                         (home_rest - away_rest)
        - season_day                             (days since season opener)
        - h2h_home_win_rate_l3                   (H2H home win rate last 3 meetings, if DB)
    """
    # ------------------------------------------------------------------
    # 1. Get game dates
    # ------------------------------------------------------------------
    if conn is not None:
        try:
            games = _load_game_dates_from_db(conn)
            logger.info("Loaded %d game dates from Postgres", len(games))
        except Exception as e:
            logger.warning("DB date load failed (%s) — approximating from game_id", e)
            games = _approximate_dates(team_features)
    else:
        logger.info("No DB connection — approximating game dates from game_id")
        games = _approximate_dates(team_features)

    games = games.sort_values("date").drop_duplicates("game_id").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 2. Per-team schedule → rest days / back-to-back
    # ------------------------------------------------------------------
    home_sched = games[["game_id", "date", "home_team"]].rename(columns={"home_team": "team"})
    away_sched = games[["game_id", "date", "away_team"]].rename(columns={"away_team": "team"})
    team_sched = (
        pd.concat([home_sched, away_sched], ignore_index=True)
        .sort_values(["team", "date", "game_id"])
        .drop_duplicates(subset=["game_id", "team"])
    )

    team_sched = team_sched.sort_values(["team", "date"])
    team_sched["prev_date"] = team_sched.groupby("team")["date"].shift(1)
    team_sched["days_rest"] = (team_sched["date"] - team_sched["prev_date"]).dt.days
    team_sched["back_to_back"] = (team_sched["days_rest"] == 1).astype(int)

    rest_home = (
        team_sched[["game_id", "team", "days_rest", "back_to_back"]]
        .merge(games[["game_id", "home_team"]], on="game_id")
        .query("team == home_team")
        .rename(columns={"days_rest": "home_rest_days", "back_to_back": "home_back_to_back"})
        [["game_id", "home_rest_days", "home_back_to_back"]]
    )
    rest_away = (
        team_sched[["game_id", "team", "days_rest", "back_to_back"]]
        .merge(games[["game_id", "away_team"]], on="game_id")
        .query("team == away_team")
        .rename(columns={"days_rest": "away_rest_days", "back_to_back": "away_back_to_back"})
        [["game_id", "away_rest_days", "away_back_to_back"]]
    )

    ctx = games[["game_id", "date", "home_team", "away_team", "home_win"]].copy()
    ctx = ctx.merge(rest_home, on="game_id", how="left")
    ctx = ctx.merge(rest_away, on="game_id", how="left")

    # rest_advantage > 0 means home team is more rested
    ctx["rest_advantage"] = ctx["home_rest_days"].fillna(2) - ctx["away_rest_days"].fillna(2)

    # ------------------------------------------------------------------
    # 3. Season day (0 = first game of the season)
    # ------------------------------------------------------------------
    season_start = (
        games.assign(season_year=games["game_id"].str[:4])
        .groupby("season_year")["date"]
        .min()
        .rename("season_start")
    )
    ctx["season_year"] = ctx["game_id"].str[:4]
    ctx = ctx.merge(season_start.reset_index(), on="season_year", how="left")
    ctx["season_day"] = (ctx["date"] - ctx["season_start"]).dt.days
    ctx = ctx.drop(columns=["season_year", "season_start"])

    # ------------------------------------------------------------------
    # 4. Head-to-head recent home win rate (last 3 matchups)
    # ------------------------------------------------------------------
    if ctx["home_win"].notna().any():
        h2h = ctx[["game_id", "date", "home_team", "away_team", "home_win"]].copy()
        h2h["home_win"] = h2h["home_win"].astype(float)
        # Canonical matchup key (alphabetically sorted so both directions match)
        h2h["matchup"] = h2h.apply(
            lambda r: "_".join(sorted([r["home_team"], r["away_team"]])), axis=1
        )
        h2h = h2h.sort_values("date")
        # Rolling mean over last 3 games between these two teams, shifted to exclude current
        h2h["h2h_home_win_rate_l3"] = (
            h2h.groupby("matchup")["home_win"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        )
        ctx = ctx.merge(h2h[["game_id", "h2h_home_win_rate_l3"]], on="game_id", how="left")
    else:
        ctx["h2h_home_win_rate_l3"] = None

    # ------------------------------------------------------------------
    # 5. Division / conference matchup flags
    # ------------------------------------------------------------------
    ctx["same_division"] = ctx.apply(
        lambda r: int(
            _DIVISIONS.get(r["home_team"], "") == _DIVISIONS.get(r["away_team"], "")
            and _DIVISIONS.get(r["home_team"], "") != ""
        ),
        axis=1,
    )
    ctx["same_conference"] = ctx.apply(
        lambda r: int(
            _CONFERENCES.get(r["home_team"], "") == _CONFERENCES.get(r["away_team"], "")
            and _CONFERENCES.get(r["home_team"], "") != ""
        ),
        axis=1,
    )

    logger.info("Context features ready: %d games", len(ctx))
    return ctx
