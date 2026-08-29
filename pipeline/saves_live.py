"""
Live goalie saves projections.

Serves :mod:`models.saves_model` for a given date, and applies the third
drift correction the model cannot: a factor fit on the current season's
completed starts.  Recency weighting and newest-season calibration both
stop at the newest *training* season, which leaves saves projecting about
one season of league drift high (+0.68 in walk-forward).  Games already
played close most of that and leak nothing.

Starter selection is a heuristic and says so.  The NHL does not publish
confirmed starters on a stable endpoint, so this takes each team's most
recent starter, which is right most nights and wrong on the second half of
a back-to-back.  Every row carries ``starter_basis`` so the caller can see
what it is trusting; a confirmed-starter feed would drop straight in here.

Usage:
    python -m pipeline.saves_live [--date YYYY-MM-DD] [--line 24.5]
"""

import argparse
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config.season import current_season
from features.goalie_props import (
    MIN_STARTS,
    OPP_TEAM_COLS,
    OWN_TEAM_COLS,
    load_goalie_game_stats,
    pregame_goalie_snapshot,
)
from ingestion.nhl_api import fetch_schedule
from models.count_model import in_season_factor, over_probability
from models.saves_model import SAVES_LINES, TARGET, load_saves_model

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"

# Same threshold the game and SOG pipelines use, for the same reason: a
# projection built mostly from training-set means is not a projection.
MIN_FEATURE_COVERAGE = 0.70


def probable_starters(history: pd.DataFrame, teams: list[str]) -> dict[str, dict]:
    """Each team's most recent starter, as a stand-in for a confirmed one.

    Returns a mapping of team -> goalie identity plus how stale the pick is,
    so a caller can down-weight a team whose last start was weeks ago.
    """
    starters: dict[str, dict] = {}
    for team in teams:
        played = history[history["team"] == team]
        if played.empty:
            logger.warning("No goalie history for %s -- no projection", team)
            continue
        last = played.sort_values(["season", "game_num"]).iloc[-1]
        starters[team] = {
            "goalie_id":   int(last["goalie_id"]),
            "goalie_name": last["goalie_name"],
            "team":        team,
            "starter_basis": "last start",
        }
    return starters


def _team_context(target_season: Optional[str] = None) -> pd.DataFrame:
    """Latest rolling team stats per team, for both sides of the join."""
    from features.team import load_team_features, pregame_snapshot

    path = PARQUET_DIR / "moneypuck_team_game_stats.parquet"
    if not path.exists():
        logger.error("No %s -- run `python -m pipeline.backfill`", path.name)
        return pd.DataFrame()

    team_stats = load_team_features(path)
    if target_season:
        scoped = team_stats[team_stats["season"] == target_season]
        if not scoped.empty:
            team_stats = scoped
    return pregame_snapshot(team_stats)


def season_for(target_date: date) -> str:
    """The season a date falls in, so a backtest calibrates against its own
    year rather than whatever season it happens to be run in.

    ``current_season`` takes the date and applies the September rollover, so
    a March game lands in the season that started the previous autumn."""
    return current_season(target_date)


def _completed_this_season(
    features: pd.DataFrame,
    season: str,
) -> pd.DataFrame:
    """Rows from that season already played, for in-season calibration."""
    return features[features["season"] == season]


def run(
    target_date: Optional[date] = None,
    lines: Optional[list[float]] = None,
    min_coverage: float = MIN_FEATURE_COVERAGE,
    calibrate_in_season: bool = True,
) -> pd.DataFrame:
    """Projected saves and P(over) for tonight's probable starters."""
    if target_date is None:
        target_date = date.today()
    date_str = target_date.strftime("%Y-%m-%d")
    lines = lines or SAVES_LINES

    games = [
        g for g in fetch_schedule(date_str)
        if g.get("game_type") == "2" and g["date"] == date_str
    ]
    if not games:
        logger.info("No regular-season games on %s", date_str)
        return pd.DataFrame()
    logger.info("Found %d games on %s", len(games), date_str)

    opponent = {}
    for g in games:
        opponent[g["home_team"]] = g["away_team"]
        opponent[g["away_team"]] = g["home_team"]

    try:
        pipeline, feature_cols, dispersion = load_saves_model()
    except FileNotFoundError as e:
        logger.error("%s", e)
        return pd.DataFrame()

    history = load_goalie_game_stats()
    starters = probable_starters(history, list(opponent))
    if not starters:
        logger.warning("No probable starters identified")
        return pd.DataFrame()

    snapshot = pregame_goalie_snapshot(
        history, goalie_ids=[s["goalie_id"] for s in starters.values()],
    )
    snapshot = snapshot[snapshot["starts"] >= MIN_STARTS]
    if snapshot.empty:
        logger.warning("No goalie has %d+ starts yet -- no projections", MIN_STARTS)
        return pd.DataFrame()

    # Schedule identity wins over history: a traded goalie's rows still
    # carry his old team, and tonight's opponent comes from tonight's game.
    identity = pd.DataFrame.from_dict(
        {s["goalie_id"]: s for s in starters.values()}, orient="index",
    )
    identity["opponent"] = identity["team"].map(opponent)
    df = snapshot.drop(columns=["goalie_name", "team"], errors="ignore").join(
        identity, how="inner",
    )
    if df.empty:
        logger.warning("No probable starter cleared the minimum-starts filter")
        return pd.DataFrame()

    season = season_for(target_date)
    context = _team_context(season)
    if not context.empty:
        own = context[OWN_TEAM_COLS].rename(columns={c: "own_" + c for c in OWN_TEAM_COLS})
        opp = context[OPP_TEAM_COLS].rename(columns={c: "opp_" + c for c in OPP_TEAM_COLS})
        df = df.join(own, on="team").join(opp, on="opponent")

    home_teams = {g["home_team"] for g in games}
    df["is_home"] = df["team"].isin(home_teams).astype(float)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        logger.warning(
            "%d/%d saves feature(s) absent: %s", len(missing), len(feature_cols), missing,
        )
        for col in missing:
            df[col] = np.nan

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df["feature_coverage"] = X.notna().mean(axis=1)

    usable = df["feature_coverage"] >= min_coverage
    if not usable.all():
        logger.error(
            "Dropping %d projection(s) below %.0f%% feature coverage",
            int((~usable).sum()), min_coverage * 100,
        )
        df, X = df[usable].copy(), X[usable]
    if df.empty:
        logger.error("No goalie had enough usable features")
        return pd.DataFrame()

    factor = 1.0
    if calibrate_in_season:
        factor = _in_season_correction(pipeline, feature_cols, season)

    df["expected_saves"] = (pipeline.predict(X.values) * factor).round(2)
    df["in_season_factor"] = round(factor, 4)

    for line in lines:
        df[f"p_over_{line}"] = over_probability(
            df["expected_saves"].values, line, dispersion,
        ).round(3)

    out_cols = (
        ["goalie_name", "team", "opponent", "starts", "expected_saves"]
        + [f"p_over_{line}" for line in lines]
        + ["starter_basis", "in_season_factor", "feature_coverage"]
    )
    out = df[[c for c in out_cols if c in df.columns]].copy()
    out = out.sort_values("expected_saves", ascending=False).reset_index(drop=True)

    logger.info("Generated saves projections for %d goalies", len(out))
    return out


def _in_season_correction(pipeline, feature_cols: list[str], season: str) -> float:
    """Fit the serve-time factor on this season's completed starts."""
    from features.goalie_props import attach_team_context, build_goalie_rolling_features
    from features.team import load_team_features

    try:
        played = build_goalie_rolling_features()
        played = attach_team_context(
            played,
            load_team_features(PARQUET_DIR / "moneypuck_team_game_stats.parquet"),
        )
    except FileNotFoundError as e:
        logger.warning("In-season calibration skipped: %s", e)
        return 1.0

    played = _completed_this_season(played, season)
    played = played[played["starts"] >= MIN_STARTS].dropna(subset=[TARGET])
    if played.empty:
        logger.info("In-season calibration skipped: no completed starts in %s", season)
        return 1.0
    return in_season_factor(pipeline, played, feature_cols, TARGET)


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Live NHL goalie saves projections")
    parser.add_argument("--date", default=None,
                        help="Date (YYYY-MM-DD). Default: today.")
    parser.add_argument("--line", type=float, action="append", default=None,
                        help="Line to price (repeatable). Default: 22.5/24.5/27.5/30.5")
    parser.add_argument("--min-coverage", type=float, default=MIN_FEATURE_COVERAGE,
                        help="Drop goalies below this fraction of features populated")
    parser.add_argument("--no-in-season", action="store_true",
                        help="Skip the serve-time drift correction")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()
    preds = run(
        target_date=target,
        lines=args.line,
        min_coverage=args.min_coverage,
        calibrate_in_season=not args.no_in_season,
    )

    if preds.empty:
        print("No projections generated.")
    else:
        print(f"\nGoalie Saves Projections -- {target}")
        print("=" * 96)
        print(preds.to_string(index=False))
        print("\nStarters are the last goalie each team started, not a "
              "confirmed lineup. Check before acting on a back-to-back.")
