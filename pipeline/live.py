"""
Live prediction pipeline — generates pre-game win probabilities for today's NHL games.

Steps:
  1. Fetch today's schedule from NHL API
  2. Load MoneyPuck parquet → compute current rolling stats per team (last N completed games)
  3. Compute context features (rest days, back-to-back) from DB or parquet approximation
  4. Load saved model + feature columns
  5. Build pre-game feature vectors for today's matchups
  6. Generate probabilities
  7. (Optional) save predictions to DB

Usage:
    python -m pipeline.live                        # today's games
    python -m pipeline.live --date 2026-03-10      # specific date
    python -m pipeline.live --dry-run              # print only, no DB save
"""

import argparse
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from features.team import (
    _add_derived_columns, _add_regulation_wins, pregame_snapshot,
)
from features.goalie_mp import GOALIE_WINDOWS
from features.special_teams import ST_WINDOWS
from features.context import _DIVISIONS, _CONFERENCES
from ingestion.nhl_api import fetch_schedule
from config.season import approximate_game_date, current_season, season_start
from pipeline.evaluate_history import HISTORY_PATH

logger = logging.getLogger(__name__)

PARQUET_DIR  = Path(__file__).parent.parent / "data" / "parquet"
SAVED_DIR    = Path(__file__).parent.parent / "models" / "saved"
DEFAULT_MODEL = "random_forest"

# Season identity comes from config.season — see docs/SEASON_ROLLOVER.md.
CURRENT_SEASON = current_season()


# ---------------------------------------------------------------------------
# Rolling stats snapshot
# ---------------------------------------------------------------------------

def _build_team_snapshot(
    parquet_path: Optional[Path] = None,
    season: Optional[str] = None,
    teams: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Return one row per team with rolling pre-game stats for their next game.

    Scoped to a single season and computed by the same code that builds the
    training matrix (``features.team.pregame_snapshot``).  Both properties
    matter: training rolls strictly within a season, so a live snapshot that
    carried form across the season boundary fed the model inputs it had never
    been trained on — worst of all in October, when every carried-over value
    came from the previous campaign.

    Teams with no games yet in the season get an all-NaN row, matching what
    training saw on opening night, instead of being dropped from the slate.

    Index: team abbreviation.
    Columns: {stat}_l5, {stat}_l10, {stat}_l20, {stat}_ewm7, {stat}_home_l10,
             {stat}_away_l10, games_played.
    """
    if parquet_path is None:
        parquet_path = PARQUET_DIR / "moneypuck_team_game_stats.parquet"
    if season is None:
        season = CURRENT_SEASON

    df = pd.read_parquet(parquet_path)
    df = _add_derived_columns(df)
    df = _add_regulation_wins(df)
    df["game_num"] = df["game_id"].str[-4:].astype(int)

    history = df[df["season"] == season].sort_values(["team", "game_num"])
    if history.empty:
        logger.warning(
            "No completed games for %s yet — team features will be NaN-imputed",
            season,
        )

    snapshot = pregame_snapshot(history, teams=teams)
    logger.info(
        "Team snapshot built: %d teams, %d stat columns (season %s, %d games played)",
        len(snapshot), len(snapshot.columns), season, len(history),
    )
    return snapshot


# ---------------------------------------------------------------------------
# Goalie snapshot (most recent starter's rolling stats per team)
# ---------------------------------------------------------------------------

def _build_goalie_snapshot() -> pd.DataFrame:
    """
    Return one row per team with the most recent starter's rolling goalie stats.
    Uses goalie_game_stats.parquet saved during backfill.

    Index: team abbreviation.
    Columns: g_save_pct_l5, g_save_pct_l10, g_gsax_l5, g_gsax_l10.
    """
    path = PARQUET_DIR / "goalie_game_stats.parquet"
    if not path.exists():
        logger.warning("No goalie_game_stats.parquet — goalie features will be NaN")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    starters = df[df["is_starter"]].copy()
    if starters.empty:
        return pd.DataFrame()

    starters["game_num"] = starters["game_id"].str[-4:].astype(int)
    starters = starters.sort_values(["goalie_id", "season", "game_num"])

    # Compute rolling per goalie
    roll_cols = ["save_pct", "gsax"]
    for col in roll_cols:
        for w in GOALIE_WINDOWS:
            starters[f"{col}_l{w}"] = (
                starters.groupby("goalie_id")[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )

    # For each team, take the most recent starter
    rows = []
    for team, grp in starters.groupby("team"):
        last = grp.iloc[-1]
        row = {"team": team}
        for col in roll_cols:
            for w in GOALIE_WINDOWS:
                row[f"g_{col}_l{w}"] = last.get(f"{col}_l{w}", np.nan)
        rows.append(row)

    snapshot = pd.DataFrame(rows).set_index("team")
    logger.info("Goalie snapshot built: %d teams", len(snapshot))
    return snapshot


# ---------------------------------------------------------------------------
# Special teams snapshot (rolling PP/PK per team)
# ---------------------------------------------------------------------------

def _build_special_teams_snapshot() -> pd.DataFrame:
    """
    Return one row per team with rolling PP/PK stats.
    Recomputes from raw MoneyPuck data cached during backfill.

    Index: team abbreviation.
    Columns: pp_goals_l10, pp_goals_l20, pp_xg_l10, pp_xg_l20,
             pk_goals_against_l10, pk_goals_against_l20, etc.
    """
    try:
        from features.special_teams import aggregate_special_teams_stats
    except ImportError:
        logger.warning("special_teams module not available — ST features will be NaN")
        return pd.DataFrame()

    st_games = aggregate_special_teams_stats()
    if st_games.empty:
        return pd.DataFrame()

    st_games = st_games.sort_values(["team", "season", "game_num"])

    roll_cols = ["pp_goals", "pp_xg", "pp_shots",
                 "pk_goals_against", "pk_xg_against", "pk_shots_against"]

    # Compute rolling within season
    for col in roll_cols:
        for w in ST_WINDOWS:
            st_games[f"{col}_l{w}"] = (
                st_games.groupby(["team", "season"])[col]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )

    # For each team, take last row
    rows = []
    for team, grp in st_games.groupby("team"):
        last = grp.iloc[-1]
        row = {"team": team}
        for col in roll_cols:
            for w in ST_WINDOWS:
                row[f"{col}_l{w}"] = last.get(f"{col}_l{w}", np.nan)
        rows.append(row)

    snapshot = pd.DataFrame(rows).set_index("team")
    logger.info("Special teams snapshot built: %d teams", len(snapshot))
    return snapshot


# ---------------------------------------------------------------------------
# ELO snapshot (current ratings per team)
# ---------------------------------------------------------------------------

def _build_elo_snapshot() -> pd.DataFrame:
    """
    Load current ELO ratings from elo_ratings.parquet.

    Index: team abbreviation.
    Columns: elo.
    """
    path = PARQUET_DIR / "elo_ratings.parquet"
    if not path.exists():
        logger.warning("No elo_ratings.parquet — ELO features will be NaN")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    snapshot = df.set_index("team")
    logger.info("ELO snapshot loaded: %d teams", len(snapshot))
    return snapshot


# ---------------------------------------------------------------------------
# Opponent quality snapshot (rolling stats vs strong/weak opponents)
# ---------------------------------------------------------------------------

def _build_opponent_quality_snapshot() -> pd.DataFrame:
    """
    Return one row per team with opponent-quality-adjusted rolling stats.
    Uses the feature matrix built during backfill — extracts the latest
    per-team values for oq columns.

    Index: team abbreviation.
    """
    fm_path = PARQUET_DIR / "feature_matrix.parquet"
    if not fm_path.exists():
        logger.warning("No feature_matrix.parquet — opponent quality features will be NaN")
        return pd.DataFrame()

    fm = pd.read_parquet(fm_path)
    oq_cols = [c for c in fm.columns if "_vs_strong_" in c or "_vs_weak_" in c]
    if not oq_cols:
        return pd.DataFrame()

    # Get home/away oq columns from the latest game per team
    home_oq = [c for c in oq_cols if c.startswith("home_")]

    rows = []
    # Extract from home-team perspective
    for team in fm["home_team"].unique():
        team_games = fm[fm["home_team"] == team].iloc[-1:]
        if team_games.empty:
            continue
        row = {"team": team}
        for col in home_oq:
            base = col.replace("home_", "", 1)
            row[base] = team_games.iloc[0].get(col, np.nan)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    snapshot = pd.DataFrame(rows).set_index("team")
    logger.info("Opponent quality snapshot built: %d teams", len(snapshot))
    return snapshot


# ---------------------------------------------------------------------------
# Last game date per team (for rest / back-to-back)
# ---------------------------------------------------------------------------

def _team_last_game_date(
    parquet_path: Optional[Path] = None,
    conn=None,
) -> dict[str, date]:
    """Return {team: last_completed_game_date}."""
    if conn is not None:
        try:
            sql = """
                SELECT team, MAX(date)::date AS last_date FROM (
                    SELECT home_team AS team, date FROM games WHERE date IS NOT NULL
                    UNION ALL
                    SELECT away_team AS team, date FROM games WHERE date IS NOT NULL
                ) sub
                GROUP BY team
            """
            df = pd.read_sql(sql, conn)
            df["last_date"] = pd.to_datetime(df["last_date"]).dt.date
            return dict(zip(df["team"], df["last_date"]))
        except Exception as e:
            logger.warning("DB last-game query failed: %s — falling back to parquet approximation", e)

    # Parquet fallback: approximate date from game_id sequence
    if parquet_path is None:
        parquet_path = PARQUET_DIR / "moneypuck_team_game_stats.parquet"

    df = pd.read_parquet(parquet_path, columns=["game_id", "team", "season"])
    df["game_num"] = df["game_id"].str[-4:].astype(int)
    df = df.sort_values(["team", "season", "game_num"])

    result: dict[str, date] = {}
    for team, grp in df.groupby("team"):
        last = grp.iloc[-1]
        try:
            result[team] = approximate_game_date(last["season"], last["game_num"])
        except (ValueError, TypeError):
            logger.warning("Cannot estimate last game date for %s", team)
    return result


# ---------------------------------------------------------------------------
# Context features
# ---------------------------------------------------------------------------

def _build_context_row(
    home_team: str,
    away_team: str,
    game_date: date,
    last_game: dict[str, date],
    season_opener: date,
) -> dict:
    """Compute context features for a single scheduled game."""
    def _rest(team: str) -> Optional[float]:
        last = last_game.get(team)
        return float((game_date - last).days) if last is not None else None

    home_rest = _rest(home_team)
    away_rest = _rest(away_team)

    # Division/conference flags
    home_div = _DIVISIONS.get(home_team, "")
    away_div = _DIVISIONS.get(away_team, "")
    same_div = int(home_div == away_div and home_div != "")
    home_conf = _CONFERENCES.get(home_team, "")
    away_conf = _CONFERENCES.get(away_team, "")
    same_conf = int(home_conf == away_conf and home_conf != "")

    return {
        "home_rest_days":    home_rest,
        "away_rest_days":    away_rest,
        "home_back_to_back": 1 if home_rest == 1.0 else 0,
        "away_back_to_back": 1 if away_rest == 1.0 else 0,
        "rest_advantage":    (home_rest or 2.0) - (away_rest or 2.0),
        "season_day":        float((game_date - season_opener).days),
        "h2h_home_win_rate_l3": np.nan,  # filled below if DB available
        "same_division":     same_div,
        "same_conference":   same_conf,
    }


def _load_h2h(conn, home_team: str, away_team: str) -> float:
    """Return average home win rate in the last 3 meetings between these two teams."""
    if conn is None:
        return np.nan
    try:
        sql = """
            SELECT home_team, home_win FROM games
            WHERE (home_team = %(h)s AND away_team = %(a)s)
               OR (home_team = %(a)s AND away_team = %(h)s)
            ORDER BY date DESC NULLS LAST
            LIMIT 3
        """
        df = pd.read_sql(sql, conn, params={"h": home_team, "a": away_team})
        if df.empty:
            return np.nan
        # Flip home_win for rows where today's home team was actually the away team
        wins = []
        for _, row in df.iterrows():
            if row["home_team"] == home_team:
                wins.append(float(row["home_win"]) if pd.notna(row["home_win"]) else 0.5)
            else:
                wins.append(1.0 - float(row["home_win"]) if pd.notna(row["home_win"]) else 0.5)
        return float(np.mean(wins))
    except Exception as e:
        logger.warning("H2H query failed: %s", e)
        return np.nan


# ---------------------------------------------------------------------------
# Build feature matrix for today's games
# ---------------------------------------------------------------------------

def build_live_features(
    games_today: list[dict],
    snapshot: pd.DataFrame,
    last_game: dict[str, date],
    game_date: date,
    conn=None,
    goalie_snapshot: pd.DataFrame | None = None,
    st_snapshot: pd.DataFrame | None = None,
    elo_snapshot: pd.DataFrame | None = None,
    oq_snapshot: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a feature matrix (one row per scheduled game) compatible with the
    training feature matrix produced by pipeline/backfill.py.

    Returns DataFrame with game_id, home_team, away_team, and all model features.
    """
    opener = season_start(CURRENT_SEASON)

    rows = []
    for g in games_today:
        home = g["home_team"]
        away = g["away_team"]

        if home not in snapshot.index or away not in snapshot.index:
            logger.warning(
                "Missing snapshot for %s or %s — skipping %s",
                home, away, g["game_id"],
            )
            continue

        row: dict = {
            "game_id":   g["game_id"],
            "home_team": home,
            "away_team": away,
        }

        h_stats = snapshot.loc[home]
        a_stats = snapshot.loc[away]

        # home_* and away_* rolling features
        for col in h_stats.index:
            row[f"home_{col}"] = h_stats[col]
        for col in a_stats.index:
            row[f"away_{col}"] = a_stats[col]

        # diff_* = home − away (for all stat columns, not games_played)
        stat_cols = [c for c in h_stats.index if c != "games_played"]
        for col in stat_cols:
            h_val = h_stats.get(col, np.nan)
            a_val = a_stats.get(col, np.nan)
            row[f"diff_{col}"] = (
                float(h_val) - float(a_val)
                if pd.notna(h_val) and pd.notna(a_val)
                else np.nan
            )

        # Context
        ctx = _build_context_row(home, away, game_date, last_game, opener)
        ctx["h2h_home_win_rate_l3"] = _load_h2h(conn, home, away)
        row.update(ctx)

        # Goalie features
        if goalie_snapshot is not None and not goalie_snapshot.empty:
            for side, team in [("home", home), ("away", away)]:
                if team in goalie_snapshot.index:
                    for col in goalie_snapshot.columns:
                        row[f"{side}_{col}"] = goalie_snapshot.loc[team, col]

        # Special teams features
        if st_snapshot is not None and not st_snapshot.empty:
            for side, team in [("home", home), ("away", away)]:
                if team in st_snapshot.index:
                    for col in st_snapshot.columns:
                        row[f"{side}_{col}"] = st_snapshot.loc[team, col]
            # Diff features for special teams
            for col in st_snapshot.columns:
                h_val = row.get(f"home_{col}", np.nan)
                a_val = row.get(f"away_{col}", np.nan)
                if pd.notna(h_val) and pd.notna(a_val):
                    row[f"diff_{col}"] = float(h_val) - float(a_val)

        # ELO features
        if elo_snapshot is not None and not elo_snapshot.empty:
            h_elo = elo_snapshot.loc[home, "elo"] if home in elo_snapshot.index else np.nan
            a_elo = elo_snapshot.loc[away, "elo"] if away in elo_snapshot.index else np.nan
            row["home_elo"] = h_elo
            row["away_elo"] = a_elo
            row["diff_elo"] = (
                float(h_elo) - float(a_elo)
                if pd.notna(h_elo) and pd.notna(a_elo)
                else np.nan
            )

        # Opponent quality features
        if oq_snapshot is not None and not oq_snapshot.empty:
            for side, team in [("home", home), ("away", away)]:
                if team in oq_snapshot.index:
                    for col in oq_snapshot.columns:
                        row[f"{side}_{col}"] = oq_snapshot.loc[team, col]
            # Diff features for opponent quality
            for col in oq_snapshot.columns:
                h_val = row.get(f"home_{col}", np.nan)
                a_val = row.get(f"away_{col}", np.nan)
                if pd.notna(h_val) and pd.notna(a_val):
                    row[f"diff_{col}"] = float(h_val) - float(a_val)

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Load model + predict
# ---------------------------------------------------------------------------

def load_model(model_name: str = DEFAULT_MODEL):
    """Load saved pipeline and feature column list from models/saved/."""
    model_path = SAVED_DIR / f"{model_name}.pkl"
    cols_path  = SAVED_DIR / f"{model_name}_feature_cols.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model at {model_path}. Run `python -m pipeline.train` first."
        )

    pipeline     = joblib.load(model_path)
    feature_cols = json.loads(cols_path.read_text(encoding="utf-8"))
    logger.info("Loaded model: %s (%d features)", model_name, len(feature_cols))
    return pipeline, feature_cols


# A prediction built from mostly-imputed features is not a prediction — it is
# the training-set mean dressed up as one.  Below this fraction of usable
# features we refuse to publish rather than emit a confident-looking number.
MIN_FEATURE_COVERAGE = 0.70


def predict(
    live_df: pd.DataFrame,
    pipeline,
    feature_cols: list[str],
    min_coverage: float = MIN_FEATURE_COVERAGE,
) -> pd.DataFrame:
    """Apply the saved model.

    Adds ``prob_home_win`` and ``feature_coverage`` (the fraction of the
    model's features that were actually populated for that game).  Rows below
    ``min_coverage`` are dropped: the imputer silently replaces every missing
    value with a column mean, so a pipeline break upstream used to surface as
    a plausible-looking 50-something percent rather than as an error.
    """
    live_df = live_df.copy()

    missing = [c for c in feature_cols if c not in live_df.columns]
    if missing:
        logger.warning(
            "%d/%d feature(s) absent from the live matrix — NaN-imputed: %s",
            len(missing), len(feature_cols), missing[:10],
        )
        for col in missing:
            live_df[col] = np.nan

    X = live_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    live_df["feature_coverage"] = X.notna().mean(axis=1)

    usable = live_df["feature_coverage"] >= min_coverage
    if not usable.all():
        for _, row in live_df[~usable].iterrows():
            logger.error(
                "Dropping %s @ %s — only %.0f%% of features available (need %.0f%%)",
                row.get("away_team"), row.get("home_team"),
                row["feature_coverage"] * 100, min_coverage * 100,
            )
        live_df = live_df[usable].copy()
        X = X[usable]

    if live_df.empty:
        logger.error(
            "No game had enough usable features — check that the feature "
            "matrix and snapshots were rebuilt (python -m pipeline.daily)",
        )
        return live_df

    worst = live_df["feature_coverage"].min()
    if worst < 1.0:
        logger.info(
            "Feature coverage: min %.1f%%, mean %.1f%%",
            worst * 100, live_df["feature_coverage"].mean() * 100,
        )

    live_df["prob_home_win"] = pipeline.predict_proba(X.values)[:, 1]
    return live_df


# ---------------------------------------------------------------------------
# Market prices
# ---------------------------------------------------------------------------

def attach_market_prices(predictions: pd.DataFrame, target_date: date) -> pd.DataFrame:
    """Add the market's no-vig probability for each game, where available.

    The model's edge is only meaningful against a price, so the price has to
    be recorded alongside the prediction.  Odds move continuously and
    historical lines are not freely available, so a line not captured at
    prediction time is lost.
    """
    if predictions.empty:
        return predictions

    from ingestion.action_network import consensus_index

    index = consensus_index(target_date)
    predictions = predictions.copy()
    if not index:
        logger.warning("No market prices for %s — edge cannot be scored later", target_date)
        predictions["market_prob_home"] = np.nan
        predictions["market_n_books"] = np.nan
        return predictions

    matched = predictions.apply(
        lambda r: index.get((r["home_team"], r["away_team"]), {}), axis=1,
    )
    predictions["market_prob_home"] = [m.get("market_prob_home", np.nan) for m in matched]
    predictions["market_n_books"] = [m.get("market_n_books", np.nan) for m in matched]

    hit = int(predictions["market_prob_home"].notna().sum())
    logger.info("Market price matched for %d/%d games", hit, len(predictions))
    if hit < len(predictions):
        for _, r in predictions[predictions["market_prob_home"].isna()].iterrows():
            logger.warning("No market price: %s @ %s", r["away_team"], r["home_team"])
    return predictions


# ---------------------------------------------------------------------------
# Save prediction history (Parquet — always works, no DB needed)
# ---------------------------------------------------------------------------

def save_prediction_history(predictions: pd.DataFrame, model_name: str) -> Path:
    """Append today's predictions to a Parquet-based history log."""
    path = HISTORY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    records = predictions[["game_id", "home_team", "away_team", "prob_home_win"]].copy()
    records["model_name"] = model_name
    records["predicted_at"] = datetime.now(timezone.utc).isoformat()
    # Include ELO and feature coverage if available
    for col in ["home_elo", "away_elo", "feature_coverage",
                "market_prob_home", "market_n_books"]:
        if col in predictions.columns:
            records[col] = predictions[col]

    if path.exists():
        existing = pd.read_parquet(path)
        # Deduplicate: keep latest prediction per (game_id, model_name)
        combined = pd.concat([existing, records], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["game_id", "model_name"], keep="last"
        )
    else:
        combined = records

    combined.to_parquet(path, index=False)
    logger.info("Prediction history saved: %d total rows → %s", len(combined), path)
    return path


# ---------------------------------------------------------------------------
# Save to DB
# ---------------------------------------------------------------------------

def save_predictions_to_db(predictions: pd.DataFrame, conn, model_name: str) -> int:
    """Upsert predictions to the predictions table. Returns number of rows saved."""
    if conn is None:
        return 0
    try:
        import psycopg2.extras
        rows = [
            {
                "game_id":       str(r["game_id"]),
                "model_name":    model_name,
                "prob_home_win": float(r["prob_home_win"]),
                "predicted_at":  datetime.now(timezone.utc),
            }
            for _, r in predictions.iterrows()
        ]
        with conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO predictions (game_id, model_name, prob_home_win, predicted_at)
                VALUES (%(game_id)s, %(model_name)s, %(prob_home_win)s, %(predicted_at)s)
                ON CONFLICT (game_id, model_name) DO UPDATE SET
                    prob_home_win = EXCLUDED.prob_home_win,
                    predicted_at  = EXCLUDED.predicted_at
                """,
                rows,
            )
        conn.commit()
        logger.info("Saved %d predictions to DB", len(rows))
        return len(rows)
    except Exception as e:
        logger.error("Failed to save predictions to DB: %s", e)
        conn.rollback()
        return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    target_date: Optional[date] = None,
    model_name: str = DEFAULT_MODEL,
    dry_run: bool = False,
    conn=None,
    min_coverage: float = MIN_FEATURE_COVERAGE,
) -> pd.DataFrame:
    """
    Full live prediction pipeline.

    Args:
        target_date:  date to predict for (default: today)
        model_name:   which saved model to use
        dry_run:      skip DB save if True
        conn:         optional psycopg2 connection
        min_coverage: drop games with fewer than this fraction of the model's
                      features populated

    Returns:
        DataFrame with game_id, home_team, away_team, prob_home_win, and context cols.
    """
    if target_date is None:
        target_date = date.today()
    date_str = target_date.strftime("%Y-%m-%d")
    logger.info("Running live predictions for %s", date_str)

    # 1. Fetch today's schedule
    games_today = fetch_schedule(date_str)
    reg_games = [
        g for g in games_today
        if g.get("game_type") == "2" and g["date"] == date_str
    ]
    if not reg_games:
        logger.info("No regular-season games scheduled for %s", date_str)
        return pd.DataFrame()
    logger.info("Found %d regular-season games", len(reg_games))

    # 2. Team rolling snapshot for the current season.  Passing today's teams
    #    guarantees a row for each, so a team that has not played yet is
    #    NaN-imputed rather than silently dropped from the slate.
    slate_teams = sorted({g["home_team"] for g in reg_games} | {g["away_team"] for g in reg_games})
    snapshot = _build_team_snapshot(teams=slate_teams)

    # 2b. Goalie, special teams, ELO, and opponent quality snapshots
    goalie_snapshot = _build_goalie_snapshot()
    st_snapshot = _build_special_teams_snapshot()
    elo_snapshot = _build_elo_snapshot()
    oq_snapshot = _build_opponent_quality_snapshot()

    # 3. Last game date per team (for rest / B2B)
    last_game = _team_last_game_date(
        parquet_path=PARQUET_DIR / "moneypuck_team_game_stats.parquet",
        conn=conn,
    )

    # 4. Build live feature matrix
    live_df = build_live_features(
        reg_games, snapshot, last_game, target_date, conn=conn,
        goalie_snapshot=goalie_snapshot,
        st_snapshot=st_snapshot,
        elo_snapshot=elo_snapshot,
        oq_snapshot=oq_snapshot,
    )
    if live_df.empty:
        logger.warning("No games could be featurized — check team abbreviation mapping")
        return pd.DataFrame()

    # 5. Load model + predict
    pipeline, feature_cols = load_model(model_name)
    predictions = predict(live_df, pipeline, feature_cols, min_coverage=min_coverage)
    if predictions.empty:
        return pd.DataFrame()

    # 5b. Market price at prediction time.  Best-effort: a missing line costs
    #     a column, never a prediction.  This is the perishable half of the
    #     record — odds move, and a price not captured now cannot be
    #     reconstructed later.
    predictions = attach_market_prices(predictions, target_date)

    # 6. Always save prediction history (Parquet)
    save_prediction_history(predictions, model_name)

    # 6b. Optionally save to DB
    if not dry_run and conn is not None:
        save_predictions_to_db(predictions, conn, model_name)

    # 7. Return display columns
    display_cols = [
        "game_id", "home_team", "away_team", "prob_home_win",
        "home_back_to_back", "away_back_to_back", "rest_advantage",
        "home_elo", "away_elo", "feature_coverage", "market_prob_home",
    ]
    keep = [c for c in display_cols if c in predictions.columns]
    return predictions[keep]


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Live NHL game predictions")
    parser.add_argument("--date", default=None,
                        help="Date to predict (YYYY-MM-DD). Default: today.")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        choices=["random_forest", "logistic_regression"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print predictions without saving to DB")
    parser.add_argument("--min-coverage", type=float, default=MIN_FEATURE_COVERAGE,
                        help="Drop games with fewer than this fraction of "
                             "model features populated (0-1)")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else date.today()

    conn = None
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        try:
            import psycopg2
            conn = psycopg2.connect(db_url)
            logger.info("Connected to Postgres")
        except Exception as e:
            logger.warning("DB connect failed: %s — running without DB", e)

    preds = run(
        target_date=target, model_name=args.model, dry_run=args.dry_run,
        conn=conn, min_coverage=args.min_coverage,
    )

    if preds.empty:
        print("No predictions generated.")
    else:
        print(f"\nPredictions for {target}:")
        print(preds.to_string(index=False))

    if conn:
        conn.close()
