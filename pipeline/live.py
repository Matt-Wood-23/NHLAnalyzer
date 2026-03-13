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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from features.team import _add_derived_columns, ROLL_STATS, WINDOWS
from features.goalie_mp import GOALIE_WINDOWS
from features.special_teams import ST_WINDOWS
from ingestion.nhl_api import fetch_schedule

logger = logging.getLogger(__name__)

PARQUET_DIR  = Path(__file__).parent.parent / "data" / "parquet"
SAVED_DIR    = Path(__file__).parent.parent / "models" / "saved"
DEFAULT_MODEL = "random_forest"

# Current season (used to scope rolling windows to recent games)
CURRENT_SEASON = "2025-2026"

_SEASON_START: dict[str, str] = {
    "2021-2022": "2021-10-12",
    "2022-2023": "2022-10-07",
    "2023-2024": "2023-10-10",
    "2024-2025": "2024-10-08",
    "2025-2026": "2025-10-08",
}


# ---------------------------------------------------------------------------
# Rolling stats snapshot
# ---------------------------------------------------------------------------

def _build_team_snapshot(parquet_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Return one row per team with rolling pre-game stats.
    Each row = rolling mean of that team's last 5 / 10 / 20 completed games
    (across season boundaries — we want their real recent form).

    Index: team abbreviation.
    Columns: {stat}_l5, {stat}_l10, {stat}_l20, games_played.
    """
    if parquet_path is None:
        parquet_path = PARQUET_DIR / "moneypuck_team_game_stats.parquet"

    df = pd.read_parquet(parquet_path)
    df = _add_derived_columns(df)
    df["game_num"] = df["game_id"].str[-4:].astype(int)
    df = df.sort_values(["team", "season", "game_num"])

    max_w = max(WINDOWS)
    rows = []
    for team, grp in df.groupby("team", sort=False):
        row: dict = {"team": team}
        recent = grp.tail(max_w)
        for col in ROLL_STATS:
            if col not in recent.columns:
                continue
            for w in WINDOWS:
                vals = recent.tail(w)[col].dropna()
                row[f"{col}_l{w}"] = float(vals.mean()) if len(vals) > 0 else np.nan
        row["games_played"] = len(grp)
        rows.append(row)

    snapshot = pd.DataFrame(rows).set_index("team")
    logger.info(
        "Team snapshot built: %d teams, %d stat columns",
        len(snapshot), len(snapshot.columns),
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
        start_str = _SEASON_START.get(last["season"])
        if start_str:
            start = pd.Timestamp(start_str).date()
            offset = int(last["game_num"] / 1230 * 185)
            result[team] = start + timedelta(days=offset)
    return result


# ---------------------------------------------------------------------------
# Context features
# ---------------------------------------------------------------------------

def _build_context_row(
    home_team: str,
    away_team: str,
    game_date: date,
    last_game: dict[str, date],
    season_start: date,
) -> dict:
    """Compute context features for a single scheduled game."""
    def _rest(team: str) -> Optional[float]:
        last = last_game.get(team)
        return float((game_date - last).days) if last is not None else None

    home_rest = _rest(home_team)
    away_rest = _rest(away_team)
    return {
        "home_rest_days":    home_rest,
        "away_rest_days":    away_rest,
        "home_back_to_back": 1 if home_rest == 1.0 else 0,
        "away_back_to_back": 1 if away_rest == 1.0 else 0,
        "rest_advantage":    (home_rest or 2.0) - (away_rest or 2.0),
        "season_day":        float((game_date - season_start).days),
        "h2h_home_win_rate_l3": np.nan,  # filled below if DB available
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
) -> pd.DataFrame:
    """
    Build a feature matrix (one row per scheduled game) compatible with the
    training feature matrix produced by pipeline/backfill.py.

    Returns DataFrame with game_id, home_team, away_team, and all model features.
    """
    season_start_str = _SEASON_START.get(CURRENT_SEASON, "2025-10-08")
    season_start = pd.Timestamp(season_start_str).date()

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
        ctx = _build_context_row(home, away, game_date, last_game, season_start)
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
    feature_cols = json.loads(cols_path.read_text())
    logger.info("Loaded model: %s (%d features)", model_name, len(feature_cols))
    return pipeline, feature_cols


def predict(live_df: pd.DataFrame, pipeline, feature_cols: list[str]) -> pd.DataFrame:
    """Apply the saved model. Returns live_df with prob_home_win column added."""
    missing = [c for c in feature_cols if c not in live_df.columns]
    if missing:
        logger.warning(
            "%d feature(s) missing — will be NaN-imputed: %s",
            len(missing), missing[:10],
        )
        for col in missing:
            live_df[col] = np.nan

    X = live_df[feature_cols].values
    live_df = live_df.copy()
    live_df["prob_home_win"] = pipeline.predict_proba(X)[:, 1]
    return live_df


# ---------------------------------------------------------------------------
# Save prediction history (Parquet — always works, no DB needed)
# ---------------------------------------------------------------------------

HISTORY_DIR = Path(__file__).parent.parent / "data" / "predictions"


def save_prediction_history(predictions: pd.DataFrame, model_name: str) -> Path:
    """Append today's predictions to a Parquet-based history log."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    path = HISTORY_DIR / "prediction_history.parquet"

    records = predictions[["game_id", "home_team", "away_team", "prob_home_win"]].copy()
    records["model_name"] = model_name
    records["predicted_at"] = datetime.utcnow().isoformat()
    # Include ELO if available
    for col in ["home_elo", "away_elo"]:
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
                "predicted_at":  datetime.utcnow(),
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
) -> pd.DataFrame:
    """
    Full live prediction pipeline.

    Args:
        target_date: date to predict for (default: today)
        model_name:  which saved model to use
        dry_run:     skip DB save if True
        conn:        optional psycopg2 connection

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

    # 2. Team rolling snapshot (last N completed games)
    snapshot = _build_team_snapshot()

    # 2b. Goalie, special teams, and ELO snapshots
    goalie_snapshot = _build_goalie_snapshot()
    st_snapshot = _build_special_teams_snapshot()
    elo_snapshot = _build_elo_snapshot()

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
    )
    if live_df.empty:
        logger.warning("No games could be featurized — check team abbreviation mapping")
        return pd.DataFrame()

    # 5. Load model + predict
    pipeline, feature_cols = load_model(model_name)
    predictions = predict(live_df, pipeline, feature_cols)

    # 6. Always save prediction history (Parquet)
    save_prediction_history(predictions, model_name)

    # 6b. Optionally save to DB
    if not dry_run and conn is not None:
        save_predictions_to_db(predictions, conn, model_name)

    # 7. Return display columns
    display_cols = [
        "game_id", "home_team", "away_team", "prob_home_win",
        "home_back_to_back", "away_back_to_back", "rest_advantage",
        "home_elo", "away_elo",
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

    preds = run(target_date=target, model_name=args.model, dry_run=args.dry_run, conn=conn)

    if preds.empty:
        print("No predictions generated.")
    else:
        print(f"\nPredictions for {target}:")
        print(preds.to_string(index=False))

    if conn:
        conn.close()
