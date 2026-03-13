"""
MoneyPuck.com shot data ingestion.

Downloads shot-level ZIP files from peter-tanner.com/moneypuck,
aggregates each shot to per-game team stats (xG, Corsi, Fenwick,
high-danger chances), and saves to Parquet + PostgreSQL.

MoneyPuck data portal: https://moneypuck.com/data.htm
Shot ZIP source: https://peter-tanner.com/moneypuck/downloads/
"""

import io
import logging
import zipfile
from pathlib import Path

import httpx
import pandas as pd
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"

# Shot ZIP URL pattern — season = start year (e.g. 2021 for 2021-2022)
SHOT_ZIP_URL = "https://peter-tanner.com/moneypuck/downloads/shots_{year}.zip"

# Seasons to ingest: display name -> URL year
MP_SEASONS = {
    "2021-2022": "2021",
    "2022-2023": "2022",
    "2023-2024": "2023",
    "2024-2025": "2024",
    "2025-2026": "2025",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://moneypuck.com/data.htm",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def download_shots_zip(season: str) -> pd.DataFrame:
    """
    Download and extract the shot-level CSV for a season.
    season: display name e.g. "2021-2022"
    Caches the extracted CSV to data/raw/.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    year = MP_SEASONS.get(season, season.split("-")[0])
    cache_path = RAW_DIR / f"moneypuck_shots_{year}.csv"

    if cache_path.exists():
        logger.info("Loading cached shots: %s", cache_path.name)
        return pd.read_csv(cache_path, low_memory=False)

    url = SHOT_ZIP_URL.format(year=year)
    logger.info("Downloading %s ...", url)

    resp = httpx.get(url, headers=HEADERS, timeout=300, follow_redirects=True)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found in zip for season {season}")
        csv_name = csv_names[0]
        logger.info("Extracting %s from zip", csv_name)
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, low_memory=False)

    df.to_csv(cache_path, index=False)
    logger.info("Cached %d shots → %s", len(df), cache_path.name)
    return df


# ---------------------------------------------------------------------------
# Aggregation: shots → per-game team stats
# ---------------------------------------------------------------------------
def aggregate_shots_to_games(shots: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Aggregate shot-level data to per-game team stats.

    Actual MoneyPuck columns used:
      game_id            — game identifier (already snake_case)
      teamCode           — shooting team abbreviation (e.g. "TBL")
      homeTeamCode       — home team abbreviation
      awayTeamCode       — away team abbreviation
      isHomeTeam         — 1 if shooting team is home
      xGoal              — expected goals for this shot
      goal               — 1 if shot was a goal
      shotWasOnGoal      — 1 if shot on goal (SOG)
      event              — SHOT | MISS | GOAL (all are shot attempts = Corsi)
      homeSkatersOnIce   — skaters on ice for home team
      awaySkatersOnIce   — skaters on ice for away team
      arenaAdjustedShotDistance — for high-danger approximation (< 20ft)
    """
    df = shots.copy()
    df["game_id"] = df["game_id"].astype(str)

    # Drop the "team" column ("HOME"/"AWAY") — we want teamCode as "team"
    df = df.drop(columns=["team"], errors="ignore")

    df = df.rename(columns={
        "teamCode":       "team",
        "homeTeamCode":   "home_team",
        "awayTeamCode":   "away_team",
        "isHomeTeam":     "is_home",
        "xGoal":          "xGoal",
        "goal":           "goal",
        "shotWasOnGoal":  "sog",
    })

    # Ensure is_home is boolean
    df["is_home"] = df["is_home"].astype(bool)

    # 5v5 xG
    mask_5v5 = (df["homeSkatersOnIce"] == 5) & (df["awaySkatersOnIce"] == 5)
    df["xg_5v5"] = df["xGoal"].where(mask_5v5, 0.0)

    # High-danger: shots within ~20 feet (MoneyPuck convention)
    df["is_hd"] = (df["arenaAdjustedShotDistance"] < 20).astype(int)
    df["hd_goal"] = df["is_hd"] * df["goal"]

    # Aggregate: one row per game × team (shooting team = "for" perspective)
    per_game = (
        df.groupby(["game_id", "team", "is_home", "home_team", "away_team"])
        .agg(
            xg_for        = ("xGoal",   "sum"),
            xg_for_5v5    = ("xg_5v5",  "sum"),
            goals_for     = ("goal",    "sum"),
            shots_for     = ("sog",     "sum"),
            corsi_for     = ("game_id", "count"),   # every row = a shot attempt
            hd_chances_for= ("is_hd",   "sum"),
            hd_goals_for  = ("hd_goal", "sum"),
        )
        .reset_index()
    )

    # Build opponent lookup to get "against" stats
    opp_cols = ["game_id", "team", "xg_for", "goals_for", "shots_for",
                "corsi_for", "hd_chances_for", "hd_goals_for"]
    opp = per_game[opp_cols].rename(columns={
        "team":           "opp_team",
        "xg_for":         "xg_against",
        "goals_for":      "goals_against",
        "shots_for":      "shots_against",
        "corsi_for":      "corsi_against",
        "hd_chances_for": "hd_chances_against",
        "hd_goals_for":   "hd_goals_against",
    })

    # Each team's opponent is the other team in the same game
    per_game["opp_team"] = per_game.apply(
        lambda r: r["away_team"] if r["team"] == r["home_team"] else r["home_team"],
        axis=1,
    )
    per_game = per_game.merge(opp, on=["game_id", "opp_team"], how="left")

    per_game["season"] = season
    per_game["source"] = "moneypuck"

    return per_game


# ---------------------------------------------------------------------------
# Parquet export
# ---------------------------------------------------------------------------
def save_parquet(df: pd.DataFrame, name: str) -> Path:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    path = PARQUET_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows → %s", len(df), path)
    return path


# ---------------------------------------------------------------------------
# Game ID conversion
# ---------------------------------------------------------------------------
def mp_game_id_to_nhl(mp_id: str, season: str) -> str:
    """
    Convert MoneyPuck game_id to NHL API format.
    MoneyPuck: "20001"  (game_type digit + 4-digit game number)
    NHL API:   "2021020001"  (4-digit season year + 2-digit game type + 4-digit game number)
    """
    season_year = season.split("-")[0]   # "2021-2022" -> "2021"
    return f"{season_year}0{mp_id}"


# ---------------------------------------------------------------------------
# Postgres upsert
# ---------------------------------------------------------------------------
def upsert_game_stubs(conn, shots: pd.DataFrame, season: str) -> None:
    """
    Insert minimal game stubs into the games table from MoneyPuck shot data.
    Runs before upsert_team_stats to satisfy the foreign key constraint.
    """
    season_year = season.split("-")[0]

    game_info = (
        shots.groupby("game_id")
        .first()
        .reset_index()[["game_id", "homeTeamCode", "awayTeamCode", "isPlayoffGame", "homeTeamWon"]]
    )

    rows = []
    for _, row in game_info.iterrows():
        nhl_id = mp_game_id_to_nhl(str(int(row["game_id"])), season)
        game_type = "P" if row.get("isPlayoffGame", 0) else "R"
        home_win = bool(row["homeTeamWon"]) if pd.notnull(row.get("homeTeamWon")) else None
        rows.append({
            "game_id":   nhl_id,
            "date":      None,          # NHL API will fill this in later
            "season":    season_year + str(int(season_year) + 1),
            "game_type": game_type,
            "home_team": row["homeTeamCode"],
            "away_team": row["awayTeamCode"],
            "home_win":  home_win,
        })

    sql = """
        INSERT INTO games (game_id, date, season, game_type, home_team, away_team, home_win)
        VALUES (%(game_id)s, %(date)s, %(season)s, %(game_type)s, %(home_team)s, %(away_team)s, %(home_win)s)
        ON CONFLICT (game_id) DO NOTHING
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    logger.info("Upserted %d game stubs for %s", len(rows), season)


def upsert_team_stats(conn, df: pd.DataFrame) -> None:
    cols = [
        "game_id", "team", "is_home", "goals_for", "goals_against",
        "shots_for", "shots_against", "corsi_for", "corsi_against",
        "xg_for", "xg_against", "hd_chances_for", "hd_goals_for", "source",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    rows = df[existing_cols].where(pd.notnull(df[existing_cols]), None).to_dict("records")
    if not rows:
        return

    col_list    = ", ".join(existing_cols)
    placeholder = ", ".join(f"%({c})s" for c in existing_cols)
    update      = ", ".join(
        f"{c} = EXCLUDED.{c}" for c in existing_cols if c not in ("game_id", "team")
    )
    sql = f"""
        INSERT INTO team_stats ({col_list})
        VALUES ({placeholder})
        ON CONFLICT (game_id, team) DO UPDATE SET
            {update},
            ingested_at = NOW()
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=500)
    logger.info("Upserted %d team_stats rows", len(rows))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def ingest_all_seasons(conn=None, seasons: list[str] | None = None) -> None:
    """
    Download MoneyPuck shot ZIPs, aggregate to per-game team stats,
    save Parquet files, and optionally load into Postgres.
    """
    if seasons is None:
        seasons = list(MP_SEASONS.keys())

    all_frames: list[pd.DataFrame] = []

    for season in seasons:
        logger.info("=== Season %s ===", season)
        shots = download_shots_zip(season)
        logger.info("Aggregating %d shots to per-game stats...", len(shots))
        per_game = aggregate_shots_to_games(shots, season)
        logger.info("Got %d team-game rows for %s", len(per_game), season)

        # Convert MoneyPuck game_ids to NHL API format before saving/upserting
        per_game["game_id"] = per_game["game_id"].apply(
            lambda gid: mp_game_id_to_nhl(str(int(float(gid))), season)
        )

        all_frames.append(per_game)

        if conn:
            # Must insert game stubs first to satisfy FK constraint
            upsert_game_stubs(conn, shots, season)
            conn.commit()
            upsert_team_stats(conn, per_game)
            conn.commit()

    combined = pd.concat(all_frames, ignore_index=True)
    save_parquet(combined, "moneypuck_team_game_stats")

    logger.info(
        "Done. %d total team-game rows across %d seasons.",
        len(combined), len(seasons),
    )


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    db_url = os.environ.get("DATABASE_URL")
    conn = psycopg2.connect(db_url) if db_url else None

    ingest_all_seasons(conn=conn)

    if conn:
        conn.close()
