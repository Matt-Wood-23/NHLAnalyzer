-- NHL ML Analyzer — PostgreSQL Schema
-- Phase 1: Data Foundation

-- ============================================================
-- GAMES
-- Core game result table. One row per completed game.
-- ============================================================
CREATE TABLE IF NOT EXISTS games (
    game_id       VARCHAR(20) PRIMARY KEY,   -- e.g. "2023020001" (NHL API format)
    date          DATE,
    season        VARCHAR(10) NOT NULL,       -- e.g. "20232024"
    game_type     VARCHAR(10) NOT NULL,       -- "R" regular, "P" playoff
    home_team     VARCHAR(5)  NOT NULL,       -- e.g. "TOR", "BOS"
    away_team     VARCHAR(5)  NOT NULL,
    home_score    SMALLINT,
    away_score    SMALLINT,
    home_win      BOOLEAN,                   -- NULL until game complete
    went_to_ot    BOOLEAN     DEFAULT FALSE,
    went_to_so    BOOLEAN     DEFAULT FALSE,
    ingested_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_games_date    ON games (date);
CREATE INDEX IF NOT EXISTS idx_games_season  ON games (season);
CREATE INDEX IF NOT EXISTS idx_games_home    ON games (home_team);
CREATE INDEX IF NOT EXISTS idx_games_away    ON games (away_team);


-- ============================================================
-- TEAM STATS
-- Per-game team-level stats. Two rows per game (one per team).
-- ============================================================
CREATE TABLE IF NOT EXISTS team_stats (
    id              SERIAL PRIMARY KEY,
    game_id         VARCHAR(20) NOT NULL REFERENCES games(game_id),
    team            VARCHAR(5)  NOT NULL,
    is_home         BOOLEAN     NOT NULL,

    -- Basic counting stats
    goals_for       SMALLINT,
    goals_against   SMALLINT,
    shots_for       SMALLINT,
    shots_against   SMALLINT,

    -- Advanced / zone stats
    corsi_for       SMALLINT,               -- shot attempts for
    corsi_against   SMALLINT,
    fenwick_for     SMALLINT,               -- unblocked shot attempts for
    fenwick_against SMALLINT,
    xg_for          NUMERIC(6,4),           -- expected goals for (5v5)
    xg_against      NUMERIC(6,4),

    -- Special teams
    pp_opportunities SMALLINT,
    pp_goals        SMALLINT,
    pk_opportunities SMALLINT,
    pk_goals_against SMALLINT,

    -- High-danger
    hd_chances_for      SMALLINT,
    hd_chances_against  SMALLINT,
    hd_goals_for        SMALLINT,
    hd_goals_against    SMALLINT,

    source          VARCHAR(30),            -- "nhl_api", "moneypuck"
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (game_id, team)
);

CREATE INDEX IF NOT EXISTS idx_team_stats_game   ON team_stats (game_id);
CREATE INDEX IF NOT EXISTS idx_team_stats_team   ON team_stats (team);


-- ============================================================
-- GOALIE STATS
-- Per-game goalie performance. One row per goalie per game.
-- ============================================================
CREATE TABLE IF NOT EXISTS goalie_stats (
    id                  SERIAL PRIMARY KEY,
    game_id             VARCHAR(20) NOT NULL REFERENCES games(game_id),
    goalie_id           INTEGER     NOT NULL,   -- NHL API player ID
    goalie_name         VARCHAR(60),
    team                VARCHAR(5)  NOT NULL,
    is_starter          BOOLEAN,

    shots_against       SMALLINT,
    saves               SMALLINT,
    goals_against       SMALLINT,
    save_pct            NUMERIC(5,4),

    -- Advanced
    xg_against          NUMERIC(6,4),           -- expected goals against
    gsax                NUMERIC(6,4),           -- goals saved above expected
    hd_shots_against    SMALLINT,
    hd_saves            SMALLINT,
    hd_save_pct         NUMERIC(5,4),

    toi_seconds         INTEGER,                -- time on ice in seconds

    ingested_at         TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (game_id, goalie_id)
);

CREATE INDEX IF NOT EXISTS idx_goalie_stats_game    ON goalie_stats (game_id);
CREATE INDEX IF NOT EXISTS idx_goalie_stats_goalie  ON goalie_stats (goalie_id);
CREATE INDEX IF NOT EXISTS idx_goalie_stats_team    ON goalie_stats (team);


-- ============================================================
-- ODDS
-- Betting market data. One row per game per sportsbook.
-- ============================================================
CREATE TABLE IF NOT EXISTS odds (
    id              SERIAL PRIMARY KEY,
    game_id         VARCHAR(20) NOT NULL REFERENCES games(game_id),
    book            VARCHAR(40) NOT NULL,       -- e.g. "draftkings", "fanduel"

    -- Moneyline (American odds format, stored as integers)
    open_home_ml    SMALLINT,
    open_away_ml    SMALLINT,
    close_home_ml   SMALLINT,
    close_away_ml   SMALLINT,

    -- Derived implied probabilities (no-vig)
    open_home_prob  NUMERIC(5,4),
    open_away_prob  NUMERIC(5,4),
    close_home_prob NUMERIC(5,4),
    close_away_prob NUMERIC(5,4),

    -- Totals
    open_total      NUMERIC(4,1),
    close_total     NUMERIC(4,1),
    open_over_ml    SMALLINT,
    close_over_ml   SMALLINT,

    ingested_at     TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (game_id, book)
);

CREATE INDEX IF NOT EXISTS idx_odds_game ON odds (game_id);


-- ============================================================
-- PREDICTIONS
-- Model output. One row per game per model version.
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    game_id         VARCHAR(20) NOT NULL REFERENCES games(game_id),
    model_version   VARCHAR(40) NOT NULL,       -- e.g. "xgb_v1", "logistic_v2"

    home_win_prob   NUMERIC(5,4) NOT NULL,      -- calibrated probability
    away_win_prob   NUMERIC(5,4),               -- 1 - home_win_prob (stored for convenience)

    -- CLV tracking (populated post-game once closing odds are known)
    close_home_prob NUMERIC(5,4),
    beat_close_line BOOLEAN,                    -- model implied prob > closing prob?

    -- Actual outcome (populated post-game)
    actual_home_win BOOLEAN,
    correct         BOOLEAN,

    predicted_at    TIMESTAMPTZ NOT NULL,
    evaluated_at    TIMESTAMPTZ,

    UNIQUE (game_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_game    ON predictions (game_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model   ON predictions (model_version);
CREATE INDEX IF NOT EXISTS idx_predictions_correct ON predictions (correct);


-- ============================================================
-- HELPER: no-vig implied probability from American moneyline
-- Usage: SELECT ml_to_prob(+150), ml_to_prob(-110)
-- ============================================================
CREATE OR REPLACE FUNCTION ml_to_prob(ml INTEGER)
RETURNS NUMERIC AS $$
BEGIN
    IF ml IS NULL THEN RETURN NULL; END IF;
    IF ml > 0 THEN
        RETURN ROUND((100.0 / (ml + 100))::NUMERIC, 4);
    ELSE
        RETURN ROUND((ABS(ml)::NUMERIC / (ABS(ml) + 100)), 4);
    END IF;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
