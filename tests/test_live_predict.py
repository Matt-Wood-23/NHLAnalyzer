"""Live prediction guardrails."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from datetime import date

from pipeline.live import MIN_FEATURE_COVERAGE, mark_started, predict

FEATURES = [f"f{i}" for i in range(10)]


@pytest.fixture
def model(rng):
    X = rng.random((200, len(FEATURES)))
    y = (X[:, 0] > 0.5).astype(int)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestClassifier(n_estimators=10, random_state=0)),
    ]).fit(X, y)


@pytest.fixture
def slate():
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "home_team": ["BOS", "TOR", "MTL"],
        "away_team": ["NYR", "OTT", "VAN"],
        **{c: [0.5, 0.5, 0.5] for c in FEATURES},
    })


class TestCoverage:
    def test_full_coverage_when_nothing_is_missing(self, model, slate):
        out = predict(slate, model, FEATURES)
        assert out["feature_coverage"].tolist() == pytest.approx([1.0] * 3)
        assert out["prob_home_win"].between(0, 1).all()

    def test_coverage_counts_nan_values(self, model, slate):
        slate.loc[0, FEATURES[:2]] = np.nan
        out = predict(slate, model, FEATURES, min_coverage=0.0)
        assert out.set_index("game_id").loc["g1", "feature_coverage"] == pytest.approx(0.8)

    def test_coverage_counts_absent_columns(self, model, slate):
        out = predict(slate.drop(columns=FEATURES[:4]), model, FEATURES, min_coverage=0.0)
        assert out["feature_coverage"].tolist() == pytest.approx([0.6] * 3)

    def test_games_below_the_threshold_are_dropped(self, model, slate):
        slate.loc[1, FEATURES[:5]] = np.nan   # 50% — below
        slate.loc[2, FEATURES[:2]] = np.nan   # 80% — above
        out = predict(slate, model, FEATURES, min_coverage=0.7)
        assert out["game_id"].tolist() == ["g1", "g3"]

    def test_a_broken_pipeline_returns_nothing_rather_than_a_fake_number(self, model, slate):
        """Every missing value becomes a column mean, so an upstream break used
        to surface as a plausible ~50% prediction instead of an error."""
        out = predict(slate[["game_id", "home_team", "away_team"]], model, FEATURES)
        assert out.empty

    def test_threshold_is_configurable(self, model, slate):
        slate.loc[1, FEATURES[:5]] = np.nan
        assert len(predict(slate, model, FEATURES, min_coverage=0.4)) == 3
        assert len(predict(slate, model, FEATURES, min_coverage=0.9)) == 2

    def test_default_threshold_is_a_fraction(self):
        assert 0.0 < MIN_FEATURE_COVERAGE <= 1.0


class TestNoMutation:
    def test_input_frame_is_not_modified(self, model, slate):
        before = slate.copy()
        predict(slate, model, FEATURES)
        pd.testing.assert_frame_equal(slate, before)

    def test_missing_columns_are_not_added_to_the_caller_s_frame(self, model, slate):
        trimmed = slate.drop(columns=FEATURES[:4])
        cols_before = list(trimmed.columns)
        predict(trimmed, model, FEATURES, min_coverage=0.0)
        assert list(trimmed.columns) == cols_before


class TestHistoryWriteGuard:
    """--dry-run used to append to the prediction log anyway."""

    def test_dry_run_and_history_are_separate_flags(self):
        import inspect
        from pipeline import live
        params = inspect.signature(live.run).parameters
        assert "dry_run" in params and "save_history" in params

    def test_dry_run_defaults_to_writing_nothing(self):
        """The intuitive meaning of the flag, and what the CLI implies."""
        import inspect
        from pipeline import live
        src = inspect.getsource(live.run)
        assert "save_history = not dry_run" in src

    def test_the_daily_pipeline_still_logs_predictions(self):
        """pipeline.daily passed dry_run=True meaning "skip Postgres" while
        relying on the history being written anyway.  Making dry_run mean
        "write nothing" would have silently stopped the daily loop recording
        anything, which /yesterday and the CLV scoring both read back."""
        import inspect
        from pipeline import daily
        src = inspect.getsource(daily.run_predictions)
        assert "save_history=True" in src

    def test_the_history_writer_is_reached_through_the_guard(self):
        import inspect
        from pipeline import live
        src = inspect.getsource(live.run)
        before = src.index("if save_history:")
        call = src.index("save_prediction_history(")
        assert call > before, "history is written before the guard is consulted"


class TestSnapshotSeasonScoping:
    """Training resets some rolling features at the season boundary and not
    others.  The live snapshots have to match, feature by feature."""

    def test_special_teams_and_opponent_quality_take_a_season(self):
        import inspect
        from pipeline import live
        for fn in (live._build_special_teams_snapshot,
                   live._build_opponent_quality_snapshot):
            assert "season" in inspect.signature(fn).parameters, fn.__name__

    def test_they_actually_filter_on_it(self):
        import inspect
        from pipeline import live
        assert 'st_games["season"] == season' in inspect.getsource(
            live._build_special_teams_snapshot)
        assert 'oq_games["season"] == season' in inspect.getsource(
            live._build_opponent_quality_snapshot)

    def test_the_goalie_snapshot_is_deliberately_not_scoped(self):
        """features.goalie_mp rolls per goalie with no season reset, so
        scoping here would feed the model something it never trained on."""
        import inspect
        from features import goalie_mp
        assert "across seasons" in inspect.getsource(
            goalie_mp.compute_goalie_rolling_features)

    def test_rest_is_deliberately_not_scoped(self):
        """features.context computes rest with groupby("team") alone, so on
        opening night training saw ~180 days.  Scoping would be the bug."""
        import inspect
        from pipeline import live
        assert "not* scoped" in inspect.getsource(live._team_last_game_date)

    def test_the_goalie_snapshot_picks_the_likely_starter(self):
        """It used to sort by goalie_id and take iloc[-1], which is the
        highest goalie_id's last start -- not the team's most recent."""
        import inspect
        from pipeline import live
        src = inspect.getsource(live._build_goalie_snapshot)
        assert "likely_starters" in src
        assert "grp.iloc[-1]" not in src

    def test_both_pipelines_share_one_starter_rule(self):
        """If they drift, the outcome model and the saves prop can put
        different goalies in net for the same game."""
        import inspect
        from pipeline import live, saves_live
        assert "likely_starters" in inspect.getsource(live._build_goalie_snapshot)
        assert "likely_starters" in inspect.getsource(saves_live.probable_starters)


class TestDailyRefreshCompleteness:
    """The daily run must refresh every file the serving path reads.

    It refreshed player_game_stats.parquet but not the _full spine that
    replaced it as the model's source, so props would have served a frozen
    snapshot indefinitely while every log line reported a successful refresh.
    """

    def test_the_daily_refresh_rebuilds_the_boxscore_spine(self):
        import inspect
        from pipeline import daily
        assert "build_boxscore_spine" in inspect.getsource(daily.refresh_data)

    def test_the_spine_is_rebuilt_after_the_moneypuck_aggregation(self):
        """The merge joins MoneyPuck xG onto the lineup, so it must run second
        or it joins yesterday's shot data onto today's lineup."""
        import inspect
        src = inspect.getsource(__import__("pipeline.daily", fromlist=["x"]).refresh_data)
        assert src.index("build_player_game_stats()") < src.index("build_boxscore_spine()")

    def test_every_parquet_the_resolver_can_pick_gets_refreshed(self):
        """If player_game_stats_path can return it, something must rebuild it."""
        import inspect
        from ingestion import player_stats
        from pipeline import daily, backfill

        resolver = inspect.getsource(player_stats.player_game_stats_path)
        refreshed = (inspect.getsource(daily.refresh_data)
                     + inspect.getsource(backfill.build_boxscore_spine)
                     + inspect.getsource(backfill.build_player_game_stats))
        for name in ("player_game_stats_full", "player_game_stats"):
            assert name in resolver
            assert name in refreshed, f"{name} is served but never rebuilt"


# ---------------------------------------------------------------------------
# Season openers
# ---------------------------------------------------------------------------

ROLLING = [f"home_roll{i}" for i in range(18)]
OPENER_FEATURES = ["home_elo", "away_elo"] + ROLLING


def _fit(rng, n_features):
    X = rng.random((200, n_features))
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestClassifier(n_estimators=10, random_state=0)),
    ]).fit(X, (X[:, 0] > X[:, 1]).astype(int))


@pytest.fixture
def opener_model(rng):
    return _fit(rng, len(OPENER_FEATURES))


def _game(game_id, home_gp, away_gp, rolling=True, elo=True):
    row = {
        "game_id": game_id, "home_team": "BOS", "away_team": "TOR",
        "home_games_played": home_gp, "away_games_played": away_gp,
        "home_elo": 1520.0 if elo else np.nan,
        "away_elo": 1480.0 if elo else np.nan,
    }
    row.update({c: 0.5 if rolling else np.nan for c in ROLLING})
    return row


class TestSeasonOpener:
    """A team's first game has ~7% coverage by construction, in training too.

    The flat 70% guard read that as a broken pipeline, dropped every game of
    the 2026-27 opening week and told the user to run the command they had
    just run.
    """

    def test_an_opener_is_published_despite_low_coverage(self, opener_model):
        slate = pd.DataFrame([_game("g1", 0, 0, rolling=False)])
        out = predict(slate, opener_model, OPENER_FEATURES)
        assert len(out) == 1
        assert out["feature_coverage"].iloc[0] == pytest.approx(0.10)
        assert out["prob_home_win"].between(0, 1).all()

    def test_it_is_marked_so_nobody_mistakes_it_for_a_real_pick(self, opener_model):
        slate = pd.DataFrame([_game("g1", 0, 4, rolling=False), _game("g2", 3, 4)])
        out = predict(slate, opener_model, OPENER_FEATURES).set_index("game_id")
        assert bool(out.loc["g1", "season_opener"]) is True
        assert bool(out.loc["g2", "season_opener"]) is False

    def test_one_side_having_played_is_still_an_opener(self, opener_model):
        """NYI @ TOR on night two: Toronto had a game, the Islanders did not."""
        slate = pd.DataFrame([_game("g1", 1, 0, rolling=False)])
        assert len(predict(slate, opener_model, OPENER_FEATURES)) == 1

    def test_an_opener_without_elo_is_dropped(self, opener_model):
        """ELO is all an opener has.  Without it there is nothing to publish."""
        slate = pd.DataFrame([_game("g1", 0, 0, rolling=False, elo=False)])
        assert predict(slate, opener_model, OPENER_FEATURES).empty

    def test_the_guard_still_bites_once_both_teams_have_played(self, opener_model):
        """The exemption must not swallow the case the guard exists for: both
        teams have games, features are missing anyway, so something broke."""
        slate = pd.DataFrame([_game("g1", 1, 1, rolling=False)])
        assert predict(slate, opener_model, OPENER_FEATURES).empty

    def test_a_mid_season_break_is_not_excused_either(self, opener_model):
        slate = pd.DataFrame([_game("g1", 40, 41, rolling=False)])
        assert predict(slate, opener_model, OPENER_FEATURES).empty

    def test_a_slate_with_no_games_played_columns_gets_the_strict_rule(self, opener_model):
        slate = pd.DataFrame([_game("g1", 0, 0, rolling=False)]).drop(
            columns=["home_games_played", "away_games_played"])
        assert predict(slate, opener_model, OPENER_FEATURES).empty

    def test_the_embed_says_so(self):
        from bot.discord_bot import format_game_embed
        row = {"home_team": "BOS", "away_team": "TOR", "prob_home_win": 0.52}
        flagged = format_game_embed({**row, "season_opener": True})["description"]
        assert "Season opener" in flagged
        assert "Season opener" not in format_game_embed({**row, "season_opener": False})["description"]
        assert "Season opener" not in format_game_embed(row)["description"]


# ---------------------------------------------------------------------------
# Games that have already started
# ---------------------------------------------------------------------------

class TestAlreadyStarted:
    def _preds(self):
        return pd.DataFrame({"game_id": ["1", "2", "3", "4", "5"],
                             "prob_home_win": [0.5] * 5})

    def test_only_pregame_states_are_recordable(self):
        games = [
            {"game_id": "1", "game_state": "FUT"},
            {"game_id": "2", "game_state": "PRE"},
            {"game_id": "3", "game_state": "LIVE"},
            {"game_id": "4", "game_state": "OFF"},
            {"game_id": "5", "game_state": "CRIT"},
        ]
        out = mark_started(self._preds(), games)
        assert out["already_started"].tolist() == [False, False, True, True, True]

    def test_an_unknown_state_does_not_block_the_log(self):
        games = [{"game_id": str(i)} for i in range(1, 6)]
        assert not mark_started(self._preds(), games)["already_started"].any()

    def test_integer_game_ids_still_match(self):
        games = [{"game_id": 3, "game_state": "LIVE"}]
        out = mark_started(self._preds(), games).set_index("game_id")
        assert bool(out.loc["3", "already_started"]) is True

    def test_the_caller_s_frame_is_left_alone(self):
        preds = self._preds()
        mark_started(preds, [{"game_id": "1", "game_state": "LIVE"}])
        assert "already_started" not in preds.columns


RUN_FEATURES = ["home_form_l5", "away_form_l5", "diff_form_l5",
                "home_elo", "away_elo", "diff_elo"]
DAY = date(2026, 10, 10)


@pytest.fixture
def wired_run(rng, monkeypatch):
    """pipeline.live.run with every data source replaced.

    Three games: one ordinary, one season opener, one already under way.
    Yields the module, the schedule, and whatever was written to the log.
    """
    from pipeline import live

    model = _fit(rng, len(RUN_FEATURES))
    schedule = [
        {"game_id": "2026020101", "date": "2026-10-10", "game_type": "2",
         "home_team": "BOS", "away_team": "TOR", "game_state": "FUT"},
        {"game_id": "2026020102", "date": "2026-10-10", "game_type": "2",
         "home_team": "MTL", "away_team": "OTT", "game_state": "PRE"},
        {"game_id": "2026020103", "date": "2026-10-10", "game_type": "2",
         "home_team": "NYR", "away_team": "NJD", "game_state": "LIVE"},
    ]
    team_snapshot = pd.DataFrame(
        {"form_l5": [0.6, 0.4, np.nan, 0.5, 0.55, 0.45],
         "games_played": [5, 5, 0, 3, 4, 4]},
        index=["BOS", "TOR", "MTL", "OTT", "NYR", "NJD"],
    )
    elo = pd.DataFrame({"elo": [1500.0 + i for i in range(6)]},
                       index=team_snapshot.index)
    saved: list[pd.DataFrame] = []

    monkeypatch.setattr(live, "fetch_schedule", lambda d: schedule)
    monkeypatch.setattr(live, "_build_team_snapshot", lambda teams=None: team_snapshot)
    monkeypatch.setattr(live, "_build_goalie_snapshot", lambda: pd.DataFrame())
    monkeypatch.setattr(live, "_build_special_teams_snapshot",
                        lambda teams=None: pd.DataFrame())
    monkeypatch.setattr(live, "_build_opponent_quality_snapshot",
                        lambda teams=None: pd.DataFrame())
    monkeypatch.setattr(live, "_build_elo_snapshot", lambda: elo)
    monkeypatch.setattr(live, "_team_last_game_date", lambda **kw: {})
    monkeypatch.setattr(live, "load_model", lambda name: (model, RUN_FEATURES))
    monkeypatch.setattr(live, "attach_market_prices", lambda preds, d: preds)
    monkeypatch.setattr(
        live, "save_prediction_history",
        lambda preds, model_name, target_date=None: saved.append(preds.copy()),
    )
    return live, schedule, saved


class TestRunEndToEnd:
    def test_every_game_is_predicted(self, wired_run):
        live, _, _ = wired_run
        out = live.run(target_date=DAY, dry_run=True, save_history=True)
        assert len(out) == 3

    def test_a_started_game_is_shown_but_never_logged(self, wired_run):
        live, _, saved = wired_run
        out = live.run(target_date=DAY, dry_run=True, save_history=True)
        assert saved[0]["game_id"].tolist() == ["2026020101", "2026020102"]
        shown = out.set_index("game_id")
        assert bool(shown.loc["2026020103", "already_started"]) is True

    def test_the_opener_is_logged_and_flagged(self, wired_run):
        live, _, saved = wired_run
        out = live.run(target_date=DAY, dry_run=True, save_history=True)
        assert bool(out.set_index("game_id").loc["2026020102", "season_opener"]) is True
        assert "2026020102" in saved[0]["game_id"].tolist()

    def test_a_dry_run_logs_nothing(self, wired_run):
        live, _, saved = wired_run
        live.run(target_date=DAY, dry_run=True)
        assert saved == []

    def test_a_late_run_cannot_overwrite_the_pregame_record(self, wired_run):
        """The log keeps the latest row per game, so a run after puck drop
        used to replace the real prediction with one carrying a live line."""
        live, schedule, saved = wired_run
        for g in schedule:
            g["game_state"] = "OFF"
        out = live.run(target_date=DAY, dry_run=True, save_history=True)
        assert len(out) == 3 and saved == []


# ---------------------------------------------------------------------------
# Context parity
# ---------------------------------------------------------------------------

class TestHeadToHead:
    def _h2h(self, monkeypatch, rows):
        from pipeline import live
        frame = pd.DataFrame(rows, columns=["home_team", "home_win"])
        captured = {}

        def fake_read_sql(sql, conn, params=None):
            captured["sql"], captured["params"] = sql, params
            return frame

        monkeypatch.setattr(live.pd, "read_sql", fake_read_sql)
        value = live._load_h2h(object(), "BOS", "TOR", before=DAY)
        return value, captured

    def test_it_is_the_home_side_s_win_rate_not_tonight_s_home_team_s(self, monkeypatch):
        """features.context averages raw home_win over the last three meetings.
        Live used to flip each result to tonight's home team: a more natural
        number, and one that matched training on a third of real games."""
        value, _ = self._h2h(monkeypatch, [("TOR", True), ("TOR", True), ("BOS", False)])
        assert value == pytest.approx(2 / 3)

    def test_unfinished_meetings_are_skipped(self, monkeypatch):
        value, _ = self._h2h(monkeypatch, [("TOR", True), ("BOS", None)])
        assert value == pytest.approx(1.0)

    def test_no_meetings_is_nan(self, monkeypatch):
        value, _ = self._h2h(monkeypatch, [])
        assert np.isnan(value)

    def test_tonight_s_game_cannot_count_as_a_previous_meeting(self, monkeypatch):
        _, captured = self._h2h(monkeypatch, [("TOR", True)])
        assert "date < %(before)s" in captured["sql"]
        assert captured["params"]["before"] == DAY


class TestLastGameDate:
    def test_only_games_before_the_target_date_count(self, monkeypatch):
        """A team that finished earlier tonight must not show zero days' rest."""
        from pipeline import live
        captured = {}

        def fake_read_sql(sql, conn, params=None):
            captured["sql"], captured["params"] = sql, params
            return pd.DataFrame({"team": ["TOR"], "last_date": ["2026-09-29"]})

        monkeypatch.setattr(live.pd, "read_sql", fake_read_sql)
        monkeypatch.setattr(live, "_warn_if_dates_are_stale", lambda *a, **k: None)
        out = live._team_last_game_date(conn=object(), before=date(2026, 9, 30))

        assert out == {"TOR": date(2026, 9, 29)}
        assert captured["sql"].count("date < %(before)s") == 2
        assert captured["params"]["before"] == date(2026, 9, 30)

    def test_run_passes_the_date_it_is_predicting(self):
        import inspect
        from pipeline import live
        assert "before=target_date" in inspect.getsource(live.run)


class TestDailyDateSync:
    """Nothing in the daily run wrote game dates, so 2026-27 opened with the
    games table a season behind and every team looking 170 days rested."""

    def test_dates_are_synced_before_the_matrix_is_rebuilt(self):
        import inspect
        from pipeline import daily
        src = inspect.getsource(daily.refresh_data)
        assert src.index("sync_game_dates(conn)") < src.index("build_feature_matrix(")

    def test_no_database_is_reported_not_fatal(self, caplog):
        from pipeline import daily
        with caplog.at_level("WARNING"):
            assert daily.sync_game_dates(None) == 0
        assert "approximate" in caplog.text

    def test_a_failed_sync_does_not_stop_the_run(self, monkeypatch, caplog):
        from ingestion import nhl_api
        from pipeline import daily

        class Conn:
            rolled_back = False

            def rollback(self):
                self.rolled_back = True

        def boom(conn, opener):
            raise RuntimeError("schedule API down")

        monkeypatch.setattr(nhl_api, "sync_recent_games", boom)
        conn = Conn()
        with caplog.at_level("ERROR"):
            assert daily.sync_game_dates(conn) == 0
        assert conn.rolled_back
        assert "stale" in caplog.text

    def test_an_empty_result_no_longer_claims_nothing_was_scheduled(self):
        """It printed "No games scheduled" directly under three log lines
        naming the games it had just dropped."""
        import inspect
        from pipeline import daily
        assert 'print(f"\nNo games scheduled' not in inspect.getsource(daily.run_predictions)
