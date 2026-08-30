"""Live prediction guardrails."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pipeline.live import MIN_FEATURE_COVERAGE, predict

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
        call = src.index("save_prediction_history(predictions")
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
        assert 'fm["season"] == season' in inspect.getsource(
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
