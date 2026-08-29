"""Player-level (SOG props) features: train/serve parity and the coverage guard."""

import numpy as np
import pandas as pd
import pytest

from features.player import (
    MIN_GAMES, PLAYER_ROLL_STATS, PLAYER_WINDOWS,
    build_player_rolling_features, pregame_player_snapshot,
)
from scipy.stats import poisson

from models.sog_model import (
    OU_LINES, RECENCY_HALF_LIFE_SEASONS, SOG_FEATURE_COLS, _apply_calibration, build_pipeline,
    evaluate_sog, fit_pipeline, over_probability, season_sample_weights,
)
from pipeline import props_live

PLAYER_FEATURES = [f"{c}_l{w}" for c in PLAYER_ROLL_STATS for w in PLAYER_WINDOWS]


def _player_games(player_id, season, n, rng, team="EDM", start=1):
    return [
        {
            "game_id": f"{season[:4]}02{i:04d}", "season": season,
            "player_id": player_id, "player_name": f"P{player_id}",
            "team": team, "position": "C",
            "sog": float(rng.integers(0, 8)),
            "xg": float(rng.random()),
            "shot_attempts": float(rng.integers(1, 12)),
            "xg_per_attempt": float(rng.random() * 0.2),
        }
        for i in range(start, start + n)
    ]


@pytest.fixture
def player_history(rng):
    rows = _player_games(8478402, "2024-2025", 20, rng)
    rows += _player_games(8478402, "2025-2026", 20, rng)
    rows += _player_games(8477934, "2025-2026", 20, rng, team="TOR")
    df = pd.DataFrame(rows)
    df["game_num"] = df["game_id"].str[-4:].astype(int)
    return df


class TestTrainServeParity:
    def test_snapshot_equals_the_training_row_for_the_next_game(self, player_history):
        one = player_history[player_history["player_id"] == 8478402]

        nxt = one.iloc[[-1]].copy()
        nxt["game_id"] = "2025020099"
        nxt["game_num"] = 99
        trained = build_player_rolling_features(
            pd.concat([one, nxt], ignore_index=True)
        ).sort_values(["season", "game_num"])
        expected = trained.iloc[-1][PLAYER_FEATURES].astype(float)

        served = pregame_player_snapshot(
            one, player_ids=[8478402],
        ).loc[8478402, PLAYER_FEATURES].astype(float)

        pd.testing.assert_series_equal(
            expected, served, check_names=False, rtol=0, atol=1e-12,
        )

    def test_rolling_spans_seasons_like_training_does(self, player_history):
        """build_player_rolling_features groups by player_id only, so the
        snapshot must not reset at the season boundary."""
        one = player_history[player_history["player_id"] == 8478402]
        served = pregame_player_snapshot(one, player_ids=[8478402])
        assert served.loc[8478402, "sog_l20"] == pytest.approx(
            one["sog"].tail(20).mean()
        )

    def test_every_model_feature_that_comes_from_player_history_is_produced(
        self, player_history,
    ):
        """The regression this guards: xg, shot_attempts and xg_per_attempt are
        absent from the NHL API, so the live path hardcoded them to NaN —
        six of the model's eleven inputs, silently mean-imputed."""
        snapshot = pregame_player_snapshot(player_history)
        opponent_features = {c for c in SOG_FEATURE_COLS if c.startswith("opp_")}
        from_player = [c for c in SOG_FEATURE_COLS if c not in opponent_features]

        assert len(from_player) == 8
        assert set(from_player) <= set(snapshot.columns)
        assert snapshot[from_player].notna().all().all()

    def test_xg_features_carry_real_values(self, player_history):
        snapshot = pregame_player_snapshot(player_history)
        for col in ("xg_l10", "xg_l20", "shot_attempts_l10", "xg_per_attempt_l20"):
            assert snapshot[col].notna().all()
            assert (snapshot[col] != 0).any()


class TestSnapshotBehaviour:
    def test_handles_multiple_players(self, player_history):
        snapshot = pregame_player_snapshot(player_history)
        assert set(snapshot.index) == {8478402, 8477934}

    def test_unknown_player_gets_a_nan_row_not_a_dropped_projection(
        self, player_history,
    ):
        snapshot = pregame_player_snapshot(player_history, player_ids=[123456])
        assert 123456 in snapshot.index
        assert snapshot.loc[123456, "games_played"] == 0
        assert snapshot.loc[123456, PLAYER_FEATURES].isna().all()

    def test_games_played_counts_prior_games_only(self, player_history):
        snapshot = pregame_player_snapshot(player_history)
        assert snapshot.loc[8477934, "games_played"] == 20

    def test_identity_comes_from_the_most_recent_game(self, player_history):
        snapshot = pregame_player_snapshot(player_history)
        assert snapshot.loc[8477934, "team"] == "TOR"

    def test_min_games_filter_is_meaningful(self, player_history, rng):
        rookie = pd.DataFrame(_player_games(999, "2025-2026", 2, rng))
        rookie["game_num"] = rookie["game_id"].str[-4:].astype(int)
        snapshot = pregame_player_snapshot(
            pd.concat([player_history, rookie], ignore_index=True)
        )
        kept = snapshot[snapshot["games_played"] >= MIN_GAMES]
        assert 999 not in kept.index
        assert 8478402 in kept.index


class TestOpponentContextParity:
    def test_opponent_columns_come_from_the_team_snapshot(self):
        """Training reads the opponent's pre-game sf_pct_l20 / xg_against_l20 /
        hd_chances_against_l20 out of the team features; the live path used to
        re-derive them as a cross-season mean of raw per-game stats."""
        from features.team import ROLL_STATS, WINDOWS

        opponent_features = [c for c in SOG_FEATURE_COLS if c.startswith("opp_")]
        assert opponent_features, "SOG model should use opponent context"

        for col in opponent_features:
            base = col.removeprefix("opp_")
            stat, _, window = base.rpartition("_l")
            assert stat in ROLL_STATS, f"{stat} is not a rolled team stat"
            assert int(window) in WINDOWS


class TestCoverageGuard:
    def test_props_has_a_threshold(self):
        from pipeline.props_live import MIN_FEATURE_COVERAGE
        assert 0.0 < MIN_FEATURE_COVERAGE <= 1.0

    def test_matches_the_game_pipeline_threshold(self):
        """Both pipelines mean-impute, so both need the same protection."""
        from pipeline.live import MIN_FEATURE_COVERAGE as game_threshold
        from pipeline.props_live import MIN_FEATURE_COVERAGE as props_threshold
        assert props_threshold == game_threshold

    def test_run_accepts_a_coverage_override(self):
        import inspect
        from pipeline.props_live import run
        assert "min_coverage" in inspect.signature(run).parameters

    def test_the_old_behaviour_would_now_be_caught(self):
        """Hardcoding xg / shot_attempts / xg_per_attempt to NaN left 5 of 11
        features populated — 45% coverage, well under the threshold."""
        from pipeline.props_live import MIN_FEATURE_COVERAGE

        nan_features = [
            c for c in SOG_FEATURE_COLS
            if c.startswith(("xg_", "shot_attempts_", "xg_per_attempt_"))
        ]
        coverage = 1 - len(nan_features) / len(SOG_FEATURE_COLS)
        assert len(nan_features) == 6
        assert coverage < MIN_FEATURE_COVERAGE


class TestPipelineWiring:
    def test_backfill_builds_the_player_stats_parquet(self):
        """It used to exist only via a manual ingestion.player_stats run, so
        the props training data drifted out of date with everything else."""
        from pipeline.backfill import build_player_game_stats
        assert callable(build_player_game_stats)

    def test_daily_refreshes_player_stats(self):
        import inspect
        from pipeline import daily
        assert "build_player_game_stats" in inspect.getsource(daily.refresh_data)


class TestOverProbability:
    """P(SOG > line) is what a bet needs; the point estimate is not."""

    def test_is_a_probability(self):
        mu = np.array([0.1, 1.0, 2.5, 6.0])
        for line in OU_LINES:
            p = over_probability(mu, line)
            assert ((p >= 0) & (p <= 1)).all()

    def test_falls_as_the_line_rises(self):
        mu = np.array([2.0])
        probs = [over_probability(mu, line)[0] for line in OU_LINES]
        assert probs == sorted(probs, reverse=True)

    def test_rises_with_the_projection(self):
        p = over_probability(np.array([1.0, 2.0, 4.0]), 2.5)
        assert p[0] < p[1] < p[2]

    def test_half_integer_line_means_strictly_more(self):
        # over 2.5 is 3+ shots, so it must equal 1 - P(0, 1 or 2)
        mu = np.array([2.3])
        expected = 1 - poisson.pmf([0, 1, 2], 2.3).sum()
        assert over_probability(mu, 2.5)[0] == pytest.approx(expected)


class TestRecencyWeights:
    """The first drift correction: old seasons should not set the intercept."""

    def test_newest_season_gets_full_weight(self):
        seasons = pd.Series(["2023-2024", "2024-2025", "2025-2026"])
        w = season_sample_weights(seasons, sorted(seasons.unique()), half_life=1.0)
        assert w[-1] == pytest.approx(1.0)

    def test_weight_halves_every_half_life(self):
        ordered = ["2022-2023", "2023-2024", "2024-2025", "2025-2026"]
        w = season_sample_weights(pd.Series(ordered), ordered, half_life=1.0)
        assert w.tolist() == pytest.approx([0.125, 0.25, 0.5, 1.0])

    def test_disabling_it_weights_every_row_equally(self):
        ordered = ["2024-2025", "2025-2026"]
        w = season_sample_weights(pd.Series(ordered), ordered, half_life=0)
        assert w.tolist() == pytest.approx([1.0, 1.0])


class TestCalibration:
    """The second drift correction, folded into the log-link intercept."""

    def test_scales_predictions_by_exactly_the_factor(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        y = rng.poisson(2.0, size=200).astype(float)
        pipe = build_pipeline()
        pipe.fit(X, y)
        before = pipe.predict(X)

        _apply_calibration(pipe, 0.9)
        assert pipe.predict(X).tolist() == pytest.approx((before * 0.9).tolist())

    def test_a_factor_of_one_changes_nothing(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(100, 2))
        pipe = build_pipeline()
        pipe.fit(X, rng.poisson(2.0, size=100).astype(float))
        before = pipe.predict(X)

        _apply_calibration(pipe, 1.0)
        assert pipe.predict(X).tolist() == pytest.approx(before.tolist())

    def test_the_saved_model_carries_the_correction(self):
        """props_live calls pipeline.predict directly, so the pickle must
        already be calibrated — nothing downstream applies a factor."""
        import inspect
        from models import sog_model

        source = inspect.getsource(sog_model.fit_pipeline)
        assert "_apply_calibration" in source

        serving = inspect.getsource(props_live.run)
        assert "calibrat" not in serving.lower(), (
            "serving must not need its own calibration step"
        )


class TestDriftCorrection:
    """A falling league rate should pull projections down, not be averaged in."""

    @staticmethod
    def _drifting_frame(rate_by_season):
        rng = np.random.default_rng(7)
        rows = []
        for season, rate in rate_by_season.items():
            for _ in range(600):
                prior = rng.poisson(rate)
                rows.append({
                    "season": season,
                    "sog": float(rng.poisson(rate)),
                    "sog_l10": float(prior),
                    "sog_l20": float(prior),
                })
        return pd.DataFrame(rows)

    def test_recency_weighting_alone_tracks_a_declining_league(self):
        """Isolated from calibration, so turning the half-life off fails here."""
        df = self._drifting_frame({
            "2021-2022": 3.0, "2022-2023": 2.6, "2023-2024": 2.2, "2024-2025": 1.8,
        })
        cols = ["sog_l10", "sog_l20"]

        flat = fit_pipeline(df, cols, half_life=0, calibrate=False)
        weighted = fit_pipeline(df, cols, half_life=0.5, calibrate=False)

        recent = df[df["season"] == "2024-2025"]
        target = recent["sog"].mean()
        assert abs(weighted.predict(recent[cols].values).mean() - target) < \
               abs(flat.predict(recent[cols].values).mean() - target)

    def test_calibration_alone_tracks_a_declining_league(self):
        df = self._drifting_frame({
            "2021-2022": 3.0, "2022-2023": 2.6, "2023-2024": 2.2, "2024-2025": 1.8,
        })
        cols = ["sog_l10", "sog_l20"]

        raw = fit_pipeline(df, cols, half_life=0, calibrate=False)
        calibrated = fit_pipeline(df, cols, half_life=0, calibrate=True)

        recent = df[df["season"] == "2024-2025"]
        target = recent["sog"].mean()
        assert abs(calibrated.predict(recent[cols].values).mean() - target) < \
               abs(raw.predict(recent[cols].values).mean() - target)

    def test_the_shipped_defaults_are_both_on(self):
        assert RECENCY_HALF_LIFE_SEASONS > 0
        df = self._drifting_frame({"2024-2025": 2.0, "2025-2026": 1.6})
        cols = ["sog_l10", "sog_l20"]
        recent = df[df["season"] == "2025-2026"]
        shipped = fit_pipeline(df, cols)
        naive = fit_pipeline(df, cols, half_life=0, calibrate=False)
        target = recent["sog"].mean()
        assert abs(shipped.predict(recent[cols].values).mean() - target) < \
               abs(naive.predict(recent[cols].values).mean() - target)


class TestHonestBaselines:
    """The old metric scored against a 0.500 coin flip it never had to beat."""

    @staticmethod
    def _scored(over_rate, n=2000):
        rng = np.random.default_rng(3)
        y = rng.poisson(1.9, size=n).astype(float)
        return evaluate_sog(y, np.full(n, 1.9), train_sog=y)

    def test_reports_every_posted_line(self):
        assert sorted(self._scored(0.3)["lines"]) == sorted(OU_LINES)

    def test_accuracy_baseline_is_the_majority_class_not_a_coin_flip(self):
        lines = self._scored(0.3)["lines"]
        assert lines[2.5]["acc_base"] > 0.6, (
            "at a 2.5 line most player-games go under; 0.500 was never the bar"
        )
        assert lines[3.5]["acc_base"] > lines[1.5]["acc_base"]

    def test_majority_baseline_equals_the_larger_side(self):
        lines = self._scored(0.3)["lines"]
        for stats in lines.values():
            assert stats["acc_base"] == pytest.approx(
                max(stats["over_rate"], 1 - stats["over_rate"])
            )

    def test_climatology_is_scored_alongside_the_model(self):
        for stats in self._scored(0.3)["lines"].values():
            assert 0 < stats["brier_base"] <= 0.25

    def test_a_base_rate_forecast_scores_exactly_climatology(self):
        rng = np.random.default_rng(5)
        y = rng.poisson(1.9, size=3000).astype(float)
        # A model that always projects the sample mean is climatology at the
        # rate level, so its Brier must land near the no-skill reference.
        m = evaluate_sog(y, np.full(len(y), y.mean()), train_sog=y)
        for stats in m["lines"].values():
            assert stats["brier"] == pytest.approx(stats["brier_base"], abs=0.02)

    def test_bias_is_reported(self):
        rng = np.random.default_rng(9)
        y = rng.poisson(2.0, size=500).astype(float)
        m = evaluate_sog(y, y + 0.3, train_sog=y)
        assert m["bias"] == pytest.approx(0.3)
