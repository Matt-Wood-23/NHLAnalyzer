"""Goalie saves props: train/serve parity, the overdispersion fix, and drift."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from features.goalie_props import (
    GOALIE_PROP_WINDOWS,
    GOALIE_ROLL_STATS,
    MIN_STARTS,
    OPP_TEAM_COLS,
    OWN_TEAM_COLS,
    attach_team_context,
    build_goalie_rolling_features,
    pregame_goalie_snapshot,
    rolling_feature_columns,
    team_context_columns,
)
from models.count_model import (
    MIN_IN_SEASON_SAMPLE,
    build_pipeline,
    estimate_dispersion,
    evaluate_counts,
    fit_pipeline,
    in_season_factor,
    over_probability,
)
from models.saves_model import SAVES_FEATURE_COLS, SAVES_LINES, TARGET
from pipeline import saves_live

GOALIE_FEATURES = rolling_feature_columns()


def _goalie_games(goalie_id, season, n, rng, team="BOS", start=1):
    rows = []
    for i in range(start, start + n):
        shots = float(rng.integers(18, 42))
        goals = float(rng.integers(0, 6))
        rows.append({
            "game_id": f"{season[:4]}02{i:04d}", "season": season,
            "goalie_id": goalie_id, "goalie_name": f"G{goalie_id}",
            "team": team, "is_starter": True, "game_num": i,
            "shots_against": shots,
            "goals_against": goals,
            "saves": shots - goals,
            "save_pct": (shots - goals) / shots,
            "gsax": float(rng.normal()),
            "xg_against": float(rng.uniform(1.5, 4.0)),
        })
    return rows


@pytest.fixture
def goalie_history(rng):
    rows = _goalie_games(8476883, "2024-2025", 20, rng)
    rows += _goalie_games(8476883, "2025-2026", 20, rng, start=21)
    rows += _goalie_games(8477424, "2025-2026", 20, rng, team="NSH")
    return pd.DataFrame(rows)


class TestTrainServeParity:
    """The one invariant this project keeps re-learning."""

    def test_snapshot_equals_the_training_row_for_the_next_game(self, goalie_history, rng):
        gid = 8476883
        played = goalie_history[goalie_history["goalie_id"] == gid]

        future = pd.DataFrame(_goalie_games(gid, "2025-2026", 1, rng, start=99))
        full = pd.concat([goalie_history, future], ignore_index=True)
        training_row = build_goalie_rolling_features(full)
        training_row = training_row[
            (training_row["goalie_id"] == gid) & (training_row["game_num"] == 99)
        ].iloc[0]

        served = pregame_goalie_snapshot(goalie_history, goalie_ids=[gid]).loc[gid]

        for col in GOALIE_FEATURES:
            assert served[col] == pytest.approx(training_row[col]), col
        assert served["starts"] == len(played)

    def test_rolling_spans_seasons_like_the_outcome_model_does(self, goalie_history):
        """A goalie's form carries over a summer; resetting would blank every
        starter for a month."""
        gid = 8476883
        snap = pregame_goalie_snapshot(goalie_history, goalie_ids=[gid]).loc[gid]
        one_season = goalie_history[
            (goalie_history["goalie_id"] == gid)
            & (goalie_history["season"] == "2025-2026")
        ]
        assert snap["starts"] > len(one_season)

    def test_every_model_feature_is_produced(self, goalie_history):
        snap = pregame_goalie_snapshot(goalie_history)
        from_goalie = [
            c for c in SAVES_FEATURE_COLS
            if not c.startswith(("own_", "opp_")) and c != "is_home"
        ]
        assert set(from_goalie).issubset(snap.columns)

    def test_workload_features_carry_real_values(self, goalie_history):
        """shots_against is the dominant driver -- it must not arrive NaN."""
        snap = pregame_goalie_snapshot(goalie_history)
        for w in GOALIE_PROP_WINDOWS:
            assert snap[f"shots_against_l{w}"].notna().all()
            assert (snap[f"shots_against_l{w}"] > 0).all()


class TestSnapshotBehaviour:
    def test_unknown_goalie_gets_a_nan_row_not_a_dropped_projection(self, goalie_history):
        snap = pregame_goalie_snapshot(goalie_history, goalie_ids=[9999999])
        assert 9999999 in snap.index
        assert snap.loc[9999999, GOALIE_FEATURES].isna().all()

    def test_starts_counts_prior_games_only(self, goalie_history):
        snap = pregame_goalie_snapshot(goalie_history, goalie_ids=[8477424])
        assert snap.loc[8477424, "starts"] == 20

    def test_identity_comes_from_the_most_recent_start(self, goalie_history):
        snap = pregame_goalie_snapshot(goalie_history, goalie_ids=[8477424])
        assert snap.loc[8477424, "team"] == "NSH"

    def test_min_starts_is_enforced_somewhere(self):
        assert MIN_STARTS >= 5


class TestTeamContext:
    def test_own_and_opponent_columns_are_not_swapped(self):
        """The whole point of the join: shots faced is a team property."""
        team_features = pd.DataFrame([
            {"game_id": "2025020001", "team": "BOS", "is_home": True,
             "sf_pct_l20": 0.55, "xg_against_l20": 2.1, "hd_chances_against_l20": 9.0,
             "xg_for_l20": 3.3, "hd_chances_for_l20": 12.0},
            {"game_id": "2025020001", "team": "TOR", "is_home": False,
             "sf_pct_l20": 0.45, "xg_against_l20": 3.4, "hd_chances_against_l20": 14.0,
             "xg_for_l20": 2.2, "hd_chances_for_l20": 8.0},
        ])
        goalies = pd.DataFrame([
            {"game_id": "2025020001", "team": "BOS", "goalie_id": 1, "saves": 30.0},
        ])
        joined = attach_team_context(goalies, team_features).iloc[0]

        assert joined["opp_team"] == "TOR"
        assert joined["is_home"] == 1.0
        # own_* describes BOS, opp_* describes the team shooting at him.
        assert joined["own_sf_pct_l20"] == pytest.approx(0.55)
        assert joined["opp_xg_for_l20"] == pytest.approx(2.2)
        assert joined["opp_hd_chances_for_l20"] == pytest.approx(8.0)

    def test_context_columns_are_namespaced_by_side(self):
        cols = team_context_columns()
        assert cols == ["own_" + c for c in OWN_TEAM_COLS] + \
                       ["opp_" + c for c in OPP_TEAM_COLS]
        assert len(set(cols)) == len(cols), "a collision would silently drop a side"


class TestOverdispersion:
    """Saves run ~1.9x variance/mean; Poisson tails are too tight for that."""

    def test_dispersion_is_zero_for_poisson_data(self, rng):
        mu = np.full(4000, 25.0)
        y = rng.poisson(25.0, size=4000).astype(float)
        assert estimate_dispersion(y, mu) < 0.005

    def test_dispersion_is_positive_for_overdispersed_data(self, rng):
        mu = np.full(4000, 25.0)
        y = rng.negative_binomial(30, 30 / (30 + 25.0), size=4000).astype(float)
        assert estimate_dispersion(y, mu) > 0.01

    def test_dispersion_widens_the_tails(self):
        """The reason it matters: Poisson would call this far more certain."""
        mu = np.array([25.0])
        tight = over_probability(mu, 30.5, dispersion=0.0)[0]
        wide  = over_probability(mu, 30.5, dispersion=0.03)[0]
        assert wide > tight

    def test_dispersion_also_widens_the_low_tail(self):
        mu = np.array([25.0])
        tight = over_probability(mu, 19.5, dispersion=0.0)[0]
        wide  = over_probability(mu, 19.5, dispersion=0.03)[0]
        assert wide < tight, "more mass below the line, so P(over) must fall"

    def test_zero_dispersion_is_exactly_poisson(self):
        from scipy.stats import poisson
        mu = np.array([25.0])
        assert over_probability(mu, 24.5, 0.0)[0] == pytest.approx(
            1 - poisson.cdf(24, 25.0)
        )

    def test_it_is_still_a_probability(self):
        mu = np.array([1.0, 25.0, 60.0])
        for line in SAVES_LINES:
            p = over_probability(mu, line, dispersion=0.03)
            assert ((p >= 0) & (p <= 1)).all()

    def test_the_saved_model_carries_its_dispersion(self):
        """P(over) is wrong without it, so it must travel with the pickle."""
        import inspect
        from models import saves_model
        assert "dispersion" in inspect.signature(
            saves_model.save_saves_model
        ).parameters
        source = inspect.getsource(saves_model.load_saves_model)
        assert "dispersion" in source


class TestInSeasonCalibration:
    """The correction weighting and newest-season calibration cannot reach."""

    @staticmethod
    def _frame(rate, n):
        rng = np.random.default_rng(11)
        prior = rng.poisson(rate, size=n).astype(float)
        return pd.DataFrame({
            "season": "2026-2027",
            "saves": rng.poisson(rate, size=n).astype(float),
            "shots_against_l10": prior,
            "saves_l10": prior,
        })

    def test_small_samples_are_left_alone(self):
        pipe = build_pipeline()
        df = self._frame(25.0, MIN_IN_SEASON_SAMPLE - 1)
        cols = ["shots_against_l10", "saves_l10"]
        pipe.fit(df[cols].values, df["saves"].values)
        assert in_season_factor(pipe, df, cols, "saves") == 1.0

    def test_none_is_handled(self):
        pipe = build_pipeline()
        assert in_season_factor(pipe, None, ["a"], "saves") == 1.0

    def test_it_pulls_a_high_model_down(self):
        cols = ["shots_against_l10", "saves_l10"]
        trained_high = self._frame(30.0, 800)
        pipe = build_pipeline()
        pipe.fit(trained_high[cols].values, trained_high["saves"].values)

        now_lower = self._frame(25.0, 800)
        factor = in_season_factor(pipe, now_lower, cols, "saves")
        assert factor < 1.0

        corrected = pipe.predict(now_lower[cols].values) * factor
        assert abs(corrected.mean() - now_lower["saves"].mean()) < \
               abs(pipe.predict(now_lower[cols].values).mean() - now_lower["saves"].mean())


class TestSavesScoring:
    def test_lines_are_half_integers(self):
        """Half-integers mean no push, so P(over) needs no tie handling."""
        for line in SAVES_LINES:
            assert line % 1 == 0.5

    def test_scoring_reports_every_line(self, rng):
        y = rng.poisson(26.0, size=800).astype(float)
        m = evaluate_counts(y, np.full(len(y), 26.0), SAVES_LINES,
                            train_target=y, dispersion=0.03)
        assert sorted(m["lines"]) == sorted(SAVES_LINES)
        assert m["dispersion"] == pytest.approx(0.03)

    def test_accuracy_baseline_is_the_majority_class(self, rng):
        y = rng.poisson(26.0, size=2000).astype(float)
        m = evaluate_counts(y, np.full(len(y), 26.0), SAVES_LINES, train_target=y)
        for stats in m["lines"].values():
            assert stats["acc_base"] == pytest.approx(
                max(stats["over_rate"], 1 - stats["over_rate"])
            )
            assert stats["acc_base"] >= 0.5

    def test_feature_list_covers_goalie_and_both_team_sides(self):
        assert set(rolling_feature_columns()).issubset(SAVES_FEATURE_COLS)
        assert set(team_context_columns()).issubset(SAVES_FEATURE_COLS)
        assert "is_home" in SAVES_FEATURE_COLS

    def test_every_rolled_stat_appears_at_every_window(self):
        expected = {f"{c}_l{w}" for c in GOALIE_ROLL_STATS for w in GOALIE_PROP_WINDOWS}
        assert expected == set(rolling_feature_columns())


class TestServingGuards:
    def test_season_follows_the_served_date_not_the_clock(self, monkeypatch):
        """A March game belongs to the season that started the previous autumn;
        keying this to `now` would calibrate a backtest against the wrong year."""
        monkeypatch.delenv("NHL_CURRENT_SEASON", raising=False)
        assert saves_live.season_for(date(2026, 3, 15)) == "2025-2026"
        assert saves_live.season_for(date(2026, 11, 2)) == "2026-2027"

    def test_coverage_threshold_matches_the_other_pipelines(self):
        from pipeline import live, props_live
        assert saves_live.MIN_FEATURE_COVERAGE == live.MIN_FEATURE_COVERAGE
        assert saves_live.MIN_FEATURE_COVERAGE == props_live.MIN_FEATURE_COVERAGE

    def test_probable_starter_is_the_most_recent_one(self, goalie_history):
        starters = saves_live.probable_starters(goalie_history, ["BOS"])
        assert starters["BOS"]["goalie_id"] == 8476883
        assert starters["BOS"]["starter_basis"] == "last start"

    def test_a_team_with_no_history_is_skipped_not_guessed(self, goalie_history):
        assert saves_live.probable_starters(goalie_history, ["VGK"]) == {}

    def test_the_starter_basis_is_carried_into_output(self):
        """Callers must be able to see that the lineup is inferred."""
        import inspect
        source = inspect.getsource(saves_live.run)
        assert "starter_basis" in source


class TestDriftMachineryIsShared:
    """SOG and saves must not grow separate copies of the same corrections."""

    def test_both_models_use_the_shared_fit(self):
        import inspect
        from models import saves_model, sog_model
        for module in (saves_model, sog_model):
            assert "count_model" in inspect.getsource(module).split("logger =")[0]

    def test_the_target_is_configurable(self, rng):
        """sog_model and saves_model differ only in target and lines."""
        df = pd.DataFrame({
            "season": ["2025-2026"] * 300,
            "saves": rng.poisson(25.0, size=300).astype(float),
            "x": rng.normal(size=300),
        })
        pipe = fit_pipeline(df, ["x"], target=TARGET)
        assert pipe.predict(df[["x"]].values).mean() == pytest.approx(
            df[TARGET].mean(), rel=0.05
        )
