"""Player-level (SOG props) features: train/serve parity and the coverage guard."""

import numpy as np
import pandas as pd
import pytest

from features.player import (
    MIN_GAMES, PLAYER_ROLL_STATS, PLAYER_WINDOWS,
    build_player_rolling_features, pregame_player_snapshot,
)
from models.sog_model import SOG_FEATURE_COLS

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
