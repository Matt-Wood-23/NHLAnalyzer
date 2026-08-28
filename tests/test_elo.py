"""ELO ratings — the model's single most important feature."""

import json

import pandas as pd
import pytest

from features import elo as elo_mod
from features.elo import (
    INITIAL_ELO, compute_elo_ratings, default_elo_params,
    load_elo_params, save_elo_params, tune_elo_parameters,
)


class TestNoLeakage:
    def test_ratings_are_pre_game(self, elo_games):
        """The first game must carry the initial rating: a rating that already
        reflected the game's result would leak the target into the feature."""
        result = compute_elo_ratings(elo_games)
        first = result.iloc[0]
        assert first["home_elo"] == INITIAL_ELO
        assert first["away_elo"] == INITIAL_ELO
        assert first["diff_elo"] == 0.0

    def test_unknown_results_do_not_move_ratings(self, elo_games):
        pending = elo_games.copy()
        pending["home_win"] = float("nan")
        _, final = compute_elo_ratings(pending, return_final=True)
        assert all(v == INITIAL_ELO for v in final.values())

    def test_one_row_per_game(self, elo_games):
        result = compute_elo_ratings(elo_games)
        assert len(result) == len(elo_games)
        assert set(result["game_id"]) == set(elo_games["game_id"])


class TestOvertimeDiscount:
    def _final(self, went_to_ot, **kw):
        game = pd.DataFrame([{
            "game_id": "2025020001", "season": "2025-2026",
            "home_team": "BOS", "away_team": "TOR",
            "home_win": 1.0, "went_to_ot": went_to_ot,
        }])
        _, final = compute_elo_ratings(game, return_final=True, **kw)
        return final

    def test_regulation_win_moves_rating_more_than_an_ot_win(self):
        reg = self._final(0.0, ot_win_value=0.6)
        ot = self._final(1.0, ot_win_value=0.6)
        assert reg["BOS"] > ot["BOS"] > INITIAL_ELO
        assert reg["TOR"] < ot["TOR"] < INITIAL_ELO

    def test_ot_value_one_reproduces_the_old_behaviour(self):
        """Kept so the tuner's grid contains the pre-change model."""
        assert self._final(1.0, ot_win_value=1.0) == self._final(0.0, ot_win_value=1.0)

    def test_ot_value_half_makes_the_winner_irrelevant(self):
        """At 0.5 an overtime finish carries no information about who won, so
        the ratings must land in the same place either way.  (They still move:
        a coin-flip finish is a below-expectation result for the home team,
        which the home-advantage term expected to win outright.)"""
        won = self._final(1.0, ot_win_value=0.5)

        lost_in_ot = pd.DataFrame([{
            "game_id": "2025020001", "season": "2025-2026",
            "home_team": "BOS", "away_team": "TOR",
            "home_win": 0.0, "went_to_ot": 1.0,
        }])
        _, lost = compute_elo_ratings(lost_in_ot, return_final=True, ot_win_value=0.5)

        assert won["BOS"] == pytest.approx(lost["BOS"])
        assert won["TOR"] == pytest.approx(lost["TOR"])

    def test_missing_ot_column_falls_back_to_win_loss(self, elo_games):
        with_ot = compute_elo_ratings(elo_games, ot_win_value=1.0)
        without = compute_elo_ratings(elo_games.drop(columns=["went_to_ot"]))
        pd.testing.assert_frame_equal(with_ot, without)


class TestRatingMechanics:
    def test_ratings_are_zero_sum(self, elo_games):
        _, final = compute_elo_ratings(elo_games, return_final=True)
        assert sum(final.values()) == pytest.approx(len(final) * INITIAL_ELO)

    def test_season_boundary_regresses_toward_the_mean(self):
        games = pd.DataFrame([
            {"game_id": "2025020001", "season": "2025-2026", "home_team": "BOS",
             "away_team": "TOR", "home_win": 1.0, "went_to_ot": 0.0},
            {"game_id": "2026020001", "season": "2026-2027", "home_team": "BOS",
             "away_team": "TOR", "home_win": 1.0, "went_to_ot": 0.0},
        ])
        no_regress = compute_elo_ratings(games, season_regress=0.0)
        regressed = compute_elo_ratings(games, season_regress=0.5)
        # Second season's pre-game rating is pulled back toward 1500.
        assert INITIAL_ELO < regressed.iloc[1]["home_elo"] < no_regress.iloc[1]["home_elo"]

    def test_expansion_team_starts_at_the_initial_rating(self, elo_games):
        expansion = pd.concat([elo_games, pd.DataFrame([{
            "game_id": "2025020005", "season": "2025-2026", "home_team": "BOS",
            "away_team": "NEW", "home_win": 1.0, "went_to_ot": 0.0,
        }])], ignore_index=True)
        result = compute_elo_ratings(expansion)
        assert result.iloc[-1]["away_elo"] == INITIAL_ELO

    def test_return_final_matches_a_replay_of_the_updates(self, elo_games):
        per_game, final = compute_elo_ratings(elo_games, return_final=True)
        assert set(final) == {"BOS", "TOR", "MTL"}
        assert len(per_game) == len(elo_games)


class TestPersistedParameters:
    def test_defaults_when_no_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.setattr(elo_mod, "ELO_PARAMS_PATH", tmp_path / "absent.json")
        assert load_elo_params() == default_elo_params()

    def test_round_trip(self, tmp_path, monkeypatch):
        path = tmp_path / "elo_params.json"
        monkeypatch.setattr(elo_mod, "ELO_PARAMS_PATH", path)
        monkeypatch.setattr(elo_mod, "SAVED_DIR", tmp_path)

        save_elo_params({"k_factor": 25.0, "ot_win_value": 0.75, "brier": 0.24})
        loaded = load_elo_params()

        assert loaded["k_factor"] == 25.0
        assert loaded["ot_win_value"] == 0.75
        # Non-parameter keys from the tuner must not leak into the kwargs.
        assert "brier" not in loaded
        assert "brier" not in json.loads(path.read_text())

    def test_corrupt_file_falls_back_to_defaults(self, tmp_path, monkeypatch):
        path = tmp_path / "elo_params.json"
        path.write_text("{ not json")
        monkeypatch.setattr(elo_mod, "ELO_PARAMS_PATH", path)
        assert load_elo_params() == default_elo_params()

    def test_loaded_params_are_valid_kwargs(self, tmp_path, monkeypatch, elo_games):
        monkeypatch.setattr(elo_mod, "ELO_PARAMS_PATH", tmp_path / "absent.json")
        compute_elo_ratings(elo_games, **load_elo_params())


class TestTuning:
    def test_returns_the_best_grid_point(self, elo_games):
        best = tune_elo_parameters(
            elo_games,
            k_values=[10.0, 20.0], ha_values=[50.0],
            regress_values=[0.33], ot_values=[0.6, 1.0],
        )
        assert best["k_factor"] in (10.0, 20.0)
        assert best["ot_win_value"] in (0.6, 1.0)
        assert 0.0 <= best["brier"] <= 1.0
