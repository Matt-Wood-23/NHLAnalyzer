"""Rolling team features: no leakage, and identical between training and serving."""

import numpy as np
import pandas as pd
import pytest

from features.team import (
    EWM_HALFLIFE, EWM_STATS, ROLL_STATS, WINDOWS,
    _rolling_team_season, pregame_snapshot, rolling_feature_columns,
)


class TestNoLeakage:
    def test_rolling_windows_exclude_the_current_game(self, team_history):
        """A rolling feature that included the current game would leak the
        result the model is being asked to predict."""
        rolled = _rolling_team_season(team_history)

        for col in ROLL_STATS:
            if col not in team_history.columns:
                continue
            for w in WINDOWS:
                got = rolled[f"{col}_l{w}"].to_numpy()
                for i in range(1, len(rolled)):
                    expected = team_history[col].iloc[max(0, i - w):i].mean()
                    assert got[i] == pytest.approx(expected), f"{col}_l{w} row {i}"

    def test_first_game_of_a_season_has_no_rolling_history(self, team_history):
        rolled = _rolling_team_season(team_history)
        for col in ROLL_STATS:
            if col in team_history.columns:
                for w in WINDOWS:
                    assert np.isnan(rolled[f"{col}_l{w}"].iloc[0])

    def test_ewm_excludes_the_current_game(self, team_history):
        rolled = _rolling_team_season(team_history)
        for col in EWM_STATS:
            expected = (
                team_history[col].shift(1)
                .ewm(halflife=EWM_HALFLIFE, min_periods=1).mean()
            )
            pd.testing.assert_series_equal(
                rolled[f"{col}_ewm{EWM_HALFLIFE}"], expected, check_names=False,
            )

    def test_venue_splits_exclude_the_current_game(self, team_history):
        rolled = _rolling_team_season(team_history)
        assert np.isnan(rolled["goal_diff_home_l10"].iloc[0])
        assert np.isnan(rolled["goal_diff_away_l10"].iloc[0])

    def test_games_played_counts_only_prior_games(self, team_history):
        rolled = _rolling_team_season(team_history)
        assert rolled["games_played"].tolist() == list(range(len(team_history)))


class TestSeasonBoundary:
    def test_rolling_never_crosses_seasons(self, two_season_history):
        """Backfill groups by (team, season); the live snapshot must too, or
        October predictions are built on last season's form."""
        parts = [
            _rolling_team_season(grp)
            for _, grp in two_season_history.groupby(["team", "season"], sort=False)
        ]
        rolled = pd.concat(parts, ignore_index=True)

        second = rolled[rolled["season"] == "2026-2027"].iloc[0]
        for w in WINDOWS:
            assert np.isnan(second[f"goal_diff_l{w}"])


class TestTrainServeParity:
    """The live pipeline used to re-implement the rolling logic and disagreed
    with training on both the season boundary and the venue-split window."""

    def test_snapshot_equals_the_training_row_for_the_next_game(self, team_history, rng):
        # What training would compute for game 26, given games 1-25.
        next_game = team_history.iloc[[-1]].copy()
        next_game["game_num"] = int(team_history["game_num"].max()) + 1
        next_game["game_id"] = "2025020026"
        full = pd.concat([team_history, next_game], ignore_index=True)

        trained = _rolling_team_season(full)
        cols = rolling_feature_columns(trained)
        expected = trained.iloc[-1][cols].astype(float)

        served = pregame_snapshot(team_history, teams=["BOS"]).loc["BOS", cols].astype(float)

        pd.testing.assert_series_equal(
            expected, served, check_names=False, rtol=0, atol=1e-12,
        )

    def test_covers_every_rolling_feature(self, team_history):
        snapshot = pregame_snapshot(team_history, teams=["BOS"])
        expected = {
            f"{c}_l{w}" for c in ROLL_STATS for w in WINDOWS
        } | {
            f"{c}_ewm{EWM_HALFLIFE}" for c in EWM_STATS
        } | {
            f"{c}_{v}_l10" for c in EWM_STATS for v in ("home", "away")
        } | {"games_played"}
        assert expected <= set(snapshot.columns)


class TestOpeningNight:
    def test_team_with_no_games_still_gets_a_row(self, team_history):
        """Opening night: a team absent from the snapshot used to have its game
        dropped from the slate entirely."""
        snapshot = pregame_snapshot(team_history, teams=["BOS", "SEA"])

        assert "SEA" in snapshot.index
        assert snapshot.loc["SEA", "games_played"] == 0
        assert snapshot.drop(columns=["games_played"]).loc["SEA"].isna().all()

    def test_empty_season_yields_all_teams_with_nan_features(self, team_history):
        empty = team_history.iloc[0:0]
        snapshot = pregame_snapshot(empty, teams=["BOS", "TOR"])

        assert list(snapshot.index) == ["BOS", "TOR"]
        assert (snapshot["games_played"] == 0).all()
        assert snapshot.drop(columns=["games_played"]).isna().all().all()

    def test_partial_season_uses_only_the_games_played(self, team_history):
        three = team_history.head(3)
        snapshot = pregame_snapshot(three, teams=["BOS"])

        assert snapshot.loc["BOS", "games_played"] == 3
        assert snapshot.loc["BOS", "goal_diff_l20"] == pytest.approx(
            three["goal_diff"].mean()
        )
