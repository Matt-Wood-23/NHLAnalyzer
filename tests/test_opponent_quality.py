"""Opponent strength must be knowable on the day.

A game's opponent used to be labelled strong or weak against the *season
median* of opponent ELO, computed over the whole season — so an October game
was classified using ratings from April.  Training had that look-ahead; the
live pipeline, with only the season to date, could not reproduce it.
"""

import numpy as np
import pandas as pd
import pytest

from features import opponent_quality
from features.elo import INITIAL_ELO

SEASON = "2025-2026"
TEAMS = ["BOS", "TOR", "MTL", "OTT"]


@pytest.fixture
def season(rng):
    """A 40-game round robin with per-game pre-game ELO for both sides.

    Ratings drift upward through the season so that the median over the whole
    season differs from the median over its first half — the condition under
    which the old classification gave different answers for the same game.
    """
    team_rows, elo_rows = [], []
    for n in range(1, 41):
        home, away = TEAMS[n % 4], TEAMS[(n + 1 + n // 4) % 4]
        if home == away:
            away = TEAMS[(TEAMS.index(home) + 1) % 4]
        gid = f"202502{n:04d}"
        drift = n * 2.0
        elo_rows.append({
            "game_id": gid,
            "home_elo": 1470 + drift + float(rng.normal(0, 25)),
            "away_elo": 1470 + drift + float(rng.normal(0, 25)),
        })
        home_won = bool(rng.integers(0, 2))
        for team, opp, is_home in ((home, away, True), (away, home, False)):
            team_rows.append({
                "game_id": gid, "season": SEASON, "team": team, "opp_team": opp,
                "home_team": home, "away_team": away, "is_home": is_home,
                "game_num": 20000 + n,
                "won": float(home_won == is_home),
                "xgf_pct": float(rng.uniform(0.4, 0.6)),
                "cf_pct": float(rng.uniform(0.4, 0.6)),
                "goal_diff": float(rng.integers(-3, 4)),
            })
    return pd.DataFrame(team_rows), pd.DataFrame(elo_rows)


def _features(team_features, elo_df):
    return (
        opponent_quality.compute_opponent_quality_features(team_features, elo_df=elo_df)
        .sort_values(["game_id", "team"]).reset_index(drop=True)
    )


class TestNoLookAhead:
    def test_later_games_cannot_change_earlier_rows(self, season):
        """Truncate the season and the surviving rows must not move.  Under the
        season-median split they did: the median itself depended on games not
        yet played."""
        team_features, elo_df = season
        full = _features(team_features, elo_df)

        early = team_features["game_num"] <= 20020
        first_half = _features(
            team_features[early], elo_df[elo_df["game_id"].isin(team_features.loc[early, "game_id"])],
        )

        same_rows = full[full["game_id"].isin(first_half["game_id"])].reset_index(drop=True)
        pd.testing.assert_frame_equal(same_rows, first_half)

    def test_the_fixture_would_have_caught_the_old_rule(self, season):
        """Guards the guard: the two medians really do differ here, so the test
        above is not passing by accident."""
        team_features, elo_df = season
        opp = pd.concat([elo_df["home_elo"], elo_df["away_elo"]])
        first = elo_df[elo_df["game_id"] <= "2025020020"]
        opp_first = pd.concat([first["home_elo"], first["away_elo"]])
        between = opp_first.between(*sorted([opp.median(), opp_first.median()]))
        assert between.any(), "no game sits between the two medians"

    def test_the_threshold_is_the_league_mean(self):
        assert opponent_quality.STRONG_OPPONENT_ELO == INITIAL_ELO


class TestClassification:
    def test_an_opponent_at_exactly_the_mean_is_strong(self, season, tmp_path, monkeypatch):
        team_features, elo_df = season
        elo_df = elo_df.assign(home_elo=float(INITIAL_ELO), away_elo=float(INITIAL_ELO))
        path = tmp_path / "oq.parquet"
        monkeypatch.setattr(opponent_quality, "OQ_GAMES_PATH", path)
        opponent_quality.compute_opponent_quality_features(
            team_features, elo_df=elo_df, save_intermediate=True,
        )
        assert pd.read_parquet(path)["opp_is_strong"].eq(1).all()

    def test_the_saved_games_are_what_the_live_snapshot_needs(self, season, tmp_path, monkeypatch):
        team_features, elo_df = season
        path = tmp_path / "oq.parquet"
        monkeypatch.setattr(opponent_quality, "OQ_GAMES_PATH", path)
        opponent_quality.compute_opponent_quality_features(
            team_features, elo_df=elo_df, save_intermediate=True,
        )
        saved = pd.read_parquet(path)
        snap = opponent_quality.pregame_snapshot(saved[saved["season"] == SEASON])
        assert set(snap.index) == set(TEAMS)
        assert list(snap.columns) == opponent_quality._rolling_columns()

    def test_nothing_is_written_unless_asked(self, season, tmp_path, monkeypatch):
        team_features, elo_df = season
        path = tmp_path / "oq.parquet"
        monkeypatch.setattr(opponent_quality, "OQ_GAMES_PATH", path)
        opponent_quality.compute_opponent_quality_features(team_features, elo_df=elo_df)
        assert not path.exists()

    def test_training_and_the_snapshot_agree_end_to_end(self, season, tmp_path, monkeypatch):
        """From raw games through the saved intermediate: the snapshot after
        game 30 is the training row for each team's next game."""
        team_features, elo_df = season
        full = opponent_quality.compute_opponent_quality_features(team_features, elo_df=elo_df)

        path = tmp_path / "oq.parquet"
        monkeypatch.setattr(opponent_quality, "OQ_GAMES_PATH", path)
        played = team_features[team_features["game_num"] <= 20030]
        opponent_quality.compute_opponent_quality_features(
            played, elo_df=elo_df[elo_df["game_id"].isin(played["game_id"])],
            save_intermediate=True,
        )
        snap = opponent_quality.pregame_snapshot(pd.read_parquet(path))

        later = team_features[team_features["game_num"] > 20030].sort_values("game_num")
        for team in TEAMS:
            nxt = later[later["team"] == team].iloc[0]
            expected = full[(full["game_id"] == nxt["game_id"]) & (full["team"] == team)].iloc[0]
            for col in snap.columns:
                assert snap.loc[team, col] == pytest.approx(expected[col], nan_ok=True), (team, col)
