"""Train/serve parity for the live snapshots.

One property, checked for each feature family: the snapshot built from a
team's first *k* games must equal what the training code computes for game
*k + 1*.  That is the whole definition of a pre-game feature.

Three of the live snapshots broke it for months.  Each re-derived the shifted
rolling mean and read the last row — the pre-game value for the game already
played — so every special-teams, goalie and opponent-quality feature was
served one game stale, and a team's second game of the season got NaN where
training had real numbers.  The tests that existed asserted on the source text
of those functions, which is why none of them noticed.
"""

import numpy as np
import pandas as pd
import pytest

from features import goalie_mp, opponent_quality, special_teams

SEASON = "2026-2027"
ST_COLS = ["pp_goals", "pp_xg", "pp_shots",
           "pk_goals_against", "pk_xg_against", "pk_shots_against"]


def _game_id(n: int, year: int = 2026) -> str:
    return f"{year}02{n:04d}"


# ---------------------------------------------------------------------------
# Special teams
# ---------------------------------------------------------------------------

@pytest.fixture
def st_games(rng):
    rows = []
    for team in ("BOS", "TOR"):
        for n in range(1, 26):
            rows.append({
                "game_id": _game_id(n), "team": team, "season": SEASON,
                "game_num": 20000 + n,
                **{c: float(rng.integers(0, 6)) for c in ST_COLS},
            })
    return pd.DataFrame(rows)


class TestSpecialTeamsParity:
    @pytest.mark.parametrize("k", [1, 2, 9, 10, 11, 20, 24])
    def test_snapshot_after_k_games_is_the_training_row_for_game_k_plus_1(self, st_games, k):
        training = special_teams.compute_special_teams_rolling(st_games)
        played = st_games[st_games["game_num"] <= 20000 + k]

        snap = special_teams.pregame_snapshot(played)

        for team in ("BOS", "TOR"):
            expected = training[
                (training["team"] == team) & (training["game_id"] == _game_id(k + 1))
            ].iloc[0]
            for col in snap.columns:
                assert snap.loc[team, col] == pytest.approx(expected[col], nan_ok=True), \
                    f"{team} {col} after {k} games"

    def test_a_team_s_second_game_sees_its_first(self, st_games):
        """The visible symptom: after one game every PP/PK feature was NaN,
        so opening-week coverage fell under the guard and games were dropped."""
        one_game = st_games[st_games["game_num"] == 20001]
        snap = special_teams.pregame_snapshot(one_game)
        bos = one_game[one_game["team"] == "BOS"].iloc[0]
        assert snap.loc["BOS", "pp_goals_l10"] == pytest.approx(bos["pp_goals"])
        assert snap.notna().all().all()

    def test_the_latest_game_is_in_the_window(self, st_games):
        """Guards the off-by-one directly: make the last game an outlier."""
        st_games.loc[
            (st_games["team"] == "BOS") & (st_games["game_num"] == 20025), "pp_goals"
        ] = 500.0
        snap = special_teams.pregame_snapshot(st_games)
        assert snap.loc["BOS", "pp_goals_l10"] > 40

    def test_a_team_with_no_games_gets_a_nan_row_not_no_row(self, st_games):
        snap = special_teams.pregame_snapshot(st_games, teams=["BOS", "SEA"])
        assert "SEA" in snap.index
        assert snap.loc["SEA"].isna().all()

    def test_nothing_played_and_nobody_asked_for(self):
        empty = pd.DataFrame(columns=["game_id", "team", "season", "game_num"] + ST_COLS)
        assert special_teams.pregame_snapshot(empty).empty


# ---------------------------------------------------------------------------
# Goalies
# ---------------------------------------------------------------------------

@pytest.fixture
def goalie_games(rng):
    """Two goalies whose starts span a season boundary, plus a relief outing."""
    rows = []
    for goalie_id, team in ((30.0, "BOS"), (31.0, "TOR")):
        for year, season in ((2025, "2025-2026"), (2026, SEASON)):
            for n in range(1, 9):
                rows.append({
                    "game_id": _game_id(n, year), "season": season,
                    "goalie_id": goalie_id, "goalie_name": f"G{int(goalie_id)}",
                    "team": team, "game_num": 20000 + n, "is_starter": True,
                    "save_pct": float(rng.uniform(0.85, 0.95)),
                    "gsax": float(rng.normal(0, 1)),
                })
    rows.append({
        "game_id": _game_id(4), "season": SEASON, "goalie_id": 32.0,
        "goalie_name": "G32", "team": "BOS", "game_num": 20004,
        "is_starter": False, "save_pct": 0.5, "gsax": -3.0,
    })
    return pd.DataFrame(rows)


class TestGoalieParity:
    @pytest.mark.parametrize("k", [1, 4, 5, 7])
    def test_snapshot_after_k_starts_is_the_training_row_for_start_k_plus_1(self, goalie_games, k):
        training = goalie_mp.compute_goalie_rolling_features(goalie_games)
        this_season = goalie_games["season"] == SEASON
        played = goalie_games[~this_season | (goalie_games["game_num"] <= 20000 + k)]

        snap = goalie_mp.pregame_snapshot(played)

        for goalie_id, team in ((30.0, "BOS"), (31.0, "TOR")):
            expected = training[
                (training["team"] == team) & (training["game_id"] == _game_id(k + 1))
            ].iloc[0]
            for col in snap.columns:
                assert snap.loc[goalie_id, col] == pytest.approx(expected[col]), \
                    f"goalie {goalie_id} {col} after {k} starts"

    def test_form_carries_across_the_season_boundary(self, goalie_games):
        """Training does not reset goalies at the boundary, so neither may
        this: a goalie entering a new season still has last season's window."""
        last_season = goalie_games[goalie_games["season"] == "2025-2026"]
        snap = goalie_mp.pregame_snapshot(last_season)
        assert snap.loc[30.0].notna().all()

    def test_the_latest_start_is_in_the_window(self, goalie_games):
        latest = (goalie_games["goalie_id"] == 30.0) & (goalie_games["season"] == SEASON) \
            & (goalie_games["game_num"] == 20008)
        goalie_games.loc[latest, "gsax"] = 100.0
        snap = goalie_mp.pregame_snapshot(goalie_games)
        assert snap.loc[30.0, "gsax_l5"] > 15

    def test_relief_appearances_are_not_starts(self, goalie_games):
        assert 32.0 not in goalie_mp.pregame_snapshot(goalie_games).index

    def test_the_refactor_left_the_training_columns_alone(self, goalie_games):
        out = goalie_mp.compute_goalie_rolling_features(goalie_games)
        assert list(out.columns) == [
            "game_id", "team",
            "save_pct_l5", "save_pct_l10", "gsax_l5", "gsax_l10",
        ]


# ---------------------------------------------------------------------------
# Opponent quality
# ---------------------------------------------------------------------------

@pytest.fixture
def oq_games(rng):
    rows = []
    for team in ("BOS", "TOR"):
        for n in range(1, 31):
            rows.append({
                "game_id": _game_id(n), "season": SEASON, "team": team,
                "game_num": 20000 + n, "opp_is_strong": int(rng.integers(0, 2)),
                **{c: float(rng.random()) for c in opponent_quality.OQ_STATS},
            })
    return pd.DataFrame(rows)


class TestOpponentQualityParity:
    @pytest.mark.parametrize("k", [3, 8, 20, 21, 29])
    def test_snapshot_after_k_games_is_the_training_row_for_game_k_plus_1(self, oq_games, k):
        training = opponent_quality._roll_by_strength(oq_games)
        played = oq_games[oq_games["game_num"] <= 20000 + k]

        snap = opponent_quality.pregame_snapshot(played)

        for team in ("BOS", "TOR"):
            expected = training[
                (training["team"] == team) & (training["game_id"] == _game_id(k + 1))
            ].iloc[0]
            for col in snap.columns:
                assert snap.loc[team, col] == pytest.approx(expected[col], nan_ok=True), \
                    f"{team} {col} after {k} games"

    def test_the_latest_game_is_in_the_window(self, oq_games):
        last = (oq_games["team"] == "BOS") & (oq_games["game_num"] == 20030)
        oq_games.loc[last, "opp_is_strong"] = 1
        oq_games.loc[last, "goal_diff"] = 1000.0
        snap = opponent_quality.pregame_snapshot(oq_games)
        assert snap.loc["BOS", "goal_diff_vs_strong_l20"] > 50

    def test_road_games_count(self, oq_games):
        """The old snapshot read each team's last *home* row off the feature
        matrix, so a team that had only played away had no row at all."""
        snap = opponent_quality.pregame_snapshot(oq_games, teams=["BOS", "TOR", "SEA"])
        assert {"BOS", "TOR", "SEA"} <= set(snap.index)
        assert snap.loc["SEA"].isna().all()

    def test_columns_match_what_training_merges(self, oq_games):
        assert list(opponent_quality.pregame_snapshot(oq_games).columns) == \
            opponent_quality._rolling_columns()


# ---------------------------------------------------------------------------
# The live builders actually use them
# ---------------------------------------------------------------------------

class TestLiveBuildersUseTheSharedCode:
    def test_the_goalie_snapshot_includes_the_last_start(self, goalie_games, tmp_path, monkeypatch):
        from pipeline import live

        latest = (goalie_games["goalie_id"] == 30.0) & (goalie_games["season"] == SEASON) \
            & (goalie_games["game_num"] == 20008)
        goalie_games.loc[latest, "gsax"] = 100.0
        goalie_games.to_parquet(tmp_path / "goalie_game_stats.parquet", index=False)
        monkeypatch.setattr(live, "PARQUET_DIR", tmp_path)

        snap = live._build_goalie_snapshot()
        assert snap.loc["BOS", "g_gsax_l5"] > 15
        assert set(snap.columns) == {
            "g_save_pct_l5", "g_save_pct_l10", "g_gsax_l5", "g_gsax_l10",
        }

    def test_the_opponent_quality_snapshot_reads_the_saved_games(self, oq_games, tmp_path, monkeypatch):
        from pipeline import live

        path = tmp_path / "opponent_quality_games.parquet"
        oq_games.to_parquet(path, index=False)
        monkeypatch.setattr(opponent_quality, "OQ_GAMES_PATH", path)

        snap = live._build_opponent_quality_snapshot(season=SEASON, teams=["BOS", "SEA"])
        expected = opponent_quality.pregame_snapshot(oq_games, teams=["BOS", "SEA"])
        pd.testing.assert_frame_equal(snap, expected)

    def test_a_missing_intermediate_degrades_to_nan_not_a_crash(self, tmp_path, monkeypatch):
        from pipeline import live

        monkeypatch.setattr(opponent_quality, "OQ_GAMES_PATH", tmp_path / "absent.parquet")
        assert live._build_opponent_quality_snapshot().empty

    def test_last_season_s_games_are_not_carried_in(self, oq_games, tmp_path, monkeypatch):
        from pipeline import live

        path = tmp_path / "opponent_quality_games.parquet"
        oq_games.assign(season="2025-2026").to_parquet(path, index=False)
        monkeypatch.setattr(opponent_quality, "OQ_GAMES_PATH", path)
        assert live._build_opponent_quality_snapshot(season=SEASON).empty

    def test_the_backfill_saves_what_the_live_snapshot_reads(self):
        """Without save_intermediate the file is never written and every
        opponent-quality feature is silently NaN in production."""
        import inspect
        from pipeline import backfill
        assert "save_intermediate=True" in inspect.getsource(backfill.build_feature_matrix)
