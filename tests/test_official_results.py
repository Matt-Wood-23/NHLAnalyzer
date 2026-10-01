"""Who won, according to the league rather than the shot chart.

Wins used to be ``goals_for > goals_against`` on MoneyPuck's shot totals.
Those contain no shootout, so a shootout game is tied in the data and *both*
teams were recorded as losing it.  Every home shootout win was a home loss in
the training target: 243 games, 3.5% of the matrix, all in one direction.  The
matrix put the home win rate at 50.2%; the league's is 53.7%, and the model
under-rated every home team by the difference.
"""

import numpy as np
import pandas as pd
import pytest

from features import team
from ingestion import boxscore
from pipeline import evaluate_history


def _box(game_id=2025020101, state="OFF", home=3, away=2, last="SO", outcome=True):
    box = {
        "id": game_id, "season": 20252026, "gameType": 2,
        "gameDate": "2025-10-20", "gameState": state,
        "homeTeam": {"abbrev": "BOS", "score": home},
        "awayTeam": {"abbrev": "TOR", "score": away},
        "periodDescriptor": {"periodType": last},
    }
    if outcome:
        box["gameOutcome"] = {"lastPeriodType": last}
    return box


class TestParseGameResult:
    def test_a_shootout_has_a_winner(self):
        """The official score carries the shootout goal; that is the point."""
        (row,) = boxscore.parse_game_result(_box(home=3, away=2, last="SO"))
        assert row["home_win"] is True
        assert (row["went_to_ot"], row["went_to_so"]) == (True, True)
        assert (row["home_score"], row["away_score"]) == (3, 2)

    def test_overtime_is_not_a_shootout(self):
        (row,) = boxscore.parse_game_result(_box(home=1, away=2, last="OT"))
        assert row["home_win"] is False
        assert (row["went_to_ot"], row["went_to_so"]) == (True, False)

    def test_regulation(self):
        (row,) = boxscore.parse_game_result(_box(home=4, away=1, last="REG"))
        assert (row["went_to_ot"], row["went_to_so"]) == (False, False)

    @pytest.mark.parametrize("state", ["LIVE", "CRIT", "FUT", "PRE"])
    def test_an_unfinished_game_has_no_result(self, state):
        """A live boxscore with a lead is not a win."""
        assert boxscore.parse_game_result(_box(state=state)) == []

    def test_a_level_score_is_not_a_result(self):
        assert boxscore.parse_game_result(_box(home=2, away=2)) == []

    def test_missing_scores_are_not_a_result(self):
        assert boxscore.parse_game_result(_box(home=None, away=None)) == []

    def test_the_period_falls_back_when_the_outcome_block_is_absent(self):
        (row,) = boxscore.parse_game_result(_box(last="SO", outcome=False))
        assert row["went_to_so"] is True

    def test_identity_fields(self):
        (row,) = boxscore.parse_game_result(_box())
        assert row["game_id"] == "2025020101"
        assert (row["home_team"], row["away_team"]) == ("BOS", "TOR")
        assert row["game_date"] == "2025-10-20"


def _team_rows(game_id, home_goals, away_goals):
    return [
        {"game_id": game_id, "team": "BOS", "is_home": True,
         "goals_for": home_goals, "goals_against": away_goals},
        {"game_id": game_id, "team": "TOR", "is_home": False,
         "goals_for": away_goals, "goals_against": home_goals},
    ]


def _with_naive_won(rows):
    df = pd.DataFrame(rows)
    df["won"] = (df["goals_for"] > df["goals_against"]).astype(float)
    return df


class TestApplyOfficialResults:
    def test_a_home_shootout_win_is_a_home_win(self):
        """The bug itself: tied on shot totals, so both sides had lost."""
        df = _with_naive_won(_team_rows("g1", 2, 2))
        assert df["won"].tolist() == [0.0, 0.0]

        out = team.apply_official_results(df, {"g1": True})
        assert out.set_index("team")["won"].to_dict() == {"BOS": 1.0, "TOR": 0.0}

    def test_an_away_shootout_win_goes_to_the_road_team(self):
        out = team.apply_official_results(_with_naive_won(_team_rows("g1", 2, 2)), {"g1": False})
        assert out.set_index("team")["won"].to_dict() == {"BOS": 0.0, "TOR": 1.0}

    def test_exactly_one_team_wins_every_decided_game(self):
        rows = _team_rows("g1", 2, 2) + _team_rows("g2", 4, 1) + _team_rows("g3", 0, 0)
        out = team.apply_official_results(
            _with_naive_won(rows), {"g1": True, "g2": True, "g3": False},
        )
        assert out.groupby("game_id")["won"].sum().tolist() == [1.0, 1.0, 1.0]

    def test_a_game_without_an_official_result_keeps_the_shot_total_answer(self):
        out = team.apply_official_results(_with_naive_won(_team_rows("g1", 4, 1)), {})
        assert out.set_index("team")["won"].to_dict() == {"BOS": 1.0, "TOR": 0.0}

    def test_an_undecidable_tie_is_reported(self, caplog):
        with caplog.at_level("WARNING"):
            team.apply_official_results(_with_naive_won(_team_rows("g1", 2, 2)), {})
        assert "no official result" in caplog.text

    def test_no_warning_when_every_tie_is_resolved(self, caplog):
        with caplog.at_level("WARNING"):
            team.apply_official_results(_with_naive_won(_team_rows("g1", 2, 2)), {"g1": True})
        assert "no official result" not in caplog.text

    def test_the_on_ice_margin_is_left_alone(self):
        df = _with_naive_won(_team_rows("g1", 2, 2))
        df["goal_diff"] = df["goals_for"] - df["goals_against"]
        out = team.apply_official_results(df, {"g1": True})
        assert out["goal_diff"].tolist() == [0, 0]

    def test_the_caller_s_frame_is_not_modified(self):
        df = _with_naive_won(_team_rows("g1", 2, 2))
        team.apply_official_results(df, {"g1": True})
        assert df["won"].tolist() == [0.0, 0.0]


class TestTheFileIsUsed:
    def _write_results(self, rows):
        team.GAME_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["game_id", "home_win"]).to_parquet(
            team.GAME_RESULTS_PATH, index=False,
        )

    def test_missing_file_means_no_overrides(self):
        assert team.official_home_wins() == {}

    def test_it_reads_what_the_boxscore_builder_writes(self):
        self._write_results([("2025020101", True), ("2025020102", False)])
        assert team.official_home_wins() == {"2025020101": True, "2025020102": False}

    def test_the_derived_columns_take_the_official_winner(self):
        """Training and the live snapshot both go through _add_derived_columns,
        so this is the one place the fix has to hold."""
        self._write_results([("2025020101", True)])
        rows = _team_rows("2025020101", 2, 2)
        for r in rows:
            r.update({
                "opp_team": "TOR" if r["team"] == "BOS" else "BOS",
                "xg_for": 2.0, "xg_against": 2.0, "xg_for_5v5": 1.5,
                "corsi_for": 50, "corsi_against": 50,
                "shots_for": 30, "shots_against": 30,
            })
        out = team._add_derived_columns(pd.DataFrame(rows))
        assert out.set_index("team")["won"].to_dict() == {"BOS": 1.0, "TOR": 0.0}

    def test_the_daily_run_refreshes_results_before_the_matrix(self):
        """Last night's games are not in the file until this runs; the matrix
        built without them would label any shootout among them by shot totals."""
        import inspect
        from pipeline import daily
        src = inspect.getsource(daily.refresh_data)
        assert src.index("build_game_results()") < src.index("build_feature_matrix(")

    def test_the_backfill_entry_point_does_too(self):
        import inspect
        from pipeline import backfill
        src = inspect.getsource(backfill)
        main = src[src.index('if __name__ == "__main__"'):]
        assert main.index("build_game_results()") < main.index("build_feature_matrix(")


class TestScoringUsesOfficialResults:
    """The prediction log was scored from the matrix target, so its stored
    outcomes inherited the same error."""

    def _write(self, name, frame):
        evaluate_history.PARQUET_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(evaluate_history.PARQUET_DIR / name, index=False)

    def _hist(self, stored=np.nan):
        return pd.DataFrame({
            "game_id": ["2025020101"], "home_team": ["BOS"], "away_team": ["TOR"],
            "prob_home_win": [0.60], "model_name": ["random_forest"],
            "actual_home_win": [stored],
        })

    def test_no_sources_means_no_outcomes(self):
        assert evaluate_history.load_outcomes() is None

    def test_official_results_win_over_the_matrix(self):
        self._write("game_results.parquet",
                    pd.DataFrame({"game_id": ["2025020101"], "home_win": [True]}))
        self._write("feature_matrix.parquet",
                    pd.DataFrame({"game_id": ["2025020101"], "target": [0.0]}))
        out = evaluate_history.load_outcomes().set_index("game_id")
        assert out.loc["2025020101", "target"] == 1.0

    def test_the_matrix_fills_games_the_results_lack(self):
        self._write("game_results.parquet",
                    pd.DataFrame({"game_id": ["2025020101"], "home_win": [True]}))
        self._write("feature_matrix.parquet",
                    pd.DataFrame({"game_id": ["2025020101", "2025020102"], "target": [0.0, 1.0]}))
        out = evaluate_history.load_outcomes().set_index("game_id")
        assert out.loc["2025020102", "target"] == 1.0
        assert len(out) == 2

    def test_a_wrongly_stored_outcome_is_corrected(self):
        """A home shootout win logged as a loss: 0.60 on the home team was
        scored as wrong when it was right."""
        self._write("game_results.parquet",
                    pd.DataFrame({"game_id": ["2025020101"], "home_win": [True]}))
        out = evaluate_history.attach_outcomes(self._hist(stored=0.0))
        assert out["actual_home_win"].iloc[0] == 1.0
        assert bool(out["correct"].iloc[0]) is True
        assert out["brier"].iloc[0] == pytest.approx(0.16)

    def test_a_stored_outcome_survives_when_no_source_knows_the_game(self):
        self._write("game_results.parquet",
                    pd.DataFrame({"game_id": ["2025020999"], "home_win": [True]}))
        out = evaluate_history.attach_outcomes(self._hist(stored=1.0))
        assert out["actual_home_win"].iloc[0] == 1.0


class TestRecapShowsTheRealRecord:
    def test_a_shootout_win_is_a_win_in_the_recap(self, tmp_path, monkeypatch):
        """/recap computed its own W/L from shot totals, so a team that won
        three shootouts in ten games was shown three wins short."""
        from bot import discord_bot

        rows = []
        for n, (gf, ga) in enumerate([(3, 1), (2, 2), (1, 4)], start=1):
            rows.append({
                "game_id": f"202502{n:04d}", "season": "2025-2026", "team": "BOS",
                "opp_team": "TOR", "is_home": True,
                "goals_for": gf, "goals_against": ga,
                "xg_for": 2.5, "xg_against": 2.0,
                "shots_for": 30, "shots_against": 28,
            })
        pd.DataFrame(rows).to_parquet(
            tmp_path / "moneypuck_team_game_stats.parquet", index=False,
        )
        monkeypatch.setattr(discord_bot, "_PARQUET_DIR", tmp_path)
        team.GAME_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "game_id": ["2025020001", "2025020002", "2025020003"],
            "home_win": [True, True, False],
        }).to_parquet(team.GAME_RESULTS_PATH, index=False)

        recap = discord_bot.build_team_recap("BOS", n_games=3)
        assert [g["result"] for g in recap["games"]] == ["W", "W", "L"]
        assert recap["record"] == (2, 1)

