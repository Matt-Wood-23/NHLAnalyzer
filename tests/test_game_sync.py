"""Keeping the games table current.

Both the training context features and the live rest-day calculation read
game dates from Postgres, and nothing in the daily run wrote them.  2026-27
opened with the table a season behind: every team looked ~170 days rested
however recently it had played.
"""

from datetime import date

import pytest

from ingestion import nhl_api


def _api_game(game_id, state, game_type=2, home=None, away=None, outcome=None):
    return {
        "id": game_id, "season": 20262027, "gameType": game_type,
        "gameState": state, "startTimeUTC": "2026-09-30T23:30:00Z",
        "homeTeam": {"abbrev": "PHI", "score": home},
        "awayTeam": {"abbrev": "PIT", "score": away},
        **({"gameOutcome": {"lastPeriodType": outcome}} if outcome else {}),
    }


class TestScheduleParsing:
    def test_state_and_outcome_are_carried_through(self, monkeypatch):
        payload = {"gameWeek": [{"date": "2026-09-30", "games": [
            _api_game(2026020006, "OFF", home=0, away=7, outcome="REG"),
            _api_game(2026020007, "LIVE", home=6, away=3),
            _api_game(2026020009, "FUT"),
        ]}]}
        monkeypatch.setattr(nhl_api, "_get", lambda url: payload)

        games = nhl_api.fetch_schedule("2026-09-30")
        assert [g["game_state"] for g in games] == ["OFF", "LIVE", "FUT"]
        assert games[0]["last_period_type"] == "REG"
        assert games[1]["last_period_type"] is None
        assert games[2]["home_score"] is None
        assert games[0]["start_time_utc"] == "2026-09-30T23:30:00Z"

    def test_the_two_state_sets_do_not_overlap(self):
        assert not nhl_api.FINAL_STATES & nhl_api.PREGAME_STATES


@pytest.fixture
def synced(monkeypatch):
    """Run sync_completed_games against a canned week; return what it wrote."""
    week = [
        {"game_id": "2026020001", "date": "2026-09-29", "season": "20262027",
         "game_type": "2", "home_team": "CAR", "away_team": "FLA",
         "home_score": 0, "away_score": 1, "game_state": "OFF", "last_period_type": "OT"},
        {"game_id": "2026020004", "date": "2026-09-29", "season": "20262027",
         "game_type": "2", "home_team": "EDM", "away_team": "VAN",
         "home_score": 5, "away_score": 6, "game_state": "FINAL", "last_period_type": "SO"},
        {"game_id": "2026020006", "date": "2026-09-30", "season": "20262027",
         "game_type": "2", "home_team": "PHI", "away_team": "PIT",
         "home_score": 0, "away_score": 7, "game_state": "OFF", "last_period_type": "REG"},
        {"game_id": "2026020007", "date": "2026-09-30", "season": "20262027",
         "game_type": "2", "home_team": "COL", "away_team": "LAK",
         "home_score": 6, "away_score": 3, "game_state": "LIVE", "last_period_type": None},
        {"game_id": "2026020009", "date": "2026-10-01", "season": "20262027",
         "game_type": "2", "home_team": "NJD", "away_team": "PHI",
         "home_score": None, "away_score": None, "game_state": "FUT", "last_period_type": None},
        {"game_id": "2026010088", "date": "2026-09-27", "season": "20262027",
         "game_type": "1", "home_team": "BOS", "away_team": "NYR",
         "home_score": 2, "away_score": 1, "game_state": "OFF", "last_period_type": "REG"},
        {"game_id": "2026030111", "date": "2027-04-14", "season": "20262027",
         "game_type": "3", "home_team": "BOS", "away_team": "TOR",
         "home_score": 3, "away_score": 2, "game_state": "OFF", "last_period_type": "REG"},
    ]
    calls = []
    written = []

    class Conn:
        commits = 0

        def commit(self):
            self.commits += 1

    def fake_schedule(day):
        calls.append(day)
        return week

    monkeypatch.setattr(nhl_api, "fetch_schedule", fake_schedule)
    monkeypatch.setattr(nhl_api, "upsert_game", lambda conn, game: written.append(game))
    conn = Conn()
    n = nhl_api.sync_completed_games(conn, date(2026, 9, 29), until=date(2026, 10, 14))
    return n, {g["game_id"]: g for g in written}, written, calls, conn


class TestSyncCompletedGames:
    def test_only_finished_games_are_written(self, synced):
        n, by_id, *_ = synced
        assert "2026020007" not in by_id, "a live game has no result yet"
        assert "2026020009" not in by_id, "a scheduled game is not a played one"
        assert n == len(by_id) == 4

    def test_preseason_is_ignored(self, synced):
        _, by_id, *_ = synced
        assert "2026010088" not in by_id

    def test_playoff_games_are_kept(self, synced):
        """The training matrix includes playoffs, so they need dates too."""
        _, by_id, *_ = synced
        assert "2026030111" in by_id

    def test_the_winner_comes_from_the_score(self, synced):
        _, by_id, *_ = synced
        assert by_id["2026020006"]["home_win"] is False     # PHI 0, PIT 7
        assert by_id["2026030111"]["home_win"] is True

    def test_overtime_and_shootout_flags(self, synced):
        _, by_id, *_ = synced
        ot, so, reg = by_id["2026020001"], by_id["2026020004"], by_id["2026020006"]
        assert (ot["went_to_ot"], ot["went_to_so"]) == (True, False)
        assert (so["went_to_ot"], so["went_to_so"]) == (True, True)
        assert (reg["went_to_ot"], reg["went_to_so"]) == (False, False)

    def test_the_date_is_the_game_s_not_the_job_s(self, synced):
        _, by_id, *_ = synced
        assert by_id["2026020001"]["date"] == "2026-09-29"

    def test_a_game_seen_in_two_weekly_calls_is_written_once(self, synced):
        _, by_id, written, calls, _ = synced
        assert len(calls) == 3, "one schedule call per week in the range"
        assert len(written) == len(by_id)

    def test_it_commits(self, synced):
        *_, conn = synced
        assert conn.commits == 1


class TestSyncRecentGames:
    def _conn(self, latest):
        class Cur:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def execute(self, sql, params):
                self.params = params

            def fetchone(self):
                return (latest,)

        class Conn:
            def cursor(self):
                return Cur()

        return Conn()

    def test_an_empty_season_starts_from_the_opener(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(nhl_api, "sync_completed_games",
                            lambda conn, since: seen.setdefault("since", since) and 0)
        nhl_api.sync_recent_games(self._conn(None), date(2026, 9, 29))
        assert seen["since"] == date(2026, 9, 29)

    def test_it_rereads_the_last_week(self, monkeypatch):
        """A postponed game can finish after later ones were already synced."""
        seen = {}
        monkeypatch.setattr(nhl_api, "sync_completed_games",
                            lambda conn, since: seen.setdefault("since", since) and 0)
        nhl_api.sync_recent_games(self._conn(date(2026, 11, 20)), date(2026, 9, 29))
        assert seen["since"] == date(2026, 11, 13)

    def test_it_never_reaches_back_before_the_opener(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(nhl_api, "sync_completed_games",
                            lambda conn, since: seen.setdefault("since", since) and 0)
        nhl_api.sync_recent_games(self._conn(date(2026, 10, 1)), date(2026, 9, 29))
        assert seen["since"] == date(2026, 9, 29)
