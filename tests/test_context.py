"""Rest, back-to-back and division context features."""

import pandas as pd
import pytest

from config.season import season_start
from features.context import _CONFERENCES, _DIVISIONS, load_context_features


def _schedule(rows):
    """Minimal team_features frame: one home row and one away row per game."""
    out = []
    for game_num, (home, away) in enumerate(rows, start=1):
        gid = f"202502{game_num:04d}"
        for team, is_home in ((home, True), (away, False)):
            out.append({
                "game_id": gid, "season": "2025-2026", "game_num": game_num,
                "team": team, "is_home": is_home,
                "home_team": home, "away_team": away,
            })
    return pd.DataFrame(out)


class TestDates:
    def test_current_season_games_get_dates(self):
        """The per-module start-date table used to lag a season behind, which
        left every rest / back-to-back / season_day value NaN."""
        ctx = load_context_features(_schedule([("BOS", "TOR")] * 5))
        assert ctx["date"].notna().all()
        assert ctx["season_day"].notna().all()

    def test_season_day_starts_at_zero_and_never_goes_negative(self):
        ctx = load_context_features(_schedule([("BOS", "TOR")] * 40))
        assert ctx["season_day"].min() == 0
        assert (ctx["season_day"] >= 0).all()

    def test_estimated_dates_land_inside_the_season(self):
        ctx = load_context_features(_schedule([("BOS", "TOR")] * 20))
        opener = pd.Timestamp(season_start("2025-2026"))
        assert (ctx["date"] >= opener).all()


class TestRest:
    def test_rest_advantage_is_the_home_minus_away_difference(self):
        ctx = load_context_features(_schedule([
            ("BOS", "TOR"), ("MTL", "OTT"), ("BOS", "MTL"),
        ]))
        row = ctx[ctx["game_id"] == "2025020003"].iloc[0]
        expected = (row["home_rest_days"] or 2) - (row["away_rest_days"] or 2)
        assert row["rest_advantage"] == pytest.approx(expected)

    def test_back_to_back_is_a_zero_one_flag(self):
        ctx = load_context_features(_schedule([("BOS", "TOR")] * 10))
        assert set(ctx["home_back_to_back"].unique()) <= {0, 1}
        assert set(ctx["away_back_to_back"].unique()) <= {0, 1}

    def test_first_game_of_the_season_has_no_prior_rest(self):
        ctx = load_context_features(_schedule([("BOS", "TOR"), ("BOS", "MTL")]))
        first = ctx.iloc[0]
        assert pd.isna(first["home_rest_days"])
        assert first["home_back_to_back"] == 0


class TestMatchupFlags:
    def test_division_and_conference_flags(self):
        ctx = load_context_features(_schedule([
            ("BOS", "TOR"),   # both Atlantic
            ("BOS", "NYR"),   # both Eastern, different division
            ("BOS", "COL"),   # different conference
        ])).set_index("game_id")

        assert ctx.loc["2025020001", ["same_division", "same_conference"]].tolist() == [1, 1]
        assert ctx.loc["2025020002", ["same_division", "same_conference"]].tolist() == [0, 1]
        assert ctx.loc["2025020003", ["same_division", "same_conference"]].tolist() == [0, 0]


class TestLeagueStructure:
    def test_all_32_current_teams_are_mapped(self):
        from ingestion.nhl_api import TEAM_ABBREVS
        missing = [t for t in TEAM_ABBREVS if t not in _DIVISIONS]
        assert not missing, f"unmapped teams: {missing}"

    def test_every_division_has_eight_current_teams(self):
        from ingestion.nhl_api import TEAM_ABBREVS
        counts: dict[str, int] = {}
        for team in TEAM_ABBREVS:
            counts[_DIVISIONS[team]] = counts.get(_DIVISIONS[team], 0) + 1
        assert counts == {
            "Atlantic": 8, "Metropolitan": 8, "Central": 8, "Pacific": 8,
        }

    def test_conferences_derive_from_divisions(self):
        for team, div in _DIVISIONS.items():
            expected = "Eastern" if div in ("Atlantic", "Metropolitan") else "Western"
            assert _CONFERENCES[team] == expected
