"""Season configuration — the thing that used to break every October."""

from datetime import date

import pytest

from config import season as cfg


class TestFormatConversion:
    def test_round_trips_between_formats(self):
        assert cfg.season_label(2025) == "2025-2026"
        assert cfg.season_api(2025) == "20252026"
        assert cfg.label_to_api("2025-2026") == "20252026"
        assert cfg.api_to_label("20252026") == "2025-2026"

    @pytest.mark.parametrize("value", ["2025-2026", "20252026", "2025", 2025])
    def test_season_year_accepts_every_format_in_the_codebase(self, value):
        assert cfg.season_year(value) == 2025

    def test_unrecognized_format_raises(self):
        with pytest.raises(ValueError):
            cfg.season_year("not-a-season")


class TestCurrentSeason:
    def test_rolls_over_in_september_not_january(self):
        assert cfg.current_season(date(2026, 8, 31)) == "2025-2026"
        assert cfg.current_season(date(2026, 9, 1)) == "2026-2027"
        assert cfg.current_season(date(2027, 1, 15)) == "2026-2027"
        assert cfg.current_season(date(2027, 6, 30)) == "2026-2027"

    def test_env_override_pins_a_season(self, monkeypatch):
        monkeypatch.setenv("NHL_CURRENT_SEASON", "2023-2024")
        assert cfg.current_season(date(2026, 12, 1)) == "2023-2024"
        assert cfg.current_season_api(date(2026, 12, 1)) == "20232024"


class TestSeasonList:
    def test_extends_automatically_without_a_code_change(self):
        assert cfg.all_seasons(date(2026, 10, 1))[-1] == "2026-2027"
        assert cfg.all_seasons(date(2027, 10, 1))[-1] == "2027-2028"

    def test_starts_at_the_first_moneypuck_season_and_has_no_gaps(self):
        seasons = cfg.all_seasons(date(2026, 10, 1))
        assert seasons[0] == cfg.season_label(cfg.FIRST_SEASON_YEAR)
        years = [cfg.season_year(s) for s in seasons]
        assert years == list(range(years[0], years[-1] + 1))

    def test_moneypuck_map_covers_every_season(self):
        today = date(2026, 10, 1)
        assert set(cfg.moneypuck_seasons(today)) == set(cfg.all_seasons(today))


class TestSeasonStarts:
    def test_known_openers_are_used_verbatim(self):
        assert cfg.season_start("2021-2022") == date(2021, 10, 12)
        assert cfg.season_start("2025-2026") == date(2025, 10, 8)

    def test_every_season_resolves_to_a_date(self):
        """The old per-module tables silently lagged a season behind, which
        turned every rest / back-to-back feature of the newest season into NaN."""
        for label in cfg.all_seasons(date(2027, 10, 1)):
            assert isinstance(cfg.season_start(label), date)

    def test_unknown_season_falls_back_to_an_october_estimate(self):
        estimated = cfg.season_start("2031-2032")
        assert estimated.year == 2031 and estimated.month == 10

    def test_game_date_estimate_stays_inside_the_season(self):
        opener = cfg.season_start("2025-2026")
        first = cfg.approximate_game_date("2025-2026", 1)
        last = cfg.approximate_game_date("2025-2026", cfg.TOTAL_GAMES)
        assert first >= opener
        assert (last - opener).days <= cfg.SEASON_DAYS

    def test_total_games_matches_a_32_team_league(self):
        assert cfg.TOTAL_GAMES == 32 * 82 // 2
