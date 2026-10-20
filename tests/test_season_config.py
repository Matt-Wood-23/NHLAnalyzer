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

    def test_2026_27_opens_in_september(self):
        """The table said 2026-10-05, annotated "confirmed".  It was not: the
        season opened on 2026-09-29, which put season_day at -6 on night one.
        The value comes from regularSeasonStartDate on /v1/schedule/{date}."""
        assert cfg.season_start("2026-2027") == date(2026, 9, 29)

    def test_no_game_is_scheduled_before_its_season_opens(self):
        """Cheap sanity net for the next wrong entry: an opener in the table
        must not fall after the rollover-to-rollover window it belongs to."""
        for year, opener in cfg.SEASON_STARTS.items():
            d = date.fromisoformat(opener)
            assert d.year == year and d.month in (9, 10), (year, opener)


class TestScheduleShape:
    """2026-27 went to 84 games per team; the date estimate assumed 82."""

    def test_the_82_game_seasons_are_unchanged(self):
        assert cfg.schedule_shape("2025-2026") == (1312, 185)
        assert cfg.schedule_shape("2021-2022") == (1312, 185)

    def test_2026_27_is_84_games(self):
        games, days = cfg.schedule_shape("2026-2027")
        assert games == 32 * 84 // 2
        assert cfg.season_start("2026-2027") + cfg.timedelta(days=days) == date(2027, 4, 10)

    def test_later_seasons_inherit_the_latest_shape(self):
        assert cfg.schedule_shape(2031) == cfg.schedule_shape(2026)

    def test_a_season_before_the_table_starts_does_not_crash(self):
        assert cfg.schedule_shape(2015) == (1312, 185)

    def test_the_last_game_lands_at_the_end_of_the_season(self):
        last = cfg.approximate_game_date("2026-2027", 1344)
        assert last == date(2027, 4, 10)


class TestApproximateDateTakesTheSortKey:
    """game_number() returns TTNNNN.  Both no-database fallbacks pass that
    straight in, and when this function only understood NNNN it read game 311
    as game 20,311 and dated it eight years after the opener."""

    def test_the_sort_key_and_the_bare_number_agree(self):
        assert cfg.approximate_game_date("2025-2026", 20311) == \
            cfg.approximate_game_date("2025-2026", 311)

    def test_a_regular_season_key_stays_inside_the_season(self):
        opener = cfg.season_start("2025-2026")
        for key in (20001, 20656, 21312):
            d = cfg.approximate_game_date("2025-2026", key)
            assert opener <= d <= opener + cfg.timedelta(days=cfg.SEASON_DAYS), key

    def test_playoffs_come_after_the_regular_season(self):
        last_regular = cfg.approximate_game_date("2025-2026", 21312)
        first_playoff = cfg.approximate_game_date("2025-2026", 30111)
        assert first_playoff > last_regular

    def test_playoff_rounds_are_in_order(self):
        keys = [30111, 30117, 30211, 30311, 30411, 30417]
        dates = [cfg.approximate_game_date("2025-2026", k) for k in keys]
        assert dates == sorted(dates)

    def test_it_composes_with_game_number(self):
        key = int(cfg.game_number(["2025020311"]).iloc[0])
        d = cfg.approximate_game_date("2025-2026", key)
        assert d.year in (2025, 2026)


class TestGameNumber:
    """The sort key every rolling feature depends on.

    game_id is SSSSTTNNNN.  Keying on the trailing four digits collides across
    game types, so playoff game 2024030311 and regular-season 2024020311 both
    became 311 and a player's playoff games sorted into the middle of his
    regular season -- leaking later results into earlier rows for the 643 of
    1471 skaters whose teams made the playoffs.
    """

    def test_playoffs_sort_after_the_regular_season(self):
        from config.season import game_number
        nums = game_number(["2024020311", "2024030311"])
        assert nums.iloc[0] < nums.iloc[1]

    def test_the_old_key_would_have_collided(self):
        """Pins why this helper exists rather than the obvious one-liner."""
        assert "2024020311"[-4:] == "2024030311"[-4:]

    def test_ordering_within_a_type_is_preserved(self):
        from config.season import game_number
        nums = game_number(["2024020001", "2024020100", "2024021312"]).tolist()
        assert nums == sorted(nums)

    def test_it_returns_plain_ints(self):
        """Callers sort and compare on it, and one builds a placeholder row
        from max() + 1."""
        from config.season import game_number
        import numpy as np
        assert np.issubdtype(game_number(["2024020001"]).dtype, np.integer)

    def test_it_accepts_a_series_or_a_list(self):
        import pandas as pd
        from config.season import game_number
        as_list = game_number(["2024020311"]).tolist()
        as_series = game_number(pd.Series(["2024020311"])).tolist()
        assert as_list == as_series == [20311]

    def test_every_module_uses_the_shared_key(self):
        """13 call sites had their own copy of the buggy expression."""
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        # Only this project's code: a virtualenv inside the repo holds
        # third-party files, some not even UTF-8.
        skip = {"tests", "__pycache__", "site-packages"}
        offenders = [
            p.relative_to(root).as_posix()
            for p in root.rglob("*.py")
            if not any(
                part in skip or part.startswith(".")
                for part in p.relative_to(root).parts
            )
            and 'str[-4:]' in p.read_text(encoding="utf-8")
        ]
        assert offenders == [], f"still deriving game_num by hand: {offenders}"
