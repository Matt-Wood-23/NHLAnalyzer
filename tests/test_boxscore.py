"""Boxscore ingestion: the dressed lineup, the cache guard, and the merge.

The point of this module is that a skater who took no shots is a real 0, not
an absent row.  Most of what is tested here is that distinction surviving the
parse and the join.
"""

import json

import numpy as np
import pandas as pd
import pytest

from ingestion import boxscore as bx


def _skater(player_id, name, sog, position="C", toi="15:00", **extra):
    row = {
        "playerId": player_id,
        "sweaterNumber": player_id % 100,
        "name": {"default": name},
        "position": position,
        "goals": 0, "assists": 0, "points": 0, "plusMinus": 0, "pim": 0,
        "hits": 1, "powerPlayGoals": 0, "sog": sog, "toi": toi,
        "blockedShots": 0, "shifts": 20, "giveaways": 0, "takeaways": 0,
    }
    row.update(extra)
    return row


def _goalie(player_id, name, starter, saves=25, toi="60:00"):
    return {
        "playerId": player_id,
        "name": {"default": name},
        "position": "G",
        "starter": starter,
        "saves": saves,
        "shotsAgainst": saves + 2,
        "goalsAgainst": 2,
        "pim": 0,
        "toi": toi,
    }


def _boxscore(game_id="2025020001", state="OFF"):
    """A complete game: 36 dressed skaters, four goalies, two starters."""
    def side(base, sogs):
        return {
            "forwards": [
                _skater(base + i, f"F. Player{base + i}", sogs[i]) for i in range(12)
            ],
            "defense": [
                _skater(base + 100 + i, f"D. Player{base + 100 + i}", 0, "D")
                for i in range(6)
            ],
            "goalies": [
                _goalie(base + 200, f"G. Starter{base}", True),
                _goalie(base + 201, f"G. Backup{base}", False, saves=0, toi="00:00"),
            ],
        }

    return {
        "id": int(game_id),
        "season": 20252026,
        "gameState": state,
        "homeTeam": {"abbrev": "BOS"},
        "awayTeam": {"abbrev": "TOR"},
        "playerByGameStats": {
            "homeTeam": side(8000000, [3, 2, 1, 0, 0, 4, 2, 1, 1, 0, 2, 1]),
            "awayTeam": side(8100000, [1, 1, 0, 2, 3, 0, 0, 1, 2, 2, 1, 0]),
        },
    }


class TestParsing:
    def test_every_dressed_skater_appears(self):
        rows = bx.parse_skaters(_boxscore())
        assert len(rows) == 36

    def test_zero_shot_skaters_are_rows_not_absences(self):
        """The bug this module exists to fix: MoneyPuck drops these entirely,
        so every rolling average was computed over a survivor set."""
        rows = bx.parse_skaters(_boxscore())
        zeros = [r for r in rows if r["sog"] == 0]
        assert len(zeros) == 19  # 12 defensemen + 7 held-off forwards
        assert all(r["sog"] == 0 for r in zeros)

    def test_home_and_away_are_labelled(self):
        rows = bx.parse_skaters(_boxscore())
        assert sum(r["is_home"] for r in rows) == 18
        assert {r["team"] for r in rows if r["is_home"]} == {"BOS"}
        assert {r["team"] for r in rows if not r["is_home"]} == {"TOR"}

    def test_the_new_stat_columns_come_through(self):
        """points/assists and hits/blocks are the props this unlocks."""
        row = bx.parse_skaters(_boxscore())[0]
        for col in ("points", "assists", "hits", "blocked_shots",
                    "shifts", "toi_seconds"):
            assert col in row

    def test_goalies_carry_the_starter_flag(self):
        rows = bx.parse_goalies(_boxscore())
        assert len(rows) == 4
        assert sum(r["starter"] for r in rows) == 2

    def test_a_missing_side_does_not_crash(self):
        box = _boxscore()
        del box["playerByGameStats"]["awayTeam"]
        assert len(bx.parse_skaters(box)) == 18

    def test_an_empty_payload_yields_nothing(self):
        assert bx.parse_skaters({}) == []
        assert bx.parse_goalies({}) == []


class TestFieldConversion:
    @pytest.mark.parametrize("toi,expected", [
        ("19:15", 1155.0), ("00:00", 0.0), ("60:00", 3600.0), ("5:07", 307.0),
    ])
    def test_toi_parses(self, toi, expected):
        assert bx.toi_to_seconds(toi) == expected

    @pytest.mark.parametrize("bad", [None, "", "--", "notatime", 15])
    def test_malformed_toi_is_zero_not_an_exception(self, bad):
        assert bx.toi_to_seconds(bad) == 0.0

    def test_name_handles_both_shapes(self):
        assert bx._player_name({"name": {"default": "S. Bennett"}}) == "S. Bennett"
        assert bx._player_name({"name": "Plain"}) == "Plain"
        assert bx._player_name({}) == ""

    def test_season_label_matches_the_parquet_convention(self):
        assert bx._season_label(20252026) == "2025-2026"
        assert bx._season_label("20212022") == "2021-2022"
        assert bx._season_label(None) == ""


class TestCaching:
    @pytest.fixture(autouse=True)
    def sandbox(self, tmp_path, monkeypatch):
        monkeypatch.setattr(bx, "CACHE_DIR", tmp_path / "boxscores")

    def test_a_final_game_is_written_once_and_read_back(self, monkeypatch):
        calls = []

        class FakeResp:
            def raise_for_status(self): pass
            def json(self): return _boxscore(state="OFF")

        def fake_get(url, timeout=None):
            calls.append(url)
            return FakeResp()

        monkeypatch.setattr(bx.httpx, "get", fake_get)
        first = bx.fetch_boxscore("2025020001")
        second = bx.fetch_boxscore("2025020001")
        assert first["id"] == second["id"]
        assert len(calls) == 1, "second call should have come from disk"

    def test_a_live_game_is_never_cached(self, monkeypatch):
        """Caching a game in progress would freeze a partial line forever."""
        class FakeResp:
            def raise_for_status(self): pass
            def json(self): return _boxscore(state="LIVE")

        monkeypatch.setattr(bx.httpx, "get", lambda url, timeout=None: FakeResp())
        bx.fetch_boxscore("2025020001")
        assert bx.cached_game_ids() == []

    def test_a_failed_request_returns_none_and_caches_nothing(self, monkeypatch):
        def boom(url, timeout=None):
            raise RuntimeError("network down")

        monkeypatch.setattr(bx.httpx, "get", boom)
        assert bx.fetch_boxscore("2025020001") is None
        assert bx.cached_game_ids() == []

    def test_backfill_skips_what_is_already_on_disk(self, monkeypatch):
        bx.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (bx.CACHE_DIR / "2025020001.json").write_text(
            json.dumps(_boxscore()), encoding="utf-8",
        )
        monkeypatch.setattr(
            bx, "fetch_boxscore",
            lambda g, **kw: pytest.fail(f"should not have fetched {g}"),
        )
        tally = bx.backfill_boxscores(["2025020001"])
        assert tally == {"cached": 1, "fetched": 0, "failed": 0}


class TestMerge:
    @pytest.fixture
    def spine(self):
        rows = bx.parse_skaters(_boxscore())
        df = pd.DataFrame(rows)
        df["season"] = "2025-2026"
        df["game_num"] = 1
        return df.drop(columns=["season_api"])

    @pytest.fixture
    def shot_data(self, spine):
        """MoneyPuck: only the skaters who generated a shot event."""
        shooters = spine[spine["sog"] > 0].copy()
        return pd.DataFrame({
            "game_id": shooters["game_id"].values,
            "player_id": shooters["player_id"].astype(float).values,
            "player_name": [f"Fullname {int(p)}" for p in shooters["player_id"]],
            "sog": shooters["sog"].values,
            "xg": np.linspace(0.1, 0.9, len(shooters)),
            "shot_attempts": shooters["sog"].values + 1,
        })

    def test_the_lineup_survives_the_join(self, spine, shot_data):
        merged = bx.merge_shot_data(spine, shot_data)
        assert len(merged) == len(spine) == 36

    def test_no_attempt_skaters_get_a_real_zero(self, spine, shot_data):
        merged = bx.merge_shot_data(spine, shot_data)
        held = merged[merged["sog"] == 0]
        assert len(held) == 19
        assert (held["shot_attempts"] == 0).all()
        assert (held["xg"] == 0.0).all()

    def test_xg_per_attempt_stays_undefined_with_no_attempts(self, spine, shot_data):
        """It is a rate.  Filling 0 would claim he generated attempts of zero
        quality; NaN means he generated none, which the rolling mean skips."""
        merged = bx.merge_shot_data(spine, shot_data)
        held = merged[merged["shot_attempts"] == 0]
        assert held["xg_per_attempt"].isna().all()
        shot = merged[merged["shot_attempts"] > 0]
        assert shot["xg_per_attempt"].notna().all()

    def test_sog_comes_from_the_official_line(self, spine, shot_data):
        """When the sources disagree the boxscore wins — it is the NHL's own."""
        shot_data = shot_data.copy()
        shot_data["sog"] = shot_data["sog"] + 5
        merged = bx.merge_shot_data(spine, shot_data)
        by_id = merged.set_index("player_id")["sog"]
        for _, row in spine.iterrows():
            assert by_id[row["player_id"]] == row["sog"]

    def test_full_names_replace_the_abbreviations(self, spine, shot_data):
        merged = bx.merge_shot_data(spine, shot_data)
        shooters = merged[merged["sog"] > 0]
        assert shooters["player_name"].str.startswith("Fullname").all()

    def test_a_never_shooting_skater_keeps_his_boxscore_name(self, spine, shot_data):
        merged = bx.merge_shot_data(spine, shot_data)
        held = merged[merged["sog"] == 0]
        assert held["player_name"].str.match(r"^[FD]\. Player").all()

    def test_a_null_player_id_does_not_break_the_join(self, spine, shot_data):
        """Three real rows in player_game_stats.parquet have a null player_id.
        The join casts ids to int64, which raises on NaN, so the whole merge
        fails rather than skipping the row."""
        junk = shot_data.iloc[[0]].copy()
        junk["player_id"] = np.nan
        merged = bx.merge_shot_data(spine, pd.concat([shot_data, junk]))
        assert len(merged) == 36

    def test_unattributed_shot_rows_are_dropped(self, spine, shot_data):
        """MoneyPuck carries a handful of player_id 0 rows for shot events it
        could not attribute.  The left join from the dressed lineup already
        makes them unreachable — this pins that they never reappear if the
        join direction is ever changed."""
        junk = shot_data.iloc[[0]].copy()
        junk["player_id"] = 0.0
        merged = bx.merge_shot_data(spine, pd.concat([shot_data, junk]))
        assert len(merged) == 36
        assert 0 not in merged["player_id"].values

    def test_duplicate_shot_rows_do_not_duplicate_the_skater(self, spine, shot_data):
        doubled = pd.concat([shot_data, shot_data], ignore_index=True)
        assert len(bx.merge_shot_data(spine, doubled)) == 36


class TestSurvivorshipIsActuallyFixed:
    """The regression guard for the whole exercise."""

    def test_rows_per_game_reaches_the_dressed_lineup(self):
        rows = bx.parse_skaters(_boxscore())
        df = pd.DataFrame(rows)
        assert len(df) / df["game_id"].nunique() == 36.0

    def test_the_minimum_shot_count_is_zero_not_one(self):
        spine = pd.DataFrame(bx.parse_skaters(_boxscore()))
        spine["season"] = "2025-2026"
        spine["game_num"] = 1
        spine = spine.drop(columns=["season_api"])
        shooters = spine[spine["sog"] > 0]
        shot_data = pd.DataFrame({
            "game_id": shooters["game_id"].values,
            "player_id": shooters["player_id"].astype(float).values,
            "player_name": None,
            "sog": shooters["sog"].values,
            "xg": 0.3,
            "shot_attempts": shooters["sog"].values + 1,
        })
        merged = bx.merge_shot_data(spine, shot_data)
        assert merged["shot_attempts"].min() == 0
        assert merged["sog"].min() == 0


class TestScratchFilter:
    """props_live projected anyone under contract with 5+ games, dressed or not."""

    def test_scratched_ids_are_collected_from_both_sides(self, monkeypatch):
        from pipeline import props_live

        payload = {
            "gameInfo": {
                "homeTeam": {"scratches": [{"id": 111}, {"id": 222}]},
                "awayTeam": {"scratches": [{"id": 333}]},
            }
        }

        class FakeResp:
            def raise_for_status(self): pass
            def json(self): return payload

        monkeypatch.setattr(
            bx.httpx.Client, "get", lambda self, url, timeout=None: FakeResp(),
        )
        out = props_live.scratched_players([{"game_id": "2025020001"}])
        assert out == {111, 222, 333}

    def test_an_unfiled_lineup_reports_nothing_scratched(self, monkeypatch):
        """Empty must mean "not published yet", never "nobody is out" — the
        pipeline logs the difference instead of silently projecting everyone."""
        from pipeline import props_live

        class FakeResp:
            def raise_for_status(self): pass
            def json(self): return {"gameInfo": {"homeTeam": {"scratches": []},
                                                 "awayTeam": {"scratches": []}}}

        monkeypatch.setattr(
            bx.httpx.Client, "get", lambda self, url, timeout=None: FakeResp(),
        )
        assert props_live.scratched_players([{"game_id": "2025020001"}]) == set()

    def test_a_failed_call_does_not_take_the_run_down(self, monkeypatch):
        from pipeline import props_live

        def boom(self, url, timeout=None):
            raise RuntimeError("right-rail unavailable")

        monkeypatch.setattr(bx.httpx.Client, "get", boom)
        assert props_live.scratched_players([{"game_id": "2025020001"}]) == set()

    def test_entries_without_an_id_are_ignored(self, monkeypatch):
        class FakeResp:
            def raise_for_status(self): pass
            def json(self):
                return {"gameInfo": {"homeTeam": {"scratches": [{"id": 111}, {}]},
                                     "awayTeam": {}}}

        monkeypatch.setattr(
            bx.httpx.Client, "get", lambda self, url, timeout=None: FakeResp(),
        )
        assert bx.fetch_scratches("2025020001", client=bx.httpx.Client()) == {
            "homeTeam": [111], "awayTeam": [],
        }

    def test_props_live_actually_applies_the_filter(self):
        import inspect
        from pipeline import props_live
        src = inspect.getsource(props_live.run)
        assert "scratched_players" in src
        assert "pid not in scratched" in src
