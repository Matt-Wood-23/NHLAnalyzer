"""Every module must take its season identity from config.season.

These are regression tests for the failure that motivated the config module:
ten copies of the season list in three formats, one of which silently fell a
season behind.  A rollover should require editing exactly one table.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("config", "features", "ingestion", "models", "pipeline", "bot")

# Season-shaped literals: "2025-2026", "20252026", or a bare recent year.
_SEASON_LITERAL = re.compile(r"^(20[2-4]\d-20[2-4]\d|20[2-4]\d20[2-4]\d|20[2-4]\d)$")


def _source_files():
    for package in PACKAGES:
        yield from sorted((ROOT / package).rglob("*.py"))


def _string_literals(path):
    """Every string constant in a file, excluding docstrings and comments."""
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value not in docstrings:
                yield node.value


class TestNoHardcodedSeasons:
    def test_no_module_outside_the_config_hardcodes_a_season(self):
        offenders = []
        for path in _source_files():
            if path.relative_to(ROOT).parts[0] == "config":
                continue
            for literal in _string_literals(path):
                if _SEASON_LITERAL.match(literal.strip()):
                    offenders.append(f"{path.relative_to(ROOT)}: {literal!r}")

        assert not offenders, (
            "Season literals belong in config/season.py:\n  "
            + "\n  ".join(offenders)
        )

    def test_config_is_the_only_place_holding_opener_dates(self):
        from config.season import SEASON_STARTS

        offenders = []
        for path in _source_files():
            if path.relative_to(ROOT).parts[0] == "config":
                continue
            for literal in _string_literals(path):
                if literal in SEASON_STARTS.values():
                    offenders.append(f"{path.relative_to(ROOT)}: {literal!r}")

        assert not offenders, "Duplicate season opener dates:\n  " + "\n  ".join(offenders)


class TestModulesResolveTheCurrentSeason:
    """Each consumer should agree with config.season at import time."""

    def test_moneypuck_season_map(self):
        from config.season import moneypuck_seasons
        from ingestion.moneypuck import MP_SEASONS

        assert MP_SEASONS == moneypuck_seasons()

    def test_model_season_lists(self):
        from config.season import all_seasons
        from models.baseline import SEASONS as baseline_seasons
        from models.sog_model import SEASONS as sog_seasons

        assert baseline_seasons == all_seasons()
        assert sog_seasons == all_seasons()

    def test_xgboost_tuning_split_advances_with_the_season_list(self):
        pytest.importorskip("xgboost")
        pytest.importorskip("lightgbm")

        from models.baseline import SEASONS
        from models.xgboost_model import TUNE_TRAIN_SEASONS, TUNE_VAL_SEASON

        assert TUNE_VAL_SEASON == SEASONS[-2]
        assert TUNE_TRAIN_SEASONS == SEASONS[:-2]
        # The newest season stays an untouched holdout for walk-forward testing.
        assert SEASONS[-1] not in TUNE_TRAIN_SEASONS
        assert SEASONS[-1] != TUNE_VAL_SEASON

    def test_live_and_props_agree_on_the_current_season(self):
        from config.season import current_season, current_season_api
        from pipeline.live import CURRENT_SEASON
        from pipeline.props_live import CURRENT_SEASON_API

        assert CURRENT_SEASON == current_season()
        assert CURRENT_SEASON_API == current_season_api()

    def test_shot_season_map_inverts_the_moneypuck_map(self):
        from config.season import moneypuck_seasons
        from ingestion.player_stats import SHOT_SEASONS

        assert SHOT_SEASONS == {y: label for label, y in moneypuck_seasons().items()}

    def test_daily_cache_targets_the_current_season(self):
        from config.season import current_season_year
        from pipeline.daily import current_season_cache_file

        assert str(current_season_year()) in current_season_cache_file().name


class TestApiDefaults:
    @pytest.mark.parametrize("func_name", [
        "fetch_player_game_log", "fetch_team_roster", "fetch_roster_with_logs",
    ])
    def test_player_endpoints_default_to_the_current_season(self, func_name):
        """These used to default to a pinned season string, so the first run of
        a new season silently queried the previous one."""
        import inspect

        from ingestion import player_stats

        sig = inspect.signature(getattr(player_stats, func_name))
        assert sig.parameters["season"].default is None
