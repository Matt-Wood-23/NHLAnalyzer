"""Shared fixtures.

Every test runs on synthetic data.  Nothing here touches ``data/``, the
network, or Postgres, so the suite runs in CI where none of those exist.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.team import EWM_STATS, ROLL_STATS  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(20262027)


def _team_game_rows(team, season, n_games, rng, start_num=1):
    """Synthetic per-game rows for one team, in the shape team.py expects."""
    stats = sorted(set(ROLL_STATS) | set(EWM_STATS))
    rows = []
    for i in range(n_games):
        num = start_num + i
        row = {
            "game_id": f"{season[:4]}02{num:04d}",
            "team": team,
            "season": season,
            "game_num": num,
            "is_home": bool(i % 2),
        }
        row.update({c: float(rng.random()) for c in stats})
        rows.append(row)
    return rows


@pytest.fixture
def team_history(rng):
    """One team, one season, 25 completed games."""
    return pd.DataFrame(_team_game_rows("BOS", "2025-2026", 25, rng))


@pytest.fixture
def two_season_history(rng):
    """One team across two seasons — used to prove rolling never crosses over."""
    rows = _team_game_rows("BOS", "2025-2026", 20, rng)
    rows += _team_game_rows("BOS", "2026-2027", 20, rng)
    return pd.DataFrame(rows)


@pytest.fixture
def elo_games():
    """Four games with a mix of regulation and overtime finishes."""
    return pd.DataFrame([
        {"game_id": "2025020001", "season": "2025-2026", "home_team": "BOS",
         "away_team": "TOR", "home_win": 1.0, "went_to_ot": 0.0},
        {"game_id": "2025020002", "season": "2025-2026", "home_team": "TOR",
         "away_team": "MTL", "home_win": 0.0, "went_to_ot": 1.0},
        {"game_id": "2025020003", "season": "2025-2026", "home_team": "MTL",
         "away_team": "BOS", "home_win": 1.0, "went_to_ot": 1.0},
        {"game_id": "2025020004", "season": "2025-2026", "home_team": "BOS",
         "away_team": "MTL", "home_win": 0.0, "went_to_ot": 0.0},
    ])
