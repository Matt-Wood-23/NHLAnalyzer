"""
Rolling goalie features for the saves prop model.

Distinct from :mod:`features.goalie_mp`, which rolls two columns
(``save_pct``, ``gsax``) for the *game outcome* matrix.  A saves prop needs
workload as well as quality: saves are shots faced times save rate, and
shots faced is the part that actually moves, driven by how much the
opponent shoots and how much the goalie's own team gives up.

Train/serve parity follows the pattern in :mod:`features.team` and
:mod:`features.player`: :func:`pregame_goalie_snapshot` appends a
placeholder row and runs the same :func:`_rolling_goalie` the training
matrix is built with, then reads the placeholder back.  Both bugs this
project has found in serving came from a parallel implementation drifting
away from the training one, so there is deliberately only the one.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config.season import game_number

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"

# Shorter than the team windows: a goalie starts ~55 games a season, so l20
# already reaches back a third of a year.
GOALIE_PROP_WINDOWS = [5, 10, 20]

# Workload first, then quality. shots_against is the dominant driver of the
# saves count; save_pct barely moves (league ~0.905) but sets the scale.
GOALIE_ROLL_STATS = [
    "saves", "shots_against", "save_pct", "gsax", "xg_against",
]

# Context for the goalie's own team: how much they concede.
OWN_TEAM_COLS = ["sf_pct_l20", "xg_against_l20", "hd_chances_against_l20"]

# Context for the team shooting at him: how much they generate.
OPP_TEAM_COLS = ["xg_for_l20", "hd_chances_for_l20", "sf_pct_l20"]

# Starts required before a goalie's rolling numbers mean anything.
MIN_STARTS = 5


# How many of a team's most recent starts the probable-starter rule looks at.
# Scored against the boxscore ``starter`` flag over 13,594 starts
# (``data/parquet/goalie_starts.parquet``):
#
#   whoever started last      45.6%
#   most common of last 3     55.7%
#   most common of last 5     58.2%
#   most common of last 10    59.7%   <- chosen
#   most common of last 20    58.9%
#   season-to-date leader     58.5%
#
# "Whoever started last" is worse than a coin flip because tandems alternate:
# the busiest goalie takes only 60.2% of his team's starts, so the previous
# game is nearly as much evidence against him going again as for it.  Past ten
# starts the window drifts, picking up goalies since traded or sent down.
STARTER_WINDOW = 10


def likely_starters(
    starts: pd.DataFrame,
    teams: Optional[list[str]] = None,
    window: int = STARTER_WINDOW,
) -> dict[str, dict]:
    """Guess each team's starter from its recent starts.

    Shared by both serving paths so they cannot drift apart: the outcome model
    needs a goalie to read rolling save stats off, and the saves prop needs one
    to project.  Neither can know the real answer -- the NHL publishes no
    probable starter before puck drop, and the boxscore ``starter`` flag only
    appears once the game is live.

    Args:
        starts: starter rows with ``team``, ``season``, ``game_num`` and a
                goalie identifier.  Accepts ``goalie_id``/``goalie_name`` or
                ``player_id``/``player_name``.
        teams:  restrict to these teams; default is every team present.
        window: how many recent starts to count over.

    Returns:
        ``{team: {goalie_id, goalie_name, team, starter_basis,
        starter_confidence}}``.  Teams with no history are omitted rather than
        guessed at.
    """
    id_col = "goalie_id" if "goalie_id" in starts.columns else "player_id"
    name_col = "goalie_name" if "goalie_name" in starts.columns else "player_name"

    wanted = teams if teams is not None else sorted(starts["team"].dropna().unique())
    out: dict[str, dict] = {}
    for team in wanted:
        played = starts[starts["team"] == team]
        if played.empty:
            logger.warning("No goalie history for %s -- no projection", team)
            continue

        recent = played.sort_values(["season", "game_num"]).tail(window)
        counts = recent[id_col].value_counts()
        goalie_id = int(counts.idxmax())
        row = recent[recent[id_col] == goalie_id].iloc[-1]

        out[team] = {
            "goalie_id":   goalie_id,
            "goalie_name": row[name_col],
            "team":        team,
            "starter_basis": f"{int(counts.max())} of last {len(recent)} starts",
            "starter_confidence": float(counts.max() / len(recent)),
        }
    return out


def rolling_feature_columns() -> list[str]:
    """The rolling columns :func:`_rolling_goalie` produces."""
    return [f"{c}_l{w}" for c in GOALIE_ROLL_STATS for w in GOALIE_PROP_WINDOWS]


def team_context_columns() -> list[str]:
    """The joined team columns, namespaced by side."""
    return (["own_" + c for c in OWN_TEAM_COLS]
            + ["opp_" + c for c in OPP_TEAM_COLS])


def _rolling_goalie(grp: pd.DataFrame) -> pd.DataFrame:
    """Rolling means within one goalie's history, shifted to stay pre-game.

    Deliberately spans seasons, matching :mod:`features.goalie_mp`: a
    goalie's form carries over a summer far better than a team's does, and
    resetting at the boundary would blank every starter for a month.
    """
    grp = grp.copy()
    for col in GOALIE_ROLL_STATS:
        if col not in grp.columns:
            continue
        shifted = grp[col].shift(1)  # exclude the current game
        for w in GOALIE_PROP_WINDOWS:
            grp[f"{col}_l{w}"] = shifted.rolling(w, min_periods=1).mean()
    grp["starts"] = np.arange(len(grp))
    return grp


def load_goalie_game_stats(path: Optional[Path] = None) -> pd.DataFrame:
    """Starter rows from the goalie game stats parquet, chronologically sorted."""
    if path is None:
        path = PARQUET_DIR / "goalie_game_stats.parquet"
    if not Path(path).exists():
        raise FileNotFoundError(
            f"No {path}. Run `python -m pipeline.backfill` to build it."
        )
    df = pd.read_parquet(path)
    df = df[df["is_starter"]].copy()
    if "game_num" not in df.columns:
        df["game_num"] = game_number(df["game_id"])
    return df.sort_values(["goalie_id", "season", "game_num"])


def build_goalie_rolling_features(
    goalie_games: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Pre-game rolling features for every starter appearance on record."""
    if goalie_games is None:
        goalie_games = load_goalie_game_stats()

    df = goalie_games.sort_values(["goalie_id", "season", "game_num"])
    parts = [_rolling_goalie(grp) for _, grp in df.groupby("goalie_id", sort=False)]
    result = pd.concat(parts, ignore_index=True)

    logger.info(
        "Goalie rolling features: %d starter rows, %d goalies, %d feature cols",
        len(result), result["goalie_id"].nunique(), len(rolling_feature_columns()),
    )
    return result


def attach_team_context(
    df: pd.DataFrame,
    team_features: pd.DataFrame,
) -> pd.DataFrame:
    """Join the goalie's own team and his opponent's rolling team stats.

    Shots faced is mostly a team property, so this is where most of the
    signal beyond the goalie's own history comes from.
    """
    lookup = team_features[["game_id", "team", "is_home"]].drop_duplicates()
    home = lookup[lookup["is_home"]].rename(columns={"team": "home_team"})
    away = lookup[~lookup["is_home"]].rename(columns={"team": "away_team"})
    game_teams = home[["game_id", "home_team"]].merge(
        away[["game_id", "away_team"]], on="game_id",
    )

    df = df.merge(game_teams, on="game_id", how="left")
    df["opp_team"] = np.where(
        df["team"] == df["home_team"], df["away_team"], df["home_team"],
    )
    df["is_home"] = (df["team"] == df["home_team"]).astype(float)

    own = team_features[["game_id", "team"] + OWN_TEAM_COLS].rename(
        columns={c: "own_" + c for c in OWN_TEAM_COLS},
    )
    df = df.merge(own, on=["game_id", "team"], how="left")

    opp = team_features[["game_id", "team"] + OPP_TEAM_COLS].rename(
        columns={**{c: "opp_" + c for c in OPP_TEAM_COLS}, "team": "opp_team"},
    )
    return df.merge(opp, on=["game_id", "opp_team"], how="left")


def pregame_goalie_snapshot(
    history: pd.DataFrame,
    goalie_ids: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Pre-game rolling features for each goalie's *next* start.

    The serving counterpart to :func:`build_goalie_rolling_features`, built
    by appending a placeholder row and running the identical rolling
    function -- so a change to one cannot silently skip the other.

    Args:
        history: starter rows from goalie_game_stats.parquet.
        goalie_ids: goalies who must appear.  Those with no history get an
            all-NaN row rather than being dropped, so the caller's coverage
            guard decides whether to publish rather than the join.

    Returns:
        DataFrame indexed by goalie_id, one column per rolling feature plus
        ``starts`` and the goalie's most recent identity.
    """
    history = history.copy()
    if "game_num" not in history.columns:
        history["game_num"] = game_number(history["game_id"])

    wanted = sorted(set(goalie_ids or []) | set(history["goalie_id"].dropna().unique()))
    feature_cols = rolling_feature_columns()
    next_game_num = int(history["game_num"].max()) + 1 if len(history) else 1

    rows = []
    for goalie_id in wanted:
        grp = history[history["goalie_id"] == goalie_id].sort_values(
            ["season", "game_num"],
        )

        placeholder = {c: np.nan for c in GOALIE_ROLL_STATS}
        placeholder.update({"goalie_id": goalie_id, "game_num": next_game_num})
        with_placeholder = pd.concat(
            [grp, pd.DataFrame([placeholder])], ignore_index=True,
        )

        rolled = _rolling_goalie(with_placeholder)
        row = rolled.iloc[-1][feature_cols].to_dict()
        row["goalie_id"] = goalie_id
        row["starts"] = len(grp)
        for col in ("goalie_name", "team"):
            row[col] = grp[col].iloc[-1] if len(grp) and col in grp.columns else None
        rows.append(row)

    snapshot = pd.DataFrame(rows).set_index("goalie_id")
    logger.info(
        "Goalie snapshot built: %d goalies, %d feature columns",
        len(snapshot), len(feature_cols),
    )
    return snapshot
