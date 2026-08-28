"""
ELO rating system for NHL teams.

Computes recursive team strength ratings that update after each game.
Pre-game ELO values are used as features (no leakage) — diff_elo is the
single most important feature in the model.

Two properties matter for hockey specifically:

* **Overtime and shootout wins are discounted.**  Roughly a quarter of NHL
  games are decided after regulation, and those outcomes are close to a coin
  flip — a 3-on-3 or shootout win says much less about team strength than a
  regulation win.  Crediting them as full wins injects that noise straight
  into the most important feature.  ``ot_win_value`` (default 0.6) is the
  credit an OT/SO winner receives instead of 1.0.

* **Ratings regress toward the mean between seasons** to account for roster
  turnover and reset expectations.

Tuned parameters are persisted to ``models/saved/elo_params.json`` by
``python -m features.elo`` and picked up automatically by the backfill and
live pipelines, so a tuning run actually changes what the pipeline uses.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"
SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"
ELO_PARAMS_PATH = SAVED_DIR / "elo_params.json"

# Default ELO parameters (tunable via grid search)
INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 50.0
SEASON_REGRESS = 0.33   # fraction regressed toward mean between seasons
OT_WIN_VALUE = 0.6      # credit for an OT/SO win (1.0 would treat it as a regulation win)

_TUNABLE = ("k_factor", "home_advantage", "season_regress", "ot_win_value")


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Standard ELO expected score for player A vs player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


# ---------------------------------------------------------------------------
# Persisted parameters
# ---------------------------------------------------------------------------

def default_elo_params() -> dict:
    """Module defaults, as a kwargs dict for :func:`compute_elo_ratings`."""
    return {
        "k_factor": K_FACTOR,
        "home_advantage": HOME_ADVANTAGE,
        "season_regress": SEASON_REGRESS,
        "ot_win_value": OT_WIN_VALUE,
    }


def save_elo_params(params: dict) -> Path:
    """Persist tuned parameters so the pipelines pick them up."""
    keep = {k: float(v) for k, v in params.items() if k in _TUNABLE}
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    ELO_PARAMS_PATH.write_text(json.dumps(keep, indent=2))
    logger.info("Saved ELO parameters → %s", ELO_PARAMS_PATH)
    return ELO_PARAMS_PATH


def load_elo_params() -> dict:
    """Tuned parameters if a tuning run has saved any, else module defaults."""
    params = default_elo_params()
    if not ELO_PARAMS_PATH.exists():
        logger.info("No tuned ELO parameters — using defaults %s", params)
        return params

    try:
        stored = json.loads(ELO_PARAMS_PATH.read_text())
    except (OSError, ValueError) as e:
        logger.warning("Could not read %s (%s) — using defaults", ELO_PARAMS_PATH, e)
        return params

    params.update({k: float(v) for k, v in stored.items() if k in _TUNABLE})
    logger.info("Loaded tuned ELO parameters: %s", params)
    return params


# ---------------------------------------------------------------------------
# Rating computation
# ---------------------------------------------------------------------------

def _actual_home_score(
    home_win: float,
    went_to_ot: float | None,
    ot_win_value: float,
) -> float:
    """Score credited to the home team for ELO purposes.

    A regulation result is a full 1/0.  An overtime or shootout result is
    pulled toward 0.5 by ``ot_win_value``, so a coin-flip finish moves the
    ratings less than a decisive one.
    """
    if went_to_ot is None or pd.isna(went_to_ot) or float(went_to_ot) == 0.0:
        return float(home_win)
    return ot_win_value if float(home_win) == 1.0 else 1.0 - ot_win_value


def _run_elo(
    game_results: pd.DataFrame,
    initial_elo: float,
    k_factor: float,
    home_advantage: float,
    season_regress: float,
    ot_win_value: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Single pass over the schedule.

    Returns ``(per_game_pregame_ratings, final_ratings)``.  Both the feature
    builder and the live snapshot need one or the other, and running the loop
    once for both keeps a backfill from replaying the whole schedule twice.
    """
    df = game_results.sort_values(["season", "game_id"]).reset_index(drop=True)

    all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    elos: dict[str, float] = {team: initial_elo for team in all_teams}

    has_ot = "went_to_ot" in df.columns

    rows = []
    prev_season = None

    for game in df.itertuples(index=False):
        season = game.season

        # Season boundary: regress all ELOs toward the mean
        if prev_season is not None and season != prev_season:
            for team in elos:
                elos[team] = elos[team] * (1 - season_regress) + initial_elo * season_regress

        home = game.home_team
        away = game.away_team
        home_win = game.home_win

        # Ensure teams exist (expansion teams, etc.)
        elos.setdefault(home, initial_elo)
        elos.setdefault(away, initial_elo)

        # Record pre-game ELOs
        home_elo = elos[home]
        away_elo = elos[away]

        rows.append({
            "game_id": game.game_id,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "diff_elo": home_elo - away_elo,
        })

        # Update ELOs (only if we know the result)
        if pd.notna(home_win):
            expected_home = _expected_score(home_elo + home_advantage, away_elo)
            actual_home = _actual_home_score(
                home_win,
                getattr(game, "went_to_ot", None) if has_ot else None,
                ot_win_value,
            )

            elos[home] = home_elo + k_factor * (actual_home - expected_home)
            elos[away] = away_elo + k_factor * ((1 - actual_home) - (1 - expected_home))

        prev_season = season

    return pd.DataFrame(rows), elos


def compute_elo_ratings(
    game_results: pd.DataFrame,
    initial_elo: float = INITIAL_ELO,
    k_factor: float = K_FACTOR,
    home_advantage: float = HOME_ADVANTAGE,
    season_regress: float = SEASON_REGRESS,
    ot_win_value: float = OT_WIN_VALUE,
    return_final: bool = False,
):
    """
    Compute ELO ratings for all teams across all seasons.

    Args:
        game_results: DataFrame with columns:
            game_id, season, home_team, away_team, home_win (1.0/0.0)
            Optionally went_to_ot (1.0/0.0) to discount OT/SO results.
            Must be sorted chronologically.
        initial_elo: Starting ELO for all teams.
        k_factor: Base K-factor for updates.
        home_advantage: ELO points added for home team expectation.
        season_regress: Fraction to regress toward mean between seasons.
        ot_win_value: Credit an OT/SO winner receives (0.5 = no credit for
            winning after regulation, 1.0 = same as a regulation win).
        return_final: also return the final per-team ratings dict.

    Returns:
        DataFrame with one row per game:
            game_id, home_elo, away_elo, diff_elo
        All ELO values are PRE-GAME (entering the game) — valid features.
        If return_final, a ``(DataFrame, dict)`` tuple.
    """
    result, final = _run_elo(
        game_results, initial_elo, k_factor, home_advantage,
        season_regress, ot_win_value,
    )

    logger.info(
        "ELO ratings computed: %d games, %d teams, "
        "home_elo range [%.0f, %.0f], diff_elo range [%.0f, %.0f]",
        len(result), len(final),
        result["home_elo"].min(), result["home_elo"].max(),
        result["diff_elo"].min(), result["diff_elo"].max(),
    )

    return (result, final) if return_final else result


def save_current_elos(
    game_results: pd.DataFrame,
    **elo_kwargs,
) -> Path:
    """
    Compute ELOs and save the final (latest) per-team ratings to parquet.
    Used by the live pipeline to look up current team ELOs.
    """
    params = default_elo_params() | {
        k: v for k, v in elo_kwargs.items() if k in _TUNABLE
    }
    _, final = _run_elo(
        game_results,
        elo_kwargs.get("initial_elo", INITIAL_ELO),
        params["k_factor"],
        params["home_advantage"],
        params["season_regress"],
        params["ot_win_value"],
    )

    return save_elo_snapshot(final)


def save_elo_snapshot(final_ratings: dict[str, float]) -> Path:
    """Write the latest per-team ratings for the live pipeline to read."""
    elo_df = pd.DataFrame(
        [{"team": team, "elo": elo} for team, elo in final_ratings.items()]
    ).sort_values("elo", ascending=False)

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    path = PARQUET_DIR / "elo_ratings.parquet"
    elo_df.to_parquet(path, index=False)
    logger.info("Saved current ELO ratings → %s", path)

    return path


# ---------------------------------------------------------------------------
# Parameter tuning
# ---------------------------------------------------------------------------

def tune_elo_parameters(
    game_results: pd.DataFrame,
    k_values: list[float] | None = None,
    ha_values: list[float] | None = None,
    regress_values: list[float] | None = None,
    ot_values: list[float] | None = None,
) -> dict:
    """
    Grid search over ELO parameters to minimize Brier score.

    Uses the ELO-implied probability as a standalone predictor:
        prob_home = expected_score(home_elo + home_advantage, away_elo)

    Returns dict with best parameters and Brier score.
    """
    if k_values is None:
        k_values = [15.0, 20.0, 25.0, 30.0]
    if ha_values is None:
        ha_values = [30.0, 50.0, 70.0]
    if regress_values is None:
        regress_values = [0.25, 0.33, 0.50]
    if ot_values is None:
        # 1.0 reproduces the old behaviour, so tuning can only improve on it.
        ot_values = [0.5, 0.6, 0.75, 1.0]

    df = game_results.sort_values(["season", "game_id"]).reset_index(drop=True)
    truth = df[["game_id", "home_win"]].dropna(subset=["home_win"])

    best_brier = float("inf")
    best_params: dict = {}

    for k in k_values:
        for ha in ha_values:
            for reg in regress_values:
                for otv in ot_values:
                    elo_df = compute_elo_ratings(
                        df, k_factor=k, home_advantage=ha,
                        season_regress=reg, ot_win_value=otv,
                    )
                    merged = truth.merge(elo_df, on="game_id")

                    prob_home = 1.0 / (
                        1.0 + 10.0 ** (
                            (merged["away_elo"] - (merged["home_elo"] + ha)) / 400.0
                        )
                    )
                    brier = float(((prob_home - merged["home_win"]) ** 2).mean())

                    if brier < best_brier:
                        best_brier = brier
                        best_params = {
                            "k_factor": k,
                            "home_advantage": ha,
                            "season_regress": reg,
                            "ot_win_value": otv,
                            "brier": brier,
                        }

    logger.info(
        "ELO tuning: best Brier=%.4f | K=%.0f, HA=%.0f, regress=%.2f, ot=%.2f",
        best_params["brier"], best_params["k_factor"],
        best_params["home_advantage"], best_params["season_regress"],
        best_params["ot_win_value"],
    )
    return best_params


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from features.team import load_team_features

    team_feats = load_team_features()
    cols = ["game_id", "season", "home_team", "away_team", "won"]
    if "went_to_ot" in team_feats.columns:
        cols.append("went_to_ot")
    home = team_feats[team_feats["is_home"]][cols].rename(columns={"won": "home_win"})

    # Tune parameters
    print("\n=== ELO Parameter Tuning ===")
    best = tune_elo_parameters(home)
    print(f"Best params: {best}")

    # Persist so backfill / live use them instead of the module defaults
    save_elo_params(best)

    # Compute with best params
    print("\n=== Computing ELO ratings ===")
    tuned = {k: v for k, v in best.items() if k in _TUNABLE}
    elo_df = compute_elo_ratings(home, **tuned)
    print(f"Shape: {elo_df.shape}")
    print(f"\nSample:\n{elo_df.head(10)}")
    print(f"\ndiff_elo stats:\n{elo_df['diff_elo'].describe().round(1)}")

    # Save current ratings
    save_current_elos(home, **tuned)
