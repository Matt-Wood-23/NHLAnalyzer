"""
ELO rating system for NHL teams.

Computes recursive team strength ratings that update after each game.
Pre-game ELO values are used as features (no leakage).

Ratings regress toward the mean between seasons to account for
roster turnover and reset expectations.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"

# Default ELO parameters (tunable via grid search)
INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADVANTAGE = 50.0
SEASON_REGRESS = 0.33   # fraction regressed toward mean between seasons


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Standard ELO expected score for player A vs player B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def compute_elo_ratings(
    game_results: pd.DataFrame,
    initial_elo: float = INITIAL_ELO,
    k_factor: float = K_FACTOR,
    home_advantage: float = HOME_ADVANTAGE,
    season_regress: float = SEASON_REGRESS,
) -> pd.DataFrame:
    """
    Compute ELO ratings for all teams across all seasons.

    Args:
        game_results: DataFrame with columns:
            game_id, season, home_team, away_team, home_win (1.0/0.0)
            Must be sorted chronologically.
        initial_elo: Starting ELO for all teams.
        k_factor: Base K-factor for updates.
        home_advantage: ELO points added for home team expectation.
        season_regress: Fraction to regress toward mean between seasons.

    Returns:
        DataFrame with one row per game:
            game_id, home_elo, away_elo, diff_elo
        All ELO values are PRE-GAME (entering the game) — valid features.
    """
    df = game_results.sort_values(["season", "game_id"]).reset_index(drop=True)

    # Initialize all teams
    all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    elos: dict[str, float] = {team: initial_elo for team in all_teams}

    rows = []
    prev_season = None

    for _, game in df.iterrows():
        season = game["season"]

        # Season boundary: regress all ELOs toward the mean
        if prev_season is not None and season != prev_season:
            for team in elos:
                elos[team] = elos[team] * (1 - season_regress) + initial_elo * season_regress

        home = game["home_team"]
        away = game["away_team"]
        home_win = game["home_win"]

        # Ensure teams exist (expansion teams, etc.)
        if home not in elos:
            elos[home] = initial_elo
        if away not in elos:
            elos[away] = initial_elo

        # Record pre-game ELOs
        home_elo = elos[home]
        away_elo = elos[away]

        rows.append({
            "game_id": game["game_id"],
            "home_elo": home_elo,
            "away_elo": away_elo,
            "diff_elo": home_elo - away_elo,
        })

        # Update ELOs (only if we know the result)
        if pd.notna(home_win):
            expected_home = _expected_score(
                home_elo + home_advantage, away_elo
            )
            actual_home = float(home_win)

            elos[home] = home_elo + k_factor * (actual_home - expected_home)
            elos[away] = away_elo + k_factor * ((1 - actual_home) - (1 - expected_home))

        prev_season = season

    result = pd.DataFrame(rows)

    logger.info(
        "ELO ratings computed: %d games, %d teams, "
        "home_elo range [%.0f, %.0f], diff_elo range [%.0f, %.0f]",
        len(result), len(all_teams),
        result["home_elo"].min(), result["home_elo"].max(),
        result["diff_elo"].min(), result["diff_elo"].max(),
    )

    return result


def save_current_elos(
    game_results: pd.DataFrame,
    **elo_kwargs,
) -> Path:
    """
    Compute ELOs and save the final (latest) per-team ratings to parquet.
    Used by the live pipeline to look up current team ELOs.
    """
    df = game_results.sort_values(["season", "game_id"]).reset_index(drop=True)
    all_teams = set(df["home_team"].unique()) | set(df["away_team"].unique())
    elos: dict[str, float] = {
        team: elo_kwargs.get("initial_elo", INITIAL_ELO) for team in all_teams
    }

    k = elo_kwargs.get("k_factor", K_FACTOR)
    ha = elo_kwargs.get("home_advantage", HOME_ADVANTAGE)
    reg = elo_kwargs.get("season_regress", SEASON_REGRESS)
    init = elo_kwargs.get("initial_elo", INITIAL_ELO)

    prev_season = None
    for _, game in df.iterrows():
        season = game["season"]
        if prev_season is not None and season != prev_season:
            for team in elos:
                elos[team] = elos[team] * (1 - reg) + init * reg

        home, away = game["home_team"], game["away_team"]
        if home not in elos:
            elos[home] = init
        if away not in elos:
            elos[away] = init

        if pd.notna(game["home_win"]):
            expected = _expected_score(elos[home] + ha, elos[away])
            actual = float(game["home_win"])
            elos[home] += k * (actual - expected)
            elos[away] += k * ((1 - actual) - (1 - expected))

        prev_season = season

    elo_df = pd.DataFrame([
        {"team": team, "elo": elo} for team, elo in elos.items()
    ]).sort_values("elo", ascending=False)

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    path = PARQUET_DIR / "elo_ratings.parquet"
    elo_df.to_parquet(path, index=False)
    logger.info("Saved current ELO ratings → %s", path)

    return path


def tune_elo_parameters(
    game_results: pd.DataFrame,
    k_values: list[float] | None = None,
    ha_values: list[float] | None = None,
    regress_values: list[float] | None = None,
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

    df = game_results.sort_values(["season", "game_id"]).reset_index(drop=True)

    best_brier = float("inf")
    best_params = {}

    for k in k_values:
        for ha in ha_values:
            for reg in regress_values:
                elo_df = compute_elo_ratings(
                    df, k_factor=k, home_advantage=ha, season_regress=reg,
                )
                merged = df[["game_id", "home_win"]].merge(elo_df, on="game_id")
                merged = merged.dropna(subset=["home_win"])

                # ELO-implied probability
                prob_home = merged.apply(
                    lambda r: _expected_score(
                        r["home_elo"] + ha, r["away_elo"]
                    ),
                    axis=1,
                )
                brier = ((prob_home - merged["home_win"]) ** 2).mean()

                if brier < best_brier:
                    best_brier = brier
                    best_params = {
                        "k_factor": k,
                        "home_advantage": ha,
                        "season_regress": reg,
                        "brier": brier,
                    }

    logger.info(
        "ELO tuning: best Brier=%.4f | K=%.0f, HA=%.0f, regress=%.2f",
        best_params["brier"], best_params["k_factor"],
        best_params["home_advantage"], best_params["season_regress"],
    )
    return best_params


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from features.team import load_team_features

    team_feats = load_team_features()
    home = team_feats[team_feats["is_home"]][
        ["game_id", "season", "home_team", "away_team", "won"]
    ].rename(columns={"won": "home_win"})

    # Tune parameters
    print("\n=== ELO Parameter Tuning ===")
    best = tune_elo_parameters(home)
    print(f"Best params: {best}")

    # Compute with best params
    print("\n=== Computing ELO ratings ===")
    elo_df = compute_elo_ratings(
        home,
        k_factor=best["k_factor"],
        home_advantage=best["home_advantage"],
        season_regress=best["season_regress"],
    )
    print(f"Shape: {elo_df.shape}")
    print(f"\nSample:\n{elo_df.head(10)}")
    print(f"\ndiff_elo stats:\n{elo_df['diff_elo'].describe().round(1)}")

    # Save current ratings
    save_current_elos(
        home,
        k_factor=best["k_factor"],
        home_advantage=best["home_advantage"],
        season_regress=best["season_regress"],
    )
