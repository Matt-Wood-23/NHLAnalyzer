"""Model vs market: the edge report, and the guards around what it claims.

The report exists to rank disagreements, not to assert edge. Several tests
below pin exactly that distinction, because it is the one that gets lost.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from bot.discord_bot import format_edge_embed
from pipeline.evaluate_history import edge_report


def _preds(rows):
    return pd.DataFrame(
        rows,
        columns=["game_id", "home_team", "away_team",
                 "prob_home_win", "market_prob_home"],
    )


@pytest.fixture
def slate():
    return _preds([
        # model likes home, market does not  -> +7 on BOS
        ("1", "BOS", "TOR", 0.62, 0.55),
        # model likes away, market likes home -> +9 on CGY
        ("2", "EDM", "CGY", 0.41, 0.50),
        # agreement                          -> filtered
        ("3", "NYR", "NJD", 0.55, 0.545),
    ])


class TestEdgeDirection:
    def test_edge_is_stated_from_the_side_the_model_picks(self, slate):
        """An away pick at 0.41 home is a 0.59 pick on CGY, and the market's
        0.50 home is 0.50 on CGY -- so +9, not -9."""
        r = edge_report(slate)
        cgy = next(e for e in r["edges"] if e["pick"] == "CGY")
        assert cgy["model_p"] == pytest.approx(0.59)
        assert cgy["market_p"] == pytest.approx(0.50)
        assert cgy["edge"] == pytest.approx(0.09)

    def test_the_opponent_is_named_too(self, slate):
        r = edge_report(slate)
        assert {e["pick"] for e in r["edges"]} == {"BOS", "CGY"}
        assert next(e for e in r["edges"] if e["pick"] == "CGY")["against"] == "EDM"

    def test_edges_are_ranked_largest_first(self, slate):
        r = edge_report(slate)
        assert [e["pick"] for e in r["edges"]] == ["CGY", "BOS"]

    def test_agreement_is_not_an_edge(self, slate):
        """0.5 points of disagreement is noise; showing it would bury the
        two that are not."""
        r = edge_report(slate)
        assert "NYR" not in {e["pick"] for e in r["edges"]}
        assert "NJD" not in {e["pick"] for e in r["edges"]}

    def test_the_threshold_is_adjustable(self, slate):
        assert len(edge_report(slate, min_edge=0.0)["edges"]) == 3
        assert len(edge_report(slate, min_edge=0.08)["edges"]) == 1

    def test_a_model_that_agrees_everywhere_yields_nothing(self):
        same = _preds([("1", "BOS", "TOR", 0.60, 0.60)])
        assert edge_report(same)["edges"] == []


class TestMissingPrices:
    def test_unpriced_games_are_counted_not_dropped_silently(self):
        """Knowing 2 of 8 games had a price is the difference between "no
        edges tonight" and "no prices tonight"."""
        df = _preds([
            ("1", "BOS", "TOR", 0.62, 0.55),
            ("2", "EDM", "CGY", 0.62, np.nan),
        ])
        r = edge_report(df)
        assert r["n_games"] == 2
        assert r["n_priced"] == 1

    def test_a_slate_with_no_prices_at_all(self):
        df = _preds([("1", "BOS", "TOR", 0.62, np.nan)])
        r = edge_report(df)
        assert r["n_priced"] == 0 and r["edges"] == []

    def test_a_frame_without_the_column_does_not_crash(self):
        df = pd.DataFrame({"game_id": ["1"], "home_team": ["BOS"],
                           "away_team": ["TOR"], "prob_home_win": [0.6]})
        assert edge_report(df)["n_priced"] == 0

    def test_an_empty_slate(self):
        r = edge_report(pd.DataFrame())
        assert r["n_games"] == 0 and r["edges"] == []


class TestRealization:
    @pytest.fixture
    def scored_history(self):
        rng = np.random.default_rng(7)
        n = 60
        model = rng.uniform(0.35, 0.75, n)
        return pd.DataFrame({
            "game_id": [str(i) for i in range(n)],
            "home_team": ["BOS"] * n,
            "away_team": ["TOR"] * n,
            "prob_home_win": model,
            "closing_prob_home": model + rng.normal(0, 0.06, n),
            "actual_home_win": rng.integers(0, 2, n).astype(float),
            "model_name": ["random_forest"] * n,
        })

    def test_realization_comes_along_when_history_is_given(self, slate, scored_history):
        r = edge_report(slate, hist=scored_history)
        assert r["realization"], "expected edge buckets from scored history"
        first = next(iter(r["realization"].values()))
        assert {"n", "model_said", "market_said", "actual"} <= set(first)

    def test_no_history_means_no_claims(self, slate):
        r = edge_report(slate)
        assert r["realization"] == {} and r["clv"] == {}

    def test_unscored_history_produces_nothing(self, slate):
        unscored = pd.DataFrame({
            "game_id": ["1"], "prob_home_win": [0.6],
            "closing_prob_home": [0.55], "actual_home_win": [np.nan],
            "model_name": ["random_forest"],
        })
        assert edge_report(slate, hist=unscored)["realization"] == {}


class TestEdgeEmbed:
    def test_it_names_the_disagreements(self, slate):
        e = format_edge_embed(edge_report(slate), date(2026, 10, 5))
        body = e["fields"][0]["value"]
        assert "CGY" in body and "BOS" in body

    def test_it_says_a_disagreement_is_not_an_edge(self, slate):
        """The single most important line on the screen. If this test is
        deleted, the embed starts reading like a tip sheet."""
        e = format_edge_embed(edge_report(slate), date(2026, 10, 5))
        assert "hypothesis" in e["description"].lower()

    def test_no_prices_says_so_rather_than_no_edges(self):
        df = _preds([("1", "BOS", "TOR", 0.62, np.nan)])
        e = format_edge_embed(edge_report(df), date(2026, 10, 5))
        assert "No market prices" in e["description"]

    def test_agreement_reads_as_a_normal_result(self):
        same = _preds([("1", "BOS", "TOR", 0.60, 0.60)])
        e = format_edge_embed(edge_report(same), date(2026, 10, 5))
        assert "agrees with the market" in e["description"]
        assert "normal result" in e["description"]

    def test_the_realization_table_uses_the_real_column_names(self, slate):
        """edge_realization returns n/model_said/market_said/actual. An earlier
        draft guessed win_rate, which silently dropped the whole field."""
        rng = np.random.default_rng(3)
        n = 60
        model = rng.uniform(0.35, 0.75, n)
        hist = pd.DataFrame({
            "game_id": [str(i) for i in range(n)],
            "prob_home_win": model,
            "closing_prob_home": model + rng.normal(0, 0.06, n),
            "actual_home_win": rng.integers(0, 2, n).astype(float),
            "model_name": ["random_forest"] * n,
        })
        e = format_edge_embed(edge_report(slate, hist=hist), date(2026, 10, 5))
        assert any("pay" in f["name"] for f in e["fields"]), \
            "realization field missing — column names likely drifted"

    def test_it_fits_in_a_discord_embed(self, slate):
        import json
        e = format_edge_embed(edge_report(slate), date(2026, 10, 5))
        assert len(json.dumps(e)) < 6000

    def test_the_footer_shows_price_coverage(self, slate):
        e = format_edge_embed(edge_report(slate), date(2026, 10, 5))
        assert "3 game(s) had a price" in e["footer"]["text"]


class TestWiring:
    def test_the_slash_command_never_writes_the_prediction_log(self):
        """/edge runs live predictions to get today's prices. Without dry_run
        that would overwrite the canonical prediction pipeline.daily logged."""
        import inspect
        from bot import discord_bot
        src = inspect.getsource(discord_bot.run_bot)
        idx = src.index('name="edge"')
        body = src[idx:idx + 1800]
        assert "dry_run=True" in body

    def test_the_webhook_can_post_edges(self):
        import inspect
        from bot import discord_bot
        assert "edge_report_data" in inspect.signature(
            discord_bot.send_predictions_webhook
        ).parameters
