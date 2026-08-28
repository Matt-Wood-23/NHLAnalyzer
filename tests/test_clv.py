"""Market prices, closing-line value, and edge realization."""

import numpy as np
import pandas as pd
import pytest

from pipeline.evaluate_history import (
    attach_outcomes, clv_summary, edge_realization, market_comparison,
)


def _history(rows):
    """rows: (game_id, model_p, market_p, closing_p, actual)"""
    return pd.DataFrame([
        {
            "game_id": gid, "home_team": "BOS", "away_team": "TOR",
            "prob_home_win": mp, "model_name": "random_forest",
            "predicted_at": f"2026-03-{i + 1:02d}T17:00:00+00:00",
            "market_prob_home": mkt, "closing_prob_home": clo,
            "actual_home_win": act,
        }
        for i, (gid, mp, mkt, clo, act) in enumerate(rows)
    ])


@pytest.fixture
def scored():
    return attach_outcomes(_history([
        ("2025020001", 0.70, 0.60, 0.65, 1.0),   # model high, line moved toward home
        ("2025020002", 0.40, 0.50, 0.45, 0.0),   # model low, line moved toward away
        ("2025020003", 0.55, 0.55, 0.55, 1.0),   # no edge, no movement
        ("2025020004", 0.65, 0.50, 0.45, 0.0),   # model high, line moved against
    ]))


class TestMarketComparison:
    def test_scores_model_and_market_on_the_same_games(self, scored):
        m = market_comparison(scored)
        assert m["n"] == 4
        y = np.array([1.0, 0.0, 1.0, 0.0])
        assert m["model_brier"] == pytest.approx((((np.array([.70,.40,.55,.65]) - y) ** 2)).mean())
        assert m["market_brier"] == pytest.approx((((np.array([.65,.45,.55,.45]) - y) ** 2)).mean())

    def test_reports_mean_absolute_edge(self, scored):
        assert market_comparison(scored)["mean_abs_edge"] == pytest.approx(
            np.mean([abs(.70-.65), abs(.40-.45), abs(.55-.55), abs(.65-.45)])
        )

    def test_uses_the_price_column_it_is_told_to(self, scored):
        closing = market_comparison(scored, price_col="closing_prob_home")
        opening = market_comparison(scored, price_col="market_prob_home")
        assert closing["market_brier"] != opening["market_brier"]

    def test_no_market_price_is_not_an_error(self):
        bare = attach_outcomes(_history([("2025020001", 0.7, np.nan, np.nan, 1.0)]))
        assert market_comparison(bare)["n"] == 0

    def test_missing_column_entirely(self, scored):
        assert market_comparison(scored.drop(columns=["closing_prob_home"]))["n"] == 0


class TestCLV:
    def test_movement_is_measured_toward_the_picked_side(self, scored):
        clv = clv_summary(scored)
        assert clv["n"] == 4
        # home picks: +0.05 and -0.05 ; away pick: +0.05 ; flat: 0.0
        assert clv["mean_clv"] == pytest.approx(np.mean([0.05, 0.05, 0.0, -0.05]))

    def test_beat_close_rate_counts_favourable_moves(self, scored):
        assert clv_summary(scored)["beat_close_rate"] == pytest.approx(0.5)

    def test_an_away_pick_counts_a_falling_home_price_as_favourable(self):
        """Picking the away side means the market moving away from home is
        movement toward us, not against us."""
        h = attach_outcomes(_history([("2025020001", 0.30, 0.50, 0.40, 0.0)]))
        assert clv_summary(h)["mean_clv"] == pytest.approx(0.10)

    def test_needs_both_prices(self, scored):
        assert clv_summary(scored.drop(columns=["market_prob_home"]))["n"] == 0


class TestEdgeRealization:
    def test_buckets_are_from_the_picked_side(self, scored):
        table = edge_realization(scored)
        assert not table.empty
        assert table["n"].sum() == 4
        for col in ("model_said", "market_said", "actual"):
            assert (table[col].between(0, 1)).all()

    def test_a_model_pick_on_the_away_side_is_not_recorded_as_a_home_edge(self):
        # Model says 30% home = a 70% away pick against a 50% market: +20% edge.
        h = attach_outcomes(_history([("2025020001", 0.30, 0.50, 0.50, 0.0)]))
        row = edge_realization(h).iloc[0]
        assert row["model_said"] == pytest.approx(0.70)
        assert row["market_said"] == pytest.approx(0.50)
        assert row["actual"] == pytest.approx(1.0)   # away won, so the pick won

    def test_empty_without_prices(self):
        bare = attach_outcomes(_history([("2025020001", 0.7, np.nan, np.nan, 1.0)]))
        assert edge_realization(bare).empty


class TestCaptureWiring:
    def test_live_attaches_market_prices(self):
        import inspect
        from pipeline.live import attach_market_prices, run
        assert "attach_market_prices" in inspect.getsource(run)
        assert callable(attach_market_prices)

    def test_market_columns_are_persisted(self):
        import inspect
        from pipeline.live import save_prediction_history
        src = inspect.getsource(save_prediction_history)
        assert "market_prob_home" in src

    def test_a_missing_line_does_not_break_a_prediction(self, monkeypatch):
        """Odds are best-effort; the prediction must still be produced."""
        import pipeline.live as live
        monkeypatch.setattr(
            "ingestion.action_network.consensus_index", lambda *a, **k: {}
        )
        preds = pd.DataFrame([{"game_id": "1", "home_team": "BOS", "away_team": "TOR"}])
        out = live.attach_market_prices(preds, pd.Timestamp("2026-03-15").date())
        assert len(out) == 1
        assert out["market_prob_home"].isna().all()

    def test_daily_captures_the_closing_price(self):
        import inspect
        from pipeline import daily
        assert "backfill_closing_odds" in inspect.getsource(daily.score_predictions)
