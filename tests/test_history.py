"""Prediction-history scoring, metrics, and the Discord /history embed."""

import pandas as pd
import pytest

from pipeline import evaluate_history as eh
from pipeline.evaluate_history import (
    add_season, attach_outcomes, calibration_table, confidence_breakdown,
    confidence_label, coverage_warning, evaluated_only, filter_season,
    recent_form, summarize,
)


_COLUMNS = ["game_id", "home_team", "away_team", "prob_home_win",
            "model_name", "predicted_at", "actual_home_win"]


def _history(rows):
    """rows: (game_id, prob_home_win, actual) — actual None = not yet played."""
    return pd.DataFrame([
        {
            "game_id": gid,
            "home_team": "BOS", "away_team": "TOR",
            "prob_home_win": prob,
            "model_name": "random_forest",
            "predicted_at": f"2026-03-{i + 1:02d}T17:00:00+00:00",
            "actual_home_win": actual,
        }
        for i, (gid, prob, actual) in enumerate(rows)
    ], columns=_COLUMNS)


@pytest.fixture
def scored():
    """Six games: four home wins, two away wins."""
    return attach_outcomes(_history([
        ("2025020001", 0.70, 1.0),
        ("2025020002", 0.65, 1.0),
        ("2025020003", 0.58, 0.0),
        ("2025020004", 0.52, 1.0),
        ("2025020005", 0.35, 0.0),
        ("2025020006", 0.45, 1.0),
    ]))


class TestSeasonTagging:
    def test_season_derives_from_the_game_id(self):
        tagged = add_season(_history([("2025020001", 0.5, None)]))
        assert tagged["season"].iloc[0] == "2025-2026"

    def test_works_retroactively_on_rows_logged_before_the_column_existed(self, scored):
        assert "season" in scored.columns
        assert scored["season"].notna().all()

    def test_separates_seasons(self):
        hist = attach_outcomes(_history([
            ("2025020001", 0.7, 1.0),
            ("2026020001", 0.7, 0.0),
        ]))
        assert len(filter_season(hist, "2025-2026")) == 1
        assert len(filter_season(hist, "2026-2027")) == 1
        assert len(filter_season(hist, "all")) == 2
        assert len(filter_season(hist, None)) == 2

    def test_malformed_game_id_does_not_raise(self):
        tagged = add_season(_history([("bogus", 0.5, None)]))
        assert tagged["season"].iloc[0] is None


class TestScoring:
    def test_correct_and_brier(self, scored):
        assert scored["correct"].tolist() == [True, True, False, True, True, False]
        assert scored["brier"].iloc[0] == pytest.approx((0.70 - 1.0) ** 2)

    def test_correct_is_a_real_boolean_column(self, scored):
        """It used to be assigned into a slice, leaving an object column."""
        assert scored["correct"].dtype == "boolean"

    def test_unplayed_games_are_not_scored(self):
        hist = attach_outcomes(_history([
            ("2025020001", 0.7, 1.0),
            ("2025020002", 0.7, None),
        ]))
        assert len(evaluated_only(hist)) == 1
        assert pd.isna(hist["correct"].iloc[1])

    def test_stored_outcomes_survive_a_missing_feature_matrix(self, scored, monkeypatch):
        """The feature matrix is gitignored and rebuilt by the daily pipeline;
        re-scoring without it must not erase results recorded earlier."""
        monkeypatch.setattr(eh, "PARQUET_DIR", eh.PARQUET_DIR / "does-not-exist")
        again = attach_outcomes(scored)
        assert again["actual_home_win"].notna().sum() == 6
        assert again["correct"].tolist() == scored["correct"].tolist()

    def test_a_fresh_lookup_fills_in_missing_outcomes(self):
        hist = _history([("2025020001", 0.7, None)])
        outcomes = pd.DataFrame([{"game_id": "2025020001", "target": 1.0}])
        assert attach_outcomes(hist, outcomes)["actual_home_win"].iloc[0] == 1.0

    def test_scoring_is_idempotent(self, scored):
        pd.testing.assert_frame_equal(attach_outcomes(scored), scored)


class TestSummary:
    def test_record_and_accuracy(self, scored):
        stats = summarize(scored)
        assert stats["n"] == 6
        assert (stats["wins"], stats["losses"]) == (4, 2)
        assert stats["accuracy"] == pytest.approx(4 / 6)

    def test_baselines_come_from_the_same_games(self, scored):
        """Accuracy is unreadable without them: home teams win ~54% of games,
        so a model at 55% has found almost nothing."""
        stats = summarize(scored)
        assert stats["home_win_rate"] == pytest.approx(4 / 6)
        assert stats["baseline_accuracy"] == pytest.approx(4 / 6)
        assert stats["baseline_brier"] == pytest.approx((4 / 6) * (2 / 6))

    def test_baseline_accuracy_covers_an_away_leaning_sample(self):
        hist = attach_outcomes(_history([
            ("2025020001", 0.7, 0.0), ("2025020002", 0.7, 0.0),
            ("2025020003", 0.7, 1.0),
        ]))
        # Always picking away would be right 2/3 of the time.
        assert summarize(hist)["baseline_accuracy"] == pytest.approx(2 / 3)

    def test_empty_history_reports_zero_rather_than_raising(self):
        assert summarize(attach_outcomes(_history([])))["n"] == 0

    def test_unscored_history_reports_zero(self):
        hist = attach_outcomes(_history([("2025020001", 0.7, None)]))
        stats = summarize(hist)
        assert stats["n"] == 0 and stats["n_logged"] == 1


class TestConfidenceTiers:
    @pytest.mark.parametrize("prob,expected", [
        (0.75, "Strong Pick"), (0.60, "Strong Pick"),
        (0.59, "Lean"), (0.55, "Lean"),
        (0.54, "Toss-Up"), (0.50, "Toss-Up"),
    ])
    def test_labels(self, prob, expected):
        assert confidence_label(prob) == expected

    def test_the_bot_and_the_report_share_one_definition(self):
        """Otherwise the breakdown would not be checking the labels shown."""
        from bot.discord_bot import _confidence_label
        for p in (0.49, 0.55, 0.62, 0.9):
            assert _confidence_label(p) == confidence_label(p)

    def test_breakdown_uses_the_favoured_side(self, scored):
        table = confidence_breakdown(scored)
        # 0.70, 0.65 and 0.35 all favour someone by >= 60%.
        assert int(table.loc["Strong Pick", "n"]) == 3

    def test_breakdown_rows_are_ordered_strongest_first(self, scored):
        assert list(confidence_breakdown(scored).index) == [
            "Strong Pick", "Lean", "Toss-Up",
        ]

    def test_breakdown_of_empty_history_is_empty(self):
        assert confidence_breakdown(_history([])).empty


class TestCalibration:
    def test_buckets_report_predicted_against_actual(self, scored):
        cal = calibration_table(scored)
        assert cal["n"].sum() == 6
        assert (cal["predicted"].between(0, 1)).all()
        assert (cal["actual"].between(0, 1)).all()

    def test_extreme_probabilities_are_binned(self):
        hist = attach_outcomes(_history([
            ("2025020001", 0.0, 0.0), ("2025020002", 1.0, 1.0),
        ]))
        assert calibration_table(hist)["n"].sum() == 2


class TestRecentForm:
    def test_takes_the_most_recent_by_time_not_row_order(self):
        """Re-predicting a game moves its row to the end of the file, so
        tail() on raw row order is not the same as most recent."""
        hist = attach_outcomes(_history([
            ("2025020001", 0.9, 1.0),
            ("2025020002", 0.9, 0.0),
            ("2025020003", 0.9, 0.0),
        ]))
        shuffled = hist.iloc[[2, 0, 1]]
        assert recent_form(shuffled, last_n=1)["wins"] == 0  # 2025020003 lost

    def test_empty_history(self):
        assert recent_form(_history([]), last_n=5)["n"] == 0


class TestCoverageWarning:
    def test_silent_when_the_column_is_absent(self, scored):
        assert coverage_warning(scored) is None

    def test_silent_when_coverage_is_complete(self, scored):
        assert coverage_warning(scored.assign(feature_coverage=1.0)) is None

    def test_flags_degraded_predictions(self, scored):
        scored = scored.copy()
        scored["feature_coverage"] = [1.0, 1.0, 0.4, 1.0, 1.0, 1.0]
        assert "1 of 6" in coverage_warning(scored)


class TestHistoryEmbed:
    """Discord rejects an embed that breaks its size limits, so check them."""

    @pytest.fixture
    def embed(self, scored, monkeypatch):
        from bot import discord_bot
        monkeypatch.setattr(discord_bot, "format_history_embed",
                            discord_bot.format_history_embed)
        monkeypatch.setattr(eh, "load_history", lambda *a, **k: scored)
        return discord_bot.format_history_embed(season="2025-2026")

    def test_reports_the_record_and_both_baselines(self, embed):
        assert "4-2" in embed["description"]
        assert "always-home" in embed["description"]
        assert "no-skill" in embed["description"]

    def test_includes_the_tables_that_were_terminal_only(self, embed):
        names = [f["name"] for f in embed["fields"]]
        assert any("pick strength" in n for n in names)
        assert any("Calibration" in n for n in names)

    def test_title_names_the_season(self, embed):
        assert "2025-2026" in embed["title"]

    def test_fits_discord_limits(self, embed):
        assert len(embed["description"]) <= 4096
        assert len(embed["fields"]) <= 25
        for field in embed["fields"]:
            assert len(field["name"]) <= 256
            assert len(field["value"]) <= 1024
        total = (
            len(embed["title"]) + len(embed["description"])
            + len(embed.get("footer", {}).get("text", ""))
            + sum(len(f["name"]) + len(f["value"]) for f in embed["fields"])
        )
        assert total <= 6000

    def test_is_json_serializable(self, embed):
        import json
        json.dumps(embed)

    def test_empty_history_gives_a_message_not_a_crash(self, monkeypatch):
        from bot import discord_bot
        monkeypatch.setattr(eh, "load_history", lambda *a, **k: pd.DataFrame())
        embed = discord_bot.format_history_embed()
        assert "No prediction history" in embed["description"]

    def test_unknown_season_lists_what_is_available(self, scored, monkeypatch):
        from bot import discord_bot
        monkeypatch.setattr(eh, "load_history", lambda *a, **k: scored)
        embed = discord_bot.format_history_embed(season="2030-2031")
        assert "2025-2026" in embed["description"]

    def test_defaults_to_the_newest_season_with_data(self, scored, monkeypatch):
        """Keeps the command useful through the off-season and opening weeks."""
        from bot import discord_bot
        monkeypatch.setattr(eh, "load_history", lambda *a, **k: scored)
        monkeypatch.setattr(discord_bot, "_current_season", lambda: "2030-2031")
        assert "2025-2026" in discord_bot.format_history_embed()["title"]


class TestReadOnly:
    def test_load_history_does_not_rewrite_the_file(self, tmp_path):
        """A Discord command should not write to a file the daily pipeline
        may be writing at the same moment."""
        path = tmp_path / "prediction_history.parquet"
        _history([("2025020001", 0.7, 1.0)]).to_parquet(path, index=False)
        before = path.stat().st_mtime_ns

        eh.load_history(path)

        assert path.stat().st_mtime_ns == before

    def test_load_history_with_no_file_returns_empty(self, tmp_path):
        assert eh.load_history(tmp_path / "absent.parquet").empty
