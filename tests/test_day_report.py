"""One night's results: date resolution, scoping, and what the embed says."""

from datetime import date

import pandas as pd
import pytest

from bot.discord_bot import format_day_embed
from pipeline.evaluate_history import (
    attach_outcomes,
    day_report,
    filter_date,
    game_dates,
    latest_played_date,
)


def _prediction(game_id, home, away, prob, game_date=None, predicted_at=None):
    row = {
        "game_id": game_id,
        "home_team": home,
        "away_team": away,
        "prob_home_win": prob,
        "model_name": "random_forest",
        "predicted_at": predicted_at or f"{game_date}T17:00:00",
    }
    if game_date:
        row["game_date"] = game_date
    return row


@pytest.fixture
def two_nights():
    hist = pd.DataFrame([
        _prediction("2025020001", "BOS", "TOR", 0.72, "2026-01-10"),
        _prediction("2025020002", "EDM", "CGY", 0.63, "2026-01-10"),
        _prediction("2025020003", "NYR", "NJD", 0.52, "2026-01-10"),
        _prediction("2025020004", "VGK", "SJS", 0.68, "2026-01-11"),
    ])
    outcomes = pd.DataFrame({
        "game_id": ["2025020001", "2025020002", "2025020003", "2025020004"],
        # Strong Pick wins, Strong Pick loses, Toss-Up loses, next night wins
        "target": [1.0, 0.0, 0.0, 1.0],
    })
    return attach_outcomes(hist, outcomes)


class TestDateResolution:
    def test_game_date_is_preferred_over_the_run_timestamp(self):
        """A run that crosses midnight UTC stamps predicted_at a day off the
        slate it predicted, which would file the results under the wrong night."""
        hist = pd.DataFrame([
            _prediction("2025020001", "BOS", "TOR", 0.6,
                        game_date="2026-01-10",
                        predicted_at="2026-01-11T02:40:00"),
        ])
        assert game_dates(hist).tolist() == ["2026-01-10"]

    def test_rows_without_game_date_fall_back_to_predicted_at(self):
        hist = pd.DataFrame([{
            "game_id": "2025020001", "home_team": "BOS", "away_team": "TOR",
            "prob_home_win": 0.6, "predicted_at": "2026-01-10T17:00:00",
        }])
        assert game_dates(hist).tolist() == ["2026-01-10"]

    def test_a_mixed_log_uses_each_row_s_best_source(self):
        hist = pd.DataFrame([
            _prediction("2025020001", "BOS", "TOR", 0.6,
                        game_date="2026-01-10",
                        predicted_at="2026-01-11T02:40:00"),
            {"game_id": "2025020002", "home_team": "EDM", "away_team": "CGY",
             "prob_home_win": 0.6, "predicted_at": "2026-01-09T17:00:00",
             "game_date": None},
        ])
        assert game_dates(hist).tolist() == ["2026-01-10", "2026-01-09"]

    def test_empty_history_is_handled(self):
        assert game_dates(pd.DataFrame()).empty
        assert latest_played_date(pd.DataFrame()) is None
        assert day_report(pd.DataFrame())["date"] is None


class TestScoping:
    def test_filter_date_keeps_only_that_night(self, two_nights):
        assert len(filter_date(two_nights, "2026-01-10")) == 3
        assert len(filter_date(two_nights, date(2026, 1, 11))) == 1

    def test_latest_is_the_most_recent_scored_night(self, two_nights):
        assert latest_played_date(two_nights) == "2026-01-11"

    def test_unscored_nights_do_not_become_the_latest(self, two_nights):
        """An unplayed slate is logged but has no results; /yesterday must not
        land on it and report nothing."""
        future = pd.DataFrame([
            _prediction("2025020099", "BOS", "MTL", 0.6, "2026-02-01"),
        ])
        combined = pd.concat([two_nights, attach_outcomes(future, pd.DataFrame(
            {"game_id": [], "target": []},
        ))], ignore_index=True)
        assert latest_played_date(combined) == "2026-01-11"

    def test_default_target_is_the_latest_scored_night(self, two_nights):
        assert day_report(two_nights)["date"] == "2026-01-11"


class TestDayReport:
    def test_record_counts_that_night_only(self, two_nights):
        r = day_report(two_nights, "2026-01-10")
        assert (r["wins"], r["losses"]) == (1, 2)
        assert r["n"] == 3

    def test_every_game_carries_the_pick_not_the_home_side(self, two_nights):
        """A 0.28 home probability is an away pick; reporting the home team
        would score the model against a bet it never made."""
        r = day_report(two_nights, "2026-01-10")
        picks = {g["pick"] for g in r["games"]}
        assert picks == {"BOS", "EDM", "NYR"}
        for g in r["games"]:
            assert g["prob"] >= 0.5

    def test_confidence_matches_the_live_label(self, two_nights):
        from pipeline.evaluate_history import confidence_label
        for g in day_report(two_nights, "2026-01-10")["games"]:
            assert g["confidence"] == confidence_label(g["prob"])

    def test_strong_misses_are_isolated(self, two_nights):
        """The whole point: 1-2 with a Strong Pick down reads differently
        from 1-2 where every miss was a coin flip."""
        r = day_report(two_nights, "2026-01-10")
        assert [g["pick"] for g in r["strong_misses"]] == ["EDM"]

    def test_a_night_of_only_tossup_misses_flags_none(self):
        hist = pd.DataFrame([
            _prediction("2025020010", "BOS", "TOR", 0.52, "2026-01-12"),
            _prediction("2025020011", "EDM", "CGY", 0.53, "2026-01-12"),
        ])
        scored = attach_outcomes(hist, pd.DataFrame({
            "game_id": ["2025020010", "2025020011"], "target": [0.0, 0.0],
        }))
        r = day_report(scored, "2026-01-12")
        assert (r["wins"], r["losses"]) == (0, 2)
        assert r["strong_misses"] == []

    def test_a_date_with_no_predictions_reports_zero_not_an_error(self, two_nights):
        r = day_report(two_nights, "2026-06-01")
        assert r["n"] == 0 and r["n_logged"] == 0

    def test_baselines_come_along(self, two_nights):
        r = day_report(two_nights, "2026-01-10")
        assert "baseline_brier" in r and "baseline_accuracy" in r


class TestDayEmbed:
    def test_a_scored_night_renders_every_pick(self, two_nights):
        e = format_day_embed(day_report(two_nights, "2026-01-10"))
        body = e["fields"][0]["value"]
        for team in ("BOS", "EDM", "NYR"):
            assert team in body
        assert "1-2" in e["fields"][0]["name"]

    def test_brier_is_shown_against_its_baseline(self, two_nights):
        e = format_day_embed(day_report(two_nights, "2026-01-10"))
        brier_field = next(f for f in e["fields"] if f["name"] == "Brier")
        assert "no-skill" in brier_field["value"]

    def test_strong_misses_get_their_own_field(self, two_nights):
        e = format_day_embed(day_report(two_nights, "2026-01-10"))
        assert any("Strong Picks that missed" in f["name"] for f in e["fields"])

    def test_a_clean_night_says_so_instead(self):
        hist = pd.DataFrame([
            _prediction("2025020010", "BOS", "TOR", 0.52, "2026-01-12"),
        ])
        scored = attach_outcomes(hist, pd.DataFrame({
            "game_id": ["2025020010"], "target": [0.0],
        }))
        e = format_day_embed(day_report(scored, "2026-01-12"))
        assert any("No Strong Picks missed" in f["name"] for f in e["fields"])

    def test_no_history_gives_a_usable_message_not_a_crash(self):
        e = format_day_embed(day_report(pd.DataFrame()))
        assert "No results yet" in e["title"]

    def test_a_date_with_nothing_logged_says_that(self, two_nights):
        e = format_day_embed(day_report(two_nights, "2026-06-01"))
        assert "No predictions were logged" in e["description"]

    def test_it_fits_in_a_discord_embed(self, two_nights):
        import json
        e = format_day_embed(day_report(two_nights, "2026-01-10"))
        assert len(json.dumps(e)) < 6000

    def test_the_verdict_colour_tracks_the_record(self, two_nights):
        losing = format_day_embed(day_report(two_nights, "2026-01-10"))["title"]
        winning = format_day_embed(day_report(two_nights, "2026-01-11"))["title"]
        assert losing != winning


class TestPipelineWiring:
    def test_live_stores_the_game_date(self):
        """Without it every night is filed by when the job ran."""
        import inspect
        from pipeline import live
        source = inspect.getsource(live.save_prediction_history)
        assert "game_date" in source
        assert "target_date" in inspect.signature(
            live.save_prediction_history
        ).parameters

    def test_live_passes_the_target_date_through(self):
        import inspect
        from pipeline import live
        assert "save_prediction_history(predictions, model_name, target_date)" \
            in inspect.getsource(live.run)

    def test_the_webhook_can_post_results(self):
        import inspect
        from bot import discord_bot
        assert "results_report" in inspect.signature(
            discord_bot.send_predictions_webhook
        ).parameters
