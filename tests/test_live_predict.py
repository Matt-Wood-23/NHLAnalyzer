"""Live prediction guardrails."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from pipeline.live import MIN_FEATURE_COVERAGE, predict

FEATURES = [f"f{i}" for i in range(10)]


@pytest.fixture
def model(rng):
    X = rng.random((200, len(FEATURES)))
    y = (X[:, 0] > 0.5).astype(int)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("model", RandomForestClassifier(n_estimators=10, random_state=0)),
    ]).fit(X, y)


@pytest.fixture
def slate():
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "home_team": ["BOS", "TOR", "MTL"],
        "away_team": ["NYR", "OTT", "VAN"],
        **{c: [0.5, 0.5, 0.5] for c in FEATURES},
    })


class TestCoverage:
    def test_full_coverage_when_nothing_is_missing(self, model, slate):
        out = predict(slate, model, FEATURES)
        assert out["feature_coverage"].tolist() == pytest.approx([1.0] * 3)
        assert out["prob_home_win"].between(0, 1).all()

    def test_coverage_counts_nan_values(self, model, slate):
        slate.loc[0, FEATURES[:2]] = np.nan
        out = predict(slate, model, FEATURES, min_coverage=0.0)
        assert out.set_index("game_id").loc["g1", "feature_coverage"] == pytest.approx(0.8)

    def test_coverage_counts_absent_columns(self, model, slate):
        out = predict(slate.drop(columns=FEATURES[:4]), model, FEATURES, min_coverage=0.0)
        assert out["feature_coverage"].tolist() == pytest.approx([0.6] * 3)

    def test_games_below_the_threshold_are_dropped(self, model, slate):
        slate.loc[1, FEATURES[:5]] = np.nan   # 50% — below
        slate.loc[2, FEATURES[:2]] = np.nan   # 80% — above
        out = predict(slate, model, FEATURES, min_coverage=0.7)
        assert out["game_id"].tolist() == ["g1", "g3"]

    def test_a_broken_pipeline_returns_nothing_rather_than_a_fake_number(self, model, slate):
        """Every missing value becomes a column mean, so an upstream break used
        to surface as a plausible ~50% prediction instead of an error."""
        out = predict(slate[["game_id", "home_team", "away_team"]], model, FEATURES)
        assert out.empty

    def test_threshold_is_configurable(self, model, slate):
        slate.loc[1, FEATURES[:5]] = np.nan
        assert len(predict(slate, model, FEATURES, min_coverage=0.4)) == 3
        assert len(predict(slate, model, FEATURES, min_coverage=0.9)) == 2

    def test_default_threshold_is_a_fraction(self):
        assert 0.0 < MIN_FEATURE_COVERAGE <= 1.0


class TestNoMutation:
    def test_input_frame_is_not_modified(self, model, slate):
        before = slate.copy()
        predict(slate, model, FEATURES)
        pd.testing.assert_frame_equal(slate, before)

    def test_missing_columns_are_not_added_to_the_caller_s_frame(self, model, slate):
        trimmed = slate.drop(columns=FEATURES[:4])
        cols_before = list(trimmed.columns)
        predict(trimmed, model, FEATURES, min_coverage=0.0)
        assert list(trimmed.columns) == cols_before
