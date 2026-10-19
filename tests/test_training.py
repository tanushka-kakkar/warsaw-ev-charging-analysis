"""Tests for metric calculations, model fitting, persistence and leakage."""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.metrics import paired_improvement, regression_metrics
from src.preprocessing import FEATURE_COLUMNS, TARGET, build_preprocessor, prepare_modeling_data, split_dataset


def test_regression_metrics_known_values():
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.0, 1.0, 5.0])
    metrics = regression_metrics(actual, predicted)

    assert metrics["n"] == 3
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt(5.0 / 3.0))
    assert metrics["r2"] == pytest.approx(-1.5)
    assert metrics["mape_pct"] == pytest.approx((0 + 0.5 + 2 / 3) / 3 * 100.0)
    assert metrics["mean_residual"] == pytest.approx(-1.0 / 3.0)


def test_paired_improvement():
    baseline = {"mae": 10.0, "rmse": 12.0, "mape_pct": 20.0, "r2": 0.0}
    model = {"mae": 8.0, "rmse": 9.0, "mape_pct": 15.0, "r2": 0.25}
    result = paired_improvement(baseline, model)
    assert result["mae_reduction_pct"] == pytest.approx(20.0)
    assert result["rmse_reduction_pct"] == pytest.approx(25.0)
    assert result["r2_gain"] == pytest.approx(0.25)


def test_pipeline_fit_predict_save_load(tmp_path, synthetic_tables):
    frame, _ = prepare_modeling_data(synthetic_tables)
    partitions, _ = split_dataset(frame)
    train, validation = partitions["train"], partitions["validation"]

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET])
    predictions = model.predict(validation[FEATURE_COLUMNS])
    assert len(predictions) == len(validation)
    assert np.isfinite(predictions).all()

    artifact = tmp_path / "model.joblib"
    joblib.dump({"model": model, "metadata": {"feature_columns": FEATURE_COLUMNS}}, artifact)
    loaded = joblib.load(artifact)
    loaded_predictions = loaded["model"].predict(validation[FEATURE_COLUMNS])
    np.testing.assert_allclose(predictions, loaded_predictions)


def test_predictions_ignore_target_column(synthetic_tables):
    frame, _ = prepare_modeling_data(synthetic_tables)
    partitions, _ = split_dataset(frame)
    train, validation = partitions["train"], partitions["validation"]
    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", Ridge(alpha=1.0)),
        ]
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET])
    first = model.predict(validation[FEATURE_COLUMNS])

    altered = validation.copy()
    altered[TARGET] = altered[TARGET] * 10.0 + 500.0
    second = model.predict(altered[FEATURE_COLUMNS])
    np.testing.assert_allclose(first, second)
