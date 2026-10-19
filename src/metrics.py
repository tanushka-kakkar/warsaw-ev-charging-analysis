"""Regression metrics, residual summaries and bootstrap uncertainty."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _finite_arrays(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(y_true, dtype=float).ravel()
    predicted = np.asarray(y_pred, dtype=float).ravel()
    if actual.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: y_true={actual.shape}, y_pred={predicted.shape}")
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual, predicted = actual[mask], predicted[mask]
    if actual.size == 0:
        raise ValueError("No finite actual/predicted pairs were provided.")
    return actual, predicted


def regression_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Compute regression metrics for the energy target in kWh."""

    actual, predicted = _finite_arrays(y_true, y_pred)
    residuals = actual - predicted
    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted)) if actual.size > 1 and np.var(actual) > 0 else None

    nonzero = np.abs(actual) > 1e-12
    mape = (
        float(np.mean(np.abs(residuals[nonzero] / actual[nonzero])) * 100.0)
        if nonzero.any()
        else None
    )
    mean_actual = float(np.mean(actual))
    return {
        "n": int(actual.size),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_pct": mape,
        "mean_actual": mean_actual,
        "mean_predicted": float(np.mean(predicted)),
        "mean_residual": float(np.mean(residuals)),
        "median_absolute_error": float(np.median(np.abs(residuals))),
        "max_absolute_error": float(np.max(np.abs(residuals))),
        "relative_rmse_pct": float(rmse / mean_actual * 100.0) if mean_actual else None,
    }


def residual_summary(y_true: Any, y_pred: Any) -> dict[str, Any]:
    actual, predicted = _finite_arrays(y_true, y_pred)
    residuals = actual - predicted
    corr = (
        float(np.corrcoef(predicted, residuals)[0, 1])
        if actual.size > 2 and np.std(predicted) > 0 and np.std(residuals) > 0
        else None
    )
    return {
        "n": int(actual.size),
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals, ddof=1)) if actual.size > 1 else 0.0,
        "median": float(np.median(residuals)),
        "p05": float(np.quantile(residuals, 0.05)),
        "p25": float(np.quantile(residuals, 0.25)),
        "p75": float(np.quantile(residuals, 0.75)),
        "p95": float(np.quantile(residuals, 0.95)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "residual_vs_prediction_correlation": corr,
    }


def bootstrap_metric_ci(
    y_true: Any,
    y_pred: Any,
    n_bootstrap: int = 400,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap intervals for RMSE, MAE and R-squared."""

    actual, predicted = _finite_arrays(y_true, y_pred)
    rng = np.random.default_rng(random_state)
    n = actual.size
    samples = {"rmse": [], "mae": [], "r2": []}
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        yt, yp = actual[indices], predicted[indices]
        samples["rmse"].append(float(np.sqrt(np.mean((yt - yp) ** 2))))
        samples["mae"].append(float(np.mean(np.abs(yt - yp))))
        samples["r2"].append(float(r2_score(yt, yp)) if np.var(yt) > 0 else np.nan)

    output: dict[str, dict[str, float]] = {}
    for name, values in samples.items():
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        output[name] = {
            "lower_95": float(np.quantile(array, 0.025)),
            "upper_95": float(np.quantile(array, 0.975)),
            "bootstrap_samples": int(array.size),
        }
    return output


def segment_metrics(
    frame: Any,
    actual: Any,
    predicted: Any,
    segment_column: str,
    minimum_rows: int = 1,
) -> list[dict[str, Any]]:
    """Compute metrics by a categorical segment (e.g. car model)."""

    import pandas as pd

    work = pd.DataFrame(
        {
            "segment": frame[segment_column].astype("object").where(frame[segment_column].notna(), "Missing"),
            "actual": np.asarray(actual, dtype=float),
            "predicted": np.asarray(predicted, dtype=float),
        }
    )
    rows: list[dict[str, Any]] = []
    for segment, group in work.groupby("segment", dropna=False):
        if len(group) < minimum_rows:
            continue
        metrics = regression_metrics(group["actual"], group["predicted"])
        rows.append({"segment": str(segment), **metrics})
    rows.sort(key=lambda row: row["n"], reverse=True)
    return rows


def paired_improvement(baseline: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Percentage error reductions of the model versus a baseline."""

    def reduction(baseline_value: float, model_value: float) -> float | None:
        if baseline_value in (None, 0) or model_value is None:
            return None
        return float((baseline_value - model_value) / baseline_value * 100.0)

    return {
        "mae_reduction_pct": reduction(baseline.get("mae"), model.get("mae")),
        "rmse_reduction_pct": reduction(baseline.get("rmse"), model.get("rmse")),
        "mape_reduction_pct": reduction(baseline.get("mape_pct"), model.get("mape_pct")),
        "r2_gain": (
            float(model["r2"] - baseline["r2"])
            if model.get("r2") is not None and baseline.get("r2") is not None
            else None
        ),
    }
