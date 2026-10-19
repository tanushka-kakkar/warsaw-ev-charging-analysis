"""Evaluate the selected model once on the chronological held-out test period.

Model selection happens in train.py using validation data only. This module
loads the frozen model, verifies the data fingerprint/split, computes test
metrics and generates figures plus a metrics JSON used by the report generator.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # Must be set before pyplot is imported.
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.dummy import DummyRegressor
from sklearn.inspection import permutation_importance

from .config import ConfigError, PipelineConfig, add_common_arguments, load_config
from .data_loader import DataValidationError, load_raw_tables
from .metrics import (
    bootstrap_metric_ci,
    paired_improvement,
    regression_metrics,
    residual_summary,
    segment_metrics,
)
from .preprocessing import (
    FEATURE_COLUMNS,
    TARGET,
    TIME_COLUMN,
    prepare_modeling_data,
    split_dataset,
)
from .reporting import generate_reports
from .utils import configure_logging, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)

BLUE = "#1f5f8b"
ORANGE = "#d97706"
GREEN = "#2f855a"
RED = "#b91c1c"
GRAY = "#64748b"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_model_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\nRun: python -m src.train --data-dir \"<dataset>\""
        )
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # joblib can raise several unpickling-related errors
        raise DataValidationError(f"Could not load model {path}: {exc}") from exc
    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise DataValidationError(f"Unexpected model bundle structure in {path}.")
    return bundle


def _check_integrity(
    metadata: dict[str, Any],
    current_fingerprint: str,
    current_split: Any,
    allow_data_change: bool,
) -> dict[str, Any]:
    fingerprint_match = current_fingerprint == metadata.get("data_fingerprint")
    expected = metadata.get("split", {})
    current = current_split.to_dict()
    split_match = all(
        current.get(key) == expected.get(key)
        for key in (
            "train_rows",
            "validation_rows",
            "test_rows",
            "test_start",
            "test_end",
        )
    )
    if (not fingerprint_match or not split_match) and not allow_data_change:
        raise DataValidationError(
            "The dataset no longer matches the data used to train this model "
            f"(fingerprint match={fingerprint_match}, split match={split_match}). "
            "Retrain the model, or pass --allow-data-change to evaluate anyway "
            "(results would no longer describe the frozen model's training data)."
        )
    return {
        "data_fingerprint_match": bool(fingerprint_match),
        "split_match": bool(split_match),
        "data_change_allowed": bool(allow_data_change),
    }


def _station_day_metrics(
    frame: pd.DataFrame,
    group_columns: list[str],
    prediction_column: str = "predicted_kwh",
) -> tuple[dict[str, Any], pd.DataFrame]:
    grouped = (
        frame.groupby(group_columns, as_index=False)[["actual_kwh", prediction_column, "baseline_kwh"]]
        .sum()
        .rename(columns={prediction_column: "predicted_energy_kwh", "baseline_kwh": "baseline_energy_kwh"})
    )
    metrics = regression_metrics(grouped["actual_kwh"], grouped["predicted_energy_kwh"])
    baseline_metrics = regression_metrics(grouped["actual_kwh"], grouped["baseline_energy_kwh"])
    metrics["baseline"] = baseline_metrics
    metrics["grouping"] = group_columns
    return metrics, grouped


def _compute_importance(
    model: Any,
    test: pd.DataFrame,
    seed: int,
) -> list[dict[str, Any]]:
    sample_size = min(4000, len(test))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(test), size=sample_size, replace=False))
    X_sample = test.iloc[indices][FEATURE_COLUMNS]
    y_sample = test.iloc[indices][TARGET].to_numpy(dtype=float)

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="neg_root_mean_squared_error",
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
    )
    rows = [
        {
            "feature": feature,
            "importance_mean": float(mean),
            "importance_std": float(std),
        }
        for feature, mean, std in zip(FEATURE_COLUMNS, result.importances_mean, result.importances_std)
    ]
    rows.sort(key=lambda row: row["importance_mean"], reverse=True)
    return rows


def create_figures(
    frame: pd.DataFrame,
    test: pd.DataFrame,
    metadata: dict[str, Any],
    station_day: pd.DataFrame,
    station_day_metrics: dict[str, Any],
    segments: dict[str, list[dict[str, Any]]],
    importance: list[dict[str, Any]],
    test_metrics: dict[str, Any],
    figures_dir: Path,
) -> list[dict[str, str]]:
    _style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, str]] = []

    # 1) Target distribution and vehicle-level means
    car_stats = (
        frame.groupby("car_model")[TARGET]
        .agg(["mean", "count"])
        .sort_values("mean")
    )
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.25]}
    )
    ax1.hist(frame[TARGET], bins=55, color=BLUE, alpha=0.85, edgecolor="white")
    mean_energy = float(frame[TARGET].mean())
    median_energy = float(frame[TARGET].median())
    ax1.axvline(mean_energy, color=ORANGE, linestyle="--", linewidth=1.8, label=f"Mean {mean_energy:.2f} kWh")
    ax1.axvline(median_energy, color=GREEN, linestyle=":", linewidth=1.8, label=f"Median {median_energy:.2f} kWh")
    ax1.set_xlabel("Delivered energy (kWh)")
    ax1.set_ylabel("Sessions")
    ax1.set_title(f"Session energy distribution (n={len(frame):,})")
    ax1.legend(frameon=False)
    ax2.barh(car_stats.index, car_stats["mean"], color=BLUE, alpha=0.88)
    ax2.set_xlabel("Mean delivered energy (kWh)")
    ax2.set_title("Mean energy by vehicle model")
    ax2.tick_params(axis="y", labelsize=8)
    for position, (_, row) in enumerate(car_stats.iterrows()):
        ax2.text(row["mean"] + 0.7, position, f"{row['mean']:.1f}", va="center", fontsize=7.5)
    ax2.set_xlim(0, car_stats["mean"].max() * 1.18)
    _save_figure(fig, figures_dir / "target_distribution.png")
    figures.append(
        {
            "file": "figures/target_distribution.png",
            "caption": "Observed session energy and mean energy by vehicle model. Vehicle capacity is the dominant observable driver.",
        }
    )

    # 2) Chronological split timeline
    daily = frame.set_index(TIME_COLUMN).resample("D").size()
    split = metadata["split"]
    validation_start = pd.Timestamp(split["validation_start"])
    test_start = pd.Timestamp(split["test_start"])
    colors = np.where(
        daily.index < validation_start,
        BLUE,
        np.where(daily.index < test_start, ORANGE, GREEN),
    )
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.bar(daily.index, daily.values, width=1.0, color=colors, alpha=0.9)
    ax.set_ylabel("Sessions per day")
    ax.set_title("Chronological train / validation / test split (no shuffling)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(
        handles=[
            Patch(color=BLUE, label=f"Train ({split['train_dates']} days)"),
            Patch(color=ORANGE, label=f"Validation ({split['validation_dates']} days)"),
            Patch(color=GREEN, label=f"Test ({split['test_dates']} days)"),
        ],
        frameon=False,
        ncol=3,
        loc="upper right",
    )
    _save_figure(fig, figures_dir / "split_timeline.png")
    figures.append(
        {
            "file": "figures/split_timeline.png",
            "caption": "Daily session volume with the chronological train/validation/test boundaries.",
        }
    )

    # 3) Candidate validation comparison
    candidates = sorted(
        metadata["candidate_results"],
        key=lambda result: result["validation_metrics"]["rmse"],
    )
    labels = []
    for result in candidates:
        if result["params"]:
            short = ", ".join(f"{key}={value}" for key, value in sorted(result["params"].items()))
            labels.append(f"{result['family']} ({short})")
        else:
            labels.append(result["family"])
    rmse_values = [result["validation_metrics"]["rmse"] for result in candidates]
    mae_values = [result["validation_metrics"]["mae"] for result in candidates]
    colors = [ORANGE if result["label"] == metadata["selected_label"] else BLUE for result in candidates]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.8), sharey=True)
    positions = np.arange(len(candidates))
    ax1.barh(positions, rmse_values, color=colors, alpha=0.9)
    ax1.set_yticks(positions, labels, fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel("Validation RMSE (kWh)")
    ax1.set_title("Validation RMSE (lower is better)")
    for position, value in zip(positions, rmse_values):
        ax1.text(value + 0.03, position, f"{value:.2f}", va="center", fontsize=7.5)
    ax2.barh(positions, mae_values, color=colors, alpha=0.9)
    ax2.set_xlabel("Validation MAE (kWh)")
    ax2.set_title("Validation MAE (lower is better)")
    for position, value in zip(positions, mae_values):
        ax2.text(value + 0.02, position, f"{value:.2f}", va="center", fontsize=7.5)
    fig.suptitle("Model comparison on the chronological validation period", fontsize=13)
    _save_figure(fig, figures_dir / "model_comparison.png")
    figures.append(
        {
            "file": "figures/model_comparison.png",
            "caption": "Validation comparison of all candidate configurations. Orange marks the selected model.",
        }
    )

    # 4) Predicted vs actual (test)
    rng = np.random.default_rng(42)
    sample_size = min(5000, len(test))
    sample = test.iloc[np.sort(rng.choice(len(test), size=sample_size, replace=False))]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(sample["actual_kwh"], sample["predicted_kwh"], s=10, alpha=0.28, color=BLUE, edgecolors="none")
    limits = [
        min(sample["actual_kwh"].min(), sample["predicted_kwh"].min()) - 2,
        max(sample["actual_kwh"].max(), sample["predicted_kwh"].max()) + 2,
    ]
    ax.plot(limits, limits, color=RED, linestyle="--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Actual energy (kWh)")
    ax.set_ylabel("Predicted energy (kWh)")
    ax.set_title("Held-out test: predicted vs actual")
    ax.text(
        0.04,
        0.96,
        f"n = {test_metrics['n']:,}\nMAE = {test_metrics['mae']:.2f} kWh\nRMSE = {test_metrics['rmse']:.2f} kWh\nR² = {test_metrics['r2']:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"},
    )
    ax.legend(frameon=False, loc="lower right")
    _save_figure(fig, figures_dir / "predicted_vs_actual.png")
    figures.append(
        {
            "file": "figures/predicted_vs_actual.png",
            "caption": "Predicted versus actual session energy on a random sample of the held-out test period.",
        }
    )

    # 5) Residual distribution
    residuals = test["actual_kwh"] - test["predicted_kwh"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.hist(residuals, bins=70, color=BLUE, alpha=0.85, edgecolor="white")
    ax.axvline(0, color=GRAY, linewidth=1.2)
    ax.axvline(float(residuals.mean()), color=ORANGE, linestyle="--", linewidth=1.8, label=f"Mean {residuals.mean():.2f}")
    ax.set_xlabel("Residual (actual − predicted, kWh)")
    ax.set_ylabel("Sessions")
    ax.set_title("Held-out test residual distribution")
    ax.legend(frameon=False)
    _save_figure(fig, figures_dir / "residual_distribution.png")
    figures.append(
        {
            "file": "figures/residual_distribution.png",
            "caption": "Test residuals. The spread reflects unobserved session-level state of charge.",
        }
    )

    # 6) Residuals vs prediction
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.scatter(sample["predicted_kwh"], sample["actual_kwh"] - sample["predicted_kwh"], s=10, alpha=0.25, color=BLUE, edgecolors="none")
    ax.axhline(0, color=RED, linestyle="--", linewidth=1.4)
    correlation = float(np.corrcoef(sample["predicted_kwh"], sample["actual_kwh"] - sample["predicted_kwh"])[0, 1])
    ax.set_xlabel("Predicted energy (kWh)")
    ax.set_ylabel("Residual (kWh)")
    ax.set_title(f"Residuals vs prediction (sample correlation {correlation:.3f})")
    _save_figure(fig, figures_dir / "residuals_vs_predicted.png")
    figures.append(
        {
            "file": "figures/residuals_vs_predicted.png",
            "caption": "Residuals against predictions; a near-zero correlation indicates no simple systematic bias by predicted level.",
        }
    )

    # 7) Permutation importance
    top = [row for row in importance if row["importance_mean"] > -1e-9][:15]
    if len(top) < 5:
        top = importance[:15]
    top = top[::-1]
    friendly = [row["feature"].replace("_", " ") for row in top]
    values = [row["importance_mean"] for row in top]
    errors = [row["importance_std"] for row in top]
    fig, ax = plt.subplots(figsize=(9, 6.4))
    ax.barh(friendly, values, xerr=errors, color=[GREEN if value >= 0 else RED for value in values], alpha=0.88, error_kw={"ecolor": GRAY, "lw": 1})
    ax.set_xlabel("Increase in test RMSE when feature is permuted (kWh)")
    ax.set_title("Permutation importance (top 15 features)")
    _save_figure(fig, figures_dir / "feature_importance.png")
    figures.append(
        {
            "file": "figures/feature_importance.png",
            "caption": "Permutation importance on up to 4,000 held-out sessions (5 repeats). Association, not causation.",
        }
    )

    # 8) Error by vehicle model
    car_segments = [row for row in segments.get("car_model", []) if row["n"] >= 30]
    car_segments.sort(key=lambda row: row["mae"])
    if car_segments:
        fig, ax = plt.subplots(figsize=(9.5, 7))
        segment_labels = [row["segment"] for row in car_segments]
        segment_mae = [row["mae"] for row in car_segments]
        ax.barh(segment_labels, segment_mae, color=BLUE, alpha=0.88)
        for position, row in enumerate(car_segments):
            ax.text(row["mae"] + 0.04, position, f"{row['mae']:.2f} (n={row['n']:,})", va="center", fontsize=7.5)
        ax.set_xlabel("Test MAE (kWh)")
        ax.set_title("Test error by vehicle model")
        ax.tick_params(axis="y", labelsize=8)
        _save_figure(fig, figures_dir / "error_by_car_model.png")
        figures.append(
            {
                "file": "figures/error_by_car_model.png",
                "caption": "Absolute test error by vehicle model. Larger batteries tend to have larger absolute errors.",
            }
        )

    # 9) Conditional station-day aggregation
    if not station_day.empty:
        fig, ax = plt.subplots(figsize=(7.4, 6.2))
        ax.scatter(station_day["actual_kwh"], station_day["predicted_energy_kwh"], s=24, alpha=0.55, color=GREEN, edgecolors="none")
        limits = [
            min(station_day["actual_kwh"].min(), station_day["predicted_energy_kwh"].min()) * 0.95,
            max(station_day["actual_kwh"].max(), station_day["predicted_energy_kwh"].max()) * 1.05,
        ]
        ax.plot(limits, limits, color=RED, linestyle="--", linewidth=1.4, label="Perfect prediction")
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel("Actual station-day energy (kWh)")
        ax.set_ylabel("Predicted station-day energy (kWh)")
        ax.set_title("Conditional station-day aggregation (test)")
        ax.text(
            0.04,
            0.96,
            f"Station-days = {station_day_metrics['n']:,}\nMAE = {station_day_metrics['mae']:.1f} kWh\nMAPE = {station_day_metrics['mape_pct']:.2f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cbd5e1"},
        )
        ax.legend(frameon=False, loc="lower right")
        _save_figure(fig, figures_dir / "station_day_aggregation.png")
        figures.append(
            {
                "file": "figures/station_day_aggregation.png",
                "caption": "Session predictions summed to station-days. This assumes the future session set is known; it is not a demand-arrival forecast.",
            }
        )

    return figures


def evaluate(
    config: PipelineConfig,
    model_path: Path | None = None,
    allow_data_change: bool = False,
) -> dict[str, Any]:
    config.ensure_output_dirs()
    started = time.perf_counter()
    artifact_path = model_path or config.model_path
    if not artifact_path.is_absolute():
        artifact_path = Path.cwd() / artifact_path
    bundle = load_model_bundle(artifact_path)
    model = bundle["model"]
    metadata = bundle["metadata"]

    expected_features = metadata.get("feature_columns", [])
    if expected_features != FEATURE_COLUMNS:
        raise DataValidationError(
            "The saved model's feature schema does not match this code version. Retrain the model."
        )

    tables = load_raw_tables(config.data_dir)
    frame, current_profile = prepare_modeling_data(tables)
    partitions, split_info = split_dataset(
        frame,
        validation_fraction=metadata["validation_fraction_configured"],
        test_fraction=metadata["test_fraction_configured"],
    )
    integrity = _check_integrity(
        metadata, tables.fingerprint, split_info, allow_data_change
    )
    if not integrity["data_fingerprint_match"]:
        LOGGER.warning("Evaluating despite a data-fingerprint mismatch (--allow-data-change).")

    test = partitions["test"].reset_index(drop=True)
    train_validation = pd.concat(
        [partitions["train"], partitions["validation"]], ignore_index=True
    )

    LOGGER.info("Evaluating selected model on %d held-out test sessions", len(test))
    test_predictions = model.predict(test[FEATURE_COLUMNS])
    test_metrics = regression_metrics(test[TARGET], test_predictions)

    # Dummy baseline is fit on train+validation, never on test.
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(train_validation[FEATURE_COLUMNS], train_validation[TARGET])
    baseline_predictions = dummy.predict(test[FEATURE_COLUMNS])
    baseline_metrics = regression_metrics(test[TARGET], baseline_predictions)

    heuristic_ratio = float(metadata["capacity_heuristic"]["train_ratio"])
    median_battery = float(metadata["capacity_heuristic"]["median_train_battery_kwh"])
    test_battery = test["battery_capacity_kwh"].where(
        test["battery_capacity_kwh"] > 0, median_battery
    ).fillna(median_battery)
    heuristic_predictions = heuristic_ratio * test_battery.to_numpy(dtype=float)
    heuristic_metrics = regression_metrics(test[TARGET], heuristic_predictions)

    test_eval = test[
        [
            "session_id",
            TIME_COLUMN,
            "customer_id",
            "station_id",
            "district_name",
            "car_model",
            "customer_income_tier",
            "operator_name",
            TARGET,
        ]
    ].copy()
    test_eval["actual_kwh"] = test_eval[TARGET]
    test_eval["predicted_kwh"] = test_predictions
    test_eval["baseline_kwh"] = baseline_predictions
    test_eval["heuristic_kwh"] = heuristic_predictions
    test_eval["residual_kwh"] = test_eval["actual_kwh"] - test_eval["predicted_kwh"]
    test_eval["session_date"] = pd.to_datetime(test_eval[TIME_COLUMN]).dt.normalize()

    residual = residual_summary(test_eval["actual_kwh"], test_eval["predicted_kwh"])
    bootstrap = bootstrap_metric_ci(
        test_eval["actual_kwh"],
        test_eval["predicted_kwh"],
        n_bootstrap=400,
        random_state=config.random_state,
    )

    segments = {
        "car_model": segment_metrics(test_eval, test_eval["actual_kwh"], test_eval["predicted_kwh"], "car_model"),
        "customer_income_tier": segment_metrics(test_eval, test_eval["actual_kwh"], test_eval["predicted_kwh"], "customer_income_tier"),
        "district_name": segment_metrics(test_eval, test_eval["actual_kwh"], test_eval["predicted_kwh"], "district_name"),
        "operator_name": segment_metrics(test_eval, test_eval["actual_kwh"], test_eval["predicted_kwh"], "operator_name"),
    }

    station_day_metrics, station_day = _station_day_metrics(
        test_eval, ["station_id", "district_name", "session_date"]
    )
    district_day_metrics, district_day = _station_day_metrics(
        test_eval, ["district_name", "session_date"]
    )

    importance = _compute_importance(model, test, config.random_state)
    figures = create_figures(
        frame=frame,
        test=test_eval,
        metadata=metadata,
        station_day=station_day,
        station_day_metrics=station_day_metrics,
        segments=segments,
        importance=importance,
        test_metrics=test_metrics,
        figures_dir=config.figures_dir,
    )

    improvement = paired_improvement(baseline_metrics, test_metrics)
    heuristic_improvement = paired_improvement(heuristic_metrics, test_metrics)
    total_actual = float(test_eval["actual_kwh"].sum())
    total_predicted = float(test_eval["predicted_kwh"].sum())
    total_baseline = float(test_eval["baseline_kwh"].sum())

    payload: dict[str, Any] = {
        "generated_utc": utc_now_iso(),
        "project": "Warsaw EV Charging - Session Energy Prediction",
        "target": TARGET,
        "target_unit": "kWh",
        "selected_model": metadata["selected_model"],
        "selected_label": metadata["selected_label"],
        "selected_params": metadata["selected_params"],
        "selection_metric": metadata["selection_metric"],
        "validation_metrics_selected": metadata["validation_metrics_selected"],
        "capacity_heuristic_train_ratio": heuristic_ratio,
        "capacity_heuristic_validation_metrics": metadata["capacity_heuristic"]["validation_metrics"],
        "candidate_results": metadata["candidate_results"],
        "split": split_info.to_dict(),
        "data_profile": metadata["data_profile"],
        "current_data_profile_summary": {
            "modeling_rows": current_profile["modeling_rows"],
            "date_min": current_profile["date_min"],
            "date_max": current_profile["date_max"],
            "fingerprint": current_profile["fingerprint"],
        },
        "data_fingerprint": metadata["data_fingerprint"],
        "evaluation_integrity": integrity,
        "test_metrics": test_metrics,
        "baseline_test_metrics": baseline_metrics,
        "heuristic_test_metrics": heuristic_metrics,
        "improvement_vs_dummy": improvement,
        "improvement_vs_heuristic": heuristic_improvement,
        "residual_summary": residual,
        "bootstrap_confidence_intervals": bootstrap,
        "segments": segments,
        "station_day_metrics": station_day_metrics,
        "district_day_metrics": district_day_metrics,
        "energy_totals": {
            "actual_kwh": total_actual,
            "predicted_kwh": total_predicted,
            "baseline_kwh": total_baseline,
            "prediction_error_kwh": total_predicted - total_actual,
            "prediction_error_pct": (total_predicted - total_actual) / total_actual * 100,
        },
        "feature_importance": importance,
        "feature_importance_method": (
            "Permutation importance on up to 4,000 random held-out sessions, "
            "5 repeats, scoring=neg_root_mean_squared_error. Increase in RMSE when "
            "a feature is randomly permuted."
        ),
        "figures": figures,
        "environment": metadata.get("environment", {}),
        "config": {
            "random_state": config.random_state,
            "validation_fraction": config.validation_fraction,
            "test_fraction": config.test_fraction,
        },
        "evaluation_seconds": float(time.perf_counter() - started),
    }

    # Persist row-level outputs (gitignored) and reusable tabular diagnostics.
    prediction_columns = [
        "session_id",
        TIME_COLUMN,
        "customer_id",
        "station_id",
        "district_name",
        "car_model",
        "customer_income_tier",
        "actual_kwh",
        "predicted_kwh",
        "baseline_kwh",
        "heuristic_kwh",
        "residual_kwh",
    ]
    test_eval[prediction_columns].to_csv(
        config.reports_dir / "test_predictions.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(importance).to_csv(
        config.reports_dir / "feature_importance.csv", index=False, encoding="utf-8"
    )
    for name, rows in segments.items():
        pd.DataFrame(rows).to_csv(
            config.reports_dir / f"segment_metrics_{name}.csv", index=False, encoding="utf-8"
        )
    station_day.to_csv(
        config.reports_dir / "station_day_predictions.csv", index=False, encoding="utf-8"
    )

    write_json(config.metrics_path, payload)
    report_paths = generate_reports(payload, config.reports_dir)
    payload["report_paths"] = report_paths
    write_json(config.metrics_path, payload)

    LOGGER.info("Saved metrics: %s", config.metrics_path)
    LOGGER.info("Saved reports: %s", report_paths)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen Warsaw EV model on the held-out test period."
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to the saved model bundle (default: <model-dir>/best_model.joblib).",
    )
    parser.add_argument(
        "--allow-data-change",
        action="store_true",
        help="Evaluate even if the dataset fingerprint/split differs from training time.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        config = load_config(args)
        payload = evaluate(
            config,
            model_path=args.model,
            allow_data_change=args.allow_data_change,
        )
    except (ConfigError, DataValidationError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        return 2

    metrics = payload["test_metrics"]
    baseline = payload["baseline_test_metrics"]
    print("\nEvaluation complete (held-out test only)")
    print(f"Selected model : {payload['selected_label']}")
    print(f"Test rows      : {metrics['n']:,}")
    print(f"Test MAE       : {metrics['mae']:.4f} kWh")
    print(f"Test RMSE      : {metrics['rmse']:.4f} kWh")
    print(f"Test R2        : {metrics['r2']:.4f}" if metrics["r2"] is not None else "Test R2        : n/a")
    print(f"Test MAPE      : {metrics['mape_pct']:.2f}%" if metrics["mape_pct"] is not None else "Test MAPE      : n/a")
    print(f"Dummy RMSE     : {baseline['rmse']:.4f} kWh")
    print(f"Markdown report: {payload['report_paths']['markdown']}")
    print(f"HTML report    : {payload['report_paths']['html']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
