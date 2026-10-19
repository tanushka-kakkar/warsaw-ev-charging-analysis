"""Tests for Markdown/HTML report rendering and offline image embedding."""

from __future__ import annotations

import base64

from src.reporting import generate_reports


def _report_payload() -> dict:
    return {
        "generated_utc": "2026-10-19T12:00:00+00:00",
        "project": "Warsaw EV Charging - Session Energy Prediction",
        "target": "kwh_charged",
        "target_unit": "kWh",
        "selected_model": "Ridge Regression",
        "selected_label": "Ridge Regression (alpha=10.0)",
        "selected_params": {"alpha": 10.0},
        "selection_metric": "validation RMSE",
        "validation_metrics_selected": {"n": 500, "mae": 3.1, "rmse": 3.9, "r2": 0.9, "mape_pct": 8.1},
        "capacity_heuristic_validation_metrics": {"n": 500, "mae": 3.2, "rmse": 4.0, "r2": 0.89, "mape_pct": 8.4},
        "candidate_results": [
            {
                "family": "Ridge Regression",
                "label": "Ridge Regression (alpha=10.0)",
                "params": {"alpha": 10.0},
                "complexity": 1,
                "fit_seconds": 0.4,
                "validation_metrics": {"n": 500, "mae": 3.1, "rmse": 3.9, "r2": 0.9, "mape_pct": 8.1},
            },
            {
                "family": "DummyRegressor (mean baseline)",
                "label": "DummyRegressor (mean baseline)",
                "params": {},
                "complexity": 0,
                "fit_seconds": 0.1,
                "validation_metrics": {"n": 500, "mae": 9.0, "rmse": 12.0, "r2": 0.0, "mape_pct": 22.0},
            },
        ],
        "split": {
            "train_rows": 700,
            "validation_rows": 150,
            "test_rows": 150,
            "train_start": "2025-01-01",
            "train_end": "2025-01-20",
            "validation_start": "2025-01-21",
            "validation_end": "2025-01-25",
            "test_start": "2025-01-26",
            "test_end": "2025-01-30",
            "train_dates": 20,
            "validation_dates": 5,
            "test_dates": 5,
            "validation_fraction_actual": 0.15,
            "test_fraction_actual": 0.15,
        },
        "data_profile": {
            "source_files": {
                "charging_sessions.csv": {
                    "rows": 1000,
                    "columns": 7,
                    "missing_cells": 0,
                    "duplicate_rows": 0,
                    "size_bytes": 10438947,
                }
            },
            "raw_sessions_rows": 1000,
            "modeling_rows": 1000,
            "raw_customers_rows": 20,
            "raw_stations_rows": 3,
            "raw_districts_rows": 2,
            "unique_customers": 20,
            "unique_stations": 3,
            "unique_districts": 2,
            "date_min": "2025-01-01",
            "date_max": "2025-01-30",
            "span_days": 29,
            "dropped_rows": {"duplicate_customer_start_time": 1, "invalid_target": 0},
            "target_summary": {"min": 13.4, "max": 76.0, "mean": 39.1, "median": 37.9},
            "target_iqr": {"outlier_count": 3, "lower_bound": 3.0, "upper_bound": 75.0},
            "ratio_to_battery": {"mean": 0.65},
            "physical_ratio": {"min": 0.5, "max": 0.8, "above_battery": 0},
            "referential_coverage": {"customer_sessions_matched": 1000, "customer_sessions_unmatched": 0, "station_sessions_matched": 1000, "station_sessions_unmatched": 0},
            "invalid_price_rows": 0,
            "invalid_battery_rows": 0,
            "invalid_coordinate_rows": 0,
            "negative_projection_rows": 0,
            "cost_identity_max_abs_error": 1e-13,
            "duplicate_station_start_time": 2,
            "final_missing_feature_values": 0,
            "sessions_per_customer": {"mean": 50.0, "median": 50.0},
            "sessions_per_station": {"mean": 333.0, "median": 333.0},
            "price_values": {"1.851397": 300, "2.704520": 400, "3.256199": 300},
        },
        "data_fingerprint": "abc123",
        "evaluation_integrity": {"data_fingerprint_match": True, "split_match": True, "data_change_allowed": False},
        "test_metrics": {"n": 150, "mae": 3.2, "rmse": 4.0, "r2": 0.895, "mape_pct": 8.3, "mean_actual": 39.0, "mean_predicted": 39.1},
        "baseline_test_metrics": {"n": 150, "mae": 9.0, "rmse": 12.1, "r2": -0.01, "mape_pct": 22.0},
        "heuristic_test_metrics": {"n": 150, "mae": 3.3, "rmse": 4.1, "r2": 0.89, "mape_pct": 8.5},
        "improvement_vs_dummy": {"mae_reduction_pct": 64.4, "rmse_reduction_pct": 66.9, "mape_reduction_pct": 62.3, "r2_gain": 0.905},
        "improvement_vs_heuristic": {"mae_reduction_pct": 3.0, "rmse_reduction_pct": 2.4, "mape_reduction_pct": 2.4, "r2_gain": 0.005},
        "residual_summary": {"mean": 0.01, "median": 0.0, "std": 4.0, "p05": -6.1, "p95": 6.2, "residual_vs_prediction_correlation": 0.01},
        "bootstrap_confidence_intervals": {
            "rmse": {"lower_95": 3.8, "upper_95": 4.2, "bootstrap_samples": 400},
            "mae": {"lower_95": 3.0, "upper_95": 3.4, "bootstrap_samples": 400},
            "r2": {"lower_95": 0.88, "upper_95": 0.91, "bootstrap_samples": 400},
        },
        "segments": {
            "car_model": [{"segment": "Test EV", "n": 150, "mae": 3.2, "rmse": 4.0, "mape_pct": 8.3}],
            "customer_income_tier": [{"segment": "High", "n": 100, "mae": 3.1, "rmse": 3.9, "mape_pct": 7.9}],
            "district_name": [{"segment": "Test District", "n": 150, "mae": 3.2, "rmse": 4.0, "mape_pct": 8.3}],
            "operator_name": [{"segment": "Test Operator", "n": 150, "mae": 3.2, "rmse": 4.0, "mape_pct": 8.3}],
        },
        "station_day_metrics": {"n": 30, "mae": 45.0, "rmse": 55.0, "r2": 0.97, "mape_pct": 3.0, "baseline": {"mape_pct": 22.0}},
        "district_day_metrics": {"n": 10, "mae": 80.0, "rmse": 95.0, "r2": 0.98, "mape_pct": 2.5, "baseline": {"mape_pct": 22.0}},
        "energy_totals": {"actual_kwh": 5850.0, "predicted_kwh": 5865.0, "prediction_error_pct": 0.26},
        "feature_importance": [
            {"feature": "battery_capacity_kwh", "importance_mean": 20.0, "importance_std": 0.5},
            {"feature": "car_model", "importance_mean": 4.0, "importance_std": 0.2},
        ],
        "figures": [
            {"file": "figures/test.png", "caption": "Synthetic test figure."},
        ],
        "evaluation_seconds": 4.2,
    }


def test_markdown_and_html_reports(tmp_path):
    payload = _report_payload()
    figure_dir = tmp_path / "figures"
    figure_dir.mkdir()
    # A tiny PNG-like blob is sufficient to test offline embedding; real runs use matplotlib PNGs.
    (figure_dir / "test.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"test")

    paths = generate_reports(payload, tmp_path)
    markdown = (tmp_path / "model_evaluation_report.md").read_text(encoding="utf-8")
    html_text = (tmp_path / "model_evaluation_report.html").read_text(encoding="utf-8")

    assert paths["markdown"].endswith("model_evaluation_report.md")
    assert paths["html"].endswith("model_evaluation_report.html")
    assert markdown.startswith("# Warsaw EV Charging")
    assert "Actual test metrics" in markdown
    assert "figures/test.png" in markdown
    assert "<!DOCTYPE html>" in html_text
    assert "Session Energy Prediction" in html_text
    assert "data:image/png;base64," in html_text
    assert "http://" not in html_text
    assert "https://" not in html_text
    assert base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"test").decode("ascii") in html_text
