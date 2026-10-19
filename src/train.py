"""Train and select the session-energy model using a chronological validation set.

The held-out test period is never used here. The selected configuration is
refit on train+validation only after model selection is complete.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from .config import ConfigError, PipelineConfig, add_common_arguments, load_config
from .data_loader import DataValidationError, load_raw_tables
from .metrics import regression_metrics
from .preprocessing import (
    FEATURE_COLUMNS,
    TARGET,
    build_preprocessor,
    prepare_modeling_data,
    split_dataset,
)
from .utils import configure_logging, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)


@dataclass
class CandidateSpec:
    family: str
    estimator_factory: Callable[[], BaseEstimator]
    param_grid: list[dict[str, Any]]
    complexity: int


def build_candidate_specs(config: PipelineConfig) -> list[CandidateSpec]:
    seed = config.random_state
    n_jobs = config.n_jobs
    return [
        CandidateSpec(
            family="DummyRegressor (mean baseline)",
            estimator_factory=lambda: DummyRegressor(strategy="mean"),
            param_grid=[{}],
            complexity=0,
        ),
        CandidateSpec(
            family="Ridge Regression",
            estimator_factory=lambda: Ridge(),
            param_grid=[{"alpha": alpha} for alpha in (0.1, 1.0, 10.0, 100.0)],
            complexity=1,
        ),
        CandidateSpec(
            family="Random Forest",
            estimator_factory=lambda: RandomForestRegressor(
                n_estimators=300,
                random_state=seed,
                n_jobs=n_jobs,
                max_features=1.0,
            ),
            param_grid=[
                {"max_depth": 12, "min_samples_leaf": 5},
                {"max_depth": 12, "min_samples_leaf": 20},
                {"max_depth": None, "min_samples_leaf": 5},
                {"max_depth": None, "min_samples_leaf": 20},
            ],
            complexity=3,
        ),
        CandidateSpec(
            family="HistGradientBoosting",
            estimator_factory=lambda: HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                min_samples_leaf=20,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=seed,
            ),
            param_grid=[
                {"max_leaf_nodes": 15},
                {"max_leaf_nodes": 31},
            ],
            complexity=2,
        ),
    ]


def _label(family: str, params: dict[str, Any]) -> str:
    if not params:
        return family
    rendered = ", ".join(f"{key}={value}" for key, value in sorted(params.items()))
    return f"{family} ({rendered})"


def _capacity_heuristic(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> tuple[float, float, np.ndarray]:
    """Transparent reference: mean train kWh-per-battery-kWh ratio × battery."""

    train_battery = train["battery_capacity_kwh"].where(train["battery_capacity_kwh"] > 0)
    ratio = float((train[TARGET] / train_battery).mean())
    median_battery = float(train_battery.median())
    evaluation_battery = evaluation["battery_capacity_kwh"].where(
        evaluation["battery_capacity_kwh"] > 0, median_battery
    )
    evaluation_battery = evaluation_battery.fillna(median_battery)
    return ratio, median_battery, ratio * evaluation_battery.to_numpy(dtype=float)


def train_pipeline(config: PipelineConfig) -> dict[str, Any]:
    config.ensure_output_dirs()
    tables = load_raw_tables(config.data_dir)
    frame, profile = prepare_modeling_data(tables)
    partitions, split_info = split_dataset(
        frame,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
    )
    train = partitions["train"]
    validation = partitions["validation"]
    train_validation = pd.concat([train, validation], ignore_index=True)

    LOGGER.info("Training rows: %d | validation rows: %d", len(train), len(validation))
    candidate_results: list[dict[str, Any]] = []
    specs = build_candidate_specs(config)

    for spec in specs:
        for params in spec.param_grid:
            label = _label(spec.family, params)
            estimator = clone(spec.estimator_factory()).set_params(**params)
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor()),
                    ("model", estimator),
                ]
            )
            started = time.perf_counter()
            pipeline.fit(train[FEATURE_COLUMNS], train[TARGET])
            fit_seconds = time.perf_counter() - started
            predictions = pipeline.predict(validation[FEATURE_COLUMNS])
            metrics = regression_metrics(validation[TARGET], predictions)
            candidate_results.append(
                {
                    "family": spec.family,
                    "label": label,
                    "params": params,
                    "complexity": spec.complexity,
                    "fit_seconds": float(fit_seconds),
                    "validation_metrics": metrics,
                }
            )
            LOGGER.info(
                "%-62s val RMSE=%7.4f | MAE=%7.4f | R2=%6.4f | %.1fs",
                label,
                metrics["rmse"],
                metrics["mae"],
                metrics["r2"] if metrics["r2"] is not None else float("nan"),
                fit_seconds,
            )

    heuristic_ratio, heuristic_median_battery, heuristic_validation_predictions = _capacity_heuristic(
        train, validation
    )
    heuristic_validation = regression_metrics(
        validation[TARGET], heuristic_validation_predictions
    )

    best = min(
        candidate_results,
        key=lambda result: (
            result["validation_metrics"]["rmse"],
            result["validation_metrics"]["mae"],
            result["complexity"],
            result["label"],
        ),
    )
    selected_spec = next(spec for spec in specs if spec.family == best["family"])
    LOGGER.info("Selected by validation RMSE: %s", best["label"])

    final_estimator = clone(selected_spec.estimator_factory()).set_params(**best["params"])
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", final_estimator),
        ]
    )
    started = time.perf_counter()
    final_pipeline.fit(train_validation[FEATURE_COLUMNS], train_validation[TARGET])
    final_fit_seconds = time.perf_counter() - started

    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "created_utc": utc_now_iso(),
        "project": "Warsaw EV Charging - Session Energy Prediction",
        "target": TARGET,
        "target_unit": "kWh",
        "selected_model": best["family"],
        "selected_label": best["label"],
        "selected_params": best["params"],
        "selection_metric": "validation RMSE (lower is better)",
        "selection_rule": "minimum validation RMSE; ties resolved by MAE then model complexity",
        "validation_metrics_selected": best["validation_metrics"],
        "candidate_results": candidate_results,
        "capacity_heuristic": {
            "description": "mean train kWh per battery kWh multiplied by session battery capacity",
            "train_ratio": heuristic_ratio,
            "median_train_battery_kwh": heuristic_median_battery,
            "validation_metrics": heuristic_validation,
        },
        "final_fit": {
            "rows": int(len(train_validation)),
            "fit_seconds": float(final_fit_seconds),
            "includes_validation": True,
        },
        "split": split_info.to_dict(),
        "data_profile": profile,
        "data_fingerprint": tables.fingerprint,
        "feature_columns": FEATURE_COLUMNS,
        "random_state": config.random_state,
        "validation_fraction_configured": config.validation_fraction,
        "test_fraction_configured": config.test_fraction,
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    bundle = {"model": final_pipeline, "metadata": metadata}
    joblib.dump(bundle, config.model_path, compress=3)
    write_json(config.metadata_path, metadata)
    LOGGER.info("Saved model: %s", config.model_path)
    LOGGER.info("Saved metadata: %s", config.metadata_path)
    return metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and validate the Warsaw EV session-energy model."
    )
    add_common_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        config = load_config(args)
        metadata = train_pipeline(config)
    except (ConfigError, DataValidationError) as exc:
        LOGGER.error("%s", exc)
        return 2
    metrics = metadata["validation_metrics_selected"]
    print("\nTraining complete")
    print(f"Selected model : {metadata['selected_label']}")
    print(f"Validation RMSE: {metrics['rmse']:.4f} kWh")
    print(f"Validation MAE : {metrics['mae']:.4f} kWh")
    print(f"Validation R2  : {metrics['r2']:.4f}" if metrics["r2"] is not None else "Validation R2  : n/a")
    print(f"Model artifact : {config.model_path}")
    print(f"Run evaluation : python -m src.evaluate --data-dir \"{config.data_dir}\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
