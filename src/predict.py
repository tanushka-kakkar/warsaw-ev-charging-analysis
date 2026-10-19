"""Batch prediction for new charging sessions using the saved pipeline.

Expected input columns:
    customer_id, station_id, session_start_time, cost_per_kwh
    session_id (optional)

Customer, station and district attributes are joined from the source lookup
files. A kwh_charged column, if present, is used only for optional post-hoc
scoring and is never passed to the model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import ConfigError, add_common_arguments, load_config
from .data_loader import DataValidationError, load_raw_tables
from .metrics import regression_metrics
from .preprocessing import FEATURE_COLUMNS, TARGET, prepare_inference_data

LOGGER = logging.getLogger(__name__)


def _read_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Prediction input not found: {path}")
    if not path.is_file():
        raise DataValidationError(f"Prediction input is not a file: {path}")
    for encoding in ("utf-8-sig", "utf-8", "cp1250", "cp1252"):
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise DataValidationError(f"Prediction input is empty: {path}") from exc
        except pd.errors.ParserError as exc:
            raise DataValidationError(f"Could not parse prediction input {path}: {exc}") from exc
    else:
        raise DataValidationError(f"Could not decode prediction input {path}.")
    if frame.empty:
        raise DataValidationError(f"Prediction input contains no rows: {path}")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def predict_file(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    data_dir: Path,
) -> pd.DataFrame:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\nRun: python -m src.train --data-dir \"<dataset>\""
        )
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise DataValidationError(f"Unexpected model bundle structure: {model_path}")
    model = bundle["model"]
    metadata = bundle["metadata"]
    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise DataValidationError(
            "The model was trained with a different feature schema. Retrain it with the current code."
        )

    input_frame = _read_input(input_path)
    tables = load_raw_tables(data_dir)
    enriched = prepare_inference_data(
        input_frame,
        tables.customers,
        tables.stations,
        tables.districts,
    )
    predictions = np.asarray(model.predict(enriched[FEATURE_COLUMNS]), dtype=float)
    if not np.isfinite(predictions).all():
        raise DataValidationError("The model produced non-finite predictions. Check input values.")
    if (predictions < 0).any():
        LOGGER.warning("%d prediction(s) are negative; inspect out-of-range inputs.", int((predictions < 0).sum()))

    output = enriched[["session_id", "customer_id", "station_id", "session_start_time", "cost_per_kwh"]].copy()
    output["predicted_kwh"] = predictions
    output["predicted_total_cost"] = predictions * output["cost_per_kwh"].to_numpy(dtype=float)
    output["model"] = metadata.get("selected_label", "n/a")
    output["target_unit"] = metadata.get("target_unit", "kWh")

    if TARGET in input_frame.columns:
        actual = pd.to_numeric(input_frame[TARGET], errors="coerce").to_numpy(dtype=float)
        if len(actual) == len(predictions) and np.isfinite(actual).any():
            valid = np.isfinite(actual)
            metrics = regression_metrics(actual[valid], predictions[valid])
            LOGGER.info(
                "Optional scoring against supplied %s: MAE=%.4f RMSE=%.4f R2=%s",
                TARGET,
                metrics["mae"],
                metrics["rmse"],
                f"{metrics['r2']:.4f}" if metrics["r2"] is not None else "n/a",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")
    LOGGER.info("Wrote %d predictions to %s", len(output), output_path)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict session energy for new records with the saved Warsaw EV model."
    )
    add_common_arguments(parser)
    parser.add_argument("--input", type=Path, required=True, help="CSV with new session records.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <reports-dir>/new_predictions.csv).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to the saved model bundle (default: <model-dir>/best_model.joblib).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from .utils import configure_logging

    configure_logging()
    args = parse_args(argv)
    try:
        config = load_config(args)
        model_path = args.model or config.model_path
        if not model_path.is_absolute():
            model_path = Path.cwd() / model_path
        output_path = args.output or (config.reports_dir / "new_predictions.csv")
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output = predict_file(
            input_path=args.input,
            output_path=output_path,
            model_path=model_path,
            data_dir=config.data_dir,
        )
    except (ConfigError, DataValidationError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        return 2

    print("\nPrediction complete")
    print(f"Rows written : {len(output):,}")
    print(f"Output file  : {output_path}")
    print(f"Mean predicted: {output['predicted_kwh'].mean():.3f} kWh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
