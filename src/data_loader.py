"""Dataset discovery, CSV loading and schema validation."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

FILES = {
    "charging_sessions.csv": {
        "session_id",
        "customer_id",
        "station_id",
        "session_start_time",
        "kwh_charged",
        "cost_per_kwh",
        "total_cost",
    },
    "charging_stations.csv": {
        "station_id",
        "district_name",
        "income_tier",
        "operator_name",
        "plugs_count",
        "latitude",
        "longitude",
    },
    "customers.csv": {
        "customer_id",
        "income_tier",
        "car_model",
        "battery_capacity_kwh",
    },
    "districts.csv": {
        "district_name",
        "income_tier",
        "projected_evs",
        "projected_plugs",
        "projected_regular_customers",
    },
}


class DataValidationError(ValueError):
    """Raised when source data do not match the expected schema."""


@dataclass
class RawTables:
    sessions: pd.DataFrame
    stations: pd.DataFrame
    customers: pd.DataFrame
    districts: pd.DataFrame
    file_info: dict[str, dict]
    fingerprint: str


def _read_csv_with_encoding(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1250", "cp1252"):
        try:
            frame = pd.read_csv(path, encoding=encoding, low_memory=False)
            LOGGER.info("Loaded %s with encoding %s (%d rows x %d columns)", path.name, encoding, *frame.shape)
            return frame
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError as exc:
            raise DataValidationError(f"{path.name} is empty.") from exc
        except pd.errors.ParserError as exc:
            raise DataValidationError(f"Could not parse {path.name}: {exc}") from exc
    raise DataValidationError(f"Could not decode {path.name} as UTF-8 or Windows Central European text.") from last_error


def _validate_schema(frame: pd.DataFrame, filename: str) -> pd.DataFrame:
    if frame.empty:
        raise DataValidationError(f"{filename} contains no data rows.")

    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DataValidationError(f"{filename} has duplicate column names: {duplicates}")

    required = FILES[filename]
    missing = sorted(required - set(frame.columns))
    if missing:
        raise DataValidationError(
            f"{filename} is missing required columns: {missing}. "
            f"Found columns: {list(frame.columns)}"
        )
    extra = sorted(set(frame.columns) - required)
    if extra:
        LOGGER.warning("%s contains extra columns that will be ignored: %s", filename, extra)
    return frame


def _file_info(frame: pd.DataFrame, path: Path) -> dict:
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "column_names": list(frame.columns),
        "size_bytes": int(path.stat().st_size),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
    }


def dataset_fingerprint(data_dir: Path) -> str:
    """Hash the four source files so a model can be tied to its training data."""

    digest = hashlib.sha256()
    for filename in sorted(FILES):
        path = data_dir / filename
        digest.update(filename.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def load_raw_tables(data_dir: Path) -> RawTables:
    """Load and validate all source CSVs without modifying them."""

    if not data_dir.exists():
        raise DataValidationError(
            f"Dataset directory not found: {data_dir}\n"
            "Pass --data-dir, set WARSAW_EV_DATA_DIR, or edit configs/config.json."
        )
    if not data_dir.is_dir():
        raise DataValidationError(f"Dataset path is not a directory: {data_dir}")

    missing_files = [name for name in FILES if not (data_dir / name).is_file()]
    if missing_files:
        raise DataValidationError(
            f"Missing required dataset file(s) in {data_dir}: {missing_files}. "
            f"Expected files: {sorted(FILES)}"
        )

    frames: dict[str, pd.DataFrame] = {}
    file_info: dict[str, dict] = {}
    for filename in FILES:
        path = data_dir / filename
        frame = _read_csv_with_encoding(path)
        frame = _validate_schema(frame, filename)
        frames[filename] = frame
        file_info[filename] = _file_info(frame, path)

    fingerprint = dataset_fingerprint(data_dir)
    LOGGER.info("Dataset fingerprint: %s", fingerprint[:16])
    return RawTables(
        sessions=frames["charging_sessions.csv"],
        stations=frames["charging_stations.csv"],
        customers=frames["customers.csv"],
        districts=frames["districts.csv"],
        file_info=file_info,
        fingerprint=fingerprint,
    )
