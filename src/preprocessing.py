"""Cleaning, feature engineering and leakage-safe preprocessing.

The model predicts energy delivered during one charging session. All learned
imputers/encoders/scalers live inside a scikit-learn pipeline, so they are fit
only on training data (or train+validation for the final refit).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import DataValidationError, RawTables

LOGGER = logging.getLogger(__name__)

TARGET = "kwh_charged"
TIME_COLUMN = "session_start_time"
ID_COLUMNS = ("session_id", "customer_id", "station_id")
LEAKAGE_COLUMNS = ("total_cost",)

# Month is deliberately excluded: a single chronological holdout falls in an
# unseen month, and 89 days are insufficient to estimate stable seasonality.
NUMERIC_FEATURES = [
    "battery_capacity_kwh",
    "cost_per_kwh",
    "plugs_count",
    "latitude",
    "longitude",
    "projected_evs",
    "projected_plugs",
    "projected_regular_customers",
    "evs_per_projected_plug",
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]
CATEGORICAL_FEATURES = [
    "customer_income_tier",
    "car_model",
    "station_income_tier",
    "operator_name",
    "district_name",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODEL_INPUT_COLUMNS = set(FEATURE_COLUMNS)

INFERENCE_INPUT_COLUMNS = [
    "customer_id",
    "station_id",
    "session_start_time",
    "cost_per_kwh",
]


@dataclass
class SplitInfo:
    train_rows: int
    validation_rows: int
    test_rows: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    train_dates: int
    validation_dates: int
    test_dates: int
    validation_fraction_actual: float
    test_fraction_actual: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "test_rows": self.test_rows,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_dates": self.train_dates,
            "validation_dates": self.validation_dates,
            "test_dates": self.test_dates,
            "validation_fraction_actual": self.validation_fraction_actual,
            "test_fraction_actual": self.test_fraction_actual,
        }


def _clean_text(series: pd.Series) -> pd.Series:
    values = series.astype("object").where(pd.notna(series), np.nan)
    return values.map(lambda value: value.strip() if isinstance(value, str) else value)


def _summary(series: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"count": 0}
    quantiles = clean.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else 0.0,
        "min": float(clean.min()),
        "p01": float(quantiles.loc[0.01]),
        "p05": float(quantiles.loc[0.05]),
        "p25": float(quantiles.loc[0.25]),
        "median": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(clean.max()),
    }


def _coerce_numeric(frame: pd.DataFrame, columns: list[str], table_name: str) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _prepare_lookups(tables: RawTables, profile: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customers = tables.customers.copy()
    stations = tables.stations.copy()
    districts = tables.districts.copy()

    customers["customer_id"] = _clean_text(customers["customer_id"])
    stations["station_id"] = pd.to_numeric(stations["station_id"], errors="coerce")
    districts["district_name"] = _clean_text(districts["district_name"])

    customers = _coerce_numeric(customers, ["battery_capacity_kwh"], "customers")
    stations = _coerce_numeric(stations, ["plugs_count", "latitude", "longitude"], "stations")
    districts = _coerce_numeric(
        districts,
        ["projected_evs", "projected_plugs", "projected_regular_customers"],
        "districts",
    )

    profile["lookup_duplicate_ids"] = {
        "customers": int(customers["customer_id"].duplicated().sum()),
        "stations": int(stations["station_id"].duplicated().sum()),
        "districts": int(districts["district_name"].duplicated().sum()),
    }
    customers = customers.drop_duplicates("customer_id", keep="first")
    stations = stations.drop_duplicates("station_id", keep="first")
    districts = districts.drop_duplicates("district_name", keep="first")

    customers["customer_income_tier"] = _clean_text(customers["income_tier"])
    customers["car_model"] = _clean_text(customers["car_model"])
    stations["station_income_tier"] = _clean_text(stations["income_tier"])
    stations["operator_name"] = _clean_text(stations["operator_name"])
    districts["district_name"] = _clean_text(districts["district_name"])

    customers = customers[["customer_id", "customer_income_tier", "car_model", "battery_capacity_kwh"]]
    stations = stations[
        [
            "station_id",
            "district_name",
            "station_income_tier",
            "operator_name",
            "plugs_count",
            "latitude",
            "longitude",
        ]
    ]
    districts = districts[
        [
            "district_name",
            "projected_evs",
            "projected_plugs",
            "projected_regular_customers",
        ]
    ]
    return customers, stations, districts


def clean_and_join(tables: RawTables) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean source tables, validate relationships and join usable attributes."""

    profile: dict[str, Any] = {
        "source_files": tables.file_info,
        "fingerprint": tables.fingerprint,
        "raw_sessions_rows": int(len(tables.sessions)),
        "raw_customers_rows": int(len(tables.customers)),
        "raw_stations_rows": int(len(tables.stations)),
        "raw_districts_rows": int(len(tables.districts)),
    }
    dropped = {
        "invalid_start_time": 0,
        "missing_identifiers": 0,
        "exact_duplicate_rows": 0,
        "duplicate_session_id": 0,
        "duplicate_customer_start_time": 0,
        "invalid_target": 0,
    }

    sessions = tables.sessions.copy()
    for column in ("session_id", "customer_id"):
        sessions[column] = _clean_text(sessions[column])
    sessions["station_id"] = pd.to_numeric(sessions["station_id"], errors="coerce")
    sessions = _coerce_numeric(sessions, ["kwh_charged", "cost_per_kwh", "total_cost"], "sessions")
    sessions[TIME_COLUMN] = pd.to_datetime(sessions[TIME_COLUMN], errors="coerce", format="mixed")

    # Cost identity is useful as a data-quality check, not as a model feature.
    cost_mask = sessions["total_cost"].notna() & sessions[TARGET].notna() & sessions["cost_per_kwh"].notna()
    profile["cost_identity_max_abs_error"] = (
        float((sessions.loc[cost_mask, "total_cost"] - sessions.loc[cost_mask, TARGET] * sessions.loc[cost_mask, "cost_per_kwh"]).abs().max())
        if cost_mask.any()
        else None
    )

    dropped["invalid_start_time"] = int(sessions[TIME_COLUMN].isna().sum())
    sessions = sessions[sessions[TIME_COLUMN].notna()]

    dropped["missing_identifiers"] = int(
        (sessions["customer_id"].isna() | sessions["station_id"].isna()).sum()
    )
    sessions = sessions[sessions["customer_id"].notna() & sessions["station_id"].notna()]

    dropped["exact_duplicate_rows"] = int(sessions.duplicated().sum())
    sessions = sessions.drop_duplicates()

    dropped["duplicate_session_id"] = int(sessions["session_id"].duplicated().sum())
    sessions = sessions.drop_duplicates("session_id", keep="first")

    # The same customer cannot begin two charging sessions at two stations in
    # the same minute. Four such contradictory pairs exist; keep the first.
    dropped["duplicate_customer_start_time"] = int(
        sessions.duplicated(["customer_id", TIME_COLUMN], keep="first").sum()
    )
    sessions = sessions.drop_duplicates(["customer_id", TIME_COLUMN], keep="first")

    invalid_target = (
        sessions[TARGET].isna()
        | ~np.isfinite(sessions[TARGET])
        | (sessions[TARGET] <= 0)
    )
    dropped["invalid_target"] = int(invalid_target.sum())
    sessions = sessions[~invalid_target].copy()

    profile["duplicate_station_start_time"] = int(
        sessions.duplicated(["station_id", TIME_COLUMN], keep=False).sum()
    )
    profile["dropped_rows"] = dropped

    invalid_price = ~np.isfinite(sessions["cost_per_kwh"]) | (sessions["cost_per_kwh"] <= 0)
    profile["invalid_price_rows"] = int(invalid_price.sum())
    sessions.loc[invalid_price, "cost_per_kwh"] = np.nan

    customers, stations, districts = _prepare_lookups(tables, profile)

    customer_ids = set(customers["customer_id"].dropna())
    station_ids = set(stations["station_id"].dropna())
    profile["referential_coverage"] = {
        "customer_sessions_matched": int(sessions["customer_id"].isin(customer_ids).sum()),
        "customer_sessions_unmatched": int((~sessions["customer_id"].isin(customer_ids)).sum()),
        "station_sessions_matched": int(sessions["station_id"].isin(station_ids).sum()),
        "station_sessions_unmatched": int((~sessions["station_id"].isin(station_ids)).sum()),
    }

    merged = sessions.merge(customers, on="customer_id", how="left", validate="many_to_one")
    merged = merged.merge(stations, on="station_id", how="left", validate="many_to_one")
    merged = merged.merge(districts, on="district_name", how="left", validate="many_to_one")

    merged["district_name"] = _clean_text(merged["district_name"])
    merged["car_model"] = _clean_text(merged["car_model"])
    merged["customer_income_tier"] = _clean_text(merged["customer_income_tier"])
    merged["station_income_tier"] = _clean_text(merged["station_income_tier"])
    merged["operator_name"] = _clean_text(merged["operator_name"])

    invalid_battery = merged["battery_capacity_kwh"].notna() & (merged["battery_capacity_kwh"] <= 0)
    profile["invalid_battery_rows"] = int(invalid_battery.sum())
    merged.loc[invalid_battery, "battery_capacity_kwh"] = np.nan

    invalid_coordinates = (
        merged["latitude"].notna()
        & ((merged["latitude"] < -90) | (merged["latitude"] > 90))
    ) | (
        merged["longitude"].notna()
        & ((merged["longitude"] < -180) | (merged["longitude"] > 180))
    )
    profile["invalid_coordinate_rows"] = int(invalid_coordinates.sum())
    merged.loc[invalid_coordinates, ["latitude", "longitude"]] = np.nan

    projection_columns = ["projected_evs", "projected_plugs", "projected_regular_customers"]
    negative_projection = (merged[projection_columns] < 0).any(axis=1)
    profile["negative_projection_rows"] = int(negative_projection.sum())
    merged.loc[negative_projection, projection_columns] = np.nan

    target = merged[TARGET]
    q1, q3 = float(target.quantile(0.25)), float(target.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    ratio = merged[TARGET] / merged["battery_capacity_kwh"].where(merged["battery_capacity_kwh"] > 0)
    profile["target_summary"] = _summary(target)
    profile["target_iqr"] = {
        "q1": q1,
        "q3": q3,
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": int(((target < lower) | (target > upper)).sum()),
    }
    profile["ratio_to_battery"] = _summary(ratio)
    profile["physical_ratio"] = {
        "below_0_499": int((ratio < 0.499).sum()),
        "above_0_801": int((ratio > 0.801).sum()),
        "above_battery": int((merged[TARGET] > merged["battery_capacity_kwh"]).sum()),
        "min": float(ratio.min()) if ratio.notna().any() else None,
        "max": float(ratio.max()) if ratio.notna().any() else None,
    }

    profile["date_min"] = merged[TIME_COLUMN].min()
    profile["date_max"] = merged[TIME_COLUMN].max()
    profile["span_days"] = int((merged[TIME_COLUMN].max() - merged[TIME_COLUMN].min()).days)
    profile["unique_customers"] = int(merged["customer_id"].nunique())
    profile["unique_stations"] = int(merged["station_id"].nunique())
    profile["unique_districts"] = int(merged["district_name"].nunique())
    profile["sessions_per_customer"] = _summary(merged.groupby("customer_id").size())
    profile["sessions_per_station"] = _summary(merged.groupby("station_id").size())
    profile["price_values"] = {
        f"{value:.6f}": int(count)
        for value, count in merged["cost_per_kwh"].value_counts().sort_index().items()
        if pd.notna(value)
    }
    profile["category_counts"] = {
        "customer_income_tier": merged["customer_income_tier"].value_counts(dropna=False).to_dict(),
        "car_model": merged["car_model"].value_counts(dropna=False).to_dict(),
        "station_income_tier": merged["station_income_tier"].value_counts(dropna=False).to_dict(),
        "operator_name": merged["operator_name"].value_counts(dropna=False).to_dict(),
        "district_name": merged["district_name"].value_counts(dropna=False).to_dict(),
    }
    profile["modeling_rows"] = int(len(merged))
    profile["missing_feature_values"] = {
        column: int(merged[column].isna().sum())
        for column in FEATURE_COLUMNS
        if column in merged.columns and merged[column].isna().any()
    }
    return merged, profile


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic time and district features (no target usage)."""

    engineered = frame.copy()
    timestamps = pd.to_datetime(engineered[TIME_COLUMN], errors="coerce")
    hour = timestamps.dt.hour.astype("float64")
    day_of_week = timestamps.dt.dayofweek.astype("float64")

    engineered["hour"] = hour
    engineered["day_of_week"] = day_of_week
    engineered["is_weekend"] = (day_of_week >= 5).astype("float64")
    engineered["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    engineered["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    engineered["dow_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    engineered["dow_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

    plugs = pd.to_numeric(engineered["projected_plugs"], errors="coerce")
    engineered["evs_per_projected_plug"] = (
        pd.to_numeric(engineered["projected_evs"], errors="coerce") / plugs.where(plugs > 0)
    )

    for column in NUMERIC_FEATURES:
        engineered[column] = pd.to_numeric(engineered[column], errors="coerce").astype("float64")
    for column in CATEGORICAL_FEATURES:
        engineered[column] = _clean_text(engineered[column])

    return engineered


def prepare_modeling_data(tables: RawTables) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return one row per usable session plus a JSON-safe data profile."""

    frame, profile = clean_and_join(tables)
    frame = engineer_features(frame)

    missing_features = sorted(MODEL_INPUT_COLUMNS - set(frame.columns))
    if missing_features:
        raise DataValidationError(f"Feature engineering did not create: {missing_features}")

    leakage = sorted(MODEL_INPUT_COLUMNS & {TARGET, TIME_COLUMN, *ID_COLUMNS, *LEAKAGE_COLUMNS})
    if leakage:
        raise DataValidationError(f"Target/identifier leakage detected in feature set: {leakage}")
    if not frame[TARGET].notna().all():
        raise DataValidationError("The cleaned modeling frame contains missing target values.")

    profile["feature_columns"] = FEATURE_COLUMNS
    profile["numeric_features"] = NUMERIC_FEATURES
    profile["categorical_features"] = CATEGORICAL_FEATURES
    profile["final_missing_feature_values"] = int(frame[FEATURE_COLUMNS].isna().sum().sum())
    profile["modeling_rows"] = int(len(frame))
    return frame.reset_index(drop=True), profile


def split_dataset(
    frame: pd.DataFrame,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> tuple[dict[str, pd.DataFrame], SplitInfo]:
    """Split by whole calendar days in chronological order.

    The test period is strictly later than validation, which is strictly later
    than training. No future information enters model fitting or selection.
    """

    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be between 0 and 1.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1.")

    dates = pd.DatetimeIndex(sorted(frame[TIME_COLUMN].dt.normalize().dropna().unique()))
    if len(dates) < 5:
        raise DataValidationError("At least five distinct session dates are required for a chronological split.")

    n_test = max(1, int(round(len(dates) * test_fraction)))
    n_validation = max(1, int(round(len(dates) * validation_fraction)))
    if len(dates) - n_test - n_validation < 2:
        raise DataValidationError("Not enough distinct dates for non-empty train/validation/test periods.")

    validation_start = dates[-(n_test + n_validation)]
    test_start = dates[-n_test]

    train = frame[frame[TIME_COLUMN] < validation_start].copy()
    validation = frame[
        (frame[TIME_COLUMN] >= validation_start) & (frame[TIME_COLUMN] < test_start)
    ].copy()
    test = frame[frame[TIME_COLUMN] >= test_start].copy()

    if train.empty or validation.empty or test.empty:
        raise DataValidationError("Chronological split produced an empty partition.")
    if train[TIME_COLUMN].max() >= validation[TIME_COLUMN].min():
        raise DataValidationError("Training period overlaps validation period.")
    if validation[TIME_COLUMN].max() >= test[TIME_COLUMN].min():
        raise DataValidationError("Validation period overlaps test period.")

    total = len(frame)
    info = SplitInfo(
        train_rows=int(len(train)),
        validation_rows=int(len(validation)),
        test_rows=int(len(test)),
        train_start=str(train[TIME_COLUMN].min().date()),
        train_end=str(train[TIME_COLUMN].max().date()),
        validation_start=str(validation[TIME_COLUMN].min().date()),
        validation_end=str(validation[TIME_COLUMN].max().date()),
        test_start=str(test[TIME_COLUMN].min().date()),
        test_end=str(test[TIME_COLUMN].max().date()),
        train_dates=int(train[TIME_COLUMN].dt.normalize().nunique()),
        validation_dates=int(validation[TIME_COLUMN].dt.normalize().nunique()),
        test_dates=int(test[TIME_COLUMN].dt.normalize().nunique()),
        validation_fraction_actual=float(len(validation) / total),
        test_fraction_actual=float(len(test) / total),
    )
    LOGGER.info(
        "Chronological split: train=%s..%s (%d), validation=%s..%s (%d), test=%s..%s (%d)",
        info.train_start,
        info.train_end,
        info.train_rows,
        info.validation_start,
        info.validation_end,
        info.validation_rows,
        info.test_start,
        info.test_end,
        info.test_rows,
    )
    return {"train": train, "validation": validation, "test": test}, info


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build an unfitted, train-only preprocessing transformer."""

    numeric = numeric_features or NUMERIC_FEATURES
    categorical = categorical_features or CATEGORICAL_FEATURES
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_inference_data(
    sessions: pd.DataFrame,
    customers: pd.DataFrame,
    stations: pd.DataFrame,
    districts: pd.DataFrame,
) -> pd.DataFrame:
    """Enrich new sessions for prediction and fail clearly on bad input."""

    required = set(INFERENCE_INPUT_COLUMNS)
    missing = sorted(required - set(sessions.columns))
    if missing:
        raise DataValidationError(
            f"Inference input is missing required columns: {missing}. "
            f"Expected at least: {INFERENCE_INPUT_COLUMNS} (session_id is optional)."
        )

    frame = sessions.copy()
    if "session_id" not in frame:
        frame["session_id"] = [f"row_{index + 1}" for index in range(len(frame))]
    frame["session_id"] = _clean_text(frame["session_id"])
    frame["customer_id"] = _clean_text(frame["customer_id"])
    frame["station_id"] = pd.to_numeric(frame["station_id"], errors="coerce")
    frame["cost_per_kwh"] = pd.to_numeric(frame["cost_per_kwh"], errors="coerce")
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], errors="coerce", format="mixed")

    invalid = (
        frame["customer_id"].isna()
        | frame["station_id"].isna()
        | frame[TIME_COLUMN].isna()
        | frame["cost_per_kwh"].isna()
        | (frame["cost_per_kwh"] <= 0)
    )
    if invalid.any():
        examples = frame.loc[invalid, ["session_id", "customer_id", "station_id", TIME_COLUMN]].head(5).to_dict("records")
        raise DataValidationError(
            f"{int(invalid.sum())} inference row(s) have invalid customer_id, station_id, "
            f"session_start_time or cost_per_kwh. Examples: {examples}"
        )

    lookup_customers = customers.copy()
    lookup_customers["customer_id"] = _clean_text(lookup_customers["customer_id"])
    lookup_customers["customer_income_tier"] = _clean_text(lookup_customers["income_tier"])
    lookup_customers["car_model"] = _clean_text(lookup_customers["car_model"])
    lookup_customers["battery_capacity_kwh"] = pd.to_numeric(
        lookup_customers["battery_capacity_kwh"], errors="coerce"
    )
    lookup_customers = lookup_customers.drop_duplicates("customer_id", keep="first")[
        ["customer_id", "customer_income_tier", "car_model", "battery_capacity_kwh"]
    ]

    lookup_stations = stations.copy()
    lookup_stations["station_id"] = pd.to_numeric(lookup_stations["station_id"], errors="coerce")
    lookup_stations["district_name"] = _clean_text(lookup_stations["district_name"])
    lookup_stations["station_income_tier"] = _clean_text(lookup_stations["income_tier"])
    lookup_stations["operator_name"] = _clean_text(lookup_stations["operator_name"])
    lookup_stations["plugs_count"] = pd.to_numeric(lookup_stations["plugs_count"], errors="coerce")
    lookup_stations["latitude"] = pd.to_numeric(lookup_stations["latitude"], errors="coerce")
    lookup_stations["longitude"] = pd.to_numeric(lookup_stations["longitude"], errors="coerce")
    lookup_stations = lookup_stations.drop_duplicates("station_id", keep="first")[
        [
            "station_id",
            "district_name",
            "station_income_tier",
            "operator_name",
            "plugs_count",
            "latitude",
            "longitude",
        ]
    ]

    lookup_districts = districts.copy()
    lookup_districts["district_name"] = _clean_text(lookup_districts["district_name"])
    for column in ("projected_evs", "projected_plugs", "projected_regular_customers"):
        lookup_districts[column] = pd.to_numeric(lookup_districts[column], errors="coerce")
    lookup_districts = lookup_districts.drop_duplicates("district_name", keep="first")[
        ["district_name", "projected_evs", "projected_plugs", "projected_regular_customers"]
    ]

    missing_customers = sorted(set(frame["customer_id"]) - set(lookup_customers["customer_id"]))
    missing_stations = sorted(set(frame["station_id"].dropna()) - set(lookup_stations["station_id"].dropna()))
    if missing_customers or missing_stations:
        raise DataValidationError(
            "Inference data reference unknown customer or station IDs. "
            f"Unknown customers: {missing_customers[:10]} ({len(missing_customers)} total); "
            f"unknown stations: {missing_stations[:10]} ({len(missing_stations)} total). "
            "Update the lookup CSVs or correct the input."
        )

    input_columns = ["session_id", "customer_id", "station_id", TIME_COLUMN, "cost_per_kwh"]
    enriched = frame[input_columns].merge(
        lookup_customers, on="customer_id", how="left", validate="many_to_one"
    )
    enriched = enriched.merge(lookup_stations, on="station_id", how="left", validate="many_to_one")
    enriched = enriched.merge(lookup_districts, on="district_name", how="left", validate="many_to_one")
    enriched = engineer_features(enriched)

    missing_features = sorted(MODEL_INPUT_COLUMNS - set(enriched.columns))
    if missing_features:
        raise DataValidationError(f"Inference feature preparation is incomplete: {missing_features}")
    return enriched.reset_index(drop=True)
