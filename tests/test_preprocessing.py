"""Tests for loading, cleaning, leakage guards, splits and inference preparation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataValidationError, load_raw_tables
from src.preprocessing import (
    FEATURE_COLUMNS,
    ID_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET,
    TIME_COLUMN,
    build_preprocessor,
    prepare_inference_data,
    prepare_modeling_data,
    split_dataset,
)


def _write_tables(tables, directory) -> None:
    tables.sessions.to_csv(directory / "charging_sessions.csv", index=False, encoding="utf-8-sig")
    tables.stations.to_csv(directory / "charging_stations.csv", index=False, encoding="utf-8-sig")
    tables.customers.to_csv(directory / "customers.csv", index=False, encoding="utf-8-sig")
    tables.districts.to_csv(directory / "districts.csv", index=False, encoding="utf-8-sig")


def test_load_raw_tables_and_schema_error(tmp_path, synthetic_tables):
    _write_tables(synthetic_tables, tmp_path)
    loaded = load_raw_tables(tmp_path)
    assert len(loaded.sessions) == len(synthetic_tables.sessions)
    assert loaded.fingerprint and len(loaded.fingerprint) == 64
    assert "Śródmieście" in set(loaded.stations.district_name)

    broken = synthetic_tables.sessions.drop(columns=["kwh_charged"])
    broken.to_csv(tmp_path / "charging_sessions.csv", index=False, encoding="utf-8-sig")
    with pytest.raises(DataValidationError, match="missing required columns"):
        load_raw_tables(tmp_path)


def test_missing_dataset_file_raises(tmp_path, synthetic_tables):
    _write_tables(synthetic_tables, tmp_path)
    (tmp_path / "districts.csv").unlink()
    with pytest.raises(DataValidationError, match="Missing required dataset file"):
        load_raw_tables(tmp_path)


def test_clean_removes_duplicates_and_invalid_targets(synthetic_tables):
    sessions = synthetic_tables.sessions.copy()
    exact_duplicate = sessions.iloc[[0]].copy()
    sessions = pd.concat([sessions, exact_duplicate], ignore_index=True)

    invalid_target = sessions.iloc[[0]].copy()
    invalid_target["session_id"] = "SINVALID"
    invalid_target["session_start_time"] = "2025-12-31 20:00:00"
    invalid_target["kwh_charged"] = -1.0
    invalid_target["total_cost"] = -1.0
    sessions = pd.concat([sessions, invalid_target], ignore_index=True)

    repeated_customer = sessions.iloc[[0]].copy()
    repeated_customer["session_id"] = "SREPEAT"
    repeated_customer["station_id"] = 2
    sessions = pd.concat([sessions, repeated_customer], ignore_index=True)

    modified = replace(synthetic_tables, sessions=sessions)
    frame, profile = prepare_modeling_data(modified)

    assert profile["dropped_rows"]["exact_duplicate_rows"] == 1
    assert profile["dropped_rows"]["invalid_target"] == 1
    assert profile["dropped_rows"]["duplicate_customer_start_time"] == 1
    assert len(frame) == len(synthetic_tables.sessions)
    assert (frame[TARGET] > 0).all()


def test_feature_engineering_has_no_leakage(synthetic_tables):
    frame, _ = prepare_modeling_data(synthetic_tables)
    forbidden = {TARGET, TIME_COLUMN, *ID_COLUMNS, *LEAKAGE_COLUMNS}
    assert not (set(FEATURE_COLUMNS) & forbidden)
    assert frame["hour"].between(0, 23).all()
    assert frame["day_of_week"].between(0, 6).all()
    assert frame["is_weekend"].isin([0.0, 1.0]).all()
    assert frame["evs_per_projected_plug"].notna().all()


def test_chronological_split_is_ordered(synthetic_tables):
    frame, _ = prepare_modeling_data(synthetic_tables)
    partitions, info = split_dataset(frame, validation_fraction=0.15, test_fraction=0.15)
    train, validation, test = partitions["train"], partitions["validation"], partitions["test"]

    assert train[TIME_COLUMN].max() < validation[TIME_COLUMN].min()
    assert validation[TIME_COLUMN].max() < test[TIME_COLUMN].min()
    assert len(train) + len(validation) + len(test) == len(frame)
    assert info.train_rows == len(train)
    assert info.test_rows == len(test)
    assert info.test_fraction_actual == pytest.approx(len(test) / len(frame))


def test_preprocessor_handles_missing_and_unknown_categories(synthetic_tables):
    frame, _ = prepare_modeling_data(synthetic_tables)
    partitions, _ = split_dataset(frame)
    train = partitions["train"].copy()
    validation = partitions["validation"].copy()

    train.loc[train.index[0], "battery_capacity_kwh"] = np.nan
    train.loc[train.index[1], "car_model"] = np.nan
    validation.loc[validation.index[0], "car_model"] = "Brand New EV"

    preprocessor = build_preprocessor()
    transformed_train = preprocessor.fit_transform(train[FEATURE_COLUMNS])
    transformed_validation = preprocessor.transform(validation[FEATURE_COLUMNS])

    assert transformed_train.shape[0] == len(train)
    assert transformed_validation.shape[0] == len(validation)
    assert np.isfinite(transformed_train).all()
    assert np.isfinite(transformed_validation).all()


def test_inference_enrichment_and_unknown_ids(synthetic_tables):
    source = synthetic_tables.sessions.head(3)[
        ["session_id", "customer_id", "station_id", "session_start_time", "cost_per_kwh"]
    ].copy()
    source["kwh_charged"] = [999.0, 999.0, 999.0]
    enriched = prepare_inference_data(
        source,
        synthetic_tables.customers,
        synthetic_tables.stations,
        synthetic_tables.districts,
    )
    assert set(FEATURE_COLUMNS).issubset(enriched.columns)
    assert TARGET not in enriched.columns
    assert enriched["car_model"].notna().all()

    bad = source.copy()
    bad.loc[bad.index[0], "customer_id"] = "C9999"
    with pytest.raises(DataValidationError, match="unknown customer"):
        prepare_inference_data(
            bad,
            synthetic_tables.customers,
            synthetic_tables.stations,
            synthetic_tables.districts,
        )
