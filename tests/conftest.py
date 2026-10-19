"""Synthetic fixtures for unit tests. These never feed the final evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import RawTables


@pytest.fixture
def synthetic_tables() -> RawTables:
    rng = np.random.default_rng(7)
    districts = pd.DataFrame(
        {
            "district_name": ["Śródmieście", "Mokotów"],
            "income_tier": ["High", "Mid-Range"],
            "projected_evs": [2700, 3500],
            "projected_plugs": [82, 106],
            "projected_regular_customers": [540, 700],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(1, 7)],
            "income_tier": ["High", "Mid-Range", "Low-Mid", "High", "Mid-Range", "Low-Mid"],
            "car_model": [
                "Tesla Model 3",
                "Hyundai Kona Electric",
                "Dacia Spring",
                "Audi e-tron",
                "Skoda Enyaq iV",
                "Fiat 500e",
            ],
            "battery_capacity_kwh": [75.0, 64.0, 26.8, 95.0, 77.0, 42.0],
        }
    )
    stations = pd.DataFrame(
        {
            "station_id": [1, 2, 3],
            "district_name": ["Śródmieście", "Mokotów", "Mokotów"],
            "income_tier": ["High", "Mid-Range", "Mid-Range"],
            "operator_name": ["GreenWay", "Tauron", "Orlen Charge"],
            "plugs_count": [6, 6, 6],
            "latitude": [52.23, 52.19, 52.20],
            "longitude": [21.01, 21.00, 20.98],
        }
    )

    rows = []
    prices = [1.8513967524, 2.7045201370, 3.2561989272]
    session_number = 1
    for day in range(30):
        for slot, hour in enumerate((7, 9, 12, 18)):
            customer_index = (day * 4 + slot) % len(customers)
            station_index = (day + slot) % len(stations)
            battery = float(customers.loc[customer_index, "battery_capacity_kwh"])
            kwh = battery * rng.uniform(0.5, 0.8)
            price = prices[(day + slot) % len(prices)]
            start = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day, hours=hour, minutes=slot * 5)
            rows.append(
                {
                    "session_id": f"S{session_number:05d}",
                    "customer_id": customers.loc[customer_index, "customer_id"],
                    "station_id": int(stations.loc[station_index, "station_id"]),
                    "session_start_time": start.strftime("%Y-%m-%d %H:%M:%S"),
                    "kwh_charged": kwh,
                    "cost_per_kwh": price,
                    "total_cost": kwh * price,
                }
            )
            session_number += 1

    sessions = pd.DataFrame(rows)
    file_info = {
        "charging_sessions.csv": {
            "rows": len(sessions),
            "columns": len(sessions.columns),
            "column_names": list(sessions.columns),
            "size_bytes": 1000,
            "missing_cells": 0,
            "duplicate_rows": 0,
        },
        "charging_stations.csv": {
            "rows": len(stations),
            "columns": len(stations.columns),
            "column_names": list(stations.columns),
            "size_bytes": 500,
            "missing_cells": 0,
            "duplicate_rows": 0,
        },
        "customers.csv": {
            "rows": len(customers),
            "columns": len(customers.columns),
            "column_names": list(customers.columns),
            "size_bytes": 500,
            "missing_cells": 0,
            "duplicate_rows": 0,
        },
        "districts.csv": {
            "rows": len(districts),
            "columns": len(districts.columns),
            "column_names": list(districts.columns),
            "size_bytes": 300,
            "missing_cells": 0,
            "duplicate_rows": 0,
        },
    }
    return RawTables(
        sessions=sessions,
        stations=stations,
        customers=customers,
        districts=districts,
        file_info=file_info,
        fingerprint="synthetic-test-fingerprint",
    )
