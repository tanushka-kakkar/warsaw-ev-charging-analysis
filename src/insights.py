"""Generate an EV-equivalent insights and recommendations report.

The assignment brief for this capstone uses airline wording ("reduce delays,
optimize operations, enhance customer experience"). The dataset and executed
analysis are for Warsaw EV charging, so this module treats the airline wording
as a template and maps each clause to its EV equivalent. No airline finding is
invented.

Inputs:
    - the four Warsaw EV CSV files (read-only)
    - reports/model_metrics.json, when available

Outputs:
    - reports/insights_metrics.json
    - reports/insights_report.md
    - reports/insights_report.html (self-contained, offline)
    - reports/figures/insights_*.png
"""

from __future__ import annotations

import argparse
import base64
import html
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ConfigError, add_common_arguments, load_config
from .data_loader import DataValidationError, RawTables, load_raw_tables
from .reporting import (
    _html_table,
    _image_data_uri,
    _inline_html,
    _integer,
    _md_table,
    _num,
    _pct,
    _slug,
)
from .utils import configure_logging, write_json

LOGGER = logging.getLogger(__name__)

PALETTE = {
    "navy": "#0f2740",
    "blue": "#1f5f8b",
    "teal": "#2c7a7b",
    "green": "#2f855a",
    "orange": "#d97706",
    "red": "#b91c1c",
    "muted": "#94a3b8",
    "grid": "#dbe4ee",
    "soft_blue": "#eef6fc",
    "soft_green": "#eefaf3",
    "soft_orange": "#fff7ed",
}


@dataclass(frozen=True)
class ReportPaths:
    metrics: Path
    markdown: Path
    html: Path
    figures: list[Path]


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------


def _load_ml_metrics(reports_dir: Path) -> dict[str, Any]:
    path = reports_dir / "model_metrics.json"
    if not path.exists():
        LOGGER.warning("ML metrics not found at %s; model section will be marked pending.", path)
        return {"available": False}
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    test = payload.get("test_metrics") or {}
    return {
        "available": bool(test),
        "selected_label": payload.get("selected_label"),
        "target": payload.get("target"),
        "test": test,
        "validation": payload.get("validation_metrics_selected") or {},
        "baseline": payload.get("baseline_test_metrics") or {},
        "heuristic": payload.get("heuristic_test_metrics") or {},
        "improvement_vs_dummy": payload.get("improvement_vs_dummy") or {},
        "station_day": payload.get("station_day_metrics") or {},
        "district_day": payload.get("district_day_metrics") or {},
        "feature_importance": payload.get("feature_importance") or [],
        "data_fingerprint": payload.get("data_fingerprint"),
        "generated_utc": payload.get("generated_utc"),
    }


def _monthly_daily_average(frame: pd.DataFrame) -> list[dict[str, Any]]:
    working = frame.copy()
    working["month"] = working["session_start_time"].dt.to_period("M").astype(str)
    grouped = working.groupby("month").agg(
        sessions=("session_id", "count"),
        active_days=("session_date", "nunique"),
    )
    grouped["sessions_per_day"] = grouped["sessions"] / grouped["active_days"]
    return [
        {
            "month": str(index),
            "sessions": int(row.sessions),
            "active_days": int(row.active_days),
            "sessions_per_day": float(row.sessions_per_day),
        }
        for index, row in grouped.iterrows()
    ]


def _weekday_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grouped = frame.groupby("day_of_week").size()
    total = int(grouped.sum())
    return [
        {
            "day_of_week": names[int(day)],
            "sessions": int(grouped.get(day, 0)),
            "share_pct": float(grouped.get(day, 0) / total * 100.0),
        }
        for day in range(7)
    ]


def _hourly_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = frame.groupby("hour").size()
    total = int(grouped.sum())
    peak_hours = list(range(8, 16))
    return [
        {
            "hour": int(hour),
            "sessions": int(grouped.get(hour, 0)),
            "share_pct": float(grouped.get(hour, 0) / total * 100.0),
            "is_peak": hour in peak_hours,
        }
        for hour in range(24)
    ]


def _district_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        frame.groupby("district_name")
        .agg(
            sessions=("session_id", "count"),
            kwh=("kwh_charged", "sum"),
            revenue=("total_cost", "sum"),
            stations=("station_id", "nunique"),
            customers=("customer_id", "nunique"),
        )
        .reset_index()
    )
    total_sessions = grouped.sessions.sum()
    total_kwh = grouped.kwh.sum()
    total_revenue = grouped.revenue.sum()
    grouped["session_share_pct"] = grouped.sessions / total_sessions * 100.0
    grouped["kwh_share_pct"] = grouped.kwh / total_kwh * 100.0
    grouped["revenue_share_pct"] = grouped.revenue / total_revenue * 100.0
    grouped = grouped.sort_values("kwh", ascending=False).reset_index(drop=True)
    grouped["rank"] = grouped.index + 1
    return grouped.to_dict(orient="records")


def _planning_summary(
    stations: pd.DataFrame,
    districts: pd.DataFrame,
    district_energy: pd.DataFrame,
) -> list[dict[str, Any]]:
    available = (
        stations.groupby("district_name")
        .agg(available_plugs=("plugs_count", "sum"), stations=("station_id", "nunique"))
        .reset_index()
    )
    energy = district_energy[["district_name", "kwh"]].rename(columns={"kwh": "actual_kwh"})
    planning = districts.merge(available, on="district_name", how="left").merge(
        energy, on="district_name", how="left"
    )
    planning["available_plugs"] = planning["available_plugs"].fillna(0).astype(int)
    planning["stations"] = planning["stations"].fillna(0).astype(int)
    planning["actual_kwh"] = planning["actual_kwh"].fillna(0.0)
    planning["coverage_pct"] = planning["available_plugs"] / planning["projected_plugs"] * 100.0
    planning["evs_per_plug"] = np.where(
        planning["available_plugs"] > 0,
        planning["projected_evs"] / planning["available_plugs"],
        np.nan,
    )
    planning["kwh_per_projected_ev"] = planning["actual_kwh"] / planning["projected_evs"]
    planning = planning.sort_values(
        ["coverage_pct", "kwh_per_projected_ev"], ascending=[False, True]
    ).reset_index(drop=True)
    planning["readiness_rank"] = planning.index + 1
    records = planning.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                record[key] = None
            elif isinstance(value, (np.integer,)):
                record[key] = int(value)
            elif isinstance(value, (np.floating,)):
                record[key] = float(value)
    return records


def _station_summary(frame: pd.DataFrame, station_count: int) -> dict[str, Any]:
    grouped = (
        frame.groupby(["station_id", "district_name", "operator_name"])
        .size()
        .reset_index(name="sessions")
        .sort_values(["sessions", "station_id"], ascending=[False, True])
    )
    total_sessions = int(grouped.sessions.sum())
    mean_sessions = total_sessions / station_count if station_count else None
    top = grouped.head(8).to_dict(orient="records")
    busiest = float(grouped.sessions.iloc[0]) if not grouped.empty else None
    return {
        "station_count": int(station_count),
        "mean_sessions": float(mean_sessions) if mean_sessions else None,
        "busiest_sessions": busiest,
        "busiest_ratio": float(busiest / mean_sessions) if busiest and mean_sessions else None,
        "top": top,
    }


def _customer_tier_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        frame.groupby("income_tier")
        .agg(
            customers=("customer_id", "nunique"),
            sessions=("session_id", "count"),
            avg_cost=("total_cost", "mean"),
            avg_price=("cost_per_kwh", "mean"),
            kwh_per_session=("kwh_charged", "mean"),
            avg_battery=("battery_capacity_kwh", "mean"),
        )
        .reset_index()
    )
    grouped["sessions_per_customer"] = grouped.sessions / grouped.customers
    grouped = grouped.sort_values("avg_cost", ascending=False).reset_index(drop=True)
    return grouped.to_dict(orient="records")


def _operator_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = (
        frame.groupby("operator_name")
        .agg(
            stations=("station_id", "nunique"),
            sessions=("session_id", "count"),
            revenue=("total_cost", "sum"),
            kwh=("kwh_charged", "sum"),
        )
        .reset_index()
    )
    grouped["revenue_per_station"] = grouped.revenue / grouped.stations
    grouped["sessions_per_station"] = grouped.sessions / grouped.stations
    grouped = grouped.sort_values("revenue_per_station", ascending=False).reset_index(drop=True)
    return grouped.to_dict(orient="records")


def _price_summary(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame["cost_per_kwh"].round(6).value_counts().sort_index()
    total = int(counts.sum())
    weighted_mean = float(frame["total_cost"].sum() / frame["kwh_charged"].sum())
    return {
        "points": [
            {"price": float(price), "sessions": int(count), "share_pct": float(count / total * 100.0)}
            for price, count in counts.items()
        ],
        "weighted_mean": weighted_mean,
        "min": float(frame["cost_per_kwh"].min()),
        "max": float(frame["cost_per_kwh"].max()),
        "std": float(frame["cost_per_kwh"].std()),
    }


def _concentration_summary(frame: pd.DataFrame) -> dict[str, Any]:
    per_customer = (
        frame.groupby("customer_id")
        .agg(kwh=("kwh_charged", "sum"), sessions=("session_id", "count"))
        .sort_values("kwh", ascending=False)
    )
    total_kwh = float(per_customer.kwh.sum())
    top10 = per_customer.head(10)
    return {
        "customer_count": int(len(per_customer)),
        "total_kwh": total_kwh,
        "top10_kwh": float(top10.kwh.sum()),
        "top10_share_pct": float(top10.kwh.sum() / total_kwh * 100.0),
        "top10_uniform_share_pct": float(10 / len(per_customer) * 100.0),
        "max_customer_kwh": float(per_customer.kwh.max()),
        "max_customer_id": str(per_customer.kwh.idxmax()),
    }


def compute_insights(tables: RawTables, ml_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute the data-backed insight payload from the raw CSV tables."""

    sessions = tables.sessions.copy()
    stations = tables.stations.copy()
    customers = tables.customers.copy()
    districts = tables.districts.copy()

    sessions["session_start_time"] = pd.to_datetime(sessions["session_start_time"], format="mixed")
    sessions["session_date"] = sessions["session_start_time"].dt.date
    sessions["hour"] = sessions["session_start_time"].dt.hour
    sessions["day_of_week"] = sessions["session_start_time"].dt.dayofweek
    for column in ("kwh_charged", "total_cost", "cost_per_kwh"):
        sessions[column] = pd.to_numeric(sessions[column], errors="coerce")

    enriched = (
        sessions.merge(
            stations[["station_id", "district_name", "operator_name", "plugs_count"]],
            on="station_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            customers[["customer_id", "income_tier", "battery_capacity_kwh", "car_model"]],
            on="customer_id",
            how="left",
            validate="many_to_one",
        )
    )

    total_sessions = int(len(sessions))
    total_kwh = float(sessions.kwh_charged.sum())
    total_revenue = float(sessions.total_cost.sum())
    daily = (
        sessions.groupby("session_date")
        .size()
        .reset_index(name="sessions")
        .sort_values("session_date")
    )
    daily_records = [
        {"date": str(row.session_date), "sessions": int(row.sessions)}
        for row in daily.itertuples()
    ]
    hourly = _hourly_summary(sessions)
    weekday = _weekday_summary(sessions)
    monthly = _monthly_daily_average(sessions)
    district_records = _district_summary(enriched)
    district_energy = pd.DataFrame(district_records)
    planning = _planning_summary(stations, districts, district_energy)

    station_summary = _station_summary(enriched, int(stations.station_id.nunique()))
    tier_records = _customer_tier_summary(enriched)
    operator_records = _operator_summary(enriched)
    price = _price_summary(sessions)
    concentration = _concentration_summary(sessions)
    district_price = enriched.groupby("district_name")["cost_per_kwh"].mean()
    operator_price = enriched.groupby("operator_name")["cost_per_kwh"].mean()

    peak_share = sum(item["share_pct"] for item in hourly if item["is_peak"])
    overnight_hours = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7]
    overnight_share = sum(item["share_pct"] for item in hourly if item["hour"] in overnight_hours)
    served_districts = sorted(stations.district_name.unique())
    unserved = [record for record in planning if record["available_plugs"] == 0]
    unserved = sorted(unserved, key=lambda record: record["projected_evs"], reverse=True)
    unserved_evs = int(sum(record["projected_evs"] for record in unserved))
    unserved_planned_plugs = int(sum(record["projected_plugs"] for record in unserved))
    total_projected_evs = int(districts.projected_evs.sum())
    total_planned_plugs = int(districts.projected_plugs.sum())
    total_available_plugs = int(stations.plugs_count.sum())
    coverage_pct = total_available_plugs / total_planned_plugs * 100.0
    top3_kwh_share = float(
        sum(record["kwh_share_pct"] for record in district_records[:3])
    )
    top3_revenue_share = float(
        sum(record["revenue_share_pct"] for record in district_records[:3])
    )
    top3_session_share = float(
        sum(record["session_share_pct"] for record in district_records[:3])
    )
    wola = next((record for record in planning if record["district_name"] == "Wola"), {})
    wesola = next((record for record in planning if record["district_name"] == "Wesoła"), {})
    weekday_max = max(weekday, key=lambda item: item["sessions"])
    weekday_min = min(weekday, key=lambda item: item["sessions"])
    weekday_spread = (
        weekday_max["sessions"] / weekday_min["sessions"] - 1.0
    ) * 100.0
    peak_per_hour = peak_share / 8.0
    overnight_per_hour = overnight_share / 12.0
    daily_mean = float(daily.sessions.mean())
    month_min = min(monthly, key=lambda item: item["sessions_per_day"])
    month_max = max(monthly, key=lambda item: item["sessions_per_day"])
    price_std = float(sessions.cost_per_kwh.std())
    quality = {
        "constant_plugs": bool(stations.plugs_count.nunique() == 1),
        "plugs_count": int(stations.plugs_count.iloc[0]) if stations.plugs_count.nunique() == 1 else None,
        "unserved_districts": len(unserved),
        "districts_total": int(districts.district_name.nunique()),
        "partial_months": [
            {"month": item["month"], "active_days": item["active_days"]}
            for item in monthly
        ],
        "duplicate_customer_start_minutes": int(
            sessions.duplicated(subset=["customer_id", "session_start_time"], keep="first").sum()
        ),
        "same_station_start_minutes": int(
            sessions.duplicated(subset=["station_id", "session_start_time"], keep="first").sum()
        ),
        "missing_cells": int(sessions[["customer_id", "station_id", "session_start_time", "kwh_charged", "cost_per_kwh", "total_cost"]].isna().sum().sum()),
        "synthetic_signal": (
            "All charged energy sits between 50% and 80% of the vehicle battery capacity "
            "and every vehicle model has a mean ratio near 0.65."
        ),
        "income_tier_columns": [
            "customers.income_tier",
            "charging_stations.income_tier",
            "districts.income_tier",
        ],
    }

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": "Warsaw EV Charging - SQL, visual, and ML insights",
        "data_fingerprint": tables.fingerprint,
        "source_rows": {
            "sessions": total_sessions,
            "stations": int(stations.station_id.nunique()),
            "customers": int(customers.customer_id.nunique()),
            "districts": int(districts.district_name.nunique()),
            "date_start": str(sessions.session_start_time.min().date()),
            "date_end": str(sessions.session_start_time.max().date()),
            "span_days": int(
                (sessions.session_start_time.max().date() - sessions.session_start_time.min().date()).days + 1
            ),
        },
        "usage": {
            "total_sessions": total_sessions,
            "total_kwh": total_kwh,
            "mean_kwh": float(sessions.kwh_charged.mean()),
            "min_kwh": float(sessions.kwh_charged.min()),
            "max_kwh": float(sessions.kwh_charged.max()),
            "sessions_per_customer": float(total_sessions / customers.customer_id.nunique()),
            "sessions_per_station": float(total_sessions / stations.station_id.nunique()),
            "sessions_per_day": daily_mean,
            "daily_min": int(daily.sessions.min()),
            "daily_max": int(daily.sessions.max()),
            "peak_share_pct": float(peak_share),
            "peak_per_hour_pct": float(peak_per_hour),
            "overnight_share_pct": float(overnight_share),
            "overnight_per_hour_pct": float(overnight_per_hour),
            "weekday_spread_pct": float(weekday_spread),
            "weekday_max": weekday_max,
            "weekday_min": weekday_min,
            "monthly": monthly,
            "monthly_min": month_min,
            "monthly_max": month_max,
            "hourly": hourly,
            "daily": daily_records,
            "weekday": weekday,
        },
        "revenue": {
            "total_revenue": total_revenue,
            "avg_session_revenue": float(sessions.total_cost.mean()),
            "revenue_per_station": float(total_revenue / stations.station_id.nunique()),
            "price": price,
            "price_std": price_std,
            "district_price_min": float(district_price.min()),
            "district_price_max": float(district_price.max()),
            "district_price_spread_pct": float(
                (district_price.max() / district_price.min() - 1.0) * 100.0
            ),
            "operator_price_spread_pct": float(
                (operator_price.max() / operator_price.min() - 1.0) * 100.0
            ),
            "top3_district_revenue_share_pct": top3_revenue_share,
            "top3_district_kwh_share_pct": top3_kwh_share,
            "top3_district_session_share_pct": top3_session_share,
            "operators": operator_records,
        },
        "districts": district_records,
        "planning": planning,
        "capacity": {
            "served_districts": len(served_districts),
            "served_district_names": served_districts,
            "unserved_districts": len(unserved),
            "unserved_records": unserved,
            "unserved_projected_evs": unserved_evs,
            "unserved_planned_plugs": unserved_planned_plugs,
            "total_projected_evs": total_projected_evs,
            "total_planned_plugs": total_planned_plugs,
            "total_available_plugs": total_available_plugs,
            "plug_plan_coverage_pct": float(coverage_pct),
            "wola": wola,
            "wesola": wesola,
        },
        "stations": station_summary,
        "customers": {
            "tiers": tier_records,
            "concentration": concentration,
        },
        "quality": quality,
        "ml": ml_metrics or {"available": False},
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _style_axes(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=13, color=PALETTE["navy"], fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=PALETTE["navy"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=PALETTE["navy"])
    ax.tick_params(colors=PALETTE["navy"], labelsize=9)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PALETTE["grid"])


def _create_coverage_figure(insights: dict[str, Any], figures_dir: Path) -> Path:
    planning = sorted(insights["planning"], key=lambda item: item["coverage_pct"])
    names = [record["district_name"] for record in planning]
    coverage = [record["coverage_pct"] for record in planning]
    projected = [record["projected_evs"] for record in planning]
    colors = [
        PALETTE["muted"] if record["available_plugs"] == 0 else PALETTE["teal"]
        for record in planning
    ]

    figure, ax = plt.subplots(figsize=(9.6, 7.2))
    bars = ax.barh(names, coverage, color=colors, height=0.66)
    ax.axvline(100.0, color=PALETTE["orange"], linestyle="--", linewidth=1.2)
    ax.text(
        100.0,
        len(names) - 0.2,
        " 100% = planned plugs built",
        color=PALETTE["orange"],
        fontsize=9,
        va="top",
    )
    for bar, value, evs in zip(bars, coverage, projected):
        ax.text(
            bar.get_width() + 3,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.0f}%  ({evs:,} proj. EVs)",
            va="center",
            fontsize=8.5,
            color=PALETTE["navy"],
        )
    ax.set_xlim(0, max(coverage) * 1.38)
    _style_axes(ax, "Plug-plan coverage by district (available ÷ planned plugs)", xlabel="Coverage (%)")
    figure.tight_layout()
    path = figures_dir / "insights_coverage.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _create_hourly_figure(insights: dict[str, Any], figures_dir: Path) -> Path:
    hourly = insights["usage"]["hourly"]
    hours = [item["hour"] for item in hourly]
    sessions = [item["sessions"] for item in hourly]
    colors = [PALETTE["blue"] if item["is_peak"] else PALETTE["muted"] for item in hourly]

    figure, ax = plt.subplots(figsize=(10.4, 5.0))
    ax.bar(hours, sessions, color=colors, width=0.72)
    ax.axvspan(7.5, 15.5, color=PALETTE["soft_blue"], zorder=0)
    ax.text(
        11.5,
        max(sessions) * 1.03,
        f"08:00-15:59 peak = {insights['usage']['peak_share_pct']:.1f}% of sessions "
        f"({insights['usage']['peak_per_hour_pct']:.2f}%/hour)",
        ha="center",
        fontsize=9.5,
        color=PALETTE["blue"],
        fontweight="bold",
    )
    ax.text(
        2.0,
        max(sessions) * 0.86,
        f"Overnight (20:00-07:59)\n{insights['usage']['overnight_share_pct']:.1f}% of sessions\n"
        f"{insights['usage']['overnight_per_hour_pct']:.2f}%/hour",
        ha="center",
        fontsize=9,
        color=PALETTE["navy"],
    )
    ax.set_xticks(hours)
    _style_axes(ax, "Sessions by hour of day — demand is concentrated in business hours", ylabel="Sessions")
    figure.tight_layout()
    path = figures_dir / "insights_hourly_demand.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _create_daily_figure(insights: dict[str, Any], figures_dir: Path) -> Path:
    daily = pd.DataFrame(insights["usage"]["daily"])
    daily["date"] = pd.to_datetime(daily["date"])
    rolling = daily["sessions"].rolling(7, min_periods=1).mean()

    figure, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.bar(daily["date"], daily["sessions"], color=PALETTE["muted"], width=0.8, alpha=0.75, label="Daily sessions")
    ax.plot(daily["date"], rolling, color=PALETTE["blue"], linewidth=2.2, label="7-day rolling mean")
    ax.axhline(insights["usage"]["sessions_per_day"], color=PALETTE["orange"], linestyle="--", linewidth=1.2, label="Overall mean")
    monthly_text = " | ".join(
        f"{item['month']}: {item['sessions_per_day']:.0f}/day" for item in insights["usage"]["monthly"]
    )
    ax.text(
        0.01,
        0.97,
        monthly_text,
        transform=ax.transAxes,
        fontsize=8.6,
        va="top",
        color=PALETTE["navy"],
    )
    _style_axes(ax, "Daily sessions, 11 Aug – 8 Nov 2025 — no growth trend", ylabel="Sessions")
    ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=3)
    figure.tight_layout()
    path = figures_dir / "insights_daily_trend.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _create_district_figure(insights: dict[str, Any], figures_dir: Path) -> Path:
    records = insights["districts"]
    names = [record["district_name"] for record in records]
    kwh = [record["kwh"] / 1000.0 for record in records]
    share = [record["kwh_share_pct"] for record in records]
    colors = [PALETTE["teal"]] * 3 + [PALETTE["muted"]] * (len(records) - 3)

    figure, ax = plt.subplots(figsize=(9.6, 5.6))
    bars = ax.bar(names, kwh, color=colors, width=0.68)
    for bar, value, pct in zip(bars, kwh, share):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(kwh) * 0.015,
            f"{pct:.1f}%",
            ha="center",
            fontsize=8.5,
            color=PALETTE["navy"],
        )
    ax.text(
        0.99,
        0.97,
        f"Top 3 districts = {insights['revenue']['top3_district_kwh_share_pct']:.1f}% of all energy",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=PALETTE["teal"],
        fontweight="bold",
    )
    _style_axes(ax, "Energy delivered by district — concentrated core", ylabel="MWh")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    figure.tight_layout()
    path = figures_dir / "insights_district_concentration.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _create_customer_figure(insights: dict[str, Any], figures_dir: Path) -> Path:
    tiers = sorted(insights["customers"]["tiers"], key=lambda item: item["avg_cost"])
    names = [record["income_tier"] for record in tiers]
    kwh = [record["kwh_per_session"] for record in tiers]
    battery = [record["avg_battery"] for record in tiers]
    cost = [record["avg_cost"] for record in tiers]

    x = np.arange(len(names))
    width = 0.36
    figure, ax = plt.subplots(figsize=(9.2, 5.2))
    bars_kwh = ax.bar(x - width / 2, kwh, width, color=PALETTE["blue"], label="kWh per session")
    ax2 = ax.twinx()
    bars_battery = ax2.bar(x + width / 2, battery, width, color=PALETTE["teal"], alpha=0.7, label="Battery capacity (kWh)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{name}\navg cost {avg_cost:.0f}" for name, avg_cost in zip(names, cost)]
    )
    ax.set_ylabel("kWh per session", color=PALETTE["blue"])
    ax2.set_ylabel("Average battery capacity (kWh)", color=PALETTE["teal"])
    _style_axes(ax, "The income gap is a garage gap — not a price or frequency gap")
    for bar, value in zip(bars_kwh, kwh):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value:.1f}", ha="center", fontsize=9, color=PALETTE["blue"])
    for bar, value in zip(bars_battery, battery):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{value:.1f}", ha="center", fontsize=9, color=PALETTE["teal"])
    ax.set_ylim(0, max(max(kwh), max(battery)) * 1.18)
    ax2.set_ylim(0, max(max(kwh), max(battery)) * 1.18)
    ax2.grid(False)
    lines = [bars_kwh, bars_battery]
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=9, loc="upper left")
    figure.tight_layout()
    path = figures_dir / "insights_income_decomposition.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _create_readiness_figure(insights: dict[str, Any], figures_dir: Path) -> Path:
    records = sorted(insights["planning"], key=lambda record: record["kwh_per_projected_ev"])
    names = [record["district_name"] for record in records]
    values = [record["kwh_per_projected_ev"] for record in records]
    coverage = [record["coverage_pct"] for record in records]
    colors = [
        PALETTE["muted"]
        if record["available_plugs"] == 0
        else (PALETTE["teal"] if record["coverage_pct"] >= 100.0 else PALETTE["blue"])
        for record in records
    ]

    figure, ax = plt.subplots(figsize=(10.8, 7.0))
    bars = ax.barh(names, values, color=colors, height=0.66)
    city_average = insights["usage"]["total_kwh"] / insights["capacity"]["total_projected_evs"]
    ax.axvline(city_average, color=PALETTE["orange"], linestyle="--", linewidth=1.2)
    ax.set_ylim(-0.6, len(names) + 0.6)
    ax.text(
        city_average + 8,
        len(names) - 0.05,
        f"city average {city_average:.0f} kWh/EV",
        color=PALETTE["orange"],
        fontsize=9,
        va="bottom",
    )
    for bar, value, cov in zip(bars, values, coverage):
        if value > 0:
            label = f"{value:,.0f} kWh/EV · {cov:.0f}% coverage"
            color = PALETTE["navy"]
        else:
            label = "no stations · 0 kWh/EV"
            color = PALETTE["red"]
        ax.text(
            bar.get_width() + 8,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8.6,
            color=color,
        )
    ax.set_xlim(0, max(values) * 1.62)
    _style_axes(
        ax,
        "Realised load per projected EV and built coverage, by district",
        xlabel="Actual kWh per projected EV",
    )
    figure.tight_layout()
    path = figures_dir / "insights_readiness.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def create_figures(insights: dict[str, Any], figures_dir: Path) -> list[dict[str, str]]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    created: list[dict[str, str]] = []
    creators = [
        (_create_coverage_figure, "Plug-plan coverage by district: five districts at 0%, Wola at 12%."),
        (_create_hourly_figure, "Sessions by hour: half of all sessions start in the 08:00-15:59 window."),
        (_create_daily_figure, "Daily sessions: stable demand around 1,267 sessions per day with no growth trend."),
        (_create_district_figure, "Energy by district: the top three districts deliver 47.9% of all kWh."),
        (_create_customer_figure, "Customer income tiers: kWh per session and battery capacity move together."),
        (_create_readiness_figure, "Readiness chart: realised kWh per projected EV, with built coverage labels."),
    ]
    for creator, caption in creators:
        path = creator(insights, figures_dir)
        created.append({"file": f"figures/{path.name}", "caption": caption})
    return created


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _fmt_money(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.0f}"


def _fmt_kwh(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f} kWh"


def _sections(insights: dict[str, Any], figures: list[dict[str, str]]) -> list[dict[str, Any]]:
    usage = insights["usage"]
    revenue = insights["revenue"]
    capacity = insights["capacity"]
    stations = insights["stations"]
    customers = insights["customers"]
    concentration = customers["concentration"]
    quality = insights["quality"]
    ml = insights["ml"]
    price_points_text = ", ".join(
        f"{record['price']:.4f}" for record in revenue["price"]["points"]
    )

    def p(text: str) -> dict[str, str]:
        return {"type": "p", "text": text}

    def h3(text: str) -> dict[str, str]:
        return {"type": "h3", "text": text}

    def ul(items: list[str]) -> dict[str, Any]:
        return {"type": "ul", "items": items}

    def table(headers: list[str], rows: list[list[Any]], caption: str | None = None) -> dict[str, Any]:
        return {"type": "table", "headers": headers, "rows": rows, "caption": caption}

    def callout(kind: str, title: str, text: str) -> dict[str, str]:
        return {"type": "callout", "kind": kind, "title": title, "text": text}

    def figure_block(file: str, caption: str) -> dict[str, str]:
        return {"type": "figure", "file": file, "caption": caption}

    figure_map = {record["file"]: record["caption"] for record in figures}

    def fig(key: str) -> dict[str, str]:
        file = f"figures/{key}"
        return figure_block(file, figure_map.get(file, key))

    district_rows = [
        [
            record["rank"],
            record["district_name"],
            _integer(record["sessions"]),
            _num(record["kwh"], 0, comma=True),
            _fmt_money(record["revenue"]),
            _pct(record["kwh_share_pct"], 1),
        ]
        for record in insights["districts"]
    ]
    planning_rows = [
        [
            record["readiness_rank"],
            record["district_name"],
            _integer(record["projected_evs"]),
            _integer(record["available_plugs"]),
            _integer(record["projected_plugs"]),
            _pct(record["coverage_pct"], 1),
            _num(record["evs_per_plug"], 1) if record["evs_per_plug"] is not None else "undefined (0 plugs)",
            _num(record["kwh_per_projected_ev"], 1),
        ]
        for record in insights["planning"]
    ]
    tier_rows = [
        [
            record["income_tier"],
            _integer(record["customers"]),
            _integer(record["sessions"]),
            _fmt_money(record["avg_cost"]),
            _num(record["avg_price"], 4),
            _num(record["kwh_per_session"], 2),
            _num(record["avg_battery"], 1),
            _num(record["sessions_per_customer"], 2),
        ]
        for record in customers["tiers"]
    ]
    operator_rows = [
        [
            record["operator_name"],
            _integer(record["stations"]),
            _integer(record["sessions"]),
            _fmt_money(record["revenue"]),
            _fmt_money(record["revenue_per_station"]),
        ]
        for record in revenue["operators"]
    ]
    unserved_rows = [
        [
            record["district_name"],
            _integer(record["projected_evs"]),
            _integer(record["available_plugs"]),
            _integer(record["projected_plugs"]),
        ]
        for record in capacity["unserved_records"]
    ]
    station_rows = [
        [
            int(record["station_id"]),
            record["district_name"],
            record["operator_name"],
            _integer(record["sessions"]),
        ]
        for record in stations["top"]
    ]
    price_rows = [
        [f"{record['price']:.4f} PLN/kWh", _integer(record["sessions"]), _pct(record["share_pct"], 1)]
        for record in revenue["price"]["points"]
    ]
    weekday_rows = [
        [record["day_of_week"], _integer(record["sessions"]), _pct(record["share_pct"], 1)]
        for record in usage["weekday"]
    ]
    monthly_rows = [
        [
            record["month"],
            _integer(record["sessions"]),
            _integer(record["active_days"]),
            _num(record["sessions_per_day"], 1),
        ]
        for record in usage["monthly"]
    ]
    quality_rows = [
        ["Missing cells in session fields used here", _integer(quality["missing_cells"])],
        ["Constant plugs per station", f"{quality['plugs_count']} on every station" if quality["constant_plugs"] else "varies"],
        ["Districts with infrastructure", f"{insights['capacity']['served_districts']} of {quality['districts_total']}"],
        ["Repeated customer + start-minute rows flagged", _integer(quality["duplicate_customer_start_minutes"])],
        ["Same station + start-minute rows flagged", _integer(quality["same_station_start_minutes"])],
        ["Observed energy-to-battery rule", quality["synthetic_signal"]],
        ["Income-tier columns to disambiguate", " / ".join(quality["income_tier_columns"])],
        ["Partial months", ", ".join(f"{item['month']} ({item['active_days']} active days)" for item in quality["partial_months"][:1] + quality["partial_months"][-1:])],
    ]

    sections: list[dict[str, Any]] = []

    def add(section_id: str, title: str, blocks: list[dict[str, Any]]) -> None:
        del section_id  # anchors must match the rendered heading, so slug the title
        sections.append({"id": _slug(title), "title": title, "blocks": blocks})

    # 1. Scope
    add(
        "Scope and airline-equivalent mapping",
        "1. Scope and airline-equivalent mapping",
        [
            callout(
                "warning",
                "Assignment wording note",
                "The brief uses airline language: reduce delays, optimize operations, enhance customer "
                "experience. The dataset and executed SQL analysis are for Warsaw EV charging. No airline "
                "data is present here, so this report maps each airline clause to its EV equivalent and keeps "
                "every number evidence-based. Airline-specific figures are not invented.",
            ),
            table(
                ["Airline brief clause", "EV-equivalent interpretation", "Evidence base"],
                [
                    [
                        "Reduce delays",
                        "Reduce access gaps and peak-time congestion; shorten the distance and wait between "
                        "demand and available infrastructure.",
                        "Q2, Q3, Q7, Q11, Q13; hourly and coverage figures",
                    ],
                    [
                        "Optimize operations",
                        "Prioritise district-level siting, load management, operator performance and "
                        "capacity sequencing instead of station-level add-ons.",
                        "Q2-Q8, Q11-Q14; district, station and operator tables",
                    ],
                    [
                        "Enhance customer experience",
                        "Improve coverage, availability, transparent flat pricing and off-peak options; "
                        "instrument waits, occupancy and state-of-charge to measure experience directly.",
                        "Q4, Q5, Q9, Q10; customer and usage tables",
                    ],
                ],
                "How the airline wording is answered",
            ),
            p(
                "The report has three evidence layers: (1) 14 SQL queries executed against MySQL 8.0.46 "
                "and reproduced in `part1_sql_analysis.md`; (2) a Power BI build specification with four "
                "pages and 24 visuals (`part2_powerbi_build_spec.md`); and (3) the session-energy ML model "
                "and its chronological holdout evaluation in this repository. The six figures in this report "
                "were generated directly from the same 114,000-session source CSVs so the SQL findings are "
                "visible in one place."
            ),
        ],
    )

    # 2. Executive summary
    add(
        "Executive summary",
        "2. Executive summary",
        [
            callout(
                "actual",
                "What the evidence says",
                f"{insights['source_rows']['sessions']:,} sessions delivered "
                f"{_num(usage['total_kwh'], 0, comma=True)} kWh and "
                f"{_num(revenue['total_revenue'], 0, comma=True)} PLN over "
                f"{insights['source_rows']['span_days']} days. The network is dense in a few districts and "
                "empty in five. Demand is stable and concentrated in business hours, individual stations are "
                "remarkably uniform, and the tariff is flat. The main growth levers are coverage and "
                "utilisation, not pricing.",
            ),
            ul(
                [
                    f"**Coverage is the largest gap.** Only {capacity['served_districts']} of "
                    f"{quality['districts_total']} districts have a station; the five unserved districts hold "
                    f"{_integer(capacity['unserved_projected_evs'])} projected EVs ("
                    f"{_pct(capacity['unserved_projected_evs'] / capacity['total_projected_evs'] * 100.0, 1)} "
                    f"of the city plan) and {_integer(capacity['unserved_planned_plugs'])} planned plugs.",
                    f"**Built coverage is 36.8%.** {_integer(capacity['total_available_plugs'])} of "
                    f"{_integer(capacity['total_planned_plugs'])} planned plugs exist. Wola is the worst "
                    f"served district at {_pct(capacity['wola']['coverage_pct'], 1)} coverage and "
                    f"{_num(capacity['wola']['evs_per_plug'], 0)} projected EVs per plug.",
                    f"**Demand is flat, not growing.** The network averages "
                    f"{_num(usage['sessions_per_day'], 0)} sessions/day; the strongest month was "
                    f"{usage['monthly_max']['month']} at {_num(usage['monthly_max']['sessions_per_day'], 0)}/day "
                    f"and the weakest was {usage['monthly_min']['month']} at "
                    f"{_num(usage['monthly_min']['sessions_per_day'], 0)}/day.",
                    f"**Demand is a business-hours phenomenon.** {_pct(usage['peak_share_pct'], 1)} of sessions "
                    f"start between 08:00 and 15:59 ({_num(usage['peak_per_hour_pct'], 2)}%/hour) versus "
                    f"{_pct(usage['overnight_share_pct'], 1)} overnight ({_num(usage['overnight_per_hour_pct'], 2)}%/hour).",
                    f"**There is no station hotspot.** The busiest station has only "
                    f"{_num(stations['busiest_ratio'], 2)}× the network average "
                    f"({_integer(stations['busiest_sessions'])} vs {_num(stations['mean_sessions'], 0)} sessions).",
                    f"**Price is not a lever.** District average price is effectively uniform at "
                    f"{_num(revenue['price']['weighted_mean'], 4)} PLN/kWh "
                    f"({_num(revenue['district_price_spread_pct'], 2)}% spread between the highest- and "
                    f"lowest-priced districts; operator spread under 1%). District revenue rank is "
                    f"identical to volume rank.",
                    f"**The income gap is a vehicle gap.** High-tier customers pay 1.89× Low-Mid per session, "
                    f"but sessions per customer are identical across tiers (about "
                    f"{_num(np.mean([tier['sessions_per_customer'] for tier in customers['tiers']]), 2)}) and "
                    f"average price per kWh is uniform at "
                    f"{_num(revenue['price']['weighted_mean'], 4)}.",
                    "**The ML model predicts session energy, not arrivals.** It reaches test RMSE "
                    f"{_num(ml.get('test', {}).get('rmse'), 3)} kWh and R² "
                    f"{_num(ml.get('test', {}).get('r2'), 4)}, but a one-feature battery benchmark ties it. "
                    "Use it for conditional energy planning, not demand forecasting.",
                ]
            ),
        ],
    )

    # 3. Evidence base
    source = insights["source_rows"]
    add(
        "Evidence base and method",
        "3. Evidence base and method",
        [
            h3("SQL analysis"),
            p(
                "Fourteen SQL queries were executed against the `warsaw_ev` MySQL 8.0.46 schema. The queries "
                "cover district inventory, station and plug counts, session and energy volumes, station "
                "utilisation, revenue, customer tiers, customer concentration, projected EV-per-plug ratios, "
                "planning shortfalls, demand intensity and a documented readiness rank. Results are reproduced "
                "verbatim in `part1_sql_analysis.md` with raw output in `sql/query_outputs.txt`."
            ),
            table(
                ["Evidence item", "Value"],
                [
                    ["Sessions analysed", _integer(source["sessions"])],
                    ["Stations", _integer(source["stations"])],
                    ["Customers", _integer(source["customers"])],
                    ["Districts", _integer(source["districts"])],
                    ["Date window", f"{source['date_start']} to {source['date_end']} ({source['span_days']} days)"],
                    ["Energy delivered", _fmt_kwh(usage["total_kwh"], 0)],
                    ["Billed revenue", f"{_num(revenue['total_revenue'], 2, comma=True)} PLN"],
                    ["Mean session", f"{_num(usage['mean_kwh'], 4)} kWh / {_num(revenue['avg_session_revenue'], 2)} PLN"],
                    ["Data fingerprint", f"`{insights['data_fingerprint']}`"],
                ],
                "Analysed dataset",
            ),
            h3("Visualisations"),
            p(
                "The Power BI specification defines four pages and 24 visuals. The six charts in this report "
                "were generated from the same raw CSV rows and follow the SQL definitions: coverage by "
                "district, hourly demand, daily trend, district energy concentration, income decomposition "
                "and the readiness chart. The ML report in `reports/model_evaluation_report.md` supplies "
                "the separate model-diagnostic figures."
            ),
            h3("Predictive model"),
            p(
                "The ML pipeline predicts `kwh_charged` for a known session using vehicle, station, district, "
                "tariff and time features. It uses a chronological split, selects on validation RMSE and "
                "evaluates the held-out test period once. The model is not an arrival or utilisation "
                "forecast; it predicts energy conditional on a session that is already starting."
            ),
        ],
    )

    # 4. Network coverage
    add(
        "Finding 1: coverage is concentrated in a dense core",
        "4. Finding 1: coverage is concentrated in a dense core",
        [
            p(
                f"{capacity['served_districts']} of {quality['districts_total']} districts have at least one "
                f"charging station. Ursynów (12 stations) and Wesoła (11) alone hold 35% of the 65 stations, "
                f"and the top three districts hold 47.7% of all stations. The five unserved districts — "
                f"{', '.join(record['district_name'] for record in capacity['unserved_records'])} — hold "
                f"{_integer(capacity['unserved_projected_evs'])} projected EVs and "
                f"{_integer(capacity['unserved_planned_plugs'])} planned plugs."
            ),
            table(
                ["District (unserved)", "Projected EVs", "Plugs now", "Planned plugs"],
                unserved_rows,
                "Unserved districts and their planned capacity",
            ),
            callout(
                "note",
                "Capacity built vs planned",
                f"Across the city, {_integer(capacity['total_available_plugs'])} of "
                f"{_integer(capacity['total_planned_plugs'])} planned plugs are built = "
                f"{_pct(capacity['plug_plan_coverage_pct'], 2)}. The unserved five account for "
                f"{_integer(capacity['unserved_planned_plugs'])} of the unbuilt plans. `plugs_count` is a "
                "constant 6 for every station, so plug count is exactly station count × 6 and must not be "
                "presented as an independent metric.",
            ),
            fig("insights_coverage.png"),
        ],
    )

    # 5. Demand patterns
    add(
        "Finding 2: demand is flat and business-hours driven",
        "5. Finding 2: demand is flat and business-hours driven",
        [
            p(
                f"The network records {_integer(usage['total_sessions'])} sessions, "
                f"{_fmt_kwh(usage['total_kwh'], 0)} and a mean session of "
                f"{_num(usage['mean_kwh'], 2)} kWh (range {_num(usage['min_kwh'], 2)}–"
                f"{_num(usage['max_kwh'], 2)} kWh). The daily rate is stable: "
                f"{_num(usage['sessions_per_day'], 0)} sessions/day overall, with monthly averages between "
                f"{_num(usage['monthly_min']['sessions_per_day'], 0)} and "
                f"{_num(usage['monthly_max']['sessions_per_day'], 0)}."
            ),
            table(
                ["Month", "Sessions", "Active days", "Sessions per active day"],
                monthly_rows,
                "Monthly demand — August and November are partial months",
            ),
            p(
                f"Half of all sessions ({_pct(usage['peak_share_pct'], 1)}) start between 08:00 and 15:59, "
                f"averaging {_num(usage['peak_per_hour_pct'], 2)}% of daily sessions per hour. The overnight "
                f"window (20:00–07:59) carries {_pct(usage['overnight_share_pct'], 1)} of sessions, or "
                f"{_num(usage['overnight_per_hour_pct'], 2)}%/hour. Weekday variation is narrow: "
                f"{usage['weekday_max']['day_of_week']} { _integer(usage['weekday_max']['sessions']) } vs "
                f"{usage['weekday_min']['day_of_week']} { _integer(usage['weekday_min']['sessions']) } "
                f"({_pct(usage['weekday_spread_pct'], 1)} spread)."
            ),
            fig("insights_hourly_demand.png"),
            fig("insights_daily_trend.png"),
            callout(
                "actual",
                "Operational implication",
                "The recoverable operational lever is time of day, not day of week. Off-peak incentives and "
                "managed charging can address the 08:00–15:59 peak; day-of-week pricing would act on a spread "
                "of less than 10% and is unlikely to be worth the complexity.",
            ),
        ],
    )

    # 6. Revenue and pricing
    add(
        "Finding 3: revenue is a pure volume business",
        "6. Finding 3: revenue is a pure volume business",
        [
            p(
                f"Total billed revenue is {_num(revenue['total_revenue'], 2, comma=True)} PLN, an average of "
                f"{_num(revenue['avg_session_revenue'], 2)} PLN per session and "
                f"{_num(revenue['revenue_per_station'], 0, comma=True)} PLN per station. District revenue "
                f"rank is identical to session and energy rank because district average price is effectively "
                f"uniform ({_num(revenue['district_price_spread_pct'], 2)}% spread). The top three "
                f"districts account for {_pct(revenue['top3_district_kwh_share_pct'], 1)} of energy and "
                f"{_pct(revenue['top3_district_revenue_share_pct'], 1)} of revenue."
            ),
            table(
                ["District", "Sessions", "kWh", "Revenue (PLN)", "Energy share"],
                [
                    [record["district_name"], _integer(record["sessions"]), _num(record["kwh"], 0, comma=True), _fmt_money(record["revenue"]), _pct(record["kwh_share_pct"], 1)]
                    for record in insights["districts"]
                ],
                "District revenue follows volume exactly",
            ),
            table(
                ["Tariff (PLN/kWh)", "Sessions", "Share"],
                price_rows,
                "Three tariff points, no district variation",
            ),
            p(
                f"The three session-level price points are {price_points_text} PLN/kWh. They are applied "
                f"network-wide: district averages span {_num(revenue['district_price_min'], 3)}–"
                f"{_num(revenue['district_price_max'], 3)} PLN/kWh "
                f"({_num(revenue['district_price_spread_pct'], 2)}% spread), operator averages span under "
                f"1%, and the three income tiers average "
                f"{_num(revenue['price']['weighted_mean'], 4)} PLN/kWh. The price points are not tied to "
                "district, operator, tier or delivered energy (mean session size is 39.0–39.2 kWh under all "
                "three). Revenue cannot be grown by repricing an existing district or tier; growth must come "
                "from added volume or higher utilisation."
            ),
            fig("insights_district_concentration.png"),
        ],
    )

    # 7. Customers
    add(
        "Finding 4: the income gap is a vehicle gap, not a pricing gap",
        "7. Finding 4: the income gap is a vehicle gap, not a pricing gap",
        [
            p(
                "Average cost per session rises sharply with customer income tier, but the decomposition "
                "shows why: high-tier customers do not charge more often and do not pay a higher rate. They "
                "drive larger-battery vehicles, so each session delivers more kWh. This is a fleet effect, "
                "not evidence of price discrimination or of a willingness-to-pay premium."
            ),
            table(
                ["Income tier", "Customers", "Sessions", "Avg cost/session", "Avg price/kWh", "kWh/session", "Avg battery kWh", "Sessions/customer"],
                tier_rows,
                "Customer-tier decomposition",
            ),
            p(
                f"Consumption is also not concentrated: the top 10 of "
                f"{_integer(concentration['customer_count'])} customers (0.125% of the base) account for "
                f"{_num(concentration['top10_kwh'], 0, comma=True)} kWh, or "
                f"{_pct(concentration['top10_share_pct'], 2)} of all energy — only "
                f"{_num(concentration['top10_share_pct'] / concentration['top10_uniform_share_pct'], 2)}× "
                "their uniform share. There is no small whale segment to target; efficiency and experience "
                "programmes should be network-wide."
            ),
            fig("insights_income_decomposition.png"),
        ],
    )

    # 8. Planning
    add(
        "Finding 5: absolute shortfalls cannot prioritise, but ratios can",
        "8. Finding 5: absolute shortfalls cannot prioritise, but ratios can",
        [
            p(
                "Every district has more projected EVs than available plugs, so the raw shortfall test has "
                "no discriminating power. The useful planning lens is relative: built coverage against the "
                "district's own plan, realised kWh per projected EV, and projected EVs per plug. The table "
                "below uses the documented readiness definition: coverage descending, then demand intensity "
                "ascending, with a five-way tie at 0% coverage."
            ),
            table(
                ["Rank", "District", "Projected EVs", "Plugs now", "Planned plugs", "Coverage", "EVs/plug", "kWh per projected EV"],
                planning_rows,
                "Planning and readiness metrics",
            ),
            callout(
                "warning",
                "Read the projections as hypotheses, not measurements",
                "Wesoła ranks first on coverage (314%) only because its projected EV count (700) is far "
                f"below its realised load ({_num(capacity['wesola']['actual_kwh'], 0, comma=True)} kWh = "
                f"{_num(capacity['wesola']['kwh_per_projected_ev'], 1)} kWh per projected EV, 3.2× the next "
                "district). Re-base any projection that produces coverage above 100% before using it for "
                "investment sequencing. The five 0% districts are indistinguishable in the ranking; break "
                "ties by projected EVs, which puts Praga Południe first.",
            ),
            fig("insights_readiness.png"),
        ],
    )

    # 9. Operations
    add(
        "Finding 6: operations are uniform at station level",
        "9. Finding 6: operations are uniform at station level",
        [
            p(
                f"The busiest station records {_integer(stations['busiest_sessions'])} sessions versus a "
                f"network mean of {_num(stations['mean_sessions'], 1)} — only "
                f"{_num(stations['busiest_ratio'], 2)}× the average. The eight busiest stations sit between "
                f"{_integer(stations['top'][-1]['sessions'])} and {_integer(stations['top'][0]['sessions'])} "
                "sessions and are spread across Wesoła, Ursynów, Włochy and Rembertów. There is no overloaded "
                "site to relieve and no station-level hotspot story to tell."
            ),
            table(
                ["Station", "District", "Operator", "Sessions"],
                station_rows,
                "Busiest stations",
            ),
            p(
                "Operator productivity is similarly tight. Revenue per station spans only about 3.4% across "
                "the five operators, so there is no operator-level performance outlier in this dataset. "
                "Operational variance is a district-coverage phenomenon, not a station or operator problem."
            ),
            table(
                ["Operator", "Stations", "Sessions", "Revenue (PLN)", "Revenue per station"],
                operator_rows,
                "Operator comparison",
            ),
        ],
    )

    # 10. ML model
    ml_available = bool(ml.get("available"))
    if ml_available:
        test = ml.get("test") or {}
        baseline = ml.get("baseline") or {}
        heuristic = ml.get("heuristic") or {}
        station_day = ml.get("station_day") or {}
        top_features = sorted(
            ml.get("feature_importance") or [],
            key=lambda record: record.get("importance_mean", 0),
            reverse=True,
        )[:5]
        ml_blocks = [
            p(
                f"The current ML pipeline selects **{ml.get('selected_label')}** on validation and reports "
                f"held-out test MAE {_num(test.get('mae'), 3)} kWh, RMSE {_num(test.get('rmse'), 3)} kWh, "
                f"R² {_num(test.get('r2'), 4)} and MAPE {_pct(test.get('mape_pct'), 2)} on "
                f"{_integer(test.get('n'))} chronological holdout sessions. The dummy-mean baseline RMSE is "
                f"{_num(baseline.get('rmse'), 3)} kWh."
            ),
            table(
                ["Model", "MAE (kWh)", "RMSE (kWh)", "R²", "MAPE"],
                [
                    ["Selected model", _num(test.get("mae"), 3), _num(test.get("rmse"), 3), _num(test.get("r2"), 4), _pct(test.get("mape_pct"), 2)],
                    ["Dummy mean baseline", _num(baseline.get("mae"), 3), _num(baseline.get("rmse"), 3), _num(baseline.get("r2"), 4), _pct(baseline.get("mape_pct"), 2)],
                    ["Battery-capacity heuristic", _num(heuristic.get("mae"), 3), _num(heuristic.get("rmse"), 3), _num(heuristic.get("r2"), 4), _pct(heuristic.get("mape_pct"), 2)],
                ],
                "Held-out test metrics",
            ),
            p(
                "The model is best read as a session-energy estimator, not as a business forecasting system. "
                f"A transparent battery-capacity benchmark ties it in RMSE, and the top feature is "
                f"`{top_features[0]['feature'] if top_features else 'battery_capacity_kwh'}` "
                f"({_num(top_features[0]['importance_mean'], 2) if top_features else 'n/a'} kWh RMSE increase). "
                f"Station-day aggregation reaches MAPE {_pct(station_day.get('mape_pct'), 2)} conditional on "
                "knowing the day's session mix; it is not an arrival forecast."
            ),
            p(
                "For demand planning, pair this energy model with a separate arrival-count or utilisation "
                "model. Until that exists, use the model to size energy for a known session mix and to test "
                "vehicle-mix scenarios, not to forecast how many drivers will arrive."
            ),
        ]
    else:
        ml_blocks = [
            callout(
                "warning",
                "Predictive model pending",
                "`reports/model_metrics.json` was not available when this report was generated. Run "
                "`python -m src.train` and `python -m src.evaluate`, then regenerate this report.",
            )
        ]
    add(
        "Finding 7: the predictive model is an energy estimator, not a demand forecast",
        "10. Finding 7: the predictive model is an energy estimator, not a demand forecast",
        ml_blocks,
    )

    # 11. Data quality
    add(
        "Data quality and interpretation limits",
        "11. Data quality and interpretation limits",
        [
            table(
                ["Check", "Finding"],
                quality_rows,
                "Data-quality constraints behind the analysis",
            ),
            ul(
                [
                    f"**{quality['duplicate_customer_start_minutes']} repeated customer + start-minute rows** "
                    "were flagged in the raw sessions. Four contradictory records were removed for ML "
                    "training; the SQL analysis uses all 114,000 raw rows. Either way, the effect on district "
                    "and hourly conclusions is negligible.",
                    "**`plugs_count` is constant at 6.** 'Stations' and 'plugs' are the same measure at a "
                    "6× scale; do not treat them as independent KPIs.",
                    "**Five districts have no infrastructure.** Fact-based charts that join sessions to "
                    "districts silently omit 26.6% of projected EV growth. Axis such charts on the full "
                    "district dimension.",
                    "**Three `income_tier` columns exist** (customers, stations, districts). Station tier is "
                    "identical to district tier, and district tier is not the customer tier; every conclusion "
                    "must state which one it uses.",
                    "**Projected EV and plug counts are static planning inputs,** not observations. They are "
                    "perfectly collinear with district and cannot identify causal coefficients.",
                    "**August and November are partial months.** Month-over-month comparisons should use the "
                    "first full months or be annotated.",
                    "**No state of charge, duration, occupancy, queue, weather or traffic data** are present. "
                    "That limits causal claims and means wait-time and customer-experience effects must be "
                    "instrumented, not inferred.",
                    f"**Observed energy follows a regular rule:** {quality['synthetic_signal']} This is "
                    "characteristic of simulated or highly standardised data and should be validated on real "
                    "network data before operational deployment.",
                ]
            ),
        ],
    )

    # 12. Recommendations
    add(
        "Recommendations",
        "12. Actionable recommendations",
        [
            p(
                "Recommendations are ordered by evidence strength and impact. They are planning and "
                "instrumentation actions, not causal claims, and they explicitly note where the data cannot "
                "yet support a decision."
            ),
            h3("R1 — Close the five-district coverage gap, sequenced by projected demand"),
            ul(
                [
                    "Praga Południe (2,500 projected EVs, 76 planned plugs), then Białołęka (2,400 / 73), "
                    "Bielany (2,300 / 70), Wawer (1,100 / 33) and Praga Północ (1,000 / 30).",
                    "Treat the projected EV counts as planning hypotheses. Validate local registration, "
                    "commuting and land-use data before committing the full planned plug count.",
                    "Because there is no station-level hotspot, a small number of well-placed stations can "
                    "be phased in without redesigning existing sites.",
                ]
            ),
            h3("R2 — Remediate Wola first among the served districts"),
            ul(
                [
                    f"Wola has {_integer(capacity['wola']['projected_evs'])} projected EVs, only "
                    f"{_integer(capacity['wola']['available_plugs'])} plugs, "
                    f"{_pct(capacity['wola']['coverage_pct'], 1)} coverage and "
                    f"{_num(capacity['wola']['evs_per_plug'], 0)} EVs per plug — the worst ratio in Warsaw.",
                    "The immediate action is a demand and site-availability study, followed by a phased "
                    "build toward the planned 100-plug network if the forecast holds.",
                    "This is the clearest existing-market investment case; it does not require new-city "
                    "infrastructure assumptions.",
                ]
            ),
            h3("R3 — Shift from station-level optimisation to district-level coverage planning"),
            ul(
                [
                    "Retire 'top stations' visualisations: the busiest station is only "
                    f"{_num(stations['busiest_ratio'], 2)}× average, so they imply hotspots that do not exist.",
                    "Use 'Plug-plan coverage %', 'projected EVs per plug' and 'kWh per projected EV' as the "
                    "primary planning KPIs; report absolute shortfall only as context.",
                    "Allocate incremental capacity to coverage gaps, not to adding plugs at already-served "
                    "sites.",
                ]
            ),
            h3("R4 — Manage the 08:00–15:59 peak with time-based levers"),
            ul(
                [
                    f"{_pct(usage['peak_share_pct'], 1)} of sessions start in the eight-hour peak window; "
                    f"overnight utilisation is {_num(usage['overnight_per_hour_pct'], 2)}%/hour.",
                    "Test off-peak price incentives, managed/delayed charging and workplace-charging "
                    "communications. Measure demand elasticity before scaling.",
                    "Do not prioritise day-of-week pricing: the weekday spread is only "
                    f"{_pct(usage['weekday_spread_pct'], 1)}.",
                ]
            ),
            h3("R5 — Do not plan revenue growth through repricing"),
            ul(
                [
                    f"District average price is effectively uniform at "
                    f"{_num(revenue['price']['weighted_mean'], 4)} PLN/kWh "
                    f"({_num(revenue['district_price_spread_pct'], 2)}% spread across districts; operator "
                    f"spread under 1%).",
                    "Revenue rank equals volume rank. Growth requires more sessions, higher utilisation, or "
                    "a tested new tariff design.",
                    "Avoid presenting the High-tier spend gap as willingness to pay: it is explained by "
                    "vehicle battery size, not by price sensitivity.",
                ]
            ),
            h3("R6 — Use ratio-based planning and re-base anomalous projections"),
            ul(
                [
                    "Use the readiness rank (coverage primary, demand intensity as tie-break) and treat the "
                    "five-way 0% tie as unresolved until validated.",
                    "Re-base Wesoła and any district whose projection produces coverage above 100%. A "
                    "projection that understates measured load by 3.2× should not drive readiness headlines.",
                    "Keep projections versioned in the data pipeline so re-based numbers are traceable.",
                ]
            ),
            h3("R7 — Pair the energy model with an arrival/utilisation forecast"),
            ul(
                [
                    "The current model predicts kWh conditional on a session starting. It cannot say how "
                    "many drivers will arrive.",
                    "Add a session-count or plug-occupancy forecast, then multiply it by expected kWh per "
                    "session to get an operational energy plan.",
                    "Monitor drift and retrain when tariffs, vehicle mix, operators or stations change.",
                ]
            ),
            h3("R8 — Instrument the data needed to manage wait times and customer experience"),
            ul(
                [
                    "The dataset has no arrival/departure state of charge, session duration, charging power, "
                    "occupancy, queue or wait time. Wait-time reduction cannot be measured from the current "
                    "columns.",
                    "Add timestamped plug occupancy, queue events and app/availability interactions; these "
                    "are the direct inputs for delay reduction and experience metrics.",
                    "Use the existing energy model as the energy dimension of that future operational system.",
                ]
            ),
            h3("R9 — Customer-experience actions supported by the current evidence"),
            ul(
                [
                    "Close coverage gaps first: availability is the primary experience variable when "
                    "five districts have no station at all.",
                    "Publish clear, uniform pricing and availability information; with three flat tariff "
                    "points, complexity is not buying revenue.",
                    "Offer off-peak incentives and managed-charging options that reduce peak waiting without "
                    "unfairly penalising customers who cannot shift their charging time.",
                ]
            ),
        ],
    )

    # 13. Conclusion
    add(
        "Conclusion",
        "13. Conclusion",
        [
            p(
                "The Warsaw EV network's problem is distribution, not station productivity. Demand is flat "
                "and concentrated in business hours; stations are uniformly utilised; the tariff is flat; "
                "and five districts have no infrastructure while the built network is 36.8% of plan. The "
                "highest-value actions are coverage sequencing, peak management and instrumentation, not "
                "station-level add-ons or price changes."
            ),
            p(
                "The ML model adds a defensible session-energy estimator for known sessions, with test RMSE "
                f"{_num(ml.get('test', {}).get('rmse'), 3)} kWh when available. It does not replace an "
                "arrival or utilisation forecast, and its accuracy is driven mainly by vehicle battery "
                "capacity. Used together with a demand-count model and better instrumentation, it can "
                "support the operational and customer-experience decisions above."
            ),
        ],
    )

    # 14. Appendix
    add(
        "Appendix",
        "14. Appendix",
        [
            h3("SQL query map"),
            table(
                ["Query", "Focus", "Where it appears"],
                [
                    ["Q1", "District income mix (9 Low-Mid / 6 Mid-Range / 3 High)", "Evidence base and segmentation"],
                    ["Q2", "Stations per district; 13 served, 5 unserved", "Finding 1"],
                    ["Q3", "Plugs per district; 390 built vs 1,061 planned", "Finding 1"],
                    ["Q4", "114,000 sessions; flat daily rate", "Finding 2"],
                    ["Q5", "Energy per session: mean 39.13 kWh, range 13.40–76.00", "Finding 2"],
                    ["Q6", "Energy per district; top 3 = 47.9%", "Findings 1 and 3"],
                    ["Q7", "Station utilisation; busiest only 1.06× mean", "Finding 6"],
                    ["Q8", "Revenue per district; 11,424,001.70 PLN, flat price", "Finding 3"],
                    ["Q9", "Customer tier spend decomposition", "Finding 4"],
                    ["Q10", "Customer concentration; top 10 = 0.36% of energy", "Finding 4"],
                    ["Q11", "Projected EVs per plug; Wola worst at 275", "Finding 5"],
                    ["Q12", "Absolute shortfall; 18/18 fail, non-discriminating", "Finding 5"],
                    ["Q13", "kWh per projected EV; Wesoła anomaly at 1,096.63", "Finding 5"],
                    ["Q14", "Readiness rank (coverage + demand intensity)", "Finding 5"],
                ],
                "How the 14 queries feed the findings",
            ),
            h3("Artifacts"),
            table(
                ["Artifact", "Path"],
                [
                    ["This report (Markdown)", "reports/insights_report.md"],
                    ["This report (offline HTML)", "reports/insights_report.html"],
                    ["Computed metrics", "reports/insights_metrics.json"],
                    ["Generated figures", "reports/figures/insights_*.png"],
                    ["SQL analysis", "part1_sql_analysis.md (capstone folder)"],
                    ["Raw SQL output", "sql/query_outputs.txt (capstone folder)"],
                    ["Power BI build specification", "part2_powerbi_build_spec.md (capstone folder)"],
                    ["ML evaluation report", "reports/model_evaluation_report.md"],
                ],
                "Deliverable map",
            ),
            p(
                "Every number in this report comes from the executed SQL analysis, the raw source CSV "
                "computations shown in the figures, or the verified ML evaluation payload. Methodological "
                "notes and recommendations are clearly separated from measured results; no airline metric "
                "has been fabricated."
            ),
        ],
    )

    return sections


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def _render_markdown(insights: dict[str, Any], sections: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Warsaw EV Charging — Insights & Recommendations Report")
    lines.append("")
    lines.append(
        f"**Generated:** {insights['generated_utc']}  |  "
        f"**Data window:** {insights['source_rows']['date_start']} to {insights['source_rows']['date_end']}  |  "
        f"**Sessions:** {insights['source_rows']['sessions']:,}  |  "
        f"**Fingerprint:** `{insights['data_fingerprint']}`"
    )
    lines.append("")
    lines.append(
        "> **Airline wording notice.** The assignment template says 'airlines reduce delays, optimize "
        "operations, enhance customer experience'. The dataset is Warsaw EV charging; Section 1 maps each "
        "airline clause to its EV equivalent. All findings below are based on the executed SQL analysis and "
        "the source data. No airline-specific figure is invented."
    )
    lines.append("")
    lines.append("## Contents")
    lines.append("")
    for section in sections:
        lines.append(f"- [{section['title']}](#{section['id']})")
    lines.append("")

    for section in sections:
        lines.append(f"## {section['title']}")
        lines.append("")
        for block in section["blocks"]:
            block_type = block["type"]
            if block_type == "p":
                lines.append(str(block["text"]))
                lines.append("")
            elif block_type == "h3":
                lines.append(f"### {block['text']}")
                lines.append("")
            elif block_type == "ul":
                for item in block["items"]:
                    lines.append(f"- {item}")
                lines.append("")
            elif block_type == "table":
                lines.append(_md_table(block["headers"], block["rows"], block.get("caption")))
                lines.append("")
            elif block_type == "callout":
                lines.append(f"> **{block['title']}**")
                lines.append(">")
                for text_line in str(block["text"]).splitlines():
                    lines.append(f"> {text_line}")
                lines.append("")
            elif block_type == "figure":
                lines.append(f"![{block['caption']}]({block['file']})")
                lines.append("")
                lines.append(f"*{block['caption']}*")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_html(
    insights: dict[str, Any],
    sections: list[dict[str, Any]],
    reports_dir: Path,
) -> str:
    css = """
:root {
  --navy: #0f2740; --blue: #1f5f8b; --orange: #d97706; --green: #2f855a;
  --ink: #1f2937; --muted: #64748b; --line: #dbe4ee; --bg: #f3f7fb;
  --soft-blue: #eef6fc; --soft-green: #eefaf3; --soft-orange: #fff7ed;
}
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: var(--bg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.62; }
.hero { background: linear-gradient(135deg, #0f2740 0%, #16466b 55%, #1f5f8b 100%); color: #fff; padding: 54px 24px 42px; }
.hero-inner { max-width: 1120px; margin: 0 auto; }
.eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: 12px; opacity: .78; font-weight: 700; }
h1 { font-size: clamp(30px, 5vw, 46px); line-height: 1.12; margin: 10px 0 12px; letter-spacing: -.02em; }
.subtitle { font-size: 18px; max-width: 820px; opacity: .92; margin: 0 0 18px; }
.chips { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 16px; }
.chip { display: inline-block; padding: 6px 11px; border-radius: 999px; background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.2); font-size: 12.5px; }
main { max-width: 1120px; margin: 0 auto; padding: 26px 22px 54px; }
.toc { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 20px 24px; margin: 0 0 22px; box-shadow: 0 2px 10px rgba(15,39,64,.05); }
.toc h2 { margin: 0 0 10px; font-size: 15px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); }
.toc ol { columns: 2; column-gap: 36px; margin: 0; padding-left: 20px; font-size: 14px; }
.toc a { color: var(--blue); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
section { background: #fff; border: 1px solid var(--line); border-radius: 14px; padding: 26px 30px; margin: 20px 0; box-shadow: 0 2px 10px rgba(15,39,64,.045); }
section h2 { margin: 0 0 14px; font-size: 26px; line-height: 1.2; color: var(--navy); letter-spacing: -.015em; }
section h3 { margin: 24px 0 8px; font-size: 18px; color: var(--navy); }
p { margin: 10px 0; }
ul { margin: 10px 0 12px; padding-left: 24px; }
li { margin: 6px 0; }
code { background: #eef2f7; padding: 1px 5px; border-radius: 4px; font-size: .9em; }
.table-wrap { overflow-x: auto; margin: 14px 0 18px; border: 1px solid var(--line); border-radius: 10px; }
table { width: 100%; border-collapse: collapse; font-size: 13.3px; }
caption { text-align: left; font-weight: 700; color: var(--muted); padding: 10px 12px; font-size: 12.5px; }
th { background: #eef4fa; color: var(--navy); text-align: left; padding: 9px 11px; border-bottom: 1px solid var(--line); white-space: nowrap; }
td { padding: 8px 11px; border-bottom: 1px solid #edf2f7; vertical-align: top; }
tr:last-child td { border-bottom: none; }
tbody tr:hover { background: #f8fbfe; }
.callout { border-left: 4px solid var(--blue); background: var(--soft-blue); padding: 15px 18px; border-radius: 8px; margin: 16px 0; }
.callout.actual { border-color: var(--green); background: var(--soft-green); }
.callout.warning { border-color: var(--orange); background: var(--soft-orange); }
.callout-title { font-weight: 800; color: var(--navy); margin-bottom: 4px; }
figure { margin: 20px 0 26px; }
figure img { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 10px; background: #fff; }
figcaption { color: var(--muted); font-size: 13px; margin-top: 7px; }
footer { max-width: 1120px; margin: 0 auto; padding: 0 22px 44px; color: var(--muted); font-size: 12.5px; }
@media (max-width: 760px) { .toc ol { columns: 1; } section { padding: 21px 18px; } .hero { padding: 40px 18px 34px; } }
@media print { body { background: #fff; } .hero { background: #0f2740 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; } section { box-shadow: none; break-inside: avoid; } .toc { break-inside: avoid; } .toc a { color: var(--ink); } }
"""
    source = insights["source_rows"]
    usage = insights["usage"]
    hero = f"""
<header class="hero">
  <div class="hero-inner">
    <div class="eyebrow">Warsaw EV Charging · Capstone Insights</div>
    <h1>Where the Network Grows Next</h1>
    <p class="subtitle">An EV-equivalent answer to the airline-style brief: findings from 14 executed SQL queries, six data visualisations, and a verified session-energy model — with prioritised, evidence-based recommendations.</p>
    <div class="chips">
      <span class="chip">Sessions: {_integer(source['sessions'])}</span>
      <span class="chip">Energy: {_num(usage['total_kwh'], 0, comma=True)} kWh</span>
      <span class="chip">Stations: {_integer(source['stations'])}</span>
      <span class="chip">Districts: {_integer(source['districts'])}</span>
      <span class="chip">Window: {_inline_html(source['date_start'])} to {_inline_html(source['date_end'])}</span>
    </div>
  </div>
</header>
"""
    toc_items = "".join(
        f'<li><a href="#{html.escape(section["id"])}">{_inline_html(section["title"])}</a></li>'
        for section in sections
    )
    toc = f"""
<nav class="toc" aria-label="Contents">
  <h2>Contents</h2>
  <ol>{toc_items}</ol>
</nav>
"""
    body_sections: list[str] = []
    for section in sections:
        block_html: list[str] = []
        for block in section["blocks"]:
            block_type = block["type"]
            if block_type == "p":
                block_html.append(f"<p>{_inline_html(block['text'])}</p>")
            elif block_type == "h3":
                block_html.append(f"<h3>{_inline_html(block['text'])}</h3>")
            elif block_type == "ul":
                items = "".join(f"<li>{_inline_html(item)}</li>" for item in block["items"])
                block_html.append(f"<ul>{items}</ul>")
            elif block_type == "table":
                block_html.append(_html_table(block["headers"], block["rows"], block.get("caption")))
            elif block_type == "callout":
                block_html.append(
                    f'<div class="callout {html.escape(block["kind"])}">'
                    f'<div class="callout-title">{_inline_html(block["title"])}</div>'
                    f'<div>{_inline_html(block["text"])}</div></div>'
                )
            elif block_type == "figure":
                source_uri = _image_data_uri(reports_dir, block["file"])
                block_html.append(
                    f'<figure><img src="{source_uri}" alt="{html.escape(block["caption"])}" loading="lazy">'
                    f'<figcaption>{_inline_html(block["caption"])}</figcaption></figure>'
                )
        body_sections.append(
            f'<section id="{html.escape(section["id"])}"><h2>{_inline_html(section["title"])}</h2>'
            + "".join(block_html)
            + "</section>"
        )
    footer = f"""
<footer>
  <p><strong>Evidence note.</strong> Generated {_inline_html(insights['generated_utc'])} from the four
  Warsaw EV CSVs, the executed SQL analysis, and the verified ML metrics. The report is self-contained:
  images are embedded, CSS is local, and no internet connection is required.</p>
  <p>Data fingerprint: <code>{_inline_html(insights['data_fingerprint'])}</code> · Generated by
  <code>src/insights.py</code> · Warsaw EV Charging capstone.</p>
</footer>
"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="Warsaw EV Charging insights pipeline">
<title>Warsaw EV Charging — Insights &amp; Recommendations Report</title>
<style>{css}</style>
</head>
<body>
{hero}
<main>
{toc}
{''.join(body_sections)}
</main>
{footer}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate_report(config: Any) -> ReportPaths:
    reports_dir = Path(config.reports_dir)
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tables = load_raw_tables(config.data_dir)
    ml_metrics = _load_ml_metrics(reports_dir)
    insights = compute_insights(tables, ml_metrics)
    figures = create_figures(insights, figures_dir)
    sections = _sections(insights, figures)

    metrics_path = reports_dir / "insights_metrics.json"
    markdown_path = reports_dir / "insights_report.md"
    html_path = reports_dir / "insights_report.html"
    write_json(
        metrics_path,
        {
            "generated_utc": insights["generated_utc"],
            "data_fingerprint": insights["data_fingerprint"],
            "source_rows": insights["source_rows"],
            "usage": insights["usage"],
            "revenue": insights["revenue"],
            "capacity": insights["capacity"],
            "stations": insights["stations"],
            "customers": insights["customers"],
            "quality": insights["quality"],
            "ml": insights["ml"],
            "planning": insights["planning"],
            "districts": insights["districts"],
            "figures": figures,
        },
    )
    markdown = _render_markdown(insights, sections)
    markdown_path.write_text(markdown, encoding="utf-8")
    html_text = _render_html(insights, sections, reports_dir)
    html_path.write_text(html_text, encoding="utf-8")
    figure_paths = [figures_dir / Path(record["file"]).name for record in figures]
    LOGGER.info("Saved insights metrics: %s", metrics_path)
    LOGGER.info("Saved insights report: %s", markdown_path)
    LOGGER.info("Saved insights report: %s", html_path)
    return ReportPaths(
        metrics=metrics_path,
        markdown=markdown_path,
        html=html_path,
        figures=figure_paths,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Warsaw EV insights and recommendations report."
    )
    add_common_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        config = load_config(args)
        paths = generate_report(config)
    except (ConfigError, DataValidationError, FileNotFoundError) as exc:
        LOGGER.error("%s", exc)
        return 2

    insights = None
    try:
        import json

        insights = json.loads(paths.metrics.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - summary only
        pass

    print("\nInsights report complete")
    print(f"Metrics JSON : {paths.metrics}")
    print(f"Markdown     : {paths.markdown}")
    print(f"HTML         : {paths.html}")
    print(f"Figures      : {len(paths.figures)}")
    if insights:
        print(
            f"Sessions     : {insights['source_rows']['sessions']:,} | "
            f"Districts served: {insights['capacity']['served_districts']}/"
            f"{insights['quality']['districts_total']} | "
            f"Plug coverage: {insights['capacity']['plug_plan_coverage_pct']:.2f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
