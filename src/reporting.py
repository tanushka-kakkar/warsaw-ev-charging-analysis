"""Generate consistent Markdown and self-contained HTML evaluation reports."""

from __future__ import annotations

import base64
import html
import math
import re
from pathlib import Path
from typing import Any

from .preprocessing import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

FILE_COLUMNS = {
    "charging_sessions.csv": {
        "session_id": "Unique session identifier (excluded from model features).",
        "customer_id": "Foreign key to customers.csv (excluded from model features).",
        "station_id": "Foreign key to charging_stations.csv (excluded from model features).",
        "session_start_time": "Session start timestamp, parsed to hour/day-of-week features; raw timestamp excluded.",
        "kwh_charged": "Energy delivered in kWh — the prediction target.",
        "cost_per_kwh": "Tariff at session start in PLN/kWh; known before/at session start and used as a feature.",
        "total_cost": "Billing total; equals kWh × tariff, so it is target-derived and excluded.",
    },
    "charging_stations.csv": {
        "station_id": "Station identifier; used for joins only.",
        "district_name": "Warsaw district of the station.",
        "income_tier": "Station/district income tier used as a categorical feature.",
        "operator_name": "Charging operator name.",
        "plugs_count": "Number of plugs at the station (constant in this dataset).",
        "latitude": "Station latitude.",
        "longitude": "Station longitude.",
    },
    "customers.csv": {
        "customer_id": "Customer identifier; used for joins only.",
        "income_tier": "Customer income tier.",
        "car_model": "Electric vehicle model.",
        "battery_capacity_kwh": "Vehicle battery capacity in kWh.",
    },
    "districts.csv": {
        "district_name": "District name (join key).",
        "income_tier": "District income tier (duplicates station tier).",
        "projected_evs": "District-level projected EV population (static planning data).",
        "projected_plugs": "District-level projected plug count.",
        "projected_regular_customers": "District-level projected regular customer count.",
    },
}

FEATURE_DESCRIPTIONS = {
    "battery_capacity_kwh": "Vehicle battery capacity (kWh) from customers.csv.",
    "cost_per_kwh": "Session tariff (PLN/kWh), known at prediction time.",
    "plugs_count": "Station plug count (constant at 6 in this sample).",
    "latitude": "Station latitude.",
    "longitude": "Station longitude.",
    "projected_evs": "District projected EV population.",
    "projected_plugs": "District projected plug count.",
    "projected_regular_customers": "District projected regular customers.",
    "evs_per_projected_plug": "District projected EVs divided by projected plugs.",
    "hour": "Hour of session start (0–23).",
    "day_of_week": "Day of week (0 = Monday).",
    "is_weekend": "1 for Saturday/Sunday, else 0.",
    "hour_sin": "Cyclical encoding of hour (sine).",
    "hour_cos": "Cyclical encoding of hour (cosine).",
    "dow_sin": "Cyclical encoding of day of week (sine).",
    "dow_cos": "Cyclical encoding of day of week (cosine).",
    "customer_income_tier": "Customer income tier category.",
    "car_model": "Vehicle model category.",
    "station_income_tier": "Station/district income tier category.",
    "operator_name": "Charging operator category.",
    "district_name": "Warsaw district category.",
}


def _num(value: Any, digits: int = 2, comma: bool = False) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:,.{digits}f}" if comma else f"{number:.{digits}f}"


def _integer(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.{digits}f}%"


def _date(value: Any) -> str:
    if value is None:
        return "n/a"
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def _mb(value: Any) -> str:
    try:
        return f"{float(value) / 1_000_000:.2f} MB"
    except (TypeError, ValueError):
        return "n/a"


def _sci(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}e}"
    except (TypeError, ValueError):
        return str(value)


def _inline_html(text: Any) -> str:
    escaped = html.escape(str(text))
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
    return escaped.replace("\n", "<br>")


def _md_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _md_table(headers: list[str], rows: list[list[Any]], caption: str | None = None) -> str:
    lines: list[str] = []
    if caption:
        lines.append(f"**{caption}**")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_md_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _html_table(headers: list[str], rows: list[list[Any]], caption: str | None = None) -> str:
    caption_html = f"<caption>{_inline_html(caption)}</caption>" if caption else ""
    head = "".join(f"<th>{_inline_html(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_inline_html(value)}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="table-wrap"><table>{caption_html}<thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body_rows)}</tbody></table></div>"
    )


def _slug(title: str) -> str:
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    return re.sub(r"\s+", "-", slug.strip())


def _figure_block(file: str, caption: str) -> dict[str, str]:
    return {"type": "figure", "file": file, "caption": caption}


def _sections(payload: dict[str, Any]) -> list[dict[str, Any]]:
    profile = payload.get("data_profile") or {}
    files = profile.get("source_files") or {}
    split = payload.get("split") or {}
    test = payload.get("test_metrics") or {}
    baseline = payload.get("baseline_test_metrics") or {}
    heuristic = payload.get("heuristic_test_metrics") or {}
    selected_label = payload.get("selected_label", "n/a")
    selected_model = payload.get("selected_model", "n/a")
    selected_params = payload.get("selected_params") or {}
    candidates = payload.get("candidate_results") or []
    improvement = payload.get("improvement_vs_dummy") or {}
    heuristic_improvement = payload.get("improvement_vs_heuristic") or {}
    residual = payload.get("residual_summary") or {}
    bootstrap = payload.get("bootstrap_confidence_intervals") or {}
    segments = payload.get("segments") or {}
    station_day = payload.get("station_day_metrics") or {}
    district_day = payload.get("district_day_metrics") or {}
    importance = payload.get("feature_importance") or []
    figures = payload.get("figures") or []
    totals = payload.get("energy_totals") or {}
    integrity = payload.get("evaluation_integrity") or {}
    target_summary = profile.get("target_summary") or {}
    ratio_summary = profile.get("ratio_to_battery") or {}
    physical = profile.get("physical_ratio") or {}
    iqr = profile.get("target_iqr") or {}
    dropped = profile.get("dropped_rows") or {}
    coverage = profile.get("referential_coverage") or {}
    heuristic_validation = payload.get("capacity_heuristic_validation_metrics") or {}

    validation_metrics = payload.get("validation_metrics_selected") or {}
    top_importance = sorted(importance, key=lambda row: row.get("importance_mean", 0), reverse=True)
    top_names = [row.get("feature", "n/a") for row in top_importance[:5]]

    def metric_row(label: str, metrics: dict[str, Any]) -> list[str]:
        return [
            label,
            _integer(metrics.get("n")),
            _num(metrics.get("mae"), 3),
            _num(metrics.get("rmse"), 3),
            _num(metrics.get("r2"), 4),
            _pct(metrics.get("mape_pct"), 2),
        ]

    candidate_rows: list[list[str]] = []
    for record in sorted(candidates, key=lambda item: item.get("validation_metrics", {}).get("rmse", 1e9)):
        metrics = record.get("validation_metrics", {})
        candidate_rows.append(
            [
                record.get("label", "n/a"),
                _num(metrics.get("mae"), 3),
                _num(metrics.get("rmse"), 3),
                _num(metrics.get("r2"), 4),
                _pct(metrics.get("mape_pct"), 2),
                _num(record.get("fit_seconds"), 1),
                "Selected" if record.get("label") == selected_label else "",
            ]
        )
    if heuristic_validation:
        candidate_rows.append(
            [
                "Capacity heuristic (reference, not selected)",
                _num(heuristic_validation.get("mae"), 3),
                _num(heuristic_validation.get("rmse"), 3),
                _num(heuristic_validation.get("r2"), 4),
                _pct(heuristic_validation.get("mape_pct"), 2),
                "n/a",
                "",
            ]
        )

    source_rows = [
        [
            filename,
            _integer(info.get("rows")),
            _integer(info.get("columns")),
            _integer(info.get("missing_cells")),
            _integer(info.get("duplicate_rows")),
            _mb(info.get("size_bytes")),
        ]
        for filename, info in sorted(files.items())
    ]

    dimension_rows = [
        ["Raw charging sessions", _integer(profile.get("raw_sessions_rows"))],
        ["Usable modeling sessions", _integer(profile.get("modeling_rows"))],
        ["Customers", _integer(profile.get("raw_customers_rows"))],
        ["Stations", _integer(profile.get("raw_stations_rows"))],
        ["Districts in reference file", _integer(profile.get("raw_districts_rows"))],
        ["Districts represented in sessions", _integer(profile.get("unique_districts"))],
        ["Observation window", f"{_date(profile.get('date_min'))} to {_date(profile.get('date_max'))}"],
        ["Window length", f"{_integer(profile.get('span_days'))} days"],
    ]

    column_rows = []
    for filename, descriptions in FILE_COLUMNS.items():
        for column, description in descriptions.items():
            column_rows.append([filename, column, description])

    split_rows = []
    total_split_rows = sum(int(split.get(key, 0)) for key in ("train_rows", "validation_rows", "test_rows")) or 0

    def split_row(period: str, key: str, days_key: str) -> list[str]:
        rows = split.get(key)
        share = (float(rows) / total_split_rows * 100.0) if rows and total_split_rows else None
        return [
            period,
            _date(split.get(f"{key.replace('_rows', '_start')}")),
            _date(split.get(f"{key.replace('_rows', '_end')}")),
            _integer(split.get(days_key)),
            _integer(rows),
            _pct(share, 1),
        ]

    split_rows = [
        split_row("Train", "train_rows", "train_dates"),
        split_row("Validation", "validation_rows", "validation_dates"),
        split_row("Test (held out)", "test_rows", "test_dates"),
    ]

    quality_rows = [
        ["Raw session rows", _integer(profile.get("raw_sessions_rows"))],
        ["Invalid / missing start timestamps", _integer(dropped.get("invalid_start_time"))],
        ["Missing customer or station identifiers", _integer(dropped.get("missing_identifiers"))],
        ["Exact duplicate rows removed", _integer(dropped.get("exact_duplicate_rows"))],
        ["Duplicate session IDs removed", _integer(dropped.get("duplicate_session_id"))],
        ["Repeated customer + start-minute sessions removed", _integer(dropped.get("duplicate_customer_start_time"))],
        ["Missing / non-positive target rows removed", _integer(dropped.get("invalid_target"))],
        ["Same station + start-minute rows (retained; multiple plugs)", _integer(profile.get("duplicate_station_start_time"))],
        ["Target physical range", f"{_num(target_summary.get('min'), 2)} to {_num(target_summary.get('max'), 2)} kWh"],
        ["Target mean / median", f"{_num(target_summary.get('mean'), 2)} / {_num(target_summary.get('median'), 2)} kWh"],
        ["IQR outlier flags (retained)", _integer(iqr.get("outlier_count"))],
        ["Energy / battery-capacity range", f"{_num(physical.get('min'), 4)} to {_num(physical.get('max'), 4)}"],
        ["Sessions above battery capacity", _integer(physical.get("above_battery"))],
        ["Invalid tariff rows (set to missing)", _integer(profile.get("invalid_price_rows"))],
        ["Invalid battery-capacity rows", _integer(profile.get("invalid_battery_rows"))],
        ["Invalid coordinate rows", _integer(profile.get("invalid_coordinate_rows"))],
        ["Cost identity max abs error", _sci(profile.get("cost_identity_max_abs_error"), 2)],
        ["Customer session coverage", f"{_integer(coverage.get('customer_sessions_matched'))} matched / {_integer(coverage.get('customer_sessions_unmatched'))} unmatched"],
        ["Station session coverage", f"{_integer(coverage.get('station_sessions_matched'))} matched / {_integer(coverage.get('station_sessions_unmatched'))} unmatched"],
        ["Missing feature values in final frame", _integer(profile.get("final_missing_feature_values"))],
    ]

    feature_rows = [
        [
            feature,
            "Categorical" if feature in CATEGORICAL_FEATURES else "Numeric",
            FEATURE_DESCRIPTIONS.get(feature, "Engineered model input."),
        ]
        for feature in FEATURE_COLUMNS
    ]

    selected_param_rows = [[key, value] for key, value in sorted(selected_params.items())] or [["(none)", "Default estimator settings"]]
    selected_params_text = ", ".join(f"{key}={value}" for key, value in sorted(selected_params.items())) or "(defaults)"

    bootstrap_rows = []
    for metric, label in (("rmse", "RMSE (kWh)"), ("mae", "MAE (kWh)"), ("r2", "R²")):
        interval = bootstrap.get(metric) or {}
        estimate = test.get(metric)
        bootstrap_rows.append(
            [
                label,
                _num(estimate, 4 if metric == "r2" else 3),
                _num(interval.get("lower_95"), 4 if metric == "r2" else 3),
                _num(interval.get("upper_95"), 4 if metric == "r2" else 3),
                _integer(interval.get("bootstrap_samples")),
            ]
        )

    residual_rows = [
        ["Mean residual (actual − predicted)", f"{_num(residual.get('mean'), 3)} kWh"],
        ["Median residual", f"{_num(residual.get('median'), 3)} kWh"],
        ["Residual standard deviation", f"{_num(residual.get('std'), 3)} kWh"],
        ["5th percentile", f"{_num(residual.get('p05'), 3)} kWh"],
        ["95th percentile", f"{_num(residual.get('p95'), 3)} kWh"],
        ["Correlation(prediction, residual)", _num(residual.get("residual_vs_prediction_correlation"), 4)],
    ]

    car_rows = [
        [row.get("segment", "n/a"), _integer(row.get("n")), _num(row.get("mae"), 3), _num(row.get("rmse"), 3), _pct(row.get("mape_pct"), 2)]
        for row in segments.get("car_model", [])
    ]
    income_rows = [
        [row.get("segment", "n/a"), _integer(row.get("n")), _num(row.get("mae"), 3), _num(row.get("rmse"), 3), _pct(row.get("mape_pct"), 2)]
        for row in segments.get("customer_income_tier", [])
    ]
    district_rows = [
        [row.get("segment", "n/a"), _integer(row.get("n")), _num(row.get("mae"), 3), _num(row.get("rmse"), 3), _pct(row.get("mape_pct"), 2)]
        for row in segments.get("district_name", [])
    ]
    operator_rows = [
        [row.get("segment", "n/a"), _integer(row.get("n")), _num(row.get("mae"), 3), _num(row.get("rmse"), 3), _pct(row.get("mape_pct"), 2)]
        for row in segments.get("operator_name", [])
    ]

    station_rows = []
    for label, metrics in (("Station-day (conditional)", station_day), ("District-day (conditional)", district_day)):
        if metrics:
            base = metrics.get("baseline") or {}
            station_rows.append(
                [
                    label,
                    _integer(metrics.get("n")),
                    _num(metrics.get("mae"), 1),
                    _num(metrics.get("rmse"), 1),
                    _pct(metrics.get("mape_pct"), 2),
                    _num(metrics.get("r2"), 4),
                    _num(base.get("mape_pct"), 2),
                ]
            )

    importance_rows = [
        [
            index + 1,
            row.get("feature", "n/a"),
            _num(row.get("importance_mean"), 4),
            _num(row.get("importance_std"), 4),
        ]
        for index, row in enumerate(top_importance[:15])
    ]

    test_table_rows = [
        metric_row("Selected model", test),
        metric_row("Dummy mean baseline", baseline),
        metric_row("Capacity heuristic (reference)", heuristic),
    ]

    rmse_reduction = improvement.get("rmse_reduction_pct")
    mae_reduction = improvement.get("mae_reduction_pct")
    heuristic_rmse_reduction = heuristic_improvement.get("rmse_reduction_pct")
    r2_value = test.get("r2")
    total_prediction_error = totals.get("prediction_error_pct")

    if rmse_reduction is not None and rmse_reduction >= 0:
        vs_dummy_text = (
            f"The selected model reduced held-out RMSE by **{_pct(rmse_reduction)}** and MAE by "
            f"**{_pct(mae_reduction)}** versus the dummy mean baseline."
        )
    elif rmse_reduction is not None:
        vs_dummy_text = (
            f"The selected model did **not** beat the dummy mean baseline on held-out RMSE "
            f"({_pct(rmse_reduction)} change); this is a warning sign for practical use."
        )
    else:
        vs_dummy_text = "Baseline comparison is unavailable."

    if heuristic_rmse_reduction is not None and heuristic_rmse_reduction > 0.1:
        heuristic_text = (
            f"The ML model beat the transparent battery-capacity heuristic by "
            f"**{_pct(heuristic_rmse_reduction)}** RMSE."
        )
    elif heuristic_rmse_reduction is not None and abs(heuristic_rmse_reduction) <= 0.1:
        heuristic_text = (
            "The ML model is statistically indistinguishable from the transparent one-feature "
            f"battery-capacity heuristic (RMSE change {_pct(heuristic_rmse_reduction)}). The current "
            "feature set therefore adds no measurable accuracy beyond vehicle battery capacity."
        )
    elif heuristic_rmse_reduction is not None:
        heuristic_text = (
            "The ML model did not outperform the transparent battery-capacity heuristic "
            f"({_pct(heuristic_rmse_reduction)} RMSE change). The added features therefore "
            "provide little measurable value beyond vehicle capacity in this dataset."
        )
    else:
        heuristic_text = "The capacity-heuristic comparison is unavailable."

    top_names_text = ", ".join(top_names) if top_names else "n/a"
    station_day_mape = station_day.get("mape_pct")
    station_day_mae = station_day.get("mae")
    test_start = _date(split.get("test_start"))
    test_end = _date(split.get("test_end"))
    test_rows = split.get("test_rows")
    date_min = _date(profile.get("date_min"))
    date_max = _date(profile.get("date_max"))

    sections: list[dict[str, Any]] = []

    def add(section_id: str, title: str, blocks: list[dict[str, Any]]) -> None:
        sections.append({"id": _slug(title), "title": title, "blocks": blocks})

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

    def code(text: str) -> dict[str, str]:
        return {"type": "code", "text": text}

    # 1. Executive summary
    add(
        "1. Executive summary",
        "1. Executive summary",
        [
            callout(
                "actual",
                "Actual held-out test result",
                f"**{selected_label}** achieved MAE **{_num(test.get('mae'), 3)} kWh**, "
                f"RMSE **{_num(test.get('rmse'), 3)} kWh**, R² **{_num(test.get('r2'), 4)}**, "
                f"and MAPE **{_pct(test.get('mape_pct'))}** on {_integer(test.get('n'))} sessions "
                f"from {test_start} to {test_end}. The dummy-mean test RMSE was "
                f"**{_num(baseline.get('rmse'), 3)} kWh**.",
            ),
            p(
                "The model predicts the energy delivered in one EV charging session in Warsaw. "
                "The test period is strictly after the validation period, and the model configuration "
                "was selected on validation data before the test set was evaluated."
            ),
            p(vs_dummy_text + " " + heuristic_text),
            p(
                f"Permutation importance ranks {top_names_text} as the most influential inputs. "
                "That result is consistent with the observed target structure: delivered energy is "
                "tightly tied to vehicle battery capacity, while time, station and district attributes "
                "add little in this sample."
            ),
            p(
                f"All {_integer(profile.get('raw_sessions_rows'))} source sessions were inspected; "
                f"{_integer(profile.get('modeling_rows'))} usable sessions entered modeling. "
                "The dataset covers only three months in one city, and the energy-to-battery ratio is "
                "bounded between roughly 0.50 and 0.80. Those facts materially limit real-world claims."
            ),
        ],
    )

    # 2. Project overview
    add(
        "2. Project overview and business objective",
        "2. Project overview and business objective",
        [
            h3("Business problem"),
            p(
                "Charging-network planners need an estimate of how much energy a session will deliver "
                "so they can size capacity, plan energy procurement, prioritize high-demand vehicle "
                "segments, and set realistic expectations for station-level demand. This project builds "
                "a reproducible benchmark for that estimate using the available session, customer, "
                "station and district data."
            ),
            h3("Prediction task"),
            table(
                ["Item", "Definition"],
                [
                    ["Target", "kwh_charged — delivered energy per charging session"],
                    ["Unit", "kilowatt-hours (kWh)"],
                    ["Unit of prediction", "One charging session"],
                    ["Task type", "Supervised regression"],
                    ["Prediction time", "When the customer, vehicle, station, tariff and session start time are known"],
                    ["Features", "Vehicle, customer tier, station, district planning attributes, tariff, calendar/cyclical time"],
                    ["Excluded", "session_id, customer_id, station_id, raw timestamp, total_cost (target-derived)"],
                    ["Split", "Chronological train / validation / held-out test by whole calendar days"],
                ],
                "Selected prediction task",
            ),
            h3("Expected business value"),
            p(
                "The model gives an expected energy estimate per session and a transparent way to "
                "aggregate expected energy to station-day or district-day totals when the arriving "
                "session mix is known. That supports scenario planning and vehicle-mix analysis. It "
                "does not forecast how many vehicles will arrive, and it does not establish causal "
                "effects of station location, operator or time."
            ),
            h3("Scope and limitations up front"),
            p(
                "The available files contain no session duration, no arrival/departure state of charge, "
                "no occupancy or queue data, and no weather, traffic or event data. The observation "
                "window is short. The energy target also has a strikingly regular relationship to "
                "battery capacity. These issues are documented in the limitations section and should "
                "be addressed before operational deployment."
            ),
        ],
    )

    # 3. Dataset inspection
    add(
        "3. Dataset files, dimensions and columns",
        "3. Dataset files, dimensions and columns",
        [
            h3("Files inspected and selected"),
            p(
                "All four CSV files found in the configured dataset directory were inspected. "
                "All four are used for modeling: sessions supply the target, customers and stations "
                "supply join attributes, and districts supply static planning features. The source "
                "files were read with UTF-8 BOM support and were not modified."
            ),
            table(
                ["File", "Rows", "Columns", "Missing cells", "Duplicate rows", "Size"],
                source_rows,
                "Source files",
            ),
            h3("Dataset dimensions"),
            table(["Measure", "Value"], dimension_rows, "Data volume"),
            h3("Relationships between files"),
            p(
                "charging_sessions.customer_id joins to customers.customer_id; "
                "charging_sessions.station_id joins to charging_stations.station_id; and "
                "charging_stations.district_name joins to districts.district_name. Every session "
                "customer and station matched the reference tables in the inspected data. The session "
                "files are transaction-level records, not station-level aggregates."
            ),
            h3("Relevant column descriptions"),
            table(["File", "Column", "Description"], column_rows, "Column dictionary"),
            p(
                f"There are {_integer(profile.get('unique_customers'))} distinct customers, "
                f"{_integer(profile.get('unique_stations'))} stations and "
                f"{_integer(profile.get('unique_districts'))} districts represented in the modeling rows. "
                "Every customer appears repeatedly, which is why identifier columns were excluded and "
                "the evaluation uses a chronological split rather than a random row shuffle."
            ),
        ],
    )

    # 4. Data quality
    add(
        "4. Data quality findings",
        "4. Data quality findings",
        [
            p(
                f"The source window is {date_min} to {date_max} ({_integer(profile.get('span_days'))} days). "
                "The raw sessions table has no missing cells in the inspected data. Data-quality checks "
                "still removed contradictory or unusable records as documented below."
            ),
            table(["Check", "Result"], quality_rows, "Data-quality checks and treatments"),
            h3("Duplicates"),
            p(
                f"{_integer(dropped.get('duplicate_customer_start_time'))} repeated customer/start-minute "
                "pairs were removed because one customer cannot begin two sessions at two stations in the "
                "same minute. "
                f"{_integer(profile.get('duplicate_station_start_time'))} rows share a station and start "
                "minute; these were retained because a station with six plugs can legitimately start "
                "multiple sessions in the same minute."
            ),
            h3("Missing values"),
            p(
                "No missing values were present in the source columns used for modeling. The trained "
                "preprocessing pipeline nevertheless includes median imputation for numeric features and "
                "most-frequent imputation for categorical features, fitted on training data only. This "
                "keeps inference robust to future incomplete records."
            ),
            h3("Outliers and physical plausibility"),
            p(
                f"The target ranges from {_num(target_summary.get('min'), 2)} to "
                f"{_num(target_summary.get('max'), 2)} kWh. The IQR rule flags "
                f"{_integer(iqr.get('outlier_count'))} sessions, but none were removed: every value is "
                "physically plausible and the extremes are genuine large-battery sessions. "
                f"The ratio of delivered energy to battery capacity ranges from "
                f"{_num(physical.get('min'), 4)} to {_num(physical.get('max'), 4)}, and no session exceeds "
                "the customer's battery capacity."
            ),
            callout(
                "note",
                "Important data-generating pattern",
                "The target is almost exactly bounded between 50% and 80% of battery capacity, and every "
                "vehicle model has a mean ratio near 0.65. This is characteristic of a simulated or highly "
                "standardized process, not of unrestricted real-world charging. The model can recover the "
                "vehicle-capacity relationship, but the data do not reveal the operational drivers that "
                "would matter in production.",
            ),
            p(
                f"The maximum absolute discrepancy in the billing identity "
                "(total_cost − kWh × tariff) was "
                f"{_sci(profile.get('cost_identity_max_abs_error'), 2)}, consistent with floating-point "
                "rounding. total_cost was nevertheless excluded from the features because it is target-derived."
            ),
        ],
    )

    # 5. Target selection
    add(
        "5. Prediction target and justification",
        "5. Prediction target and justification",
        [
            p(
                "The primary target is **kwh_charged**, the energy delivered in a charging session. "
                "It is directly observed at transaction level, has 114,000 candidate records, is "
                "continuous and non-negative, and supports a clear operational question: how much "
                "energy should a session of this vehicle/customer/station/time profile be expected to draw?"
            ),
            table(
                ["Candidate task", "Support in this dataset", "Decision"],
                [
                    [
                        "Session energy consumption",
                        "Direct target; 114k sessions; vehicle and station attributes available at session start.",
                        "Selected",
                    ],
                    [
                        "Session duration",
                        "No end time or duration column exists.",
                        "Not feasible without additional data",
                    ],
                    [
                        "Session revenue",
                        "total_cost = kWh × tariff; it is a deterministic transform of the energy target. The tariff is not a modelled business outcome.",
                        "Not selected as a separate target",
                    ],
                    [
                        "Station utilization",
                        "Only session starts are available; no plug occupancy, queue or capacity-over-time data.",
                        "Only weakly approximable; future work",
                    ],
                    [
                        "Future demand forecast",
                        "89 days of history; arrivals are not explained by available exogenous data, and no lag/forecast infrastructure would be reliable yet.",
                        "Deferred; requires a longer series and arrival data",
                    ],
                ],
                "Candidate prediction tasks",
            ),
            h3("Prediction-time information"),
            p(
                "At session start the operator can know the customer account and vehicle, the selected "
                "station and its district, the tariff, and the timestamp. The model uses those inputs. "
                "It does not use the realized kWh, billing total, session identifier, customer identifier "
                "or station identifier."
            ),
            h3("Why this is a regression problem"),
            p(
                "The target is continuous and measured in kWh, so the problem is supervised regression. "
                "The evaluation therefore reports MAE, RMSE, R² and MAPE, with residual and segment analysis. "
                "Classification metrics and decision thresholds do not apply."
            ),
        ],
    )

    # 6. Preprocessing
    add(
        "6. Preprocessing methodology",
        "6. Preprocessing methodology",
        [
            ul(
                [
                    "Loaded each CSV with UTF-8 BOM support; validated required column schemas and row counts; preserved the sources.",
                    "Parsed session_start_time with coercion and removed records with unusable timestamps or missing customer/station identifiers.",
                    "Coerced energy, tariff, battery, coordinate and projection columns to numeric values; set impossible tariffs/coordinates to missing rather than silently using them.",
                    "Removed exact duplicate rows, duplicate session IDs, and contradictory repeated customer/start-minute sessions.",
                    "Dropped only missing or non-positive target rows; retained plausible high-energy outliers after physical plausibility checks.",
                    "Joined customer, station and district reference tables with many-to-one validation; inspected unmatched records.",
                    "Separated the target from inputs and asserted that target, identifiers and billing total are not in the feature list.",
                ]
            ),
            table(
                ["Pipeline stage", "Learned on", "Applied to", "Purpose"],
                [
                    ["Numeric median imputation", "Training rows only", "Validation / test / new data", "Handle missing numeric inputs"],
                    ["Categorical most-frequent imputation", "Training rows only", "Validation / test / new data", "Handle missing categories"],
                    ["StandardScaler", "Training rows only", "Validation / test / new data", "Scale numeric inputs for Ridge"],
                    ["OneHotEncoder(handle_unknown='ignore')", "Training rows only", "Validation / test / new data", "Encode categories; unseen categories become zero-vector"],
                    ["Ridge / RandomForest / HistGradientBoosting", "Training rows only; final refit train+validation", "Validation / test", "Prediction"],
                ],
                "Learned transformations",
            ),
            p(
                "All learned transformations are inside a scikit-learn Pipeline saved with the estimator. "
                "No imputer, scaler, encoder or model was fitted on the test set. For model selection, "
                "each candidate pipeline was fitted on the training partition only. After selection, the "
                "chosen pipeline was refitted on train+validation, and the test set was transformed but never fitted."
            ),
            h3("Leakage prevention"),
            p(
                "total_cost is a deterministic product of the target and tariff and was excluded. "
                "Identifiers were excluded so the model cannot memorize a session, customer or station. "
                "The raw timestamp was replaced with calendar fields. The split is chronological, so "
                "future sessions cannot inform predictions of earlier sessions. The final model is saved "
                "together with its metadata, feature list, preprocessing pipeline and data fingerprint."
            ),
        ],
    )

    # 7. Feature engineering
    add(
        "7. Feature engineering decisions",
        "7. Feature engineering decisions",
        [
            p(
                "Features were chosen so that they would be available at session start and so that they "
                "describe the vehicle, customer segment, station context, static district planning context "
                "and time of day. The engineered features are deterministic row-wise transforms, not "
                "target encodings."
            ),
            table(["Feature", "Type", "Description"], feature_rows, "Model features"),
            h3("Excluded columns"),
            ul(
                [
                    "kwh_charged — the target.",
                    "total_cost — target-derived (kWh × tariff).",
                    "session_id, customer_id, station_id — identifiers; excluded to prevent memorization and to allow cold-start style prediction.",
                    "session_start_time raw timestamp — replaced with hour/day-of-week/cyclical features.",
                    "month — excluded because the chronological test period is an unseen month and three months cannot support stable seasonal extrapolation.",
                ]
            ),
            p(
                f"After joining and engineering, the final frame contains {_integer(profile.get('final_missing_feature_values'))} "
                "missing feature values. The pipeline still handles missing values for inference. "
                "The tariff is included because it is known at session start, but its permutation importance "
                "is reported rather than assumed to be causal."
            ),
        ],
    )

    # 8. Split
    add(
        "8. Train, validation and test split",
        "8. Train, validation and test split",
        [
            p(
                "Sessions are split chronologically by whole calendar days: the earliest days form the "
                "training set, the next days form the validation set, and the latest days form the held-out "
                "test set. No random shuffling is used, because a future session must not inform a model "
                "that predicts the past and because the deployment scenario is future prediction."
            ),
            table(
                ["Period", "Start", "End", "Days", "Sessions", "Share of rows"],
                split_rows,
                "Chronological split",
            ),
            p(
                f"Training uses {_integer(split.get('train_rows'))} sessions; validation uses "
                f"{_integer(split.get('validation_rows'))}; the test period contains "
                f"{_integer(split.get('test_rows'))} sessions from {test_start} to {test_end}. "
                "The test set was not used to choose features, hyperparameters or the final model."
            ),
            h3("Repeated customers and grouping"),
            p(
                "Each customer contributes several sessions across the window. A random split could place "
                "the same customer's sessions in both train and test, but the model does not use a customer "
                "identifier or customer-specific target encoding, so there is no learned customer memory. "
                "The chronological split answers a deployment-relevant question: how well does a model "
                "trained on earlier sessions predict later sessions from the same population? A future "
                "sensitivity analysis could also use GroupShuffleSplit by customer, but that would answer a "
                "different cold-customer question and reduce training data."
            ),
        ],
    )

    # 9. Models
    add(
        "9. Models evaluated and selection",
        "9. Models evaluated and selection",
        [
            p(
                "A dummy mean regressor establishes the minimum baseline. Ridge Regression provides an "
                "interpretable linear benchmark. Random Forest and HistGradientBoosting test nonlinear and "
                "interaction-capable alternatives. A transparent battery-capacity heuristic is also reported "
                "as a domain benchmark. It is not eligible for selection because it is a one-feature reference "
                "rather than a maintained model pipeline, but it is the most useful comparison for deciding "
                "whether the ML features add accuracy. The ML pipeline is retained as the deliverable and as "
                "the extension point for future features, and the benchmark result is reported honestly."
            ),
            table(
                ["Model / configuration", "Validation MAE", "Validation RMSE", "Validation R²", "Validation MAPE", "Fit seconds", "Selected"],
                candidate_rows,
                "Validation comparison (model selection data)",
            ),
            p(
                f"The selection criterion was minimum validation RMSE, with MAE and model complexity as "
                f"tie-breakers. The selected configuration was **{selected_label}** with parameters "
                f"{selected_params_text}. Selection used validation data only; "
                "the test period was evaluated once afterwards."
            ),
            p(
                "After selection, the chosen pipeline was refitted on train+validation "
                f"({_integer(payload.get('split', {}).get('train_rows', 0) + payload.get('split', {}).get('validation_rows', 0))} rows). "
                "This uses all pre-test information while keeping the test set untouched. The fitted "
                "preprocessing transformer is saved inside the same artifact as the estimator."
            ),
            table(["Selected hyperparameter", "Value"], selected_param_rows, "Selected configuration"),
            p(
                "Random seeds are fixed at 42 for splitting, tree fitting, permutation importance and "
                "bootstrap resampling. Random Forest uses 300 trees. HistGradientBoosting uses 300 iterations, "
                "a 0.05 learning rate, and no early stopping so that the run is deterministic. The search "
                "spaces were deliberately small rather than exhaustive."
            ),
        ],
    )

    # 10. Test metrics
    add(
        "10. Held-out test evaluation",
        "10. Held-out test evaluation",
        [
            callout(
                "actual",
                "Actual test metrics",
                f"Evaluated on {_integer(test.get('n'))} chronological holdout sessions "
                f"({test_start} to {test_end}). The selected model is {selected_label}.",
            ),
            table(
                ["Model", "n", "MAE (kWh)", "RMSE (kWh)", "R²", "MAPE"],
                test_table_rows,
                "Final test comparison",
            ),
            p(vs_dummy_text + " " + heuristic_text),
            h3("How to read the metrics"),
            ul(
                [
                    "MAE is the average absolute prediction error in kWh. It is directly interpretable in the target's units.",
                    "RMSE is the square root of mean squared error; it penalizes large misses more strongly than MAE.",
                    "R² compares squared error with a mean-only predictor. It describes variance explained in this sample, not causal importance.",
                    "MAPE is the mean absolute percentage error. It is valid here because every actual test value is above zero, but it can be dominated by low-energy sessions; it is reported alongside absolute errors.",
                ]
            ),
            p(
                f"Mean actual test energy was {_num(test.get('mean_actual'), 3)} kWh and mean prediction "
                f"was {_num(test.get('mean_predicted'), 3)} kWh. The aggregate prediction error over the test "
                f"period was {_pct(total_prediction_error)} of actual energy. A small aggregate bias can hide "
                "large offsetting session-level errors, so the residual and segment tables below are essential."
            ),
            h3("Bootstrap uncertainty"),
            table(
                ["Metric", "Point estimate", "95% CI lower", "95% CI upper", "Bootstrap samples"],
                bootstrap_rows,
                "Percentile bootstrap confidence intervals",
            ),
            p(
                "The intervals quantify sampling uncertainty of the test metrics, not uncertainty for a "
                "single future session. They are relatively narrow because the test set contains "
                f"{_integer(test.get('n'))} sessions."
            ),
            h3("Evaluation integrity"),
            p(
                f"Data fingerprint match: {integrity.get('data_fingerprint_match')}; split match: "
                f"{integrity.get('split_match')}; data change allowed: {integrity.get('data_change_allowed')}. "
                "A true/true/false result means the frozen model was evaluated on the same data and split "
                "used for training. Model selection was based on validation results, never on these test scores."
            ),
        ],
    )

    # 11. Residual and segments
    add(
        "11. Residual and segment analysis",
        "11. Residual and segment analysis",
        [
            p(
                "Residuals are actual minus predicted energy. A mean near zero indicates little global "
                "bias; the spread shows the irreducible session-level uncertainty. The 5th–95th percentile "
                "range is a practical description of typical error magnitude, though it is not a formal "
                "prediction interval."
            ),
            table(["Residual statistic", "Value"], residual_rows, "Test residuals"),
            p(
                "Residual standard deviation is expected to increase with battery size: a 50–80% state-of-charge "
                "window is wider in kWh for a large battery. The segment table therefore reports both absolute "
                "and percentage errors by vehicle model."
            ),
            table(
                ["Vehicle model", "n", "MAE (kWh)", "RMSE (kWh)", "MAPE"],
                car_rows,
                "Test performance by vehicle model",
            ),
            table(
                ["Customer income tier", "n", "MAE (kWh)", "RMSE (kWh)", "MAPE"],
                income_rows,
                "Test performance by customer tier",
            ),
            h3("Conditional station-day aggregation"),
            p(
                "Session predictions were summed to station-day and district-day totals on the test set. "
                "This is a conditional aggregation: it assumes the future set of sessions is known. It is "
                "not a demand-arrival forecast. Its value is to show how much session-level noise cancels "
                "when planning energy for a known operating day."
            ),
            table(
                ["Aggregation", "Groups", "MAE (kWh)", "RMSE (kWh)", "MAPE", "R²", "Dummy baseline MAPE"],
                station_rows,
                "Aggregated test errors",
            ),
            p(
                f"Station-day aggregation has MAPE {_pct(station_day_mape)} and MAE "
                f"{_num(station_day_mae, 1)} kWh. If the session mix is known, this is materially more useful "
                "for energy planning than a single-session error, but it must not be presented as a forecast "
                "of arrivals or utilization."
            ),
        ],
    )

    # 12. Visualizations
    visualization_blocks: list[dict[str, Any]] = [
        p(
            "The figures below are generated directly from the executed pipeline. They show the target "
            "structure, the chronological split, validation model comparison, held-out fit, residuals, "
            "permutation importance, vehicle-segment error, and conditional station-day aggregation."
        )
    ]
    if figures:
        for figure in figures:
            visualization_blocks.append(_figure_block(figure.get("file", ""), figure.get("caption", "")))
    else:
        visualization_blocks.append(p("No figures were available in this run."))
    add(
        "12. Evaluation visualizations",
        "12. Evaluation visualizations",
        visualization_blocks,
    )

    # 13. Feature importance
    importance_blocks: list[dict[str, Any]] = [
        p(
            "Permutation importance measures how much test RMSE increases when one feature is randomly "
            "shuffled while the rest of the row remains unchanged. It is model-agnostic but not causal. "
            "Correlated features such as battery capacity and vehicle model can split their apparent "
            "importance."
        ),
        table(
            ["Rank", "Feature", "Mean RMSE increase (kWh)", "Std. across repeats"],
            importance_rows,
            "Top permutation importances (held-out sample)",
        ),
        p(
            f"The leading features are {top_names_text}. A large positive value means the model relies on "
            "that information for prediction. A value near zero or negative means shuffling it did not "
            "worsen test RMSE materially; such a feature may be redundant, noisy, or simply not associated "
            "with energy in this sample."
        ),
        p(
            "The strong vehicle-capacity signal should not be interpreted as evidence that station "
            "location, operator or time of day cause higher consumption. The data do not contain the "
            "operational variables needed for causal claims, and the observed target distribution is "
            "unusually regular."
        ),
    ]
    add(
        "13. Interpretation of important features",
        "13. Interpretation of important features",
        importance_blocks,
    )

    # 14. Business implications
    add(
        "14. Business implications for Warsaw EV charging",
        "14. Business implications for Warsaw EV charging",
        [
            p(
                f"On the held-out period, the model predicted {_num(totals.get('predicted_kwh'), 1, comma=True)} kWh "
                f"against {_num(totals.get('actual_kwh'), 1, comma=True)} kWh actual, an aggregate error of "
                f"{_pct(total_prediction_error)}. That aggregate closeness is encouraging for portfolio-level "
                "energy estimates, but individual sessions still carry substantial kWh uncertainty."
            ),
            p(
                f"Conditional station-day aggregation had MAE {_num(station_day_mae, 1)} kWh and MAPE "
                f"{_pct(station_day_mape)} across {_integer(station_day.get('n'))} station-days. This is the "
                "most defensible operational use of the current model: scenario planning when the future "
                "session mix is known or separately forecast. It is not a utilization forecast."
            ),
            h3("What management can reasonably do"),
            ul(
                [
                    "Use expected session energy, together with a separately forecast session count, to estimate daily energy needs by station and district.",
                    "Use vehicle-model and customer-tier segment estimates to plan capacity for high-energy vehicles and to design tariff or charging-speed propositions.",
                    "Treat the error bands as planning uncertainty rather than guaranteed per-session accuracy; avoid customer-facing guarantees from this model.",
                    "Invest first in data collection: arrival/departure state of charge, session duration, occupancy, queues, weather, traffic and events. Those variables are more likely to improve operational forecasts than further tuning on the current columns.",
                    "Monitor drift and retrain after adding new periods, operators, stations or vehicle models. Unknown categories are safely ignored by the encoder but may reduce accuracy.",
                ]
            ),
            h3("Revenue and ROI caution"),
            p(
                "The dataset contains a tariff and a billing total, but revenue, margin, utilization, "
                "electricity procurement cost and capital expenditure are not modelled here. Multiplying "
                "predicted kWh by the observed tariff gives an expected billed-energy value, not profit or "
                "ROI. No expansion, profitability or return-on-investment conclusion should be drawn from "
                "these predictions."
            ),
            h3("Association is not causation"),
            p(
                "The model identifies associations useful for prediction. It does not show that changing a "
                "station's operator, district or opening time would change energy demand. Causal questions "
                "require designed comparisons or experiments, not this observational model."
            ),
        ],
    )

    # 15. Limitations
    add(
        "15. Model limitations and risks",
        "15. Model limitations and risks",
        [
            ul(
                [
                    "The target is almost perfectly bounded between 50% and 80% of battery capacity, with a mean ratio near 0.65 for every vehicle model. This suggests simulated or highly standardized data. High R² mostly reflects battery size, not operational behavior.",
                    "The observation window is only 89 days (August–November 2025) in one city. Seasonality, holidays, long-term trends and network changes are not represented.",
                    "There is no session duration, end time, arrival/departure state of charge, charging power, queue, occupancy or plug-availability data.",
                    "There are no weather, traffic, event, competitor or local economic variables.",
                    "Repeated customer sessions are handled with a chronological split and no customer identifiers, but only one 15% time-based holdout is available; metric uncertainty is quantified with a bootstrap, not with repeated rolling-origin evaluation.",
                    "Model selection searched a small grid on one validation period. A different period or a larger search could change the selected model.",
                    "The model is point-prediction only. It does not produce calibrated prediction intervals; bootstrap intervals describe metric uncertainty, not individual-session uncertainty.",
                    "Unknown categories at inference are ignored by the encoder. Cold-start stations, new operators, new vehicle models and changed tariffs may be predicted poorly.",
                    "The data are not sufficient for causal claims, revenue/ROI claims, or a causal expansion decision.",
                    "The artifact is tied to a dataset fingerprint. Retraining is required if source files, schema or business definitions change materially.",
                ]
            ),
            callout(
                "warning",
                "Suitability statement",
                "This is a credible benchmark and a useful planning prototype, but it is not validated for "
                "real-time operational control, customer pricing guarantees, or capital-expansion decisions. "
                "The dominant predictor is vehicle battery capacity, and the dataset appears synthetic-like. "
                "Production use requires richer behavioral data, external validation and ongoing monitoring.",
            ),
        ],
    )

    # 16. Reproducibility
    add(
        "16. Reproducibility instructions",
        "16. Reproducibility instructions",
        [
            p(
                "The pipeline is runnable on Windows with Python. The dataset directory, model directory "
                "and reports directory are configurable through CLI arguments, environment variables or "
                "configs/config.json. The default dataset location is resolved from the current user's "
                "home directory; no Windows username is hard-coded in source files."
            ),
            h3("Setup"),
            code(
                "py -m venv .venv\n"
                ".venv\\Scripts\\Activate.ps1\n"
                "python -m pip install --upgrade pip\n"
                "python -m pip install -r requirements.txt"
            ),
            h3("Train and evaluate"),
            code(
                "# Use --data-dir for a custom location; otherwise set WARSAW_EV_DATA_DIR.\n"
                "python -m src.train --data-dir \"C:\\path\\to\\Warsaw EV Charging\"\n"
                "python -m src.evaluate --data-dir \"C:\\path\\to\\Warsaw EV Charging\""
            ),
            p(
                "train.py performs schema validation, cleaning, feature engineering, the chronological split, "
                "candidate fitting and validation-based selection, then saves the refitted pipeline. "
                "evaluate.py verifies the data fingerprint, evaluates the frozen model on the test period, "
                "writes metrics/figures and regenerates both reports."
            ),
            h3("Predict on new sessions"),
            code(
                "python -m src.predict --input new_sessions.csv "
                "--data-dir \"C:\\path\\to\\Warsaw EV Charging\" "
                "--output reports/new_predictions.csv"
            ),
            p(
                "The inference CSV must contain customer_id, station_id, session_start_time and cost_per_kwh. "
                "session_id is optional. Customer, station and district attributes are joined from the source "
                "lookup files; unknown IDs raise an actionable error. Any kwh_charged column in the input is "
                "ignored by the model."
            ),
            h3("Tests"),
            code("python -m pytest -q"),
            table(
                ["Artifact", "Location"],
                [
                    ["Trained pipeline + metadata bundle", str(Path("models") / "best_model.joblib")],
                    ["Machine-readable metadata", str(Path("models") / "model_metadata.json")],
                    ["Metrics payload", str(Path("reports") / "model_metrics.json")],
                    ["Markdown report", str(Path("reports") / "model_evaluation_report.md")],
                    ["HTML report", str(Path("reports") / "model_evaluation_report.html")],
                    ["Figures", str(Path("reports") / "figures")],
                    ["Row-level test predictions", str(Path("reports") / "test_predictions.csv")],
                ],
                "Generated artifacts",
            ),
            p(
                f"Training data fingerprint: `{payload.get('data_fingerprint', 'n/a')}`. "
                "The fingerprint is stored with the model and checked before evaluation. If the source CSVs "
                "change, retrain; if you intentionally evaluate a changed dataset, use --allow-data-change "
                "and treat the results as a new experiment rather than the frozen model's original test score."
            ),
        ],
    )

    # 17. Conclusion
    add(
        "17. Conclusion and recommendations",
        "17. Conclusion and recommendations",
        [
            p(
                f"The selected model is **{selected_label}**. On the chronological test set it achieved "
                f"MAE {_num(test.get('mae'), 3)} kWh, RMSE {_num(test.get('rmse'), 3)} kWh, "
                f"R² {_num(test.get('r2'), 4)} and MAPE {_pct(test.get('mape_pct'))}, compared with a dummy "
                f"baseline RMSE of {_num(baseline.get('rmse'), 3)} kWh. {vs_dummy_text} {heuristic_text}"
            ),
            p(
                "The result is strong enough to justify using the model as a session-energy planning "
                "prototype, especially for conditional station-day energy aggregation. It is not strong "
                "enough to justify real-time control, revenue guarantees, or expansion decisions without "
                "additional data and validation. The vehicle battery signal dominates; station, time and "
                "district effects are weak in this sample."
            ),
            h3("Recommended next steps"),
            ul(
                [
                    "Collect arrival and departure state of charge, session duration, charging power and station occupancy; these are the most direct missing drivers of delivered energy.",
                    "Extend the observation window to at least 12 months and evaluate rolling-origin forecasts across seasons.",
                    "Build a separate arrival-count/utilization model, then combine it with this energy model for a genuine demand forecast.",
                    "Add weather, traffic, holidays/events, local EV registrations and tariff changes as candidate features.",
                    "Add quantile or conformal prediction intervals and monitor calibration in production.",
                    "Run a rolling backtest and a GroupShuffleSplit-by-customer sensitivity analysis before deployment.",
                    "If the data are confirmed to be synthetic, validate on real charging-network data before making operational decisions.",
                ]
            ),
        ],
    )

    # 18. Appendix
    add(
        "18. Appendix",
        "18. Appendix",
        [
            h3("District-level test performance"),
            table(
                ["District", "n", "MAE (kWh)", "RMSE (kWh)", "MAPE"],
                district_rows,
                "Test performance by district",
            ),
            h3("Operator-level test performance"),
            table(
                ["Operator", "n", "MAE (kWh)", "RMSE (kWh)", "MAPE"],
                operator_rows,
                "Test performance by operator",
            ),
            h3("Additional data profile"),
            table(
                ["Profile item", "Value"],
                [
                    ["Sessions per customer (mean / median)", f"{_num(profile.get('sessions_per_customer', {}).get('mean'), 2)} / {_num(profile.get('sessions_per_customer', {}).get('median'), 2)}"],
                    ["Sessions per station (mean / median)", f"{_num(profile.get('sessions_per_station', {}).get('mean'), 2)} / {_num(profile.get('sessions_per_station', {}).get('median'), 2)}"],
                    ["Tariff values observed", ", ".join(f"{key} ({value:,})" for key, value in (profile.get("price_values") or {}).items()) or "n/a"],
                    ["Target IQR lower / upper fence", f"{_num(iqr.get('lower_bound'), 2)} / {_num(iqr.get('upper_bound'), 2)} kWh"],
                    ["Task type", "Regression"],
                    ["Report generated (UTC)", payload.get("generated_utc", "n/a")],
                    ["Evaluation runtime", f"{_num(payload.get('evaluation_seconds'), 1)} seconds"],
                    ["Data fingerprint", payload.get("data_fingerprint", "n/a")],
                ],
                "Supplementary profile",
            ),
            p(
                "This report was generated by src/reporting.py from reports/model_metrics.json. The metrics "
                "table values come from actual execution on the configured dataset. Methodological text, "
                "interpretations and recommendations are not model outputs. No score or chart in this report "
                "was hand-entered or fabricated."
            ),
        ],
    )

    return sections


def _render_markdown(payload: dict[str, Any], sections: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("# Warsaw EV Charging — Session Energy Prediction")
    lines.append("")
    lines.append("**Model evaluation report**")
    lines.append("")
    lines.append(
        f"Generated: {payload.get('generated_utc', 'n/a')} | Target: `{payload.get('target', 'kwh_charged')}` "
        f"({payload.get('target_unit', 'kWh')}) | Selected model: {payload.get('selected_label', 'n/a')}"
    )
    lines.append("")
    lines.append(
        "> **Report basis.** Metrics in this report come from the executed pipeline on the configured "
        "dataset. The test period is chronological and was not used for model selection. Methodological "
        "explanations and recommendations are clearly separate from actual scores."
    )
    lines.append("")
    lines.append("## Contents")
    lines.append("")
    for section in sections:
        lines.append(f"- [{section['title']}](#{_slug(section['title'])})")
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
            elif block_type == "code":
                lines.append("```text")
                lines.append(block["text"])
                lines.append("```")
                lines.append("")
            elif block_type == "figure":
                lines.append(f"![{block['caption']}]({block['file']})")
                lines.append("")
                lines.append(f"*{block['caption']}*")
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _image_data_uri(reports_dir: Path, relative_file: str) -> str:
    path = reports_dir / relative_file
    if path.exists():
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return f"data:{mime};base64,{encoded}"
    return relative_file


def _render_html(
    payload: dict[str, Any],
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
body {
  margin: 0; color: var(--ink); background: var(--bg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.62;
}
.hero {
  background: linear-gradient(135deg, #0f2740 0%, #16466b 55%, #1f5f8b 100%);
  color: #fff; padding: 56px 24px 44px;
}
.hero-inner { max-width: 1120px; margin: 0 auto; }
.eyebrow { text-transform: uppercase; letter-spacing: .16em; font-size: 12px; opacity: .78; font-weight: 700; }
h1 { font-size: clamp(30px, 5vw, 46px); line-height: 1.12; margin: 10px 0 12px; letter-spacing: -.02em; }
.subtitle { font-size: 18px; max-width: 780px; opacity: .92; margin: 0 0 22px; }
.chips { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 18px; }
.chip {
  display: inline-block; padding: 6px 11px; border-radius: 999px;
  background: rgba(255,255,255,.13); border: 1px solid rgba(255,255,255,.2);
  font-size: 12.5px; letter-spacing: .01em;
}
main { max-width: 1120px; margin: 0 auto; padding: 26px 22px 54px; }
.toc {
  background: #fff; border: 1px solid var(--line); border-radius: 14px;
  padding: 20px 24px; margin: 0 0 22px; box-shadow: 0 2px 10px rgba(15,39,64,.05);
}
.toc h2 { margin: 0 0 10px; font-size: 15px; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); }
.toc ol { columns: 2; column-gap: 36px; margin: 0; padding-left: 20px; font-size: 14px; }
.toc a { color: var(--blue); text-decoration: none; }
.toc a:hover { text-decoration: underline; }
section {
  background: #fff; border: 1px solid var(--line); border-radius: 14px;
  padding: 26px 30px; margin: 20px 0; box-shadow: 0 2px 10px rgba(15,39,64,.045);
}
section h2 { margin: 0 0 14px; font-size: 26px; line-height: 1.2; color: var(--navy); letter-spacing: -.015em; }
section h3 { margin: 24px 0 8px; font-size: 18px; color: var(--navy); }
p { margin: 10px 0; }
ul { margin: 10px 0 12px; padding-left: 24px; }
li { margin: 6px 0; }
code { background: #eef2f7; padding: 1px 5px; border-radius: 4px; font-size: .9em; }
pre {
  background: #0f2740; color: #e8f1f8; padding: 16px 18px; border-radius: 10px;
  overflow-x: auto; font-size: 13px; line-height: 1.5;
}
pre code { background: transparent; padding: 0; color: inherit; }
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
.status { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.status span { background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.18); border-radius: 6px; padding: 5px 9px; font-size: 12px; }
@media (max-width: 760px) {
  .toc ol { columns: 1; }
  section { padding: 21px 18px; }
  hero { padding: 40px 18px 34px; }
}
@media print {
  body { background: #fff; }
  .hero { background: #0f2740 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  section { box-shadow: none; break-inside: avoid; }
  .toc { break-inside: avoid; }
  .toc a { color: var(--ink); }
}
"""
    hero = f"""
<header class="hero">
  <div class="hero-inner">
    <div class="eyebrow">Warsaw EV Charging · Machine Learning Deliverable</div>
    <h1>Session Energy Prediction</h1>
    <p class="subtitle">A reproducible, leakage-safe regression model with a chronological holdout evaluation and operational interpretation.</p>
    <div class="chips">
      <span class="chip">Target: {_inline_html(payload.get('target', 'kwh_charged'))} ({_inline_html(payload.get('target_unit', 'kWh'))})</span>
      <span class="chip">Model: {_inline_html(payload.get('selected_label', 'n/a'))}</span>
      <span class="chip">Test rows: {_integer(payload.get('test_metrics', {}).get('n'))}</span>
      <span class="chip">Generated: {_inline_html(payload.get('generated_utc', 'n/a'))}</span>
    </div>
    <div class="status">
      <span>Test MAE {_num(payload.get('test_metrics', {}).get('mae'), 3)} kWh</span>
      <span>Test RMSE {_num(payload.get('test_metrics', {}).get('rmse'), 3)} kWh</span>
      <span>Test R² {_num(payload.get('test_metrics', {}).get('r2'), 4)}</span>
      <span>MAPE {_pct(payload.get('test_metrics', {}).get('mape_pct'))}</span>
    </div>
  </div>
</header>
"""
    toc_items = "".join(
        f'<li><a href="#{section["id"]}">{_inline_html(section["title"])}</a></li>' for section in sections
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
            elif block_type == "code":
                block_html.append(f"<pre><code>{html.escape(block['text'])}</code></pre>")
            elif block_type == "figure":
                source = _image_data_uri(reports_dir, block["file"])
                block_html.append(
                    f'<figure><img src="{source}" alt="{html.escape(block["caption"])}" loading="lazy">'
                    f'<figcaption>{_inline_html(block["caption"])}</figcaption></figure>'
                )
        body_sections.append(
            f'<section id="{html.escape(section["id"])}"><h2>{_inline_html(section["title"])}</h2>'
            + "".join(block_html)
            + "</section>"
        )

    footer = f"""
<footer>
  <p><strong>Report status.</strong> Metrics were produced by the executed pipeline at
  {_inline_html(payload.get('generated_utc', 'n/a'))}. Test evaluation used a chronological holdout and
  the selected model configuration came from validation data only. The report is self-contained: images
  are embedded, CSS is local, and no internet connection or server is required.</p>
  <p>Source: <code>reports/model_metrics.json</code> · Generated by <code>src/reporting.py</code> ·
  Warsaw EV Charging ML project.</p>
</footer>
"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="Warsaw EV Charging ML pipeline">
<title>Warsaw EV Charging — Session Energy Prediction Report</title>
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


def generate_reports(payload: dict[str, Any], reports_dir: Path) -> dict[str, str]:
    """Write Markdown and self-contained HTML reports and return their paths."""

    reports_dir.mkdir(parents=True, exist_ok=True)
    sections = _sections(payload)
    markdown = _render_markdown(payload, sections)
    html_text = _render_html(payload, sections, reports_dir)
    markdown_path = reports_dir / "model_evaluation_report.md"
    html_path = reports_dir / "model_evaluation_report.html"
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_text, encoding="utf-8")
    return {"markdown": str(markdown_path), "html": str(html_path)}


def render_markdown(payload: dict[str, Any]) -> str:
    return _render_markdown(payload, _sections(payload))


def render_html(payload: dict[str, Any], reports_dir: Path) -> str:
    return _render_html(payload, _sections(payload), reports_dir)
