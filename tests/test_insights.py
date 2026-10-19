"""Smoke tests for the insights report generator.

These tests use the synthetic fixture only; the production report is generated
from the real Warsaw EV CSVs and the verified model metrics.
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import RawTables
from src.insights import (
    _render_html,
    _render_markdown,
    _sections,
    compute_insights,
    create_figures,
)


def _with_wola_and_wesola(tables: RawTables) -> RawTables:
    """Add the two districts referenced by the report narrative."""

    districts = pd.concat(
        [
            tables.districts,
            pd.DataFrame(
                {
                    "district_name": ["Wola", "Wesoła"],
                    "income_tier": ["Mid-Range", "Low-Mid"],
                    "projected_evs": [3300, 700],
                    "projected_plugs": [100, 21],
                    "projected_regular_customers": [660, 140],
                }
            ),
        ],
        ignore_index=True,
    )
    stations = pd.concat(
        [
            tables.stations,
            pd.DataFrame(
                {
                    "station_id": [91, 92],
                    "district_name": ["Wola", "Wesoła"],
                    "income_tier": ["Mid-Range", "Low-Mid"],
                    "operator_name": ["GreenWay", "Tauron"],
                    "plugs_count": [6, 6],
                    "latitude": [52.24, 52.25],
                    "longitude": [20.96, 21.05],
                }
            ),
        ],
        ignore_index=True,
    )
    return RawTables(
        sessions=tables.sessions,
        stations=stations,
        customers=tables.customers,
        districts=districts,
        file_info=tables.file_info,
        fingerprint=tables.fingerprint,
    )


def test_compute_insights_on_synthetic_data(synthetic_tables: RawTables) -> None:
    tables = _with_wola_and_wesola(synthetic_tables)
    insights = compute_insights(tables, {"available": False})

    assert insights["source_rows"]["sessions"] == len(tables.sessions)
    assert insights["capacity"]["served_districts"] == 4
    assert insights["capacity"]["unserved_districts"] == 0
    assert insights["usage"]["total_sessions"] == len(tables.sessions)
    assert insights["revenue"]["price"]["points"]
    assert insights["customers"]["tiers"]
    assert insights["ml"]["available"] is False


def test_report_rendering_and_figures(synthetic_tables: RawTables, tmp_path) -> None:
    tables = _with_wola_and_wesola(synthetic_tables)
    insights = compute_insights(tables, {"available": False})
    figures = create_figures(insights, tmp_path / "figures")

    assert len(figures) == 6
    for record in figures:
        path = tmp_path / record["file"]
        assert path.exists()
        assert path.stat().st_size > 0

    sections = _sections(insights, figures)
    markdown = _render_markdown(insights, sections)
    html = _render_html(insights, sections, tmp_path)

    assert "Airline wording notice" in markdown
    assert "Actionable recommendations" in markdown
    assert "data:image/png;base64," in html
    assert "http://" not in html
    assert "https://" not in html
    assert markdown.count("](figures/insights_") == len(figures)
