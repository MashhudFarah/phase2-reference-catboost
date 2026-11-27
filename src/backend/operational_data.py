"""Validated access to the stable ward-capacity CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPERATIONAL_CAPACITY_PATH = PROJECT_ROOT / "data" / "processed" / "operational_capacity.csv"
WARD_FIELDS = (
    "ward_id", "city", "population_density_sqkm", "water_logging_risk_index",
    "sanitation_score", "vulnerability_index", "mean_weekly_cases_2016_2023",
    "estimated_bed_capacity", "actual_available_beds", "saline_stock_available",
    "diagnostic_kits_stock_available", "crew_count_available", "last_updated",
    "data_source", "bed_capacity_method", "operational_data_status",
)


class OperationalDataError(ValueError):
    """The ward reference CSV is missing, malformed, or inconsistent."""


def read_ward_capacity(path: Path | None = None) -> pd.DataFrame:
    """Read the CSV and validate its existing 16-column, one-row-per-ward contract."""
    source = path or OPERATIONAL_CAPACITY_PATH
    if not source.is_file():
        raise OperationalDataError("Ward operational capacity data is unavailable")
    try:
        data = pd.read_csv(source, keep_default_na=True)
    except (OSError, pd.errors.ParserError) as exc:
        raise OperationalDataError("Ward operational capacity CSV cannot be read") from exc
    if list(data.columns) != list(WARD_FIELDS):
        raise OperationalDataError("Ward operational capacity CSV does not match the preserved schema")
    if len(data) != 60 or data["ward_id"].isna().any() or data["ward_id"].duplicated().any():
        raise OperationalDataError("Ward operational capacity CSV must contain 60 unique wards")
    if data["city"].isna().any() or data["city"].nunique() != 5:
        raise OperationalDataError("Ward operational capacity CSV must contain five cities")
    if pd.to_numeric(data["estimated_bed_capacity"], errors="coerce").isna().any():
        raise OperationalDataError("Estimated bed capacity contains invalid values")
    return data


