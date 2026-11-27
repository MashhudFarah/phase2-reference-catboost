"""Canonical Phase 2 feature engineering and forecast-target contract."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.feature_schema import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION

TARGET_COLUMN = "dengue_cases"
CATEGORICAL_FEATURES: tuple[str, ...] = ()
NUMERIC_FEATURES = FEATURE_COLUMNS
SUPPORTED_HORIZONS = (2, 3, 4)
TARGET_CONSTRUCTION_MAX_WEEKS = 52
SOURCE_COLUMNS = (
    "date", "city", "ward_id", "population_density_sqkm",
    "water_logging_risk_index", "sanitation_score", "avg_temp_C",
    "rainfall_mm", "relative_humidity_pct", "dengue_cases",
)


def validate_feature_schema(frame: pd.DataFrame) -> None:
    """Check required ordered features and reject legacy/ineligible feature lists."""
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Processed data is missing required feature columns: {missing}")
    if len(FEATURE_COLUMNS) != 34 or len(set(FEATURE_COLUMNS)) != 34:
        raise ValueError("Active feature schema is not the canonical 34-feature contract")


def usable_feature_mask(frame: pd.DataFrame) -> pd.Series:
    """Identify complete finite feature rows; early lag warm-up rows stay historical."""
    validate_feature_schema(frame)
    matrix = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    values = matrix.to_numpy(dtype=float)
    return pd.Series(np.isfinite(values).all(axis=1), index=frame.index)


def engineer_features(frame: pd.DataFrame, *, require_weekly: bool = True) -> pd.DataFrame:
    """Build the shared 34 inputs from raw weekly rows, independently per ward.

    Rolling rainfall includes the current observation. Temperature and humidity
    means use the four observations strictly before the forecast origin.
    """
    missing = [column for column in SOURCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing source columns: {missing}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise", utc=True).dt.tz_convert(None).dt.normalize()
    if data.duplicated(["ward_id", "date"]).any():
        raise ValueError("Dataset contains duplicate ward/date observations")
    if not data["date"].dt.day_of_week.eq(0).all():
        raise ValueError("Weekly observations must be aligned to Mondays")
    data = data.sort_values(["ward_id", "date"], kind="stable").reset_index(drop=True)
    grouped = data.groupby("ward_id", sort=False, observed=True)
    if require_weekly:
        gap = grouped["date"].diff().dropna()
        if not gap.eq(pd.Timedelta(days=7)).all():
            raise ValueError("Weekly continuity failed within one or more wards")
    numeric_sources = [column for column in SOURCE_COLUMNS if column not in {"date", "city", "ward_id"}]
    for column in numeric_sources:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if not np.isfinite(data[numeric_sources].to_numpy(dtype=float)).all():
        raise ValueError("Source features contain missing or non-finite values")
    if (data["population_density_sqkm"] <= 0).any():
        raise ValueError("population_density_sqkm must be positive")
    for column in ("water_logging_risk_index", "sanitation_score"):
        if not data[column].between(0, 1).all():
            raise ValueError(f"{column} must be within [0, 1]")
    if not data["relative_humidity_pct"].between(0, 100).all():
        raise ValueError("relative_humidity_pct must be within [0, 100]")
    if not data["avg_temp_C"].between(-90, 60).all():
        raise ValueError("avg_temp_C must be within plausible bounds [-90, 60] °C")
    if (data[["rainfall_mm", "dengue_cases"]] < 0).any().any():
        raise ValueError("Rainfall and dengue cases must be non-negative")
    if not np.equal(data["dengue_cases"], np.floor(data["dengue_cases"])).all():
        raise ValueError("dengue_cases must contain non-negative integer counts")

    for source, prefix in (("dengue_cases", "dengue_cases"),
                           ("rainfall_mm", "rain"),
                           ("avg_temp_C", "avg_temp_C"),
                           ("relative_humidity_pct", "relative_humidity_pct")):
        for lag in range(1, 5):
            data[f"{prefix}_lag_{lag}w"] = grouped[source].shift(lag)

    grouped = data.groupby("ward_id", sort=False, observed=True)
    for window in (2, 3, 4, 6, 8):
        data[f"rain_sum_{window}w"] = grouped["rainfall_mm"].transform(
            lambda values, size=window: values.rolling(size, min_periods=size).apply(
                math.fsum, raw=True,
            )
        )
    for source, name in (("avg_temp_C", "avg_temp_C_mean_prev_4w"),
                         ("relative_humidity_pct", "relative_humidity_pct_mean_prev_4w")):
        data[name] = grouped[source].transform(
            lambda values: values.shift(1).rolling(4, min_periods=4).apply(
                lambda window: math.fsum(window) / 4, raw=True,
            )
        )
    iso_week = data["date"].dt.isocalendar().week.astype(int)
    data["calendar_week"] = iso_week
    data["calendar_month"] = data["date"].dt.month
    data["week_sin"] = np.sin(2 * np.pi * data["calendar_week"] / 52)
    data["week_cos"] = np.cos(2 * np.pi * data["calendar_week"] / 52)
    return data


def construct_targets(frame: pd.DataFrame, horizons: Iterable[int] = SUPPORTED_HORIZONS) -> pd.DataFrame:
    """Add same-ward future dengue values and dates after chronological sorting.

    Input is expected to have passed weekly-continuity validation. That makes a
    grouped shift of *h* observations exactly an h-week forecast.
    """
    result = frame.copy()
    if "date" not in result or "ward_id" not in result or TARGET_COLUMN not in result:
        raise ValueError("Target construction requires date, ward_id, and dengue_cases")
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values(["ward_id", "date"], kind="stable").reset_index(drop=True)
    grouped = result.groupby("ward_id", sort=False, observed=True)
    for horizon in horizons:
        if not isinstance(horizon, (int, np.integer)) or not 1 <= horizon <= TARGET_CONSTRUCTION_MAX_WEEKS:
            raise ValueError(
                f"Target-construction horizon must be 1–{TARGET_CONSTRUCTION_MAX_WEEKS} weeks"
            )
        result[f"target_{horizon}w"] = grouped[TARGET_COLUMN].shift(-horizon)
        result[f"target_date_{horizon}w"] = grouped["date"].shift(-horizon)
    return result
