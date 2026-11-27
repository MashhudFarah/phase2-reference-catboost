"""Transparent prototype risk bands and observed-input context."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import pandas as pd

RISK_LEVELS = ("LOW", "MODERATE", "HIGH", "CRITICAL")
GLOBAL_FALLBACK_THRESHOLDS = {"low_max": 5.0, "medium_max": 9.0, "high_max": 14.0}
RISK_METHOD = "city-pooled positive weekly dengue cases, 2016-2023; empirical, not clinically validated"


@lru_cache(maxsize=4)
def load_risk_calibration_data(path: str) -> pd.DataFrame:
    """Load the complete historical case series, including feature warm-up rows."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Risk calibration data is unavailable: {source}")
    frame = pd.read_csv(source, usecols=["date", "city", "dengue_cases"])
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["dengue_cases"] = pd.to_numeric(frame["dengue_cases"], errors="raise")
    if frame[["date", "city", "dengue_cases"]].isna().any().any():
        raise ValueError("Risk calibration data contains missing required values")
    return frame


def risk_level_index(level: str) -> int:
    """Return the ordinal position used only for tier-distance metrics."""
    try:
        return RISK_LEVELS.index(str(level))
    except ValueError as exc:
        raise ValueError(f"Unknown risk level {level!r}") from exc


def derive_city_thresholds(frame: pd.DataFrame, city: str) -> dict[str, float]:
    """Derive common city-wide P50/P75/P90 cutoffs using 2016-2023 active weeks."""
    required = {"date", "city", "dengue_cases"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Risk calibration data is missing columns: {sorted(required-set(frame.columns))}")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    cases = pd.to_numeric(frame["dengue_cases"], errors="coerce")
    selected = frame["city"].astype(str).eq(str(city)) & dates.dt.year.between(2016, 2023) & cases.gt(0)
    active = cases.loc[selected].dropna()
    if active.empty:
        raise ValueError(f"Cannot derive city-pooled risk cutoffs for {city!r} from 2016-2023 data")
    low, moderate, high = (float(active.quantile(q)) for q in (0.50, 0.75, 0.90))
    return {"low_max": low, "medium_max": moderate, "high_max": high}


def risk_thresholds(_historical_cases: pd.Series | None = None) -> dict[str, float]:
    """Return a fixed pooled-data fallback; production callers pass city cutoffs."""
    if _historical_cases is not None and pd.to_numeric(_historical_cases, errors="coerce").dropna().empty:
        raise ValueError("Cannot classify risk without historical case observations")
    return dict(GLOBAL_FALLBACK_THRESHOLDS)


def classify_risk(
    predicted_cases: float,
    historical_cases: pd.Series | None = None,
    *,
    thresholds: dict[str, float] | None = None,
) -> dict:
    """Classify against absolute pooled cutoffs, never ward-relative percentiles."""
    value = float(predicted_cases)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Predicted cases must be a finite, non-negative number")
    if thresholds is None:
        thresholds = risk_thresholds(historical_cases)
    required = {"low_max", "medium_max", "high_max"}
    if not required.issubset(thresholds) or not (
        0 <= float(thresholds["low_max"]) <= float(thresholds["medium_max"]) <= float(thresholds["high_max"])
    ):
        raise ValueError("Risk cutoffs must be ordered, non-negative absolute case counts")
    if value < 1:
        level = "LOW"
    elif value <= thresholds["low_max"]:
        level = "LOW"
    elif value <= thresholds["medium_max"]:
        level = "MODERATE"
    elif value <= thresholds["high_max"]:
        level = "HIGH"
    else:
        level = "CRITICAL"
    return {
        "level": level,
        "method": RISK_METHOD,
        "thresholds": thresholds,
        "warning": (
            "Empirical pooled case bands for prioritization only; not clinically validated public-health alerts or probabilities."
        ),
    }


def ward_priority_score(vulnerability_index: float, predicted_cases: float, thresholds: dict[str, float]) -> float:
    """Equal-weight planning index combining structural vulnerability and forecast risk."""
    vulnerability = float(vulnerability_index)
    cases = float(predicted_cases)
    if not (0 <= vulnerability <= 1) or not math.isfinite(cases) or cases < 0:
        raise ValueError("Priority inputs must be finite; vulnerability must be within [0,1]")
    risk_scale = max(float(thresholds["high_max"]), 1.0)
    normalized_cases = min(cases / risk_scale, 1.0)
    return 0.5 * vulnerability + 0.5 * normalized_cases


def _relative_label(value: float, historical: pd.Series, unit: str) -> str:
    values = pd.to_numeric(historical, errors="coerce").dropna()
    if values.empty:
        return f"{value:.1f}{unit}; historical comparison unavailable"
    relation = "above" if value > float(values.quantile(0.75)) else "within or below"
    return f"{value:.1f}{unit}; {relation} the ward's historical 75th percentile"


def describe_drivers(ward_history: pd.DataFrame, latest: pd.Series) -> dict[str, str]:
    """Summarize observed context without claiming feature attribution or causality."""
    return {
        "rainfall": _relative_label(float(latest["rainfall_mm"]), ward_history["rainfall_mm"], " mm"),
        "temperature": _relative_label(float(latest["avg_temp_C"]), ward_history["avg_temp_C"], " °C"),
        "humidity": _relative_label(float(latest["relative_humidity_pct"]), ward_history["relative_humidity_pct"], "%"),
        "recentCases": (
            f"{float(latest['dengue_cases']):g} new cases in the observed reporting week; "
            "context only, not a causal explanation"
        ),
    }
