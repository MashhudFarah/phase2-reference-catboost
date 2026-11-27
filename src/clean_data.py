#!/usr/bin/env python3
"""Validate the Phase 2 weekly dengue CSV and prepare leakage-aware history.

This cleaner preserves every valid historical row. Missing source observations
or malformed records stop the run instead of being imputed or silently dropped.
Rainfall values are validated as finite and non-negative; no outlier filtering
is applied. Time-dependent features and future targets are grouped by ward.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.feature_engineering import construct_targets, engineer_features
from src.feature_schema import FEATURE_COLUMNS

DEFAULT_INPUT = ROOT / "data" / "raw" / "bangladesh_dengue_large_dataset_onsite - bangladesh_dengue_large_dataset.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "processed_historical_data.csv"

EXPECTED_ROWS = 31_320
EXPECTED_WARDS = 60
EXPECTED_CITIES = 5

REQUIRED_COLUMNS = [
    "date", "city", "ward_id", "population_density_sqkm",
    "water_logging_risk_index", "sanitation_score", "avg_temp_c",
    "max_temp_c", "min_temp_c", "rainfall_mm",
    "relative_humidity_pct", "dengue_cases", "saline_bags_needed",
    "diagnostic_kits_needed", "hospital_beds_required",
]
NUMERIC_COLUMNS = [column for column in REQUIRED_COLUMNS if column not in {"date", "city", "ward_id"}]
SUPPLY_COLUMNS = (
    "saline_bags_needed", "diagnostic_kits_needed", "hospital_beds_required",
)
IDENTIFIER_COLUMNS = ("date", "ward_id", "city")
CITY_PREFIXES = {
    "Chattogram": "CCC",
    "Dhaka North": "DNCC",
    "Dhaka South": "DSCC",
    "Gazipur": "GCC",
    "Narayanganj": "NCC",
}
WARD_PATTERN = re.compile(r"^([A-Z]+)_Ward_(\d{2})$")
MISSING_MARKERS = {"", "na", "n/a", "null", "none", "nan"}
HORIZONS = (2, 3, 4)
RAIN_SUM_WINDOWS = (2, 3, 4, 6, 8)
LAG_WEEKS = (1, 2, 3, 4)

GENERATED_FEATURE_COLUMNS = [
    *(f"dengue_cases_lag_{week}w" for week in LAG_WEEKS),
    *(f"rain_lag_{week}w" for week in LAG_WEEKS),
    *(f"avg_temp_C_lag_{week}w" for week in LAG_WEEKS),
    *(f"relative_humidity_pct_lag_{week}w" for week in LAG_WEEKS),
    *(f"rain_sum_{window}w" for window in RAIN_SUM_WINDOWS),
    "avg_temp_C_mean_prev_4w",
    "relative_humidity_pct_mean_prev_4w",
    "calendar_week", "calendar_month", "week_sin", "week_cos",
]
TARGET_COLUMNS = [
    *(f"target_{horizon}w" for horizon in HORIZONS),
    *(f"target_date_{horizon}w" for horizon in HORIZONS),
]
OUTPUT_RAW_COLUMNS = [
    "date", "ward_id", "city", "population_density_sqkm",
    "water_logging_risk_index", "sanitation_score", "avg_temp_C",
    "max_temp_c", "min_temp_c", "rainfall_mm",
    "relative_humidity_pct", "dengue_cases", *SUPPLY_COLUMNS,
]
# Model inputs are owned by the shared feature schema. Reference identifiers,
# supplies, extrema, and all future targets remain outside that allowlist.
MODEL_FEATURE_COLUMNS = list(FEATURE_COLUMNS)


class DatasetValidationError(ValueError):
    """Raised when the source cannot safely be prepared without guessing."""


def _format_number(value: float) -> str:
    if not math.isfinite(value):
        raise DatasetValidationError("An engineered numeric value is non-finite")
    return str(int(value)) if value.is_integer() else format(value, ".12g")


def _read_and_validate_source(path: Path) -> tuple[list[dict[str, str]], list[date]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input dataset does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise DatasetValidationError("Input CSV has no header")
            columns = list(reader.fieldnames)
            if len(columns) != len(set(columns)):
                raise DatasetValidationError("Input CSV has duplicate column names")
            missing_headers = sorted(set(REQUIRED_COLUMNS) - set(columns))
            extra_headers = sorted(set(columns) - set(REQUIRED_COLUMNS))
            if missing_headers or extra_headers:
                raise DatasetValidationError(
                    f"Unexpected input schema; missing={missing_headers}, extra={extra_headers}"
                )
            rows: list[dict[str, str]] = []
            malformed_lines: list[int] = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    malformed_lines.append(reader.line_num)
                    continue
                rows.append(row)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DatasetValidationError(f"Could not read input CSV: {exc}") from exc
    if malformed_lines:
        raise DatasetValidationError(f"Malformed CSV row(s) at line(s): {malformed_lines[:10]}")
    if len(rows) != EXPECTED_ROWS:
        raise DatasetValidationError(f"Expected {EXPECTED_ROWS:,} input rows, found {len(rows):,}")
    if not rows:
        raise DatasetValidationError("Input CSV contains no data rows")

    missing_counts = Counter()
    for line_no, row in enumerate(rows, start=2):
        for column in REQUIRED_COLUMNS:
            if row[column].strip().lower() in MISSING_MARKERS:
                missing_counts[column] += 1
    if missing_counts:
        raise DatasetValidationError(f"Missing source values by column: {dict(missing_counts)}")

    dates: list[date] = []
    numeric_rows: list[dict[str, float]] = []
    bad_values: list[str] = []
    for line_no, row in enumerate(rows, start=2):
        text = row["date"].strip()
        try:
            parsed = date.fromisoformat(text)
        except ValueError:
            bad_values.append(f"line {line_no}: invalid date")
            continue
        if parsed.isoformat() != text:
            bad_values.append(f"line {line_no}: date must use YYYY-MM-DD")
        elif parsed.weekday() != 0:
            bad_values.append(f"line {line_no}: date is not Monday")
        dates.append(parsed)
        parsed_numbers: dict[str, float] = {}
        for column in NUMERIC_COLUMNS:
            try:
                value = float(row[column])
            except (TypeError, ValueError):
                bad_values.append(f"line {line_no}: non-numeric {column}")
                continue
            if not math.isfinite(value):
                bad_values.append(f"line {line_no}: non-finite {column}")
                continue
            parsed_numbers[column] = value
        numeric_rows.append(parsed_numbers)
    if bad_values:
        raise DatasetValidationError("Invalid source values: " + "; ".join(bad_values[:12]))

    # Detect both exact duplicate rows and duplicate ward/date keys before any
    # feature generation. Invalid duplicates stop processing; no rows are dropped.
    seen_full_rows: set[tuple[str, ...]] = set()
    duplicate_full_count = 0
    seen_keys: set[tuple[str, date]] = set()
    duplicate_keys: list[tuple[str, str]] = []
    for row, parsed_date in zip(rows, dates):
        full_key = tuple(row[column] for column in REQUIRED_COLUMNS)
        if full_key in seen_full_rows:
            duplicate_full_count += 1
        seen_full_rows.add(full_key)
        ward_date = (row["ward_id"], parsed_date)
        if ward_date in seen_keys:
            duplicate_keys.append((row["ward_id"], parsed_date.isoformat()))
        seen_keys.add(ward_date)
    if duplicate_full_count or duplicate_keys:
        raise DatasetValidationError(
            f"Duplicate records found: exact_full_rows={duplicate_full_count}, "
            f"ward_date_keys={len(duplicate_keys)}, examples={duplicate_keys[:5]}"
        )

    city_values = {row["city"] for row in rows}
    ward_values = {row["ward_id"] for row in rows}
    if len(ward_values) != EXPECTED_WARDS or len(city_values) != EXPECTED_CITIES:
        raise DatasetValidationError(
            f"Expected {EXPECTED_WARDS} wards and {EXPECTED_CITIES} cities; "
            f"found {len(ward_values)} wards and {len(city_values)} cities"
        )
    unexpected_cities = sorted(city_values - set(CITY_PREFIXES))
    missing_cities = sorted(set(CITY_PREFIXES) - city_values)
    if unexpected_cities or missing_cities:
        raise DatasetValidationError(
            f"Unexpected city labels: unknown={unexpected_cities}, missing={missing_cities}"
        )
    ward_city: dict[str, str] = {}
    for row in rows:
        city, ward = row["city"], row["ward_id"]
        match = WARD_PATTERN.fullmatch(ward)
        if not match or match.group(1) != CITY_PREFIXES[city]:
            raise DatasetValidationError(
                f"Ward ID/city-prefix mismatch for ward {ward!r} and city {city!r}"
            )
        previous_city = ward_city.setdefault(ward, city)
        if previous_city != city:
            raise DatasetValidationError(f"Ward {ward!r} appears under multiple cities")

    # Per-field physical/domain validation. Rainfall has no upper bound here so
    # legitimate extreme rainfall observations are preserved exactly.
    errors: list[str] = []
    for line_no, values in enumerate(numeric_rows, start=2):
        if values["population_density_sqkm"] <= 0:
            errors.append(f"line {line_no}: population_density_sqkm must be positive")
        for column in ("water_logging_risk_index", "sanitation_score"):
            if not 0 <= values[column] <= 1:
                errors.append(f"line {line_no}: {column} must be within [0, 1]")
        if values["rainfall_mm"] < 0:
            errors.append(f"line {line_no}: rainfall_mm cannot be negative")
        if not 0 <= values["relative_humidity_pct"] <= 100:
            errors.append(f"line {line_no}: relative_humidity_pct must be within [0, 100]")
        for column in ("min_temp_c", "avg_temp_c", "max_temp_c"):
            if not -90 <= values[column] <= 60:
                errors.append(f"line {line_no}: {column} outside plausible Celsius bounds [-90, 60]")
        if not values["min_temp_c"] <= values["avg_temp_c"] <= values["max_temp_c"]:
            errors.append(f"line {line_no}: min/average/max temperature ordering is invalid")
        for column in ("dengue_cases", *SUPPLY_COLUMNS):
            value = values[column]
            if value < 0 or not value.is_integer():
                errors.append(f"line {line_no}: {column} must be a non-negative integer")
    if errors:
        raise DatasetValidationError("Invalid physical/count values: " + "; ".join(errors[:12]))

    indices_by_ward: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_ward[row["ward_id"]].append(index)
    continuity_errors: list[str] = []
    for ward, indices in indices_by_ward.items():
        indices.sort(key=lambda index: dates[index])
        ward_dates = [dates[index] for index in indices]
        broken = [
            (left, right) for left, right in zip(ward_dates, ward_dates[1:])
            if (right - left).days != 7
        ]
        if broken:
            continuity_errors.append(
                f"{ward}: {len(broken)} non-seven-day interval(s), "
                f"examples={[(a.isoformat(), b.isoformat()) for a, b in broken[:3]]}"
            )
    if continuity_errors:
        raise DatasetValidationError("Ward weekly continuity failed: " + "; ".join(continuity_errors[:10]))

    return rows, dates


def _engineer_rows(
    rows: list[dict[str, str]],
    dates: list[date],
) -> tuple[list[dict[str, str]], list[str]]:
    # Feature formulas live in feature_engineering.py and are shared with all
    # inference/training backends. This cleaner only handles CSV validation and
    # serialization, so it cannot drift into a competing lag/rolling contract.
    source_frame = pd.DataFrame([
        {**row, "date": dates[index].isoformat(), "avg_temp_C": row["avg_temp_c"]}
        for index, row in enumerate(rows)
    ])
    engineered = construct_targets(engineer_features(source_frame), HORIZONS)
    output_rows: list[dict[str, str]] = []
    for row in engineered.to_dict(orient="records"):
        day = pd.Timestamp(row["date"]).date().isoformat()
        result: dict[str, str] = {
            "date": day, "ward_id": str(row["ward_id"]), "city": str(row["city"]),
        }
        for column in OUTPUT_RAW_COLUMNS:
            if column in IDENTIFIER_COLUMNS:
                continue
            source_column = "avg_temp_c" if column == "avg_temp_C" else column
            result[column] = _format_number(float(row[source_column]))
        for column in GENERATED_FEATURE_COLUMNS:
            value = row[column]
            result[column] = "" if pd.isna(value) else _format_number(float(value))
        for horizon in HORIZONS:
            target = row[f"target_{horizon}w"]
            target_date = row[f"target_date_{horizon}w"]
            result[f"target_{horizon}w"] = "" if pd.isna(target) else _format_number(float(target))
            result[f"target_date_{horizon}w"] = (
                "" if pd.isna(target_date) else pd.Timestamp(target_date).date().isoformat()
            )
        output_rows.append(result)
    output_rows.sort(key=lambda row: (row["ward_id"], row["date"]))
    output_columns = [*OUTPUT_RAW_COLUMNS, *GENERATED_FEATURE_COLUMNS, *TARGET_COLUMNS]
    if len(output_columns) != len(set(output_columns)):
        raise AssertionError("Output schema contains duplicate columns")
    if set(SUPPLY_COLUMNS) & set(MODEL_FEATURE_COLUMNS):
        raise AssertionError("Supply-derived columns leaked into the model feature allowlist")
    forbidden_features = set(IDENTIFIER_COLUMNS) | set(SUPPLY_COLUMNS) | {
        "max_temp_c", "min_temp_c", *TARGET_COLUMNS,
    }
    if forbidden_features & set(MODEL_FEATURE_COLUMNS):
        raise AssertionError("A forbidden reference/target column is in the model feature allowlist")
    return output_rows, output_columns


def _verify_output(
    output_path: Path,
    source_rows: list[dict[str, str]],
    source_dates: list[date],
    expected_columns: list[str],
) -> tuple[int, set[str], set[str]]:
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise DatasetValidationError("Generated CSV header differs from the prepared schema")
        output_rows = list(reader)
    if len(output_rows) != len(source_rows):
        raise DatasetValidationError(
            f"Generated row count mismatch: expected {len(source_rows)}, found {len(output_rows)}"
        )
    seen: set[tuple[str, str]] = set()
    ward_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in output_rows:
        key = (row["ward_id"], row["date"])
        if key in seen:
            raise DatasetValidationError(f"Generated output duplicates ward/date key {key}")
        seen.add(key)
        ward_rows[row["ward_id"]].append(row)
    if len(ward_rows) != EXPECTED_WARDS:
        raise DatasetValidationError(f"Output has {len(ward_rows)} wards, expected {EXPECTED_WARDS}")

    # Recheck target alignment, lag direction, current-inclusive rainfall sums,
    # and prior-only rolling means against each ward's ordered source records.
    by_ward_source: dict[str, list[int]] = defaultdict(list)
    for index, raw in enumerate(source_rows):
        by_ward_source[raw["ward_id"]].append(index)
    output_by_key = {(row["ward_id"], row["date"]): row for row in output_rows}
    for ward, indices in by_ward_source.items():
        indices.sort(key=lambda i: source_dates[i])
        for pos, raw_index in enumerate(indices):
            current = output_by_key[(ward, source_dates[raw_index].isoformat())]
            for lag in LAG_WEEKS:
                expected = _format_number(float(source_rows[indices[pos - lag]]["dengue_cases"])) if pos >= lag else ""
                if current[f"dengue_cases_lag_{lag}w"] != expected:
                    raise DatasetValidationError(f"Dengue lag verification failed for {ward} at {current['date']}")
            for window in RAIN_SUM_WINDOWS:
                start = pos - window + 1
                expected = _format_number(sum(float(source_rows[indices[j]]["rainfall_mm"]) for j in range(start, pos + 1))) if start >= 0 else ""
                if current[f"rain_sum_{window}w"] != expected:
                    raise DatasetValidationError(f"Rainfall sum verification failed for {ward} at {current['date']}")
            for horizon in HORIZONS:
                target_pos = pos + horizon
                if target_pos < len(indices):
                    target_raw = source_rows[indices[target_pos]]
                    if target_raw["ward_id"] != ward:
                        raise DatasetValidationError("Target crossed a ward boundary")
                    if current[f"target_{horizon}w"] != _format_number(float(target_raw["dengue_cases"])):
                        raise DatasetValidationError(f"{horizon}-week target verification failed for {ward} at {current['date']}")
                    if current[f"target_date_{horizon}w"] != source_dates[indices[target_pos]].isoformat():
                        raise DatasetValidationError(f"{horizon}-week target date verification failed for {ward} at {current['date']}")
                elif current[f"target_{horizon}w"] or current[f"target_date_{horizon}w"]:
                    raise DatasetValidationError("Unavailable future target was not preserved as blank")
    cities = {row["city"] for row in output_rows}
    return len(output_rows), set(ward_rows), cities


def clean_dataset(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> Path:
    if input_path.resolve() == output_path.resolve():
        raise DatasetValidationError("Input and output paths must differ")
    rows, dates = _read_and_validate_source(input_path)
    engineered, output_columns = _engineer_rows(rows, dates)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=output_columns, extrasaction="raise")
            writer.writeheader()
            writer.writerows(engineered)
            handle.flush()
            os.fsync(handle.fileno())
        _verify_output(temp_path, rows, dates, output_columns)
        os.replace(temp_path, output_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    output_rows, wards, cities = _verify_output(output_path, rows, dates, output_columns)
    print(f"Input rows: {len(rows):,}")
    print("Removed rows: 0 (no rows were removed or imputed)")
    print("Removal reasons: none; any missing, duplicate, invalid, or non-weekly source record stops processing")
    print("Duplicate full rows: 0")
    print("Duplicate ward/date records: 0")
    print(f"Wards: {len(wards)} (expected {EXPECTED_WARDS})")
    print(f"Cities: {len(cities)} (expected {EXPECTED_CITIES})")
    print(f"Date range: {min(dates).isoformat()} through {max(dates).isoformat()}")
    print(f"Output rows: {output_rows:,}")
    print(f"Generated features: {', '.join(GENERATED_FEATURE_COLUMNS)}")
    print(f"Target columns: {', '.join(TARGET_COLUMNS)}")
    print(f"Model feature allowlist columns: {len(MODEL_FEATURE_COLUMNS)}")
    print(f"Supply columns excluded from model features: {', '.join(SUPPLY_COLUMNS)}")
    print(f"Output CSV: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        clean_dataset(args.input, args.output)
    except (DatasetValidationError, FileNotFoundError) as exc:
        print(f"Data preparation stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
