"""Single ordered model-input contract shared by cleaning and inference."""

from __future__ import annotations

FEATURE_SCHEMA_VERSION = "phase2-dengue-34-v1"

# Order is part of the saved-artifact contract. Keep this list aligned with
# the reviewed Phase 2 schema; identifiers, temperature extrema, supplies,
# and all targets are deliberately excluded.
FEATURE_COLUMNS = (
    "population_density_sqkm",
    "water_logging_risk_index",
    "sanitation_score",
    "avg_temp_C",
    "rainfall_mm",
    "relative_humidity_pct",
    "dengue_cases",
    *(f"dengue_cases_lag_{week}w" for week in range(1, 5)),
    *(f"rain_lag_{week}w" for week in range(1, 5)),
    *(f"rain_sum_{window}w" for window in (2, 3, 4, 6, 8)),
    *(f"avg_temp_C_lag_{week}w" for week in range(1, 5)),
    "avg_temp_C_mean_prev_4w",
    *(f"relative_humidity_pct_lag_{week}w" for week in range(1, 5)),
    "relative_humidity_pct_mean_prev_4w",
    "calendar_week",
    "calendar_month",
    "week_sin",
    "week_cos",
)

EXCLUDED_MODEL_COLUMNS = frozenset({
    "date", "ward_id", "city", "max_temp_c", "min_temp_c",
    "saline_bags_needed", "diagnostic_kits_needed", "hospital_beds_required",
    "target_2w", "target_3w", "target_4w",
    "target_date_2w", "target_date_3w", "target_date_4w",
})

if len(FEATURE_COLUMNS) != 34 or len(set(FEATURE_COLUMNS)) != 34:
    raise RuntimeError("The canonical Phase 2 model schema must contain 34 unique ordered features")
if EXCLUDED_MODEL_COLUMNS.intersection(FEATURE_COLUMNS):
    raise RuntimeError("A forbidden identifier, supply, or target column entered the model schema")
