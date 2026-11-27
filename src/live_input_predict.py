"""Live weekly-input inference for the reference CatBoost early-warning models.

The live endpoint accepts the six values that can change from week to week:
current weekly dengue cases and the current week's weather measurements.
Ward/city/static context is taken from the latest historical row for that ward.
All lag/rolling/cyclic features are then rebuilt by the exact same feature
engineering function used when the reference CatBoost models were trained.
"""
from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import SUPPORTED_HORIZONS
from src.reference_catboost_features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    RAW_COLUMNS,
    engineer_features,
    usable_feature_mask,
)
from src.risk_engine import (classify_risk, derive_city_thresholds, describe_drivers,
                             load_risk_calibration_data)
from src.train import display_case_count

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "processed_historical_data.csv"

# The reference models require at least 8 previous weekly observations for
# their rolling features. We also require a one-week continuation from the
# latest stored observation so the weekly lag contract remains valid.
MIN_HISTORY_WEEKS = 8


class LiveInputError(ValueError):
    """The submitted live observation cannot satisfy the feature contract."""


class LiveModelUnavailableError(RuntimeError):
    """A compatible CatBoost artifact is not available."""


@lru_cache(maxsize=1)
def load_historical_source() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise LiveInputError(f"Historical source data was not found: {DATA_PATH}")

    try:
        raw = pd.read_csv(DATA_PATH)
        engineered = engineer_features(raw)
        usable = usable_feature_mask(engineered)
        data = engineered.loc[usable].copy()
    except Exception as exc:
        raise LiveInputError("Historical source data cannot produce valid reference features") from exc

    if data.empty:
        raise LiveInputError("Historical source data contains no usable feature rows")

    return data.sort_values(["ward_id", "date"], kind="stable").reset_index(drop=True)


@lru_cache(maxsize=3)
def load_live_model(horizon_weeks: int):
    if horizon_weeks not in SUPPORTED_HORIZONS:
        raise LiveModelUnavailableError(
            f"Unsupported horizon; choose one of {SUPPORTED_HORIZONS} weeks"
        )
    try:
        # Live-input and what-if inference consume the same validated bundle as
        # the ordinary Phase 2 and reference-style forecasting APIs.
        from src.predict import _run_directory, load_model_bundle

        bundle = load_model_bundle(horizon_weeks)
        run = _run_directory(horizon_weeks)
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        info = metadata["horizons"][str(horizon_weeks)]
        if metadata.get("backend") != "phase2_catboost" or info.get("horizon_weeks") != horizon_weeks:
            raise ValueError("Run does not use the canonical Phase 2 CatBoost trainer")
        normalized = {
            **info,
            "training_cutoff": info["training_data_cutoff"],
            "tree_count": int(bundle["model"].tree_count_),
            "metrics": info.get("test_metrics", {}),
        }
        return bundle["model"], normalized, run
    except Exception as exc:
        raise LiveModelUnavailableError(
            f"CatBoost model for {horizon_weeks} weeks is unavailable or incompatible"
        ) from exc


def get_live_areas() -> dict[str, list[str]]:
    data = load_historical_source()
    return {
        str(city): sorted(group.ward_id.astype(str).unique().tolist())
        for city, group in data.groupby("city", sort=True, observed=True)
    }


def get_live_defaults(city: str, ward_id: str) -> dict:
    """Return static ward context and the next expected weekly input date."""
    data = load_historical_source()
    ward = data.loc[
        data.city.astype(str).eq(str(city))
        & data.ward_id.astype(str).eq(str(ward_id))
    ].sort_values("date")

    if ward.empty:
        raise LiveInputError("Unknown city/ward combination")
    if len(ward) < MIN_HISTORY_WEEKS:
        raise LiveInputError("This ward does not have enough weekly history for live inference")

    latest = ward.iloc[-1]
    next_date = pd.Timestamp(latest["date"]) + pd.Timedelta(weeks=1)

    return {
        "city": str(latest["city"]),
        "wardId": str(latest["ward_id"]),
        "latestHistoricalDate": pd.Timestamp(latest["date"]).date().isoformat(),
        "nextExpectedWeeklyDate": next_date.date().isoformat(),
        "populationDensity": float(latest["population_density_sqkm"]),
        "waterLoggingRisk": float(latest["water_logging_risk_index"]),
        "sanitationScore": float(latest["sanitation_score"]),
        "previousCases": float(latest["dengue_cases"]),
        "previousRainfall": float(latest["rainfall_mm"]),
        "previousTemperature": float(latest["avg_temp_C"]),
        "previousHumidity": float(latest["relative_humidity_pct"]),
    }


def _validate_number(name: str, value: float, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise LiveInputError(f"{name} must be numeric") from exc
    if not np.isfinite(number):
        raise LiveInputError(f"{name} must be finite")
    if minimum is not None and number < minimum:
        raise LiveInputError(f"{name} must be at least {minimum}")
    return number


def predict_live(
    *,
    city: str,
    ward_id: str,
    observation_date: date,
    horizon_weeks: int,
    avg_temp_c: float,
    max_temp_c: float,
    min_temp_c: float,
    rainfall_mm: float,
    relative_humidity_pct: float,
    dengue_cases: float,
    today: date | None = None,
) -> dict:
    """Build one live feature row and run the selected CatBoost horizon model."""
    if horizon_weeks not in SUPPORTED_HORIZONS:
        raise LiveInputError(f"Unsupported horizon; choose one of {SUPPORTED_HORIZONS} weeks")

    data = load_historical_source()
    ward = data.loc[
        data.city.astype(str).eq(str(city))
        & data.ward_id.astype(str).eq(str(ward_id))
    ].sort_values("date")

    if ward.empty:
        raise LiveInputError("Unknown city/ward combination")
    if len(ward) < MIN_HISTORY_WEEKS:
        raise LiveInputError("This ward does not have enough weekly history for live inference")

    latest_date = pd.Timestamp(ward.date.iloc[-1])
    live_date = pd.Timestamp(observation_date)
    expected_date = latest_date + pd.Timedelta(weeks=1)

    if live_date != expected_date:
        raise LiveInputError(
            "Live observation date must be exactly one week after the latest stored observation "
            f"({expected_date.date().isoformat()}). Update the historical dataset first if a week is missing."
        )
    current_date = today or date.today()
    if abs((current_date - live_date.date()).days) > 14:
        raise LiveInputError(
            "Current live input must be dated within 14 days of today. "
            "Historical observations belong in the historical replay workflow."
        )

    # Static ward-level values are taken from the latest stored row. They do not
    # need to be re-entered by the user every week.
    latest = ward.iloc[-1]
    live_row = {
        "date": live_date,
        "city": str(city),
        "ward_id": str(ward_id),
        "population_density_sqkm": float(latest["population_density_sqkm"]),
        "water_logging_risk_index": float(latest["water_logging_risk_index"]),
        "sanitation_score": float(latest["sanitation_score"]),
        "avg_temp_C": _validate_number("Average temperature", avg_temp_c),
        "max_temp_c": _validate_number("Maximum temperature", max_temp_c),
        "min_temp_c": _validate_number("Minimum temperature", min_temp_c),
        "rainfall_mm": _validate_number("Rainfall", rainfall_mm, 0.0),
        "relative_humidity_pct": _validate_number("Relative humidity", relative_humidity_pct, 0.0),
        "dengue_cases": _validate_number("Current weekly dengue cases", dengue_cases, 0.0),
    }
    temperatures = [live_row[name] for name in ("min_temp_c", "avg_temp_C", "max_temp_c")]
    if any(value < -90 or value > 60 for value in temperatures) or not (
        temperatures[0] <= temperatures[1] <= temperatures[2]
    ):
        raise LiveInputError("Temperatures must be plausible Celsius values with min ≤ average ≤ max")
    if live_row["relative_humidity_pct"] > 100:
        raise LiveInputError("Relative humidity must be within [0, 100]")

    # engineer_features() is deliberately reused here. This guarantees the
    # live row gets exactly the same lag, rolling and calendar transformations
    # as the rows used to train the reference CatBoost models.
    source = pd.concat(
        [ward.loc[:, list(RAW_COLUMNS)].copy(), pd.DataFrame([live_row])],
        ignore_index=True,
    )
    engineered = engineer_features(source)
    live_features = engineered.loc[engineered.date.eq(live_date)].copy()

    if len(live_features) != 1:
        raise LiveInputError("The live observation could not be converted into exactly one feature row")
    if not bool(usable_feature_mask(live_features).iloc[0]):
        raise LiveInputError(
            "The live observation does not have enough complete historical information to build all model features"
        )

    model, info, run = load_live_model(horizon_weeks)
    matrix = live_features.loc[:, list(FEATURE_COLUMNS)].copy()
    for column in CATEGORICAL_FEATURES:
        matrix[column] = matrix[column].astype(str)
    numeric = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]
    matrix[numeric] = matrix[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(matrix[numeric].to_numpy(dtype=float)).all():
        raise LiveInputError("Live model features contain missing or non-finite values")

    raw_prediction = float(np.asarray(model.predict(matrix), dtype=float).reshape(-1)[0])
    if not np.isfinite(raw_prediction):
        raise LiveModelUnavailableError("CatBoost produced a non-finite prediction")

    nonnegative = max(0.0, raw_prediction)
    predicted_cases = display_case_count(nonnegative)

    # Risk bands are calculated from the historical observations only. The
    # current live case count must not change the threshold used for its own alert.
    risk = classify_risk(
        predicted_cases, ward.dengue_cases,
        thresholds=derive_city_thresholds(load_risk_calibration_data(str(DATA_PATH.resolve())), city),
    )
    history = ward.tail(16)

    target_date = live_date + pd.Timedelta(weeks=horizon_weeks)
    return {
        "city": str(city),
        "wardId": str(ward_id),
        "horizonWeeks": int(horizon_weeks),
        "observationDate": live_date.date().isoformat(),
        "targetDate": target_date.date().isoformat(),
        "predictedCases": predicted_cases,
        "rawPredictedCases": raw_prediction,
        "expectedPredictedCases": nonnegative,
        "riskCategory": risk["level"],
        "riskMethod": risk["method"],
        "riskThresholds": risk["thresholds"],
        "riskWarning": risk["warning"],
        "earlyWarning": risk["level"] in {"HIGH", "CRITICAL"},
        "warningMessage": (
            "Early warning triggered: predicted dengue activity is in the HIGH/CRITICAL range."
            if risk["level"] in {"HIGH", "CRITICAL"}
            else "No HIGH/CRITICAL early warning was triggered by the current model prediction."
        ),
        "model": "CatBoost",
        "predictionBackend": "reference_catboost_live_input",
        "experimentId": run.name,
        "modelArtifact": info["model_artifact"],
        "modelTreeCount": info["tree_count"],
        "modelTrainingDataThrough": info["training_cutoff"],
        "modelInputFeatureCount": len(FEATURE_COLUMNS),
        "expectedFeatureCount": len(FEATURE_COLUMNS),
        "featureContract": list(FEATURE_COLUMNS),
        "staticContext": {
            "populationDensity": float(latest["population_density_sqkm"]),
            "waterLoggingRisk": float(latest["water_logging_risk_index"]),
            "sanitationScore": float(latest["sanitation_score"]),
        },
        "liveInput": {
            "avgTempC": live_row["avg_temp_C"],
            "maxTempC": live_row["max_temp_c"],
            "minTempC": live_row["min_temp_c"],
            "rainfallMm": live_row["rainfall_mm"],
            "humidityPct": live_row["relative_humidity_pct"],
            "dengueCases": live_row["dengue_cases"],
        },
        "drivers": describe_drivers(ward, pd.Series(live_row)),
        "history": [
            {"date": pd.Timestamp(r.date).date().isoformat(), "cases": float(r.dengue_cases)}
            for r in history.itertuples(index=False)
        ],
        "forecast": [{"date": target_date.date().isoformat(), "cases": predicted_cases}],
        "observedFeatures": {
            name: float(live_features.iloc[0][name])
            for name in (
                "avg_temp_C", "relative_humidity_pct", "rainfall_mm",
                "rain_lag_1w", "rain_lag_2w", "rain_lag_3w", "rain_lag_4w",
                "rain_sum_2w", "rain_sum_4w", "rain_sum_8w",
                "population_density_sqkm", "water_logging_risk_index", "sanitation_score",
                "dengue_cases_lag_1w", "dengue_cases_lag_2w", "dengue_cases_lag_3w", "dengue_cases_lag_4w",
            )
            if name in live_features.columns and pd.notna(live_features.iloc[0][name])
        },
    }


def simulate_what_if(
    *,
    city: str,
    ward_ids: list[str],
    forecast_date: date,
    hypothetical_crew_count: int,
    intervention_effectiveness_pct: float,
) -> dict:
    """Run baseline/scenario inference on copied canonical rows; never persist scenario data."""
    if not ward_ids or len(set(ward_ids)) != len(ward_ids):
        raise LiveInputError("Select one or more distinct treated wards")
    if isinstance(hypothetical_crew_count, bool) or int(hypothetical_crew_count) != hypothetical_crew_count or hypothetical_crew_count < 0:
        raise LiveInputError("Hypothetical crew count must be a non-negative integer")
    effectiveness = _validate_number("Intervention effectiveness percentage", intervention_effectiveness_pct, 0.0)
    if effectiveness > 100:
        raise LiveInputError("Intervention effectiveness percentage must not exceed 100")

    data = load_historical_source()
    from src.reference_catboost_predict import resolve_feature_row
    from src.risk_engine import derive_city_thresholds, load_risk_calibration_data
    chosen = []
    for ward_id in ward_ids:
        ward = data.loc[data.city.astype(str).eq(city) & data.ward_id.astype(str).eq(str(ward_id))].sort_values("date")
        if ward.empty:
            raise LiveInputError(f"Ward {ward_id!r} is not part of {city!r}")
        try:
            origin, row = resolve_feature_row(ward, forecast_date)
        except ValueError as exc:
            raise LiveInputError(str(exc)) from exc
        chosen.append((str(ward_id), origin, row))

    models = {h: load_live_model(h) for h in SUPPORTED_HORIZONS}
    if len({str(entry[2].resolve()) for entry in models.values()}) != 1:
        raise LiveModelUnavailableError("Scenario horizons must use one compatible model run")
    latest_origin = max(origin for _ward, origin, _row in chosen)
    for horizon, (_model, info, _run) in models.items():
        if pd.Timestamp(info.get("training_cutoff")) > latest_origin:
            raise LiveInputError(f"The {horizon}-week model was trained after the selected origin; historical scenario replay is unavailable.")

    from src.backend.operational_data import read_ward_capacity
    ward_reference = read_ward_capacity().set_index("ward_id")
    risk_calibration = load_risk_calibration_data(str(DATA_PATH.resolve()))
    rain_columns = [name for name in FEATURE_COLUMNS if name == "rainfall_mm" or name.startswith("rain_lag_") or name.startswith("rain_sum_")]
    factor = 1.0 - effectiveness / 100.0
    output = []
    baseline_priority = []
    for ward_id, origin, row in chosen:
        thresholds = derive_city_thresholds(risk_calibration, city)
        predicted_by_horizon = {}
        for horizon in SUPPORTED_HORIZONS:
            model, info, run = models[horizon]
            baseline = row.loc[list(FEATURE_COLUMNS)].to_frame().T.copy()
            scenario = baseline.copy()
            for name in rain_columns:
                scenario[name] = pd.to_numeric(scenario[name], errors="coerce") * factor
            for matrix in (baseline, scenario):
                matrix.loc[:, list(FEATURE_COLUMNS)] = matrix.loc[:, list(FEATURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
                if not np.isfinite(matrix.to_numpy(dtype=float)).all():
                    raise LiveInputError("Scenario feature row contains missing or non-finite values")
            base_cases = max(0.0, float(np.asarray(model.predict(baseline), dtype=float).reshape(-1)[0]))
            scenario_cases = max(0.0, float(np.asarray(model.predict(scenario), dtype=float).reshape(-1)[0]))
            if not np.isfinite([base_cases, scenario_cases]).all():
                raise LiveModelUnavailableError("Scenario model produced a non-finite prediction")
            predicted_by_horizon[horizon] = base_cases
            output.append({
                "city": city, "ward_id": ward_id,
                "forecast_origin": origin.date().isoformat(),
                "target_date": (origin + pd.Timedelta(weeks=horizon)).date().isoformat(),
                "horizon_weeks": horizon,
                "baseline_expected_cases": base_cases,
                "scenario_expected_cases": scenario_cases,
                "expected_case_change": scenario_cases - base_cases,
                "baseline_saline_bags": base_cases * 2.385,
                "scenario_saline_bags": scenario_cases * 2.385,
                "baseline_diagnostic_kits": base_cases * 1.165,
                "scenario_diagnostic_kits": scenario_cases * 1.165,
                "baseline_hospital_beds": base_cases * 0.3,
                "scenario_hospital_beds": scenario_cases * 0.3,
                "baseline_risk_tier": classify_risk(base_cases, thresholds=thresholds)["level"],
                "scenario_risk_tier": classify_risk(scenario_cases, thresholds=thresholds)["level"],
                "model_version": run.name,
                "model_training_cutoff": info.get("training_cutoff"),
                "rainfall_proxy_factor": factor,
            })
        baseline_priority.append({
            "ward_id": ward_id,
            "vulnerability_index": float(ward_reference.loc[ward_id, "vulnerability_index"]),
            "baseline_cases": predicted_by_horizon[2],
        })

    # A hypothetical team pool is apportioned by the equal-weight planning
    # score; crew counts do not directly modify rainfall or the model output.
    priorities = pd.DataFrame(baseline_priority)
    if len(priorities):
        priority_cutoff = max(float(derive_city_thresholds(risk_calibration, city)["high_max"]), 1.0)
        priorities["priority_score"] = 0.5 * priorities.vulnerability_index + 0.5 * (priorities.baseline_cases / priority_cutoff).clip(upper=1)
        weights = np.maximum(priorities.priority_score.to_numpy(dtype=float), 1e-9)
        exact = int(hypothetical_crew_count) * weights / weights.sum()
        shares = np.floor(exact).astype(int)
        remainder = int(hypothetical_crew_count) - int(shares.sum())
        order = sorted(range(len(priorities)), key=lambda i: (-(exact[i] - shares[i]), -weights[i], priorities.iloc[i].ward_id))
        for index in order[:remainder]:
            shares[index] += 1
        share_by_ward = dict(zip(priorities.ward_id, shares))
        priority_by_ward = dict(zip(priorities.ward_id, priorities.priority_score))
        for item in output:
            item["hypothetical_crews_assigned"] = int(share_by_ward[item["ward_id"]])
            item["priority_score"] = float(priority_by_ward[item["ward_id"]])

    return {
        "mode": "assumption-based historical what-if scenario",
        "selected_date": forecast_date.isoformat(),
        "treated_wards": list(ward_ids),
        "hypothetical_crew_count": int(hypothetical_crew_count),
        "intervention_effectiveness_pct": effectiveness,
        "rainfall_proxy_factor": factor,
        "forecasts": output,
        "warning": (
            "Rainfall-feature damping is a hypothetical environmental proxy. Crews do not directly reduce rainfall; "
            "there is no intervention history in the supplied data, so case changes are not causal or clinically validated."
        ),
    }
