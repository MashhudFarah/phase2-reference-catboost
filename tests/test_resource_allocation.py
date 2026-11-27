"""Non-fitting checks for resource shortage and city pooling scenarios."""

import pandas as pd
import pytest

from src.resource_allocation import calculate_resource_allocation
from src.backend import operational_data


def fixture_inputs():
    forecasts = pd.DataFrame([
        {"ward_id": "w1", "city": "City A", "horizon_weeks": 2,
         "expected_cases": 20.0, "saline_bags_expected": 47.7,
         "diagnostic_kits_expected": 23.3, "hospital_beds_expected": 6.0,
         "vulnerability_index": 0.2, "estimated_bed_capacity": 10,
         "risk_tier": "LOW"},
        {"ward_id": "w2", "city": "City A", "horizon_weeks": 2,
         "expected_cases": 40.0, "saline_bags_expected": 95.4,
         "diagnostic_kits_expected": 46.6, "hospital_beds_expected": 12.0,
         "vulnerability_index": 0.8, "estimated_bed_capacity": 8,
         "risk_tier": "CRITICAL"},
        {"ward_id": "w3", "city": "City B", "horizon_weeks": 2,
         "expected_cases": 5.0, "saline_bags_expected": 11.925,
         "diagnostic_kits_expected": 5.825, "hospital_beds_expected": 1.5,
         "vulnerability_index": 0.5, "estimated_bed_capacity": 4,
         "risk_tier": "MODERATE"},
    ])
    operations = pd.DataFrame([
        {"ward_id": "w1", "city": "City A", "estimated_bed_capacity": 10,
         "crew_count_available": None, "actual_available_beds": 20,
         "saline_stock_available": 100, "diagnostic_kits_stock_available": 30,
         "vulnerability_index": 0.2},
        {"ward_id": "w2", "city": "City A", "estimated_bed_capacity": 8,
         "crew_count_available": 1, "actual_available_beds": 5,
         "saline_stock_available": 20, "diagnostic_kits_stock_available": 10,
         "vulnerability_index": 0.8},
        {"ward_id": "w3", "city": "City B", "estimated_bed_capacity": 4,
         "crew_count_available": None, "actual_available_beds": None,
         "saline_stock_available": None, "diagnostic_kits_stock_available": None,
         "vulnerability_index": 0.5},
    ])
    return forecasts, operations


def test_allocation_preserves_unknowns_limits_transfers_to_city_and_assigns_crew_pool():
    forecasts, operations = fixture_inputs()
    result = calculate_resource_allocation(
        forecasts, operations, horizon_weeks=2,
        stock_availability_pct=70, bed_capacity_adjustment_pct=10,
        hypothetical_crew_count=5,
    )
    by_ward = {row["ward_id"]: row for row in result["wards"]}
    assert by_ward["w1"]["saline_stock_available_shortage"] == 0
    assert by_ward["w2"]["saline_stock_available_shortage"] == pytest.approx(75.4)
    assert by_ward["w3"]["saline_stock_available_shortage"] is None
    assert by_ward["w3"]["actual_bed_shortage"] is None
    assert sum(row["hypothetical_crews_assigned"] for row in result["wards"]) == 5
    assert all(move["city"] == "City A" for move in result["proposed_transfers"])
    assert all("requires operational confirmation" in move["decision_status"]
               for move in result["proposed_transfers"])
    city_a = next(row for row in result["city_summary"] if row["city"] == "City A")
    assert city_a["estimated_capacity_headroom"] == pytest.approx(19.8 - 18.0)
    assert result["stock_availability_pct_assumption"] == 70
    city_b = next(row for row in result["city_summary"] if row["city"] == "City B")
    assert city_b["remaining_actual_bed_shortage"] is None


def test_allocation_rejects_invalid_horizon_scenario_and_duplicate_operations():
    forecasts, operations = fixture_inputs()
    with pytest.raises(ValueError, match="horizon"):
        calculate_resource_allocation(forecasts, operations, horizon_weeks=5)
    with pytest.raises(ValueError, match="30 and 100"):
        calculate_resource_allocation(forecasts, operations, horizon_weeks=2, stock_availability_pct=20)
    with pytest.raises(ValueError, match="unique"):
        calculate_resource_allocation(
            forecasts, pd.concat([operations, operations.iloc[[0]]]), horizon_weeks=2,
        )


def test_ward_capacity_csv_is_read_only_and_schema_validated(tmp_path):
    path = tmp_path / "operational_capacity.csv"
    rows = []
    for city_no in range(5):
        for ward_no in range(12):
            rows.append({
                "ward_id": f"W{city_no}_{ward_no}", "city": f"City {city_no}",
                "population_density_sqkm": 10, "water_logging_risk_index": .5,
                "sanitation_score": .5, "vulnerability_index": .5,
                "mean_weekly_cases_2016_2023": 2, "estimated_bed_capacity": 4,
                "actual_available_beds": None, "saline_stock_available": None,
                "diagnostic_kits_stock_available": None, "crew_count_available": None,
                "last_updated": None, "data_source": "fixture",
                "bed_capacity_method": "fixture", "operational_data_status": "unknown",
            })
    pd.DataFrame(rows, columns=operational_data.WARD_FIELDS).to_csv(path, index=False)
    reread = operational_data.read_ward_capacity(path)
    assert reread.loc[reread.ward_id.eq("W2_4"), "crew_count_available"].isna().all()
    assert reread.actual_available_beds.isna().all()
    assert len(reread) == 60


def test_reference_batch_builds_complete_180_records_without_model_fit(monkeypatch):
    from pathlib import Path
    from src import reference_catboost_predict as service
    from src.feature_engineering import SUPPORTED_HORIZONS

    wards = {f"City {city}": [f"W{city}_{ward}" for ward in range(12)] for city in range(5)}
    data = pd.DataFrame({
        "date": pd.to_datetime(["2023-12-25", "2025-12-29"]),
        "dengue_cases": [1, 2], "city": ["City 0", "City 0"],
    })
    capacity = []
    for city, city_wards in wards.items():
        for ward in city_wards:
            capacity.append({
                "ward_id": ward, "city": city, "population_density_sqkm": 10,
                "water_logging_risk_index": .5, "sanitation_score": .5,
                "vulnerability_index": .5, "estimated_bed_capacity": 4,
            })
    run = Path("/runs/reference_catboost_compatible_test")
    model_meta = {h: (object(), {"training_cutoff": "2023-12-31", "metrics": {}}, run)
                  for h in SUPPORTED_HORIZONS}
    monkeypatch.setattr(service, "load_data", lambda: data)
    monkeypatch.setattr(service, "get_supported_areas", lambda: wards)
    monkeypatch.setattr(service, "load_model", lambda horizon: model_meta[horizon])
    monkeypatch.setattr(service, "generate_forecast", lambda city, ward, horizon, origin, today=None: {
        "forecastOrigin": "2025-08-18", "targetDate": (pd.Timestamp("2025-08-18") + pd.Timedelta(weeks=horizon)).date().isoformat(),
        "expectedPredictedCases": 4.25, "observationDate": "2025-08-18",
    })
    monkeypatch.setattr(service, "derive_city_thresholds", lambda _data, _city: {
        "low_max": 2, "medium_max": 4, "high_max": 8,
    })
    monkeypatch.setattr(operational_data, "read_ward_capacity", lambda: pd.DataFrame(capacity))
    result = service.generate_batch_forecast("2025-08-18")
    assert len(result) == 180
    assert result.ward_id.nunique() == 60
    assert set(result.horizon_weeks) == {2, 3, 4}
    assert not result.duplicated(["ward_id", "forecast_origin", "horizon_weeks"]).any()
    row = result.iloc[0]
    assert row.target_date == (pd.Timestamp("2025-08-18") + pd.Timedelta(weeks=row.horizon_weeks)).date().isoformat()
    assert row.saline_bags_expected == pytest.approx(4.25 * 2.385)
    assert row.diagnostic_kits_expected == pytest.approx(4.25 * 1.165)
    assert row.hospital_beds_expected == pytest.approx(4.25 * 0.3)
    assert result.lower_cases.isna().all() and result.upper_cases.isna().all()

