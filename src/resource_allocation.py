"""Conservative city-level resource planning from forecast and reported data."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

FORECAST_KEYS = {"ward_id", "city", "horizon_weeks", "expected_cases",
                 "saline_bags_expected", "diagnostic_kits_expected", "hospital_beds_expected",
                 "vulnerability_index", "estimated_bed_capacity", "risk_tier"}
OPERATIONAL_KEYS = {"ward_id", "city", "estimated_bed_capacity", "crew_count_available",
                    "actual_available_beds", "saline_stock_available", "diagnostic_kits_stock_available"}


def _finite_nonnegative(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError("Operational resource values must be finite and non-negative")
    return number


def _pair_surplus_to_deficit(group: pd.DataFrame, resource: str, demand: str) -> tuple[list[dict], float | None]:
    """Suggest transfers within one city only; no geographic travel is inferred."""
    if group[resource].isna().any():
        return [], None
    surplus = [[str(row.ward_id), max(float(getattr(row, resource)) - float(getattr(row, demand)), 0.0)]
               for row in group.itertuples()]
    deficit = [[str(row.ward_id), max(float(getattr(row, demand)) - float(getattr(row, resource)), 0.0)]
               for row in group.itertuples()]
    sources = sorted((x for x in surplus if x[1] > 0), key=lambda x: (-x[1], x[0]))
    destinations = sorted((x for x in deficit if x[1] > 0), key=lambda x: (-x[1], x[0]))
    transfers = []
    for source in sources:
        for destination in destinations:
            amount = min(source[1], destination[1])
            if amount <= 0:
                continue
            transfers.append({"from_ward": source[0], "to_ward": destination[0],
                              "quantity": amount, "resource": resource,
                              "decision_status": "proposed; requires operational confirmation",
                              "routing": "same-city pooling only; no distance or travel time available"})
            source[1] -= amount
            destination[1] -= amount
    remaining = sum(x[1] for x in destinations)
    return transfers, float(remaining)


def _distribute_crew_pool(rows: pd.DataFrame, count: int) -> dict[str, int]:
    """Allocate an explicitly hypothetical crew pool proportionally to priority."""
    if isinstance(count, bool) or int(count) != count or count < 0:
        raise ValueError("Hypothetical crew count must be a non-negative integer")
    if count == 0 or rows.empty:
        return {str(ward): 0 for ward in rows.ward_id}
    weights = rows.priority_score.to_numpy(dtype=float)
    if not np.isfinite(weights).all():
        raise ValueError("Crew deployment priorities must be finite")
    # Give zero-priority wards a small positive share to avoid a zero denominator.
    weights = np.maximum(weights, 1e-9)
    exact = count * weights / weights.sum()
    base = np.floor(exact).astype(int)
    remainder = count - int(base.sum())
    order = sorted(range(len(rows)), key=lambda i: (-(exact[i] - base[i]), -weights[i], str(rows.iloc[i].ward_id)))
    for index in order[:remainder]:
        base[index] += 1
    return {str(rows.iloc[i].ward_id): int(base[i]) for i in range(len(rows))}


def calculate_resource_allocation(
    forecasts: pd.DataFrame,
    ward_operations: pd.DataFrame,
    *,
    horizon_weeks: int,
    stock_availability_pct: float = 100.0,
    bed_capacity_adjustment_pct: float = 0.0,
    selected_wards: Iterable[str] | None = None,
    hypothetical_crew_count: int = 0,
) -> dict:
    """Calculate transparent shortages, city pooling proposals, and crew priorities.

    Actual supply comparisons remain unknown when source values are blank.
    Scenario stocks and bed adjustments are returned as assumptions only.
    """
    missing_forecast = FORECAST_KEYS - set(forecasts.columns)
    missing_operations = OPERATIONAL_KEYS - set(ward_operations.columns)
    if missing_forecast or missing_operations:
        raise ValueError(f"Allocation inputs are missing columns: {sorted(missing_forecast | missing_operations)}")
    if horizon_weeks not in (2, 3, 4):
        raise ValueError("Forecast horizon must be 2, 3, or 4 weeks")
    stock_pct = float(stock_availability_pct)
    bed_pct = float(bed_capacity_adjustment_pct)
    if not math.isfinite(stock_pct) or not 30 <= stock_pct <= 100:
        raise ValueError("Stock availability scenario must be between 30 and 100 percent")
    if not math.isfinite(bed_pct) or not -100 <= bed_pct <= 100:
        raise ValueError("Bed capacity adjustment must be between -100 and 100 percent")
    batch = forecasts.loc[pd.to_numeric(forecasts.horizon_weeks, errors="coerce").eq(horizon_weeks)].copy()
    if selected_wards is not None:
        selected = {str(x) for x in selected_wards}
        batch = batch.loc[batch.ward_id.astype(str).isin(selected)]
    if batch.empty:
        raise ValueError("No compatible forecast records are available for this horizon and selection")
    if ward_operations.ward_id.astype(str).duplicated().any():
        raise ValueError("Ward operational records must be unique by ward_id")
    operations = ward_operations.copy()
    if operations.empty:
        raise ValueError("No unique ward operational records are available")
    rows = batch.merge(operations, on=["ward_id", "city"], how="left", suffixes=("", "_ops"), validate="one_to_one")
    if rows.estimated_bed_capacity_ops.isna().any() or rows.vulnerability_index_ops.isna().any():
        raise ValueError("Ward operational reference is missing for one or more forecasts")
    rows["estimated_bed_capacity_adjusted"] = (
        pd.to_numeric(rows.estimated_bed_capacity_ops) * (1 + bed_pct / 100)
    ).clip(lower=0)
    rows["estimated_capacity_headroom"] = rows.estimated_bed_capacity_adjusted - pd.to_numeric(rows.hospital_beds_expected)
    rows["actual_bed_shortage"] = [
        max(float(demand) - float(available), 0.0) if pd.notna(available) else np.nan
        for demand, available in zip(rows.hospital_beds_expected, rows.actual_available_beds)
    ]
    for stock, demand in (("saline_stock_available", "saline_bags_expected"),
                          ("diagnostic_kits_stock_available", "diagnostic_kits_expected")):
        rows[f"{stock}_shortage"] = [
            max(float(need) - float(available), 0.0) if pd.notna(available) else np.nan
            for need, available in zip(rows[demand], rows[stock])
        ]
        rows[f"scenario_{stock}"] = pd.to_numeric(rows[demand]) * stock_pct / 100
        rows[f"scenario_{stock}_shortage"] = pd.to_numeric(rows[demand]) - rows[f"scenario_{stock}"]
    risk_rank = {"LOW": 0.0, "MODERATE": 1 / 3, "HIGH": 2 / 3, "CRITICAL": 1.0}
    for city, group in rows.groupby("city", sort=True):
        # Risk categories were assigned from common city-pooled case thresholds,
        # not from ward-specific ranks or quantiles.
        risk_component = group.risk_tier.astype(str).str.upper().map(risk_rank)
        if risk_component.isna().any():
            raise ValueError(f"Forecast contains an unsupported risk tier for {city}")
        rows.loc[group.index, "priority_score"] = (
            0.5 * pd.to_numeric(group.vulnerability_index_ops)
            + 0.5 * risk_component
        )
    rows["hypothetical_crews_assigned"] = 0
    # The request represents one pool for the selected scenario, so allocation
    # must conserve that pool across the complete selection (not multiply it by
    # the number of cities). This is a planning allocation, not a dispatch.
    allocation = _distribute_crew_pool(rows, hypothetical_crew_count)
    for ward_id, crew_count in allocation.items():
        rows.loc[rows.ward_id.astype(str).eq(ward_id), "hypothetical_crews_assigned"] = crew_count

    transfers = []
    city_summary = []
    for city, group in rows.groupby("city", sort=True):
        bed_moves, remaining_bed_shortage = _pair_surplus_to_deficit(group, "actual_available_beds", "hospital_beds_expected")
        saline_moves, remaining_saline_shortage = _pair_surplus_to_deficit(group, "saline_stock_available", "saline_bags_expected")
        kit_moves, remaining_kit_shortage = _pair_surplus_to_deficit(group, "diagnostic_kits_stock_available", "diagnostic_kits_expected")
        transfers.extend({**move, "city": city} for move in bed_moves + saline_moves + kit_moves)
        city_summary.append({
            "city": city,
            "ward_count": len(group),
            "estimated_bed_capacity": float(group.estimated_bed_capacity_adjusted.sum()),
            "expected_bed_demand": float(group.hospital_beds_expected.sum()),
            "estimated_capacity_headroom": float(group.estimated_capacity_headroom.sum()),
            "actual_bed_availability_known": bool(group.actual_available_beds.notna().all()),
            "remaining_actual_bed_shortage": remaining_bed_shortage,
            "remaining_actual_saline_shortage": remaining_saline_shortage,
            "remaining_actual_diagnostic_kit_shortage": remaining_kit_shortage,
            "redistribution_scope": "same city only; proposals require operational confirmation",
        })
    return {
        "horizon_weeks": horizon_weeks,
        "stock_availability_pct_assumption": stock_pct,
        "bed_capacity_adjustment_pct_assumption": bed_pct,
        "hypothetical_crew_pool": int(hypothetical_crew_count),
        "wards": rows.replace({np.nan: None}).to_dict(orient="records"),
        "city_summary": city_summary,
        "proposed_transfers": transfers,
        "notes": [
            "Unknown actual inventory remains null and is not treated as zero.",
            "Estimated bed capacity is not actual staffed or available capacity.",
            "Redistribution proposals use city membership only; no coordinates, route, distance, or travel time are available.",
            "Crew distribution is hypothetical and weighted by an unvalidated equal-weight priority score.",
        ],
    }
