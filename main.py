
"""FastAPI application entry point for DengueGuard."""

import os
import logging
import uuid
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

# Resolve paths independently of the terminal's current directory.
PROJECT_ROOT = Path(__file__).resolve().parent

# Load environment variables before importing database configuration.
load_dotenv(PROJECT_ROOT / ".env", override=False)

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from sqlalchemy import func, text
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
# --------------------------------------------------
# APPLICATION ENVIRONMENT AND SESSION CONFIGURATION
# --------------------------------------------------

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()

if APP_ENV not in {"development", "test", "production"}:
    raise RuntimeError(
        "APP_ENV must be development, test, or production."
    )

SESSION_SECRET = os.getenv("SESSION_SECRET", "").strip()

if (
    not SESSION_SECRET
    or SESSION_SECRET == "replace_with_a_random_secret"
    or len(SESSION_SECRET) < 32
):
    raise RuntimeError(
        "SESSION_SECRET is missing or too short. "
        "Generate a secure secret and add it to your .env file."
    )

try:
    SESSION_MAX_AGE = int(os.getenv("SESSION_MAX_AGE", "3600"))
except ValueError as exc:
    raise RuntimeError("SESSION_MAX_AGE must be an integer.") from exc

if SESSION_MAX_AGE <= 0:
    raise RuntimeError("SESSION_MAX_AGE must be greater than zero.")

# Secure cookies default to enabled in production.
cookie_secure_raw = os.getenv("COOKIE_SECURE")

if cookie_secure_raw is None:
    COOKIE_SECURE = APP_ENV == "production"
else:
    cookie_secure_value = cookie_secure_raw.strip().lower()

    if cookie_secure_value not in {
        "true", "false", "1", "0", "yes", "no", "on", "off"
    }:
        raise RuntimeError(
            "COOKIE_SECURE must be a valid boolean value."
        )

    COOKIE_SECURE = cookie_secure_value in {
        "true", "1", "yes", "on"
    }

if APP_ENV == "production" and not COOKIE_SECURE:
    raise RuntimeError(
        "COOKIE_SECURE must be true in production. "
        "Production session cookies must only be sent over HTTPS."
    )

from src.backend.database import Base, engine, get_db
from src.backend.models import (
    AuditLog, HospitalMembership, HospitalOperationalRecord,
    HospitalWardAssignment, OperationsUserScope, User, WardOperationsRecord,
)
from src.backend.auth import router as auth_router
from src.backend.security import get_csrf_token, verify_csrf_token

from src.backend.dependencies import (
    get_current_hospital_user,
    get_current_admin_user,
)
PREDICTION_BACKEND = os.getenv("PREDICTION_BACKEND", "reference_catboost").strip().lower()
if PREDICTION_BACKEND == "reference_catboost":
    from src.reference_catboost_predict import (
        ForecastUnavailableError, ModelUnavailableError, data_freshness,
        generate_forecast, get_available_horizons, get_data_through,
        get_supported_date_bounds, get_supported_areas, DATA_PATH,
    )
elif PREDICTION_BACKEND == "phase2":
    from src.predict import (
        ForecastUnavailableError, ModelUnavailableError, data_freshness,
        generate_forecast, get_available_horizons, get_data_through,
        get_supported_date_bounds, get_supported_areas, DATA_PATH,
    )
else:
    raise RuntimeError("PREDICTION_BACKEND must be 'reference_catboost' or 'phase2'.")
from src.feature_engineering import SUPPORTED_HORIZONS

logger = logging.getLogger(__name__)


# --------------------------------------------------
# APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="DengueGuard",
    description="Urban Dengue Early Warning System",
    version="1.0.0",
)


# --------------------------------------------------
# SESSION MIDDLEWARE
# --------------------------------------------------

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    session_cookie="dengue_session",
    max_age=SESSION_MAX_AGE,
    same_site="lax",
    https_only=COOKIE_SECURE,
)


# --------------------------------------------------
# STATIC FILES AND TEMPLATES
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "static")),
    name="static",
)

templates = Jinja2Templates(
    directory=str(PROJECT_ROOT / "templates")
)


class ForecastRequest(BaseModel):
    city: str = Field(min_length=1, max_length=100)
    ward: str = Field(min_length=1, max_length=100)
    horizonWeeks: int = Field(ge=2, le=4)
    asOf: Optional[date] = None
    strictEvaluation: bool = False


class LivePredictionRequest(BaseModel):
    city: str = Field(min_length=1, max_length=100)
    ward: str = Field(min_length=1, max_length=100)
    observationDate: date
    horizonWeeks: int = Field(ge=2, le=4)
    avgTempC: float
    maxTempC: float
    minTempC: float
    rainfallMm: float = Field(ge=0)
    humidityPct: float = Field(ge=0, le=100)
    dengueCases: float = Field(ge=0)


class HospitalMembershipPayload(BaseModel):
    hospital_name: str = Field(min_length=1, max_length=255)


class HospitalWardAssignmentPayload(BaseModel):
    city: str = Field(min_length=1, max_length=100)
    ward_id: str = Field(min_length=1, max_length=100)
    assigned: bool = True


class OperationsScopePayload(BaseModel):
    city: str = Field(min_length=1, max_length=100)
    ward_id: Optional[str] = Field(default=None, max_length=100)
    authorized: bool = True


class HospitalInventoryPayload(BaseModel):
    actual_available_beds: Optional[int] = Field(default=None, ge=0)
    saline_stock_available: Optional[int] = Field(default=None, ge=0)
    diagnostic_kits_stock_available: Optional[int] = Field(default=None, ge=0)


class HospitalInventoryCorrectionPayload(HospitalInventoryPayload):
    correction_reason: str = Field(min_length=8, max_length=300)


class WardCrewPayload(BaseModel):
    crew_count_available: int = Field(ge=0)


class BatchForecastPayload(BaseModel):
    asOf: Optional[date] = None


class AllocationRequest(BaseModel):
    horizonWeeks: int = Field(ge=2, le=4)
    asOf: Optional[date] = None
    stockAvailabilityPct: float = Field(default=100, ge=30, le=100)
    bedCapacityAdjustmentPct: float = Field(default=0, ge=-100, le=100)
    selectedWards: Optional[list[str]] = None
    hypotheticalCrewCount: int = Field(default=0, ge=0)


class WhatIfRequest(BaseModel):
    city: str = Field(min_length=1, max_length=100)
    selectedWards: list[str] = Field(min_length=1, max_length=60)
    forecastDate: date
    hypotheticalCrewCount: int = Field(default=0, ge=0)
    interventionEffectivenessPct: float = Field(ge=0, le=100)


@app.post("/api/predict")
async def predict_ward(
    payload: ForecastRequest,
    request: Request,
    x_csrf_token: str = Header(default=""),
    current_user: User = Depends(get_current_hospital_user),
):
    """Serve an authenticated forecast from the selected trained model run."""
    request_id = uuid.uuid4().hex
    if not verify_csrf_token(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")
    if payload.horizonWeeks not in SUPPORTED_HORIZONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported forecast horizon; choose one of {SUPPORTED_HORIZONS} weeks.",
        )
    try:
        result = generate_forecast(
            payload.city,
            payload.ward,
            payload.horizonWeeks,
            payload.asOf.isoformat() if payload.asOf else None,
            strict_evaluation=payload.strictEvaluation,
        )
        result["requestId"] = request_id
        logger.info(
            "forecast request_id=%s city=%s ward=%s horizon_weeks=%s origin=%s mode=%s "
            "model=%s experiment=%s features=%s expected_features=%s missing_features=%s "
            "raw_prediction=%s expected_prediction=%s rounded_prediction=%s freshness=%s",
            request_id,
            result.get("city"),
            result.get("wardId"),
            result.get("horizonWeeks"),
            result.get("forecastOrigin"),
            result.get("mode"),
            result.get("model"),
            result.get("experimentId"),
            result.get("modelInputFeatureCount"),
            result.get("expectedFeatureCount"),
            result.get("missingFeatureCount"),
            result.get("rawPredictedCases"),
            result.get("expectedPredictedCases"),
            result.get("predictedCases"),
            result.get("dataFreshness"),
        )
        return result
    except (FileNotFoundError, ForecastUnavailableError, ModelUnavailableError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc) if isinstance(exc, ForecastUnavailableError) else "A compatible forecasting model or required data is unavailable.",
        ) from exc
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "forecast request failed request_id=%s city=%s ward=%s horizon_weeks=%s origin=%s",
            request_id, payload.city, payload.ward, payload.horizonWeeks,
            payload.asOf.isoformat() if payload.asOf else None,
        )
        raise HTTPException(
            status_code=500,
            detail="Prediction failed because the model or processed data could not be used.",
        ) from exc


# --------------------------------------------------
# AUTHENTICATION ROUTER
# --------------------------------------------------

app.include_router(auth_router)


# --------------------------------------------------
# PUBLIC HOME PAGE
# --------------------------------------------------

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home/index.html",
        context={},
    )


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

@app.get("/login")
async def login_page(request: Request):

    csrf_token = get_csrf_token(request)

    return templates.TemplateResponse(
        request=request,
        name="auth/login/login.html",
        context={
            "csrf_token": csrf_token,
        },
        headers={"Cache-Control": "no-store, private"},
    )


# --------------------------------------------------
# REGISTRATION PAGE
# --------------------------------------------------

@app.get("/register")
async def register_page(request: Request):

    csrf_token = get_csrf_token(request)

    return templates.TemplateResponse(
        request=request,
        name="auth/register/register.html",
        context={
            "csrf_token": csrf_token,
        },
        headers={"Cache-Control": "no-store, private"},
    )


# --------------------------------------------------
# HOSPITAL USER DASHBOARD
# --------------------------------------------------

@app.get("/user_dashboard")
async def user_dashboard(
    request: Request,
    current_user: User = Depends(get_current_hospital_user),
):

    csrf_token = get_csrf_token(request)
    data_error = None
    model_horizons = get_available_horizons()
    data_through = None
    freshness = None
    date_bounds = {}
    try:
        areas = get_supported_areas()
        data_through = get_data_through()
        freshness = data_freshness()
        date_bounds = get_supported_date_bounds()
    except (FileNotFoundError, ValueError, ForecastUnavailableError):
        areas = {}
        data_error = "Processed ward data is unavailable or invalid. Check the server configuration."

    return templates.TemplateResponse(
        request=request,
        name="user/user_dashboard.html",
        context={
            "current_user": current_user,
            "csrf_token": csrf_token,
            "areas": areas,
            "data_error": data_error,
            "model_horizons": model_horizons,
            "data_through": data_through,
            "freshness": freshness,
            "date_bounds": date_bounds,
        },
        headers={"Cache-Control": "no-store, private"},
    )


@app.get("/live-input")
async def live_input_page(
    request: Request,
    current_user: User = Depends(get_current_hospital_user),
):
    """Render the weekly observation form; the model is checked at submission."""
    from src.live_input_predict import get_live_areas

    try:
        areas = get_live_areas()
        data_error = None
    except Exception:
        areas = {}
        data_error = "Historical ward data is unavailable for building live input features."
    return templates.TemplateResponse(
        request=request,
        name="user/live_input.html",
        context={
            "current_user": current_user,
            "csrf_token": get_csrf_token(request),
            "areas": areas,
            "data_error": data_error,
        },
        headers={"Cache-Control": "no-store, private"},
    )


@app.get("/api/live-input/defaults")
async def live_input_defaults(
    city: str = Query(min_length=1, max_length=100),
    ward: str = Query(min_length=1, max_length=100),
    current_user: User = Depends(get_current_hospital_user),
):
    """Return historical context for the authenticated hospital member's form."""
    from src.live_input_predict import LiveInputError, get_live_defaults

    try:
        return get_live_defaults(city, ward)
    except LiveInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Live input history is unavailable.") from exc


@app.post("/api/live-predict")
async def live_input_prediction(
    payload: LivePredictionRequest,
    request: Request,
    x_csrf_token: str = Header(default=""),
    current_user: User = Depends(get_current_hospital_user),
):
    """Run inference from a one-week observation; never train or refit here."""
    if not verify_csrf_token(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")
    from src.live_input_predict import (
        LiveInputError, LiveModelUnavailableError, predict_live,
    )

    try:
        return predict_live(
            city=payload.city,
            ward_id=payload.ward,
            observation_date=payload.observationDate,
            horizon_weeks=payload.horizonWeeks,
            avg_temp_c=payload.avgTempC,
            max_temp_c=payload.maxTempC,
            min_temp_c=payload.minTempC,
            rainfall_mm=payload.rainfallMm,
            relative_humidity_pct=payload.humidityPct,
            dengue_cases=payload.dengueCases,
        )
    except LiveInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LiveModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# --------------------------------------------------
# ADMIN DASHBOARD
# --------------------------------------------------

@app.get("/admin_dashboard")
async def admin_dashboard(
    request: Request,
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Render the admin dashboard using live account records from the database."""

    csrf_token = get_csrf_token(request)

    # Keep the dashboard statistics aligned with the same users table shown below.
    # Hospital accounts exclude administrators; administrator count includes both
    # active and inactive administrator accounts.
    total_users = db.query(func.count(User.id)).scalar() or 0
    active_hospital_accounts = (
        db.query(func.count(User.id))
        .filter(User.is_admin.is_(False), User.is_active.is_(True))
        .scalar()
        or 0
    )
    administrator_accounts = (
        db.query(func.count(User.id))
        .filter(User.is_admin.is_(True))
        .scalar()
        or 0
    )
    inactive_accounts = (
        db.query(func.count(User.id))
        .filter(User.is_active.is_(False))
        .scalar()
        or 0
    )

    # Do not expose password hashes or other internal fields to the template.
    users = (
        db.query(User)
        .order_by(User.created_at.desc(), User.id.desc())
        .all()
    )

    return templates.TemplateResponse(
        request=request,
        name="admin/admin_dashboard.html",
        context={
            "current_user": current_user,
            "csrf_token": csrf_token,
            "total_users": total_users,
            "active_hospital_accounts": active_hospital_accounts,
            "administrator_accounts": administrator_accounts,
            "inactive_accounts": inactive_accounts,
            "users": users,
        },
        headers={"Cache-Control": "no-store, private"},
    )


# --------------------------------------------------
# ADMIN-ONLY JSON API
# --------------------------------------------------


def _user_payload(user: User) -> dict:
    """Return only fields intended for the administrator user table."""
    return {
        "id": user.id,
        "full_name": user.full_name,
        "hospital_name": user.hospital_name,
        "email": user.email,
        "is_admin": bool(user.is_admin),
        "is_active": bool(user.is_active),
        "created_at": user.created_at.isoformat() if user.created_at else None,
    }


def _hospital_inventory_payload(record: HospitalOperationalRecord | None) -> dict:
    return {
        "actual_available_beds": record.actual_available_beds if record else None,
        "saline_stock_available": record.saline_stock_available if record else None,
        "diagnostic_kits_stock_available": record.diagnostic_kits_stock_available if record else None,
        "updated_at": record.updated_at.isoformat() if record and record.updated_at else None,
    }


def _ward_operational_summary(db: Session) -> list[dict]:
    """Combine stable ward CSV data with complete hospital-reported ward aggregates."""
    from src.backend import operational_data

    try:
        capacity = operational_data.read_ward_capacity()
    except operational_data.OperationalDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    assignments = db.query(HospitalWardAssignment).all()
    crew_records = {record.ward_id: record for record in db.query(WardOperationsRecord).all()}
    records = {
        record.hospital_name: record
        for record in db.query(HospitalOperationalRecord).all()
    }
    by_ward: dict[str, list[str]] = {}
    for assignment in assignments:
        by_ward.setdefault(assignment.ward_id, []).append(assignment.hospital_name)

    forecasts_by_ward: dict[str, dict[str, dict]] = {}
    forecast_path = PROJECT_ROOT / "data" / "processed" / "ward_resource_demand.csv"
    try:
        from src.reference_catboost_predict import get_available_horizons, load_model
        if set(get_available_horizons()) == set(SUPPORTED_HORIZONS) and forecast_path.is_file():
            runs = {load_model(horizon)[2].name for horizon in SUPPORTED_HORIZONS}
            if len(runs) == 1:
                forecast_table = pd.read_csv(forecast_path)
                required = {"forecast_origin", "ward_id", "horizon_weeks", "expected_cases", "risk_tier", "model_version"}
                if required.issubset(forecast_table.columns) and set(forecast_table.model_version.dropna().astype(str)) == runs:
                    latest_origin = forecast_table.forecast_origin.max()
                    forecast_table = forecast_table.loc[forecast_table.forecast_origin.eq(latest_origin)]
                    for row in forecast_table.to_dict(orient="records"):
                        forecasts_by_ward.setdefault(str(row["ward_id"]), {})[str(int(row["horizon_weeks"]))] = {
                            "origin": str(row["forecast_origin"]),
                            "expected_cases": float(row["expected_cases"]),
                            "risk_tier": str(row["risk_tier"]),
                            "mode": "historical replay" if str(row["forecast_origin"]) < date.today().isoformat() else "current forecast",
                        }
    except Exception:
        # Forecasts stay explicitly unavailable when a compatible central batch cannot be read.
        forecasts_by_ward = {}

    result = []
    for row in capacity.to_dict(orient="records"):
        ward_id = str(row["ward_id"])
        linked = by_ward.get(ward_id, [])
        ward_records = [records[name] for name in linked if name in records]
        report = {"hospital_count": len(linked), "reporting_hospital_count": len(ward_records)}
        for field in ("actual_available_beds", "saline_stock_available", "diagnostic_kits_stock_available"):
            values = [getattr(item, field) for item in ward_records]
            # Do not present a partial sum as total availability for the ward.
            complete = bool(linked) and len(ward_records) == len(linked) and all(value is not None for value in values)
            report[field] = sum(values) if complete else None
            report[f"{field}_status"] = "reported" if complete else ("partial" if any(v is not None for v in values) else "not_provided")
        crew = crew_records.get(ward_id)
        if crew is not None:
            row["crew_count_available"] = crew.crew_count_available
            row["last_updated"] = crew.updated_at.isoformat() if crew.updated_at else row.get("last_updated")
            row["operational_data_status"] = (
                "Ward crew availability reported by scoped operations staff; hospital inventory is reported separately."
            )
        result.append({**row, **report, "forecast_by_horizon": forecasts_by_ward.get(ward_id, {})})
    # Starlette's JSON encoder rejects NaN. In this data contract blanks mean
    # unknown, so expose them as JSON null rather than zero or non-standard NaN.
    for record in result:
        for key, value in tuple(record.items()):
            if value is not None and not isinstance(value, (dict, list)) and pd.isna(value):
                record[key] = None
    return result


@app.get("/api/wards/operational-summary")
async def ward_operational_summary(
    current_user: User = Depends(get_current_hospital_user),
    db: Session = Depends(get_db),
):
    """Provide ward reference, vulnerability, capacity, and known inventory to dashboard users."""
    return {"wards": _ward_operational_summary(db)}


@app.get("/api/hospital/operational-data")
async def hospital_operational_data(
    current_user: User = Depends(get_current_hospital_user),
    db: Session = Depends(get_db),
):
    membership = db.get(HospitalMembership, current_user.id)
    if membership is None:
        raise HTTPException(status_code=403, detail="A verified hospital association is required before inventory access.")
    record = db.query(HospitalOperationalRecord).filter_by(hospital_name=membership.hospital_name).one_or_none()
    assignments = db.query(HospitalWardAssignment).filter_by(hospital_name=membership.hospital_name).all()
    return {
        "hospital_name": membership.hospital_name,
        "inventory": _hospital_inventory_payload(record),
        "wards": [{"ward_id": item.ward_id, "city": item.city} for item in assignments],
        "operational_data_status": "Blank values are unknown until this hospital reports them.",
    }


@app.put("/api/hospital/operational-data")
async def update_hospital_operational_data(
    payload: HospitalInventoryPayload,
    request: Request,
    x_csrf_token: str = Header(default=""),
    current_user: User = Depends(get_current_hospital_user),
    db: Session = Depends(get_db),
):
    if not verify_csrf_token(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")
    membership = db.get(HospitalMembership, current_user.id)
    if membership is None:
        raise HTTPException(status_code=403, detail="A verified hospital association is required before inventory updates.")
    fields = payload.model_fields_set
    if not fields:
        raise HTTPException(status_code=422, detail="Submit at least one hospital inventory field.")
    record = db.query(HospitalOperationalRecord).filter_by(hospital_name=membership.hospital_name).one_or_none()
    if record is None:
        record = HospitalOperationalRecord(hospital_name=membership.hospital_name)
        db.add(record)
    for field in fields:
        setattr(record, field, getattr(payload, field))
    record.updated_by_user_id = current_user.id
    db.add(AuditLog(
        event_type="hospital_inventory_updated",
        description=f"Hospital inventory fields updated for {membership.hospital_name}: {', '.join(sorted(fields))}.",
        actor_user_id=current_user.id,
    ))
    try:
        db.commit()
        db.refresh(record)
    except Exception as exc:
        db.rollback()
        logger.exception("Hospital inventory update failed for actor %s", current_user.id)
        raise HTTPException(status_code=503, detail="Hospital inventory update could not be saved.") from exc
    return {"hospital_name": membership.hospital_name, "inventory": _hospital_inventory_payload(record)}


@app.put("/api/admin/hospital-operational-data/{hospital_name}")
async def correct_hospital_operational_data(
    hospital_name: str,
    payload: HospitalInventoryCorrectionPayload,
    request: Request,
    x_csrf_token: str = Header(default=""),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Apply a specifically reasoned, audited correction under admin oversight."""
    if not verify_csrf_token(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")
    normalized = " ".join(hospital_name.split())
    record = db.query(HospitalOperationalRecord).filter_by(hospital_name=normalized).one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="Hospital operational record was not found.")
    fields = payload.model_fields_set - {"correction_reason"}
    if not fields:
        raise HTTPException(status_code=422, detail="Submit at least one inventory field to correct.")
    for field in fields:
        setattr(record, field, getattr(payload, field))
    record.updated_by_user_id = current_user.id
    reason = " ".join(payload.correction_reason.split())
    db.add(AuditLog(
        event_type="admin_hospital_inventory_corrected",
        description=(f"Administrator corrected {', '.join(sorted(fields))} for {normalized}; "
                     f"reason: {reason}"),
        actor_user_id=current_user.id,
    ))
    try:
        db.commit()
        db.refresh(record)
    except Exception as exc:
        db.rollback()
        logger.exception("Administrative hospital data correction failed for actor %s", current_user.id)
        raise HTTPException(status_code=503, detail="Hospital data correction could not be saved.") from exc
    return {"hospital_name": normalized, "inventory": _hospital_inventory_payload(record), "corrected_fields": sorted(fields)}


@app.get("/api/hospital/operational-history")
async def hospital_operational_history(
    limit: int = Query(default=50, ge=1, le=100),
    current_user: User = Depends(get_current_hospital_user),
    db: Session = Depends(get_db),
):
    membership = db.get(HospitalMembership, current_user.id)
    if membership is None:
        raise HTTPException(status_code=403, detail="A verified hospital association is required.")
    member_ids = [row.user_id for row in db.query(HospitalMembership.user_id).filter_by(hospital_name=membership.hospital_name).all()]
    events = (
        db.query(AuditLog).filter(
            AuditLog.event_type == "hospital_inventory_updated",
            AuditLog.actor_user_id.in_(member_ids),
        ).order_by(AuditLog.created_at.desc(), AuditLog.id.desc()).limit(limit).all()
    )
    return {"events": [{"event_type": row.event_type, "description": row.description,
                         "timestamp": row.created_at.isoformat()} for row in events]}


@app.get("/api/admin/operations")
async def admin_operations_overview(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    memberships = db.query(HospitalMembership).all()
    records = db.query(HospitalOperationalRecord).all()
    assignments = db.query(HospitalWardAssignment).all()
    scopes = db.query(OperationsUserScope).all()
    users = db.query(User).filter(User.is_admin.is_(False)).all()
    return {
        "wards": _ward_operational_summary(db),
        "hospital_users": [{"id": u.id, "full_name": u.full_name, "hospital_name": u.hospital_name,
                             "email": u.email, "is_active": bool(u.is_active),
                             "verified_hospital": next((m.hospital_name for m in memberships if m.user_id == u.id), None)}
                            for u in users],
        "hospital_inventory": [{"hospital_name": row.hospital_name, **_hospital_inventory_payload(row)} for row in records],
        "hospital_ward_assignments": [{"hospital_name": x.hospital_name, "ward_id": x.ward_id, "city": x.city} for x in assignments],
        "operations_scopes": [{"user_id": x.user_id, "city": x.city, "ward_id": x.ward_id} for x in scopes],
    }


@app.put("/api/admin/hospital-memberships/{user_id}")
async def assign_hospital_membership(
    user_id: int,
    payload: HospitalMembershipPayload,
    request: Request,
    x_csrf_token: str = Header(default=""),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    if not verify_csrf_token(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")
    user = db.get(User, user_id)
    hospital_name = " ".join(payload.hospital_name.split())
    if user is None or user.is_admin or not user.is_active:
        raise HTTPException(status_code=404, detail="Active hospital user was not found.")
    if user.hospital_name.casefold() != hospital_name.casefold():
        raise HTTPException(status_code=422, detail="Membership must match the hospital name supplied by the account.")
    membership = db.get(HospitalMembership, user_id)
    if membership is None:
        membership = HospitalMembership(user_id=user_id)
        db.add(membership)
    membership.hospital_name = hospital_name
    membership.approved_by_user_id = current_user.id
    if db.query(HospitalOperationalRecord).filter_by(hospital_name=hospital_name).one_or_none() is None:
        db.add(HospitalOperationalRecord(hospital_name=hospital_name))
    db.add(AuditLog(event_type="hospital_membership_assigned",
                    description=f"Verified hospital association assigned to user {user_id}.", actor_user_id=current_user.id))
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("Hospital membership assignment failed for actor %s", current_user.id)
        raise HTTPException(status_code=503, detail="Hospital membership and audit event could not be saved.") from exc
    return {"user_id": user_id, "hospital_name": hospital_name, "verified": True}


@app.put("/api/admin/hospital-ward-assignments/{hospital_name}")
async def manage_hospital_ward_assignment(
    hospital_name: str,
    payload: HospitalWardAssignmentPayload,
    request: Request,
    x_csrf_token: str = Header(default=""),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    if not verify_csrf_token(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")
    from src.backend import operational_data
    data = operational_data.read_ward_capacity()
    valid = data.loc[data.ward_id.astype(str).eq(payload.ward_id) & data.city.astype(str).eq(payload.city)]
    if valid.empty:
        raise HTTPException(status_code=422, detail="Ward does not belong to the selected city.")
    if db.query(HospitalOperationalRecord).filter_by(hospital_name=hospital_name).one_or_none() is None:
        raise HTTPException(status_code=404, detail="Hospital inventory record is not established.")
    assignment = db.query(HospitalWardAssignment).filter_by(hospital_name=hospital_name, ward_id=payload.ward_id).one_or_none()
    if payload.assigned and assignment is None:
        db.add(HospitalWardAssignment(hospital_name=hospital_name, ward_id=payload.ward_id,
                                      city=payload.city, assigned_by_user_id=current_user.id))
    elif not payload.assigned and assignment is not None:
        db.delete(assignment)
    db.add(AuditLog(event_type="hospital_ward_assignment_changed",
                    description=f"Hospital ward association updated for {hospital_name} / {payload.ward_id}.", actor_user_id=current_user.id))
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("Hospital ward assignment failed for actor %s", current_user.id)
        raise HTTPException(status_code=503, detail="Hospital ward assignment and audit event could not be saved.") from exc
    return {"hospital_name": hospital_name, "ward_id": payload.ward_id, "city": payload.city, "assigned": payload.assigned}


@app.put("/api/admin/operations-scopes/{user_id}")
async def manage_operations_scope(
    user_id: int,
    payload: OperationsScopePayload,
    request: Request,
    x_csrf_token: str = Header(default=""),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    if not verify_csrf_token(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")
    from src.backend import operational_data
    ward_data = operational_data.read_ward_capacity()
    scoped = ward_data.loc[ward_data.city.astype(str).eq(payload.city)]
    if payload.ward_id:
        scoped = scoped.loc[scoped.ward_id.astype(str).eq(payload.ward_id)]
    user = db.get(User, user_id)
    if user is None or user.is_admin or not user.is_active or scoped.empty:
        raise HTTPException(status_code=422, detail="Active operations user and valid city/ward scope are required.")
    query = db.query(OperationsUserScope).filter_by(user_id=user_id, city=payload.city)
    scopes = query.all()
    if payload.ward_id:
        item = next((x for x in scopes if x.ward_id == payload.ward_id), None)
        if payload.authorized and item is None:
            db.add(OperationsUserScope(user_id=user_id, city=payload.city, ward_id=payload.ward_id, granted_by_user_id=current_user.id))
        elif not payload.authorized and item is not None:
            db.delete(item)
    elif payload.authorized:
        for item in scopes:
            db.delete(item)
        db.add(OperationsUserScope(user_id=user_id, city=payload.city, ward_id=None, granted_by_user_id=current_user.id))
    else:
        for item in scopes:
            db.delete(item)
    db.add(AuditLog(event_type="operations_scope_changed",
                    description=f"Operations scope changed for user {user_id} in {payload.city}.", actor_user_id=current_user.id))
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("Operations scope change failed for actor %s", current_user.id)
        raise HTTPException(status_code=503, detail="Operations scope and audit event could not be saved.") from exc
    return {"user_id": user_id, "city": payload.city, "ward_id": payload.ward_id, "authorized": payload.authorized}


@app.get("/api/operations/wards")
async def operations_scoped_wards(
    current_user: User = Depends(get_current_hospital_user),
    db: Session = Depends(get_db),
):
    from src.backend import operational_data
    scopes = db.query(OperationsUserScope).filter_by(user_id=current_user.id).all()
    if not scopes:
        raise HTTPException(status_code=403, detail="No ward operations scope is assigned to this account.")
    data = operational_data.read_ward_capacity()
    allowed = set()
    for scope in scopes:
        rows = data.loc[data.city.astype(str).eq(scope.city)]
        if scope.ward_id:
            rows = rows.loc[rows.ward_id.astype(str).eq(scope.ward_id)]
        allowed.update(rows.ward_id.astype(str))
    return {"wards": [row for row in _ward_operational_summary(db) if row["ward_id"] in allowed]}


@app.put("/api/operations/wards/{ward_id}/crew")
async def update_scoped_ward_crew(
    ward_id: str,
    payload: WardCrewPayload,
    request: Request,
    x_csrf_token: str = Header(default=""),
    current_user: User = Depends(get_current_hospital_user),
    db: Session = Depends(get_db),
):
    if not verify_csrf_token(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")
    from src.backend import operational_data
    data = operational_data.read_ward_capacity()
    rows = data.loc[data.ward_id.astype(str).eq(ward_id)]
    if rows.empty:
        raise HTTPException(status_code=404, detail="Unknown ward identifier.")
    city = str(rows.iloc[0].city)
    scopes = db.query(OperationsUserScope).filter_by(user_id=current_user.id, city=city).all()
    if not any(scope.ward_id is None or scope.ward_id == ward_id for scope in scopes):
        raise HTTPException(status_code=403, detail="This account is not authorized to manage the selected ward.")
    record = db.get(WardOperationsRecord, ward_id)
    if record is None:
        record = WardOperationsRecord(ward_id=ward_id)
        db.add(record)
    record.crew_count_available = payload.crew_count_available
    record.updated_by_user_id = current_user.id
    db.add(AuditLog(event_type="ward_crew_updated",
                    description=f"Available response crew count updated for {ward_id}.", actor_user_id=current_user.id))
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("Crew update audit failed for actor %s", current_user.id)
        raise HTTPException(status_code=503, detail="Crew update and audit event could not be saved.") from exc
    return {"ward_id": ward_id, "city": city, "crew_count_available": int(record.crew_count_available)}


@app.post("/api/admin/forecasts/batch")
async def create_batch_forecast(
    payload: BatchForecastPayload,
    request: Request,
    x_csrf_token: str = Header(default=""),
    current_user: User = Depends(get_current_admin_user),
):
    """Create an all-ward batch only from compatible trained models and safe origins."""
    if not verify_csrf_token(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")
    from src.reference_catboost_predict import generate_batch_forecast, write_batch_forecast
    try:
        frame = generate_batch_forecast(payload.asOf.isoformat() if payload.asOf else None)
        path = write_batch_forecast(frame)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ForecastUnavailableError, ModelUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, LookupError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"rows": len(frame), "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "forecast_origin": sorted(frame.forecast_origin.unique()),
            "horizons": sorted(int(value) for value in frame.horizon_weeks.unique())}


@app.post("/api/resources/allocation")
async def resource_allocation(
    payload: AllocationRequest,
    request: Request,
    x_csrf_token: str = Header(default=""),
    current_user: User = Depends(get_current_hospital_user),
    db: Session = Depends(get_db),
):
    """Calculate resource gaps and proposed same-city pooling from compatible forecasts."""
    if not verify_csrf_token(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")
    from src.reference_catboost_predict import generate_batch_forecast
    from src.resource_allocation import calculate_resource_allocation
    try:
        forecasts = generate_batch_forecast(payload.asOf.isoformat() if payload.asOf else None)
        operations = pd.DataFrame(_ward_operational_summary(db))
        allocation = calculate_resource_allocation(
            forecasts, operations, horizon_weeks=payload.horizonWeeks,
            stock_availability_pct=payload.stockAvailabilityPct,
            bed_capacity_adjustment_pct=payload.bedCapacityAdjustmentPct,
            selected_wards=payload.selectedWards,
            hypothetical_crew_count=payload.hypotheticalCrewCount,
        )
        allocation["forecast_mode"] = "historical replay" if payload.asOf else "current forecast"
        allocation["forecast_origin"] = sorted(forecasts.forecast_origin.unique())
        return allocation
    except (ForecastUnavailableError, ModelUnavailableError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/what-if")
async def what_if_simulation(
    payload: WhatIfRequest,
    request: Request,
    x_csrf_token: str = Header(default=""),
    current_user: User = Depends(get_current_hospital_user),
):
    """Run baseline and assumption-based rainfall-proxy model scenarios."""
    if not verify_csrf_token(request, x_csrf_token):
        raise HTTPException(status_code=403, detail="Invalid or missing CSRF token.")
    from src.live_input_predict import LiveInputError, LiveModelUnavailableError, simulate_what_if
    try:
        return simulate_what_if(
            city=payload.city,
            ward_ids=payload.selectedWards,
            forecast_date=payload.forecastDate,
            hypothetical_crew_count=payload.hypotheticalCrewCount,
            intervention_effectiveness_pct=payload.interventionEffectivenessPct,
        )
    except LiveInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LiveModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/admin/dashboard/summary")
async def admin_dashboard_summary(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Return live account totals for the administrator dashboard."""
    return {
        "total_users": db.query(func.count(User.id)).scalar() or 0,
        "active_hospital_accounts": (
            db.query(func.count(User.id))
            .filter(User.is_admin.is_(False), User.is_active.is_(True))
            .scalar() or 0
        ),
        "administrator_accounts": (
            db.query(func.count(User.id))
            .filter(User.is_admin.is_(True))
            .scalar() or 0
        ),
        "inactive_accounts": (
            db.query(func.count(User.id))
            .filter(User.is_active.is_(False))
            .scalar() or 0
        ),
    }


@app.get("/api/admin/users")
async def admin_list_users(
    search: str = Query(default="", max_length=255),
    role: str = Query(default="", pattern="^(|admin|hospital)$"),
    status_filter: str = Query(default="", alias="status", pattern="^(|active|inactive)$"),
    limit: int = Query(default=500, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Search and filter user accounts without exposing password hashes."""
    query = db.query(User)
    term = search.strip().casefold()
    if term:
        pattern = f"%{term}%"
        query = query.filter(
            func.lower(User.full_name).like(pattern)
            | func.lower(User.hospital_name).like(pattern)
            | func.lower(User.email).like(pattern)
        )
    if role == "admin":
        query = query.filter(User.is_admin.is_(True))
    elif role == "hospital":
        query = query.filter(User.is_admin.is_(False))
    if status_filter == "active":
        query = query.filter(User.is_active.is_(True))
    elif status_filter == "inactive":
        query = query.filter(User.is_active.is_(False))

    total = query.count()
    records = (
        query.order_by(User.created_at.desc(), User.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "users": [_user_payload(user) for user in records],
    }


@app.get("/api/admin/activity")
async def admin_recent_activity(
    limit: int = Query(default=20, ge=1, le=50),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Return persisted, non-sensitive audit events newest first."""
    events = (
        db.query(AuditLog)
        .order_by(AuditLog.created_at.desc(), AuditLog.id.desc())
        .limit(limit)
        .all()
    )
    return {
        "events": [
            {
                "id": event.id,
                "event_type": event.event_type,
                "description": event.description,
                "timestamp": event.created_at.isoformat(),
            }
            for event in events
        ]
    }


@app.get("/api/admin/system-status")
async def admin_system_status(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
):
    """Report observed database, dataset, and model-artifact availability."""
    try:
        db.execute(text("SELECT 1"))
        database_status = "available"
    except Exception:
        database_status = "unavailable"

    processed_status = "missing"
    data_through = None
    freshness_status = "unknown"
    if DATA_PATH.is_file():
        processed_status = "available"
        try:
            freshness = data_freshness()
            data_through = freshness.get("dataThrough")
            freshness_status = freshness.get("status", "unknown")
        except Exception:
            processed_status = "invalid"
            freshness_status = "invalid"

    try:
        available_horizons = get_available_horizons()
    except Exception:
        available_horizons = []

    return {
        "backend": "available",
        "database": database_status,
        "raw_dataset": "not_configured",
        "processed_dataset": processed_status,
        "data_through": data_through,
        "data_freshness": freshness_status,
        "model_artifacts": "available" if available_horizons else "unavailable",
        "available_horizons": available_horizons,
        "prediction_service": "available" if available_horizons else "unavailable",
        "application_version": app.version,
    }


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

Base.metadata.create_all(bind=engine)
