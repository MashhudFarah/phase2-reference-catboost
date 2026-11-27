"""Direct non-fitting checks for authorization and audited operations routes."""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["SESSION_SECRET"] = "operations-api-test-session-secret-32chars"
os.environ["COOKIE_SECURE"] = "false"

import pandas as pd
import pytest
from pydantic import ValidationError
from fastapi import HTTPException
from starlette.requests import Request

import main
from src.backend.database import Base, SessionLocal, engine
from src.backend.models import (
    AuditLog, HospitalMembership, HospitalOperationalRecord, OperationsUserScope,
    User, WardOperationsRecord,
)


def request_with_csrf(token="operations-test-csrf-token-which-is-long-enough"):
    return Request({
        "type": "http", "http_version": "1.1", "method": "PUT", "scheme": "http",
        "path": "/api/operations/test", "raw_path": b"/api/operations/test",
        "query_string": b"", "headers": [], "client": ("127.0.0.1", 1),
        "server": ("test", 80), "root_path": "", "session": {"_csrf_token": token},
    })


@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    for model in (AuditLog, WardOperationsRecord, OperationsUserScope, HospitalMembership,
                  HospitalOperationalRecord, User):
        db.query(model).delete()
    db.commit()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


def add_user(db, *, admin=False, hospital="Hospital A"):
    user = User(full_name="Test Account", hospital_name=hospital,
                email=f"user-{admin}-{hospital}@example.invalid", hashed_password="hash",
                is_admin=admin, is_active=True, terms_accepted=True)
    db.add(user)
    db.commit()
    return user


def test_hospital_member_can_only_update_verified_hospital_inventory(db_session):
    db = db_session
    hospital_user = add_user(db)
    db.add(HospitalMembership(user_id=hospital_user.id, hospital_name="Hospital A"))
    own = HospitalOperationalRecord(hospital_name="Hospital A", actual_available_beds=3)
    other = HospitalOperationalRecord(hospital_name="Hospital B", actual_available_beds=9)
    db.add_all([own, other]); db.commit()
    token = "operations-test-csrf-token-which-is-long-enough"
    result = asyncio.run(main.update_hospital_operational_data(
        main.HospitalInventoryPayload(actual_available_beds=5, saline_stock_available=11),
        request_with_csrf(token), token, hospital_user, db,
    ))
    db.refresh(other)
    assert result["hospital_name"] == "Hospital A"
    assert result["inventory"]["actual_available_beds"] == 5
    assert result["inventory"]["saline_stock_available"] == 11
    assert other.actual_available_beds == 9
    assert db.query(AuditLog).filter_by(event_type="hospital_inventory_updated").count() == 1
    with pytest.raises(HTTPException) as denied:
        asyncio.run(main.update_hospital_operational_data(
            main.HospitalInventoryPayload(actual_available_beds=1), request_with_csrf(token),
            "invalid", hospital_user, db,
        ))
    assert denied.value.status_code == 403


def test_admin_correction_requires_reason_and_is_audited(db_session):
    db = db_session
    admin = add_user(db, admin=True, hospital="Platform")
    db.add(HospitalOperationalRecord(hospital_name="Hospital A", actual_available_beds=2)); db.commit()
    token = "operations-test-csrf-token-which-is-long-enough"
    result = asyncio.run(main.correct_hospital_operational_data(
        "Hospital A", main.HospitalInventoryCorrectionPayload(
            actual_available_beds=4, correction_reason="Verified by facility call",
        ), request_with_csrf(token), token, admin, db,
    ))
    assert result["inventory"]["actual_available_beds"] == 4
    audit = db.query(AuditLog).filter_by(event_type="admin_hospital_inventory_corrected").one()
    assert "Verified by facility call" in audit.description
    with pytest.raises(ValidationError):
        main.HospitalInventoryCorrectionPayload(correction_reason="short")


def test_scoped_crew_api_checks_scope_and_records_audit(monkeypatch, db_session):
    db = db_session
    user = add_user(db)
    db.add(OperationsUserScope(user_id=user.id, city="City A", ward_id="WARD_01")); db.commit()
    from src.backend import operational_data
    fake = pd.DataFrame([{"ward_id": "WARD_01", "city": "City A", "crew_count_available": None}])
    monkeypatch.setattr(operational_data, "read_ward_capacity", lambda: fake.copy())
    token = "operations-test-csrf-token-which-is-long-enough"
    result = asyncio.run(main.update_scoped_ward_crew(
        "WARD_01", main.WardCrewPayload(crew_count_available=2),
        request_with_csrf(token), token, user, db,
    ))
    assert result["crew_count_available"] == 2
    assert db.get(WardOperationsRecord, "WARD_01").crew_count_available == 2
    assert db.query(AuditLog).filter_by(event_type="ward_crew_updated").count() == 1
    with pytest.raises(HTTPException) as denied:
        asyncio.run(main.update_scoped_ward_crew(
            "WARD_02", main.WardCrewPayload(crew_count_available=2),
            request_with_csrf(token), token, user, db,
        ))
    assert denied.value.status_code == 404


def test_ward_summary_is_json_safe_and_keeps_unreported_values_unknown(db_session):
    records = main._ward_operational_summary(db_session)
    encoded = json.dumps(records, allow_nan=False)
    assert len(records) == 60
    assert len(encoded) > 100
    assert all(row["actual_available_beds"] is None for row in records)
    assert all(row["saline_stock_available"] is None for row in records)
    assert all(row["diagnostic_kits_stock_available"] is None for row in records)
    assert all(row["forecast_by_horizon"] == {} for row in records)

