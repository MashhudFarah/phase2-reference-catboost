"""Async ASGI integration checks that avoid the installed TestClient hang."""

from __future__ import annotations

import asyncio
import os
import re
from types import SimpleNamespace

os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["SESSION_SECRET"] = "isolated-asgi-test-session-secret-32chars"
os.environ["COOKIE_SECURE"] = "false"

import httpx
from src.backend.database import Base, SessionLocal, engine
import main


def test_authenticated_pages_operations_apis_csrf_and_missing_model_responses():
    Base.metadata.create_all(bind=engine)
    hospital = SimpleNamespace(
        id=99101, full_name="ASGI Test User", hospital_name="Test Hospital",
        email="asgi-user@example.invalid", is_admin=False, is_active=True,
    )
    admin = SimpleNamespace(
        id=99102, full_name="ASGI Test Admin", hospital_name="Platform",
        email="asgi-admin@example.invalid", is_admin=True, is_active=True,
    )

    async def hospital_override():
        return hospital

    async def admin_override():
        return admin

    async def db_override():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()

    main.app.dependency_overrides[main.get_current_hospital_user] = hospital_override
    main.app.dependency_overrides[main.get_current_admin_user] = admin_override
    main.app.dependency_overrides[main.get_db] = db_override

    async def exercise():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test",
        ) as client:
            for path in ("/", "/login", "/register", "/user_dashboard", "/live-input", "/admin_dashboard"):
                response = await client.get(path)
                assert response.status_code == 200, (path, response.status_code)
            token = re.search(
                r'name="csrf_token"\s+value="([^"]+)', (await client.get("/login")).text,
            ).group(1)
            defaults = await client.get("/api/live-input/defaults", params={
                "city": "Chattogram", "ward": "CCC_Ward_01",
            })
            assert defaults.status_code == 200
            assert defaults.json()["latestHistoricalDate"] == "2025-12-29"
            live_payload = {
                "city": "Chattogram", "ward": "CCC_Ward_01",
                "observationDate": "2026-01-05", "horizonWeeks": 2,
                "avgTempC": 25, "maxTempC": 30, "minTempC": 20,
                "rainfallMm": 0, "humidityPct": 60, "dengueCases": 1,
            }
            live_csrf = await client.post("/api/live-predict", json=live_payload,
                                          headers={"x-csrf-token": "invalid"})
            assert live_csrf.status_code == 403
            live_result = await client.post("/api/live-predict", json=live_payload,
                                             headers={"x-csrf-token": token})
            assert live_result.status_code == 422
            assert "within 14 days" in live_result.json()["detail"]
            wards = await client.get("/api/wards/operational-summary")
            assert wards.status_code == 200 and len(wards.json()["wards"]) == 60
            admin_data = await client.get("/api/admin/operations")
            assert admin_data.status_code == 200 and len(admin_data.json()["wards"]) == 60
            hospital_inventory = await client.get("/api/hospital/operational-data")
            assert hospital_inventory.status_code == 403
            unrestricted_operations = await client.get("/api/operations/wards")
            assert unrestricted_operations.status_code == 403
            denied = await client.post("/api/predict", json={
                "city": "Chattogram", "ward": "CCC_Ward_01", "horizonWeeks": 2,
            }, headers={"x-csrf-token": "invalid"})
            assert denied.status_code == 403
            unavailable = await client.post("/api/predict", json={
                "city": "Chattogram", "ward": "CCC_Ward_01", "horizonWeeks": 2,
                "asOf": "2025-08-18",
            }, headers={"x-csrf-token": token})
            assert unavailable.status_code == 503
            batch = await client.post("/api/admin/forecasts/batch", json={
                "asOf": "2025-08-18",
            }, headers={"x-csrf-token": token})
            assert batch.status_code == 503
            allocation = await client.post("/api/resources/allocation", json={
                "horizonWeeks": 2, "asOf": "2025-08-18",
            }, headers={"x-csrf-token": token})
            assert allocation.status_code == 503
            what_if = await client.post("/api/what-if", json={
                "city": "Chattogram", "selectedWards": ["CCC_Ward_01"],
                "forecastDate": "2025-08-18", "hypotheticalCrewCount": 1,
                "interventionEffectivenessPct": 20,
            }, headers={"x-csrf-token": token})
            assert what_if.status_code == 503

    try:
        asyncio.run(exercise())
    finally:
        main.app.dependency_overrides.clear()
