# tests/test_api.py
# FastAPI endpoint tests using Starlette's TestClient.
# Each test gets its own temporary SQLite database via the temp_db fixture.

import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Fresh temp database for every test — same pattern as test_database.py."""
    db_file = str(tmp_path / "test_edgepulse.db")
    import config
    import database.db as db_module
    monkeypatch.setattr(config, "DATABASE_PATH", db_file)
    monkeypatch.setattr(db_module, "DATABASE_PATH", db_file)
    db_module.init_db()
    return db_file


import database.db as db_module
from api.main import app

SAMPLE = {
    "timestamp": "2026-08-27T10:00:00",
    "vehicle_id": "car_01",
    "speed": 100.0,
    "rpm": 3000.0,
    "engine_temp": 90.0,
}


@pytest.fixture
def client():
    return TestClient(app)


def test_root(client):
    """GET / returns 200 with service info."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "EdgePulse Telemetry API"
    assert body["status"] == "running"


def test_health(client):
    """GET /health returns 200 when database is reachable."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_latest_telemetry_empty(client):
    """GET /telemetry/latest returns 404 when no data exists."""
    response = client.get("/telemetry/latest")
    assert response.status_code == 404


def test_latest_telemetry(client):
    """GET /telemetry/latest returns the most recently inserted record."""
    db_module.insert_telemetry(SAMPLE)
    response = client.get("/telemetry/latest")
    assert response.status_code == 200
    body = response.json()
    assert body["vehicle_id"] == "car_01"
    assert body["speed"] == 100.0


def test_list_telemetry_empty(client):
    """GET /telemetry returns an empty list when no data exists."""
    response = client.get("/telemetry")
    assert response.status_code == 200
    assert response.json() == []


def test_list_telemetry(client):
    """GET /telemetry returns inserted records."""
    db_module.insert_telemetry(SAMPLE)
    db_module.insert_telemetry({**SAMPLE, "speed": 80.0})
    response = client.get("/telemetry")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_list_telemetry_limit(client):
    """GET /telemetry?limit=1 respects the limit parameter."""
    for i in range(5):
        db_module.insert_telemetry({**SAMPLE, "speed": float(i * 10)})
    response = client.get("/telemetry?limit=1")
    assert response.status_code == 200
    assert len(response.json()) == 1


def test_list_telemetry_invalid_limit(client):
    """GET /telemetry?limit=0 returns 422 Unprocessable Entity."""
    response = client.get("/telemetry?limit=0")
    assert response.status_code == 422


def test_list_telemetry_limit_too_large(client):
    """GET /telemetry?limit=200 exceeds allowed max of 100 — 422."""
    response = client.get("/telemetry?limit=200")
    assert response.status_code == 422


def test_vehicle_telemetry(client):
    """GET /telemetry/{vehicle_id} returns only that vehicle's records."""
    db_module.insert_telemetry({**SAMPLE, "vehicle_id": "car_01"})
    db_module.insert_telemetry({**SAMPLE, "vehicle_id": "car_02"})
    response = client.get("/telemetry/car_01")
    assert response.status_code == 200
    records = response.json()
    assert len(records) == 1
    assert records[0]["vehicle_id"] == "car_01"


def test_vehicle_telemetry_not_found(client):
    """GET /telemetry/unknown_car returns 404."""
    response = client.get("/telemetry/unknown_car")
    assert response.status_code == 404
