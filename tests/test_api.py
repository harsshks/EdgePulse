# tests/test_api.py
# FastAPI endpoint tests.
# Uses Starlette's TestClient — no real HTTP server or broker needed.
# Each test gets a fresh temporary database.

import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_edgepulse.db")
    import config
    import database.db as db_module
    monkeypatch.setattr(config, "DATABASE_PATH", db_file)
    monkeypatch.setattr(db_module, "DATABASE_PATH", db_file)
    db_module.init_db()
    return db_file


import database.db as db_module
from api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def make_record(**overrides) -> dict:
    base = {
        "message_id":          "msg-001",
        "event_timestamp":     "2026-09-01T10:00:00+00:00",
        "ingestion_timestamp": "2026-09-01T10:00:01+00:00",
        "vehicle_id":          "car_01",
        "speed":               100.0,
        "rpm":                 3000.0,
        "engine_temp":         90.0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

def test_root_returns_200(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "EdgePulse Telemetry API"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_returns_healthy(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# GET /telemetry/latest
# ---------------------------------------------------------------------------

def test_latest_returns_404_when_empty(client):
    r = client.get("/telemetry/latest")
    assert r.status_code == 404


def test_latest_returns_most_recent_record(client):
    db_module.insert_telemetry(make_record(message_id="m1", speed=60.0))
    db_module.insert_telemetry(make_record(message_id="m2", speed=120.0))
    r = client.get("/telemetry/latest")
    assert r.status_code == 200
    assert r.json()["speed"] == 120.0


def test_latest_response_has_both_timestamps(client):
    db_module.insert_telemetry(make_record())
    r = client.get("/telemetry/latest")
    body = r.json()
    assert "event_timestamp" in body
    assert "ingestion_timestamp" in body


# ---------------------------------------------------------------------------
# GET /telemetry
# ---------------------------------------------------------------------------

def test_list_returns_empty_list_when_no_data(client):
    r = client.get("/telemetry")
    assert r.status_code == 200
    assert r.json() == []


def test_list_returns_all_records(client):
    db_module.insert_telemetry(make_record(message_id="m1"))
    db_module.insert_telemetry(make_record(message_id="m2"))
    r = client.get("/telemetry")
    assert len(r.json()) == 2


def test_list_limit_respected(client):
    for i in range(5):
        db_module.insert_telemetry(make_record(message_id=f"m{i}"))
    r = client.get("/telemetry?limit=2")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_list_limit_zero_returns_422(client):
    r = client.get("/telemetry?limit=0")
    assert r.status_code == 422


def test_list_limit_above_100_returns_422(client):
    r = client.get("/telemetry?limit=101")
    assert r.status_code == 422


def test_list_limit_non_integer_returns_422(client):
    r = client.get("/telemetry?limit=abc")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /telemetry/{vehicle_id}
# ---------------------------------------------------------------------------

def test_vehicle_returns_only_matching_records(client):
    db_module.insert_telemetry(make_record(message_id="m1", vehicle_id="car_01"))
    db_module.insert_telemetry(make_record(message_id="m2", vehicle_id="car_02"))
    r = client.get("/telemetry/car_01")
    assert r.status_code == 200
    records = r.json()
    assert len(records) == 1
    assert records[0]["vehicle_id"] == "car_01"


def test_vehicle_returns_404_for_unknown_id(client):
    r = client.get("/telemetry/nonexistent_vehicle")
    assert r.status_code == 404


def test_vehicle_limit_respected(client):
    for i in range(5):
        db_module.insert_telemetry(make_record(message_id=f"m{i}", vehicle_id="car_01"))
    r = client.get("/telemetry/car_01?limit=2")
    assert r.status_code == 200
    assert len(r.json()) == 2
