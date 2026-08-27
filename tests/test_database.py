# tests/test_database.py
# Tests for the SQLite persistence layer.
# Each test gets a temporary database file (via pytest's tmp_path fixture)
# so tests are fully isolated and never touch the real edgepulse.db.

import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """
    Point DATABASE_PATH at a fresh temp file for every test.
    monkeypatch reverts the change automatically after each test.
    We also call init_db() to create the schema in that fresh file.
    """
    db_file = str(tmp_path / "test_edgepulse.db")
    import config
    import database.db as db_module
    monkeypatch.setattr(config, "DATABASE_PATH", db_file)
    monkeypatch.setattr(db_module, "DATABASE_PATH", db_file)
    db_module.init_db()
    return db_file


import database.db as db_module

SAMPLE = {
    "timestamp": "2026-08-27T10:00:00",
    "vehicle_id": "car_01",
    "speed": 100.0,
    "rpm": 3000.0,
    "engine_temp": 90.0,
}


def test_insert_and_retrieve():
    """Inserted record is retrievable via get_telemetry."""
    db_module.insert_telemetry(SAMPLE)
    records = db_module.get_telemetry(limit=10)
    assert len(records) == 1
    assert records[0]["vehicle_id"] == "car_01"
    assert records[0]["speed"] == 100.0


def test_get_latest_telemetry():
    """get_latest_telemetry returns the most recently inserted row."""
    db_module.insert_telemetry({**SAMPLE, "speed": 60.0})
    db_module.insert_telemetry({**SAMPLE, "speed": 120.0})
    latest = db_module.get_latest_telemetry()
    assert latest is not None
    assert latest["speed"] == 120.0


def test_get_latest_returns_none_when_empty():
    """get_latest_telemetry returns None on an empty table."""
    result = db_module.get_latest_telemetry()
    assert result is None


def test_limit_is_respected():
    """get_telemetry honours the limit parameter."""
    for i in range(5):
        db_module.insert_telemetry({**SAMPLE, "speed": float(i * 10)})
    records = db_module.get_telemetry(limit=3)
    assert len(records) == 3


def test_get_by_vehicle():
    """get_telemetry_by_vehicle filters correctly."""
    db_module.insert_telemetry({**SAMPLE, "vehicle_id": "car_01"})
    db_module.insert_telemetry({**SAMPLE, "vehicle_id": "car_02"})
    records = db_module.get_telemetry_by_vehicle("car_01")
    assert all(r["vehicle_id"] == "car_01" for r in records)
    assert len(records) == 1
