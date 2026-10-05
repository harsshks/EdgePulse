# tests/test_database.py
# Unit tests for the SQLite persistence layer.
# Every test uses a temporary database file — the real edgepulse.db is never touched.

import sys
import os
import sqlite3
import pytest

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

# ---------------------------------------------------------------------------
# Sample records
# ---------------------------------------------------------------------------

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
# Insert and retrieve
# ---------------------------------------------------------------------------

def test_insert_and_retrieve():
    db_module.insert_telemetry(make_record())
    records = db_module.get_telemetry(limit=10)
    assert len(records) == 1
    assert records[0]["vehicle_id"] == "car_01"
    assert records[0]["speed"] == 100.0


def test_inserted_record_has_both_timestamps():
    db_module.insert_telemetry(make_record())
    records = db_module.get_telemetry(limit=1)
    assert "event_timestamp" in records[0]
    assert "ingestion_timestamp" in records[0]


def test_get_latest_returns_most_recent():
    db_module.insert_telemetry(make_record(message_id="m1", speed=60.0))
    db_module.insert_telemetry(make_record(message_id="m2", speed=120.0))
    latest = db_module.get_latest_telemetry()
    assert latest["speed"] == 120.0


def test_get_latest_returns_none_when_empty():
    assert db_module.get_latest_telemetry() is None


def test_limit_is_respected():
    for i in range(5):
        db_module.insert_telemetry(make_record(message_id=f"m{i}", speed=float(i * 10)))
    records = db_module.get_telemetry(limit=3)
    assert len(records) == 3


def test_results_are_newest_first():
    db_module.insert_telemetry(make_record(message_id="m1", speed=50.0))
    db_module.insert_telemetry(make_record(message_id="m2", speed=100.0))
    records = db_module.get_telemetry(limit=10)
    assert records[0]["speed"] == 100.0
    assert records[1]["speed"] == 50.0


# ---------------------------------------------------------------------------
# Vehicle filtering
# ---------------------------------------------------------------------------

def test_get_by_vehicle_filters_correctly():
    db_module.insert_telemetry(make_record(message_id="m1", vehicle_id="car_01"))
    db_module.insert_telemetry(make_record(message_id="m2", vehicle_id="car_02"))
    records = db_module.get_telemetry_by_vehicle("car_01")
    assert all(r["vehicle_id"] == "car_01" for r in records)
    assert len(records) == 1


def test_get_by_vehicle_returns_empty_for_unknown():
    records = db_module.get_telemetry_by_vehicle("unknown_vehicle")
    assert records == []


# ---------------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------------

def test_duplicate_message_id_is_ignored():
    """Same message_id inserted twice — second insert is silently dropped."""
    db_module.insert_telemetry(make_record(message_id="dup-001", speed=80.0))
    db_module.insert_telemetry(make_record(message_id="dup-001", speed=999.0))
    records = db_module.get_telemetry(limit=10)
    assert len(records) == 1
    assert records[0]["speed"] == 80.0  # first value preserved


def test_insert_returns_true_for_new_record():
    result = db_module.insert_telemetry(make_record())
    assert result is True


def test_insert_returns_false_for_duplicate():
    db_module.insert_telemetry(make_record(message_id="dup"))
    result = db_module.insert_telemetry(make_record(message_id="dup"))
    assert result is False


def test_none_message_id_allows_multiple_inserts():
    """NULL message_id records are not considered duplicates of each other."""
    db_module.insert_telemetry(make_record(message_id=None))
    db_module.insert_telemetry(make_record(message_id=None))
    records = db_module.get_telemetry(limit=10)
    assert len(records) == 2
