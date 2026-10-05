# tests/test_consumer.py
# Tests for telemetry validation and consumer processing logic.
# No MQTT broker required — all tests are pure unit tests.

import sys
import os
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge.validation import validate_telemetry
from edge.consumer import process_telemetry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def valid_payload(**overrides) -> dict:
    base = {
        "vehicle_id": "car_01",
        "speed": 80,
        "rpm": 3000,
        "engine_temp": 90,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

def test_valid_payload_passes():
    ok, reason = validate_telemetry(valid_payload())
    assert ok is True
    assert reason == ""


def test_missing_vehicle_id():
    d = valid_payload()
    del d["vehicle_id"]
    ok, reason = validate_telemetry(d)
    assert ok is False
    assert "vehicle_id" in reason


def test_missing_speed():
    d = valid_payload()
    del d["speed"]
    ok, reason = validate_telemetry(d)
    assert ok is False
    assert "speed" in reason


def test_missing_rpm():
    d = valid_payload()
    del d["rpm"]
    ok, reason = validate_telemetry(d)
    assert ok is False
    assert "rpm" in reason


def test_missing_engine_temp():
    d = valid_payload()
    del d["engine_temp"]
    ok, reason = validate_telemetry(d)
    assert ok is False
    assert "engine_temp" in reason


# ---------------------------------------------------------------------------
# vehicle_id format
# ---------------------------------------------------------------------------

def test_empty_vehicle_id():
    ok, reason = validate_telemetry(valid_payload(vehicle_id=""))
    assert ok is False


def test_vehicle_id_with_spaces():
    ok, reason = validate_telemetry(valid_payload(vehicle_id="car 01"))
    assert ok is False


def test_vehicle_id_with_valid_hyphen():
    ok, reason = validate_telemetry(valid_payload(vehicle_id="car-01"))
    assert ok is True


def test_vehicle_id_with_valid_underscore():
    ok, reason = validate_telemetry(valid_payload(vehicle_id="truck_99"))
    assert ok is True


# ---------------------------------------------------------------------------
# Numeric type validation
# ---------------------------------------------------------------------------

def test_speed_string():
    ok, reason = validate_telemetry(valid_payload(speed="fast"))
    assert ok is False
    assert "speed" in reason


def test_rpm_none():
    ok, reason = validate_telemetry(valid_payload(rpm=None))
    assert ok is False
    assert "rpm" in reason


def test_engine_temp_string():
    ok, reason = validate_telemetry(valid_payload(engine_temp="hot"))
    assert ok is False
    assert "engine_temp" in reason


# ---------------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------------

def test_negative_speed():
    ok, reason = validate_telemetry(valid_payload(speed=-1))
    assert ok is False
    assert "speed" in reason


def test_speed_at_zero():
    ok, _ = validate_telemetry(valid_payload(speed=0))
    assert ok is True


def test_speed_at_maximum():
    ok, _ = validate_telemetry(valid_payload(speed=300))
    assert ok is True


def test_speed_above_maximum():
    ok, reason = validate_telemetry(valid_payload(speed=301))
    assert ok is False
    assert "speed" in reason


def test_negative_rpm():
    ok, reason = validate_telemetry(valid_payload(rpm=-1))
    assert ok is False
    assert "rpm" in reason


def test_rpm_at_zero():
    ok, _ = validate_telemetry(valid_payload(rpm=0))
    assert ok is True


def test_rpm_above_maximum():
    ok, reason = validate_telemetry(valid_payload(rpm=9001))
    assert ok is False
    assert "rpm" in reason


def test_engine_temp_too_low():
    ok, reason = validate_telemetry(valid_payload(engine_temp=-11))
    assert ok is False
    assert "engine_temp" in reason


def test_engine_temp_too_high():
    ok, reason = validate_telemetry(valid_payload(engine_temp=131))
    assert ok is False
    assert "engine_temp" in reason


def test_engine_temp_boundary_low():
    ok, _ = validate_telemetry(valid_payload(engine_temp=-10))
    assert ok is True


def test_engine_temp_boundary_high():
    ok, _ = validate_telemetry(valid_payload(engine_temp=130))
    assert ok is True


# ---------------------------------------------------------------------------
# Optional event_timestamp
# ---------------------------------------------------------------------------

def test_valid_event_timestamp_accepted():
    ok, _ = validate_telemetry(
        valid_payload(event_timestamp="2026-09-01T12:00:00.000000+00:00")
    )
    assert ok is True


def test_invalid_event_timestamp_rejected():
    ok, reason = validate_telemetry(
        valid_payload(event_timestamp="not-a-timestamp")
    )
    assert ok is False
    assert "event_timestamp" in reason


def test_missing_event_timestamp_is_fine():
    """event_timestamp is optional — payload without it is valid."""
    ok, _ = validate_telemetry(valid_payload())
    assert ok is True


# ---------------------------------------------------------------------------
# Non-dict payload
# ---------------------------------------------------------------------------

def test_list_payload_rejected():
    ok, reason = validate_telemetry([1, 2, 3])
    assert ok is False


def test_none_payload_rejected():
    ok, reason = validate_telemetry(None)
    assert ok is False


# ---------------------------------------------------------------------------
# process_telemetry (consumer-level enrichment)
# ---------------------------------------------------------------------------

def test_process_adds_ingestion_timestamp():
    result = process_telemetry(valid_payload())
    assert result is not None
    assert "ingestion_timestamp" in result


def test_process_adds_event_timestamp_fallback():
    """If vehicle doesn't send event_timestamp, consumer falls back to ingestion time."""
    result = process_telemetry(valid_payload())
    assert result is not None
    assert "event_timestamp" in result


def test_process_preserves_vehicle_event_timestamp():
    ts = "2026-09-01T12:00:00+00:00"
    result = process_telemetry(valid_payload(event_timestamp=ts))
    assert result is not None
    assert result["event_timestamp"] == ts


def test_process_generates_message_id():
    result = process_telemetry(valid_payload())
    assert result is not None
    assert "message_id" in result
    assert len(result["message_id"]) > 0


def test_process_rejects_invalid():
    result = process_telemetry(valid_payload(speed=-1))
    assert result is None


def test_process_rejects_missing_field():
    d = valid_payload()
    del d["rpm"]
    result = process_telemetry(d)
    assert result is None
