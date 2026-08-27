# tests/test_consumer.py
# Telemetry validation tests.
# Imports from edge/validation.py — the extracted module — rather than
# from consumer.py directly. This is cleaner: tests should test the logic,
# not the MQTT wiring.

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge.validation import validate_telemetry
from edge.consumer import process_telemetry


# --- Validation tests ---

def test_valid_telemetry():
    """Valid payload passes all checks."""
    is_valid, reason = validate_telemetry({
        "vehicle_id": "car_01",
        "speed": 100,
        "rpm": 3000,
        "engine_temp": 90,
    })
    assert is_valid is True
    assert reason == ""


def test_missing_field():
    """Payload missing a required field is rejected."""
    is_valid, reason = validate_telemetry({
        "vehicle_id": "car_01",
        # speed missing
        "rpm": 3000,
        "engine_temp": 90,
    })
    assert is_valid is False
    assert "speed" in reason


def test_missing_vehicle_id():
    """vehicle_id is required."""
    is_valid, reason = validate_telemetry({
        "speed": 80,
        "rpm": 3000,
        "engine_temp": 90,
    })
    assert is_valid is False
    assert "vehicle_id" in reason


def test_negative_speed():
    """Speed cannot be negative."""
    is_valid, reason = validate_telemetry({
        "vehicle_id": "car_01",
        "speed": -10,
        "rpm": 3000,
        "engine_temp": 90,
    })
    assert is_valid is False
    assert "speed" in reason


def test_negative_rpm():
    """RPM cannot be negative."""
    is_valid, reason = validate_telemetry({
        "vehicle_id": "car_01",
        "speed": 60,
        "rpm": -500,
        "engine_temp": 90,
    })
    assert is_valid is False
    assert "rpm" in reason


def test_invalid_numeric_value():
    """Non-numeric speed is rejected."""
    is_valid, reason = validate_telemetry({
        "vehicle_id": "car_01",
        "speed": "fast",
        "rpm": 3000,
        "engine_temp": 90,
    })
    assert is_valid is False
    assert "numeric" in reason


def test_process_telemetry_adds_timestamp():
    """process_telemetry returns the dict with a timestamp added."""
    result = process_telemetry({
        "vehicle_id": "car_01",
        "speed": 80,
        "rpm": 3000,
        "engine_temp": 88,
    })
    assert result is not None
    assert "timestamp" in result


def test_process_telemetry_rejects_invalid():
    """process_telemetry returns None for invalid data."""
    result = process_telemetry({
        "vehicle_id": "car_01",
        "speed": -5,
        "rpm": 3000,
        "engine_temp": 88,
    })
    assert result is None
