# edge/validation.py
# Telemetry validation for EdgePulse.
#
# Design principles:
# - Validation is pure logic: no I/O, no MQTT, no database.
# - Returns (bool, str) so callers can decide how to handle failures.
# - All rules are in one place — consumer, API, and tests all import from here.
# - Ranges are justified by real vehicle physics, not arbitrary limits.
#
# Schema (Phase 0):
#   vehicle_id        str   — which vehicle sent this (non-empty, alphanumeric + underscore/hyphen)
#   speed             float — km/h, 0–300 (300 km/h is an extreme but valid track speed)
#   rpm               float — 0–9000 (most road vehicles redline below 8000)
#   engine_temp       float — °C, -10–130 (cold start to severe overheating)
#   event_timestamp   str   — ISO-8601 UTC timestamp from the vehicle (optional in Phase 0)
#   message_id        str   — unique message identifier for deduplication (optional in Phase 0)

import re
from datetime import datetime, timezone

# Fields that MUST be present in every telemetry payload
REQUIRED_FIELDS = ["vehicle_id", "speed", "rpm", "engine_temp"]

# vehicle_id: alphanumeric, underscores, hyphens; 1–64 characters
_VEHICLE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

# Numeric sensor ranges — chosen to represent physically plausible values
_SPEED_MIN = 0.0       # km/h — vehicle cannot travel backwards in this schema
_SPEED_MAX = 300.0     # km/h — extreme but valid (e.g. Bugatti, track vehicles)
_RPM_MIN = 0.0         # engine can be at rest (idle off)
_RPM_MAX = 9000.0      # above 9000 indicates sensor fault or non-road vehicle
_TEMP_MIN = -10.0      # °C — cold start in winter conditions
_TEMP_MAX = 130.0      # °C — above this is severe overheating / sensor fault


def validate_telemetry(data: dict) -> tuple[bool, str]:
    """
    Validates a raw telemetry dictionary.

    Returns:
        (True, "")           on success
        (False, reason_str)  on any validation failure

    Does NOT mutate the input dict.
    """
    if not isinstance(data, dict):
        return False, f"Payload must be a JSON object, got {type(data).__name__}"

    # 1. Required fields
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing required field: '{field}'"

    # 2. vehicle_id format
    vehicle_id = data["vehicle_id"]
    if not isinstance(vehicle_id, str) or not vehicle_id.strip():
        return False, "Field 'vehicle_id' must be a non-empty string"
    if not _VEHICLE_ID_PATTERN.match(vehicle_id):
        return False, (
            f"Field 'vehicle_id' contains invalid characters: '{vehicle_id}'. "
            "Use alphanumeric, underscore, or hyphen only (max 64 chars)."
        )

    # 3. Numeric type coercion and range checks
    try:
        speed = float(data["speed"])
    except (TypeError, ValueError):
        return False, f"Field 'speed' must be numeric, got: {data['speed']!r}"

    try:
        rpm = float(data["rpm"])
    except (TypeError, ValueError):
        return False, f"Field 'rpm' must be numeric, got: {data['rpm']!r}"

    try:
        engine_temp = float(data["engine_temp"])
    except (TypeError, ValueError):
        return False, f"Field 'engine_temp' must be numeric, got: {data['engine_temp']!r}"

    if not (_SPEED_MIN <= speed <= _SPEED_MAX):
        return False, f"Field 'speed' out of range [{_SPEED_MIN}, {_SPEED_MAX}]: got {speed}"

    if not (_RPM_MIN <= rpm <= _RPM_MAX):
        return False, f"Field 'rpm' out of range [{_RPM_MIN}, {_RPM_MAX}]: got {rpm}"

    if not (_TEMP_MIN <= engine_temp <= _TEMP_MAX):
        return False, (
            f"Field 'engine_temp' out of range [{_TEMP_MIN}, {_TEMP_MAX}]: got {engine_temp}"
        )

    # 4. Optional: event_timestamp format validation (if provided by the vehicle)
    if "event_timestamp" in data:
        ts = data["event_timestamp"]
        if not _is_valid_iso8601(ts):
            return False, f"Field 'event_timestamp' is not a valid ISO-8601 timestamp: {ts!r}"

    return True, ""


def _is_valid_iso8601(value: str) -> bool:
    """
    Returns True if `value` is a parseable ISO-8601 datetime string.
    Accepts both naive (no timezone) and timezone-aware strings.
    Used to validate optional event_timestamp from the vehicle.
    """
    if not isinstance(value, str):
        return False
    # Try the two most common formats
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            datetime.strptime(value[:len(fmt) + 6], fmt)
            return True
        except ValueError:
            continue
    # Fallback: try fromisoformat (Python 3.7+)
    try:
        datetime.fromisoformat(value)
        return True
    except ValueError:
        return False
