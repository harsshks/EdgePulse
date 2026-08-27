# edge/validation.py
# Telemetry validation logic, extracted from consumer.py so it can be
# imported independently by the consumer, the API, and the tests.

REQUIRED_FIELDS = ["vehicle_id", "speed", "rpm", "engine_temp"]


def validate_telemetry(data: dict) -> tuple[bool, str]:
    """
    Validates a telemetry dictionary.

    Returns a (is_valid, reason) tuple.
    - is_valid: True if the data passes all checks.
    - reason:   Empty string on success; a human-readable error message on failure.

    Keeping validation separate from the consumer means:
    - Tests can import ONLY this function without spinning up MQTT.
    - The FastAPI layer can re-use the same rules if it ever accepts POST payloads.
    - Rules are defined in exactly one place (no drift between files).
    """

    # 1. Check that every required field is present
    for field in REQUIRED_FIELDS:
        if field not in data:
            return False, f"Missing required field: '{field}'"

    # 2. Numeric range checks — catch obviously bad sensor readings
    try:
        speed = float(data["speed"])
        rpm = float(data["rpm"])
        engine_temp = float(data["engine_temp"])
    except (TypeError, ValueError):
        return False, "Fields 'speed', 'rpm', and 'engine_temp' must be numeric"

    if speed < 0:
        return False, f"Invalid value: speed cannot be negative (got {speed})"

    if rpm < 0:
        return False, f"Invalid value: rpm cannot be negative (got {rpm})"

    if engine_temp < -50 or engine_temp > 300:
        return False, f"Invalid value: engine_temp out of range (got {engine_temp})"

    return True, ""
