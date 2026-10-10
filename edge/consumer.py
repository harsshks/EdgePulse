# edge/consumer.py
# MQTT Edge Consumer for EdgePulse.
#
# Responsibilities:
#   1. Connect to the MQTT broker and subscribe to the telemetry topic.
#   2. Parse, validate, and timestamp every incoming message.
#   3. Persist valid telemetry to SQLite (event + ingestion timestamps).
#   4. Append to the JSONL audit log as a secondary record.
#   5. Survive bad messages, transient broker outages, and DB errors.
#
# One bad message must never kill this process.
#
# MQTT QoS choice:
#   QoS 0 ("at most once") — fire and forget, no redelivery.
#   QoS 1 ("at least once") — broker retries until acknowledged; duplicates possible.
#
#   We use QoS 1 for the subscription so we don't silently lose telemetry
#   if the consumer is momentarily busy. Duplicates are handled by the
#   message_id UNIQUE constraint in the database (INSERT OR IGNORE).
#
# Reconnect behaviour:
#   paho-mqtt's loop_forever() automatically reconnects when the broker
#   drops the connection (with exponential back-off up to reconnect_delay_max).
#   on_connect re-subscribes every time so the topic is never lost after reconnect.

import json
import logging
import os
import uuid
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from config import (
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_TOPIC,
    LOG_FILE,
    LOG_LEVEL,
)
from edge.validation import validate_telemetry
from database.db import init_db, insert_telemetry
from edge.pipeline import FeaturePipeline

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("edgepulse.consumer")

# Module-level pipeline instance — shared across all MQTT callbacks.
# Initialised in main() after the database is ready.
_pipeline: FeaturePipeline | None = None


# ---------------------------------------------------------------------------
# Telemetry processing
# ---------------------------------------------------------------------------

def process_telemetry(data: dict) -> dict | None:
    """
    Validates and enriches a raw telemetry dictionary.

    Enrichment adds:
      ingestion_timestamp — UTC time EdgePulse received the message.
      event_timestamp     — taken from the vehicle payload if present;
                            falls back to ingestion_timestamp so the field
                            is always populated (Phase 0 simulator doesn't
                            send event_timestamp yet).
      message_id          — taken from payload if present; otherwise a new
                            UUID is generated so every record has a unique ID.

    Returns the enriched dict on success, None if validation fails.
    Does NOT mutate the original dict.
    """
    is_valid, reason = validate_telemetry(data)
    if not is_valid:
        logger.warning("Validation failed: %s | payload: %s", reason, data)
        return None

    ingestion_ts = datetime.now(timezone.utc).isoformat()

    def _optional_float(key: str) -> float | None:
        val = data.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    enriched = {
        "message_id":          data.get("message_id") or str(uuid.uuid4()),
        "event_timestamp":     data.get("event_timestamp") or ingestion_ts,
        "ingestion_timestamp": ingestion_ts,
        "vehicle_id":          data["vehicle_id"],
        "speed":               float(data["speed"]),
        "rpm":                 float(data["rpm"]),
        "engine_temp":         float(data["engine_temp"]),
        "battery_voltage":     _optional_float("battery_voltage"),
        "fuel_level":          _optional_float("fuel_level"),
    }
    return enriched


def log_to_jsonl(record: dict) -> None:
    """Appends one processed record to the JSONL audit log."""
    try:
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError as e:
        logger.error("JSONL log write failed: %s", e)


# ---------------------------------------------------------------------------
# MQTT callbacks
# ---------------------------------------------------------------------------

def on_connect(client, userdata, flags, reason_code, properties):
    """
    Fires on every (re)connection to the broker.
    Re-subscribing here ensures the topic subscription survives reconnects.
    """
    if reason_code == 0:
        logger.info(
            "Connected to MQTT broker at %s:%d", MQTT_BROKER_HOST, MQTT_BROKER_PORT
        )
        result, _ = client.subscribe(MQTT_TOPIC, qos=1)
        if result == mqtt.MQTT_ERR_SUCCESS:
            logger.info("Subscribed to topic: %s (QoS 1)", MQTT_TOPIC)
        else:
            logger.error("Failed to subscribe to topic: %s (code %d)", MQTT_TOPIC, result)
    else:
        logger.error("MQTT connection refused — reason code: %d", reason_code)


def on_disconnect(client, userdata, flags, reason_code, properties):
    """
    Fires when the broker connection drops.
    loop_forever() will attempt automatic reconnection.
    """
    if reason_code != 0:
        logger.warning(
            "Unexpected MQTT disconnect (code %d) — will attempt reconnect.", reason_code
        )
    else:
        logger.info("MQTT disconnected cleanly.")


def on_message(client, userdata, message):
    """
    Fires for every received MQTT message.

    Pipeline:
      raw bytes → decode → JSON parse → validate → enrich → SQLite → JSONL

    Any failure at any step logs a warning and returns early.
    The consumer loop continues regardless.
    """
    # 1. Decode bytes
    try:
        raw_payload = message.payload.decode("utf-8")
    except UnicodeDecodeError as e:
        logger.warning("Could not decode message payload: %s", e)
        return

    # 2. Parse JSON
    try:
        data = json.loads(raw_payload)
    except json.JSONDecodeError as e:
        logger.warning("Malformed JSON received: %s | raw: %.200s", e, raw_payload)
        return

    if not isinstance(data, dict):
        logger.warning("Payload is not a JSON object: %r", data)
        return

    # 3. Validate + enrich
    record = process_telemetry(data)
    if record is None:
        return  # warning already logged inside process_telemetry

    # 4. Persist to SQLite — raw telemetry MUST be stored before feature pipeline runs
    try:
        inserted = insert_telemetry(record)
        if inserted:
            logger.info(
                "Stored telemetry | vehicle=%s speed=%.1f rpm=%.0f temp=%.1f",
                record["vehicle_id"], record["speed"], record["rpm"], record["engine_temp"],
            )
    except Exception as e:
        logger.error(
            "Database write failed for vehicle '%s': %s — continuing.", record["vehicle_id"], e
        )
        # Fall through to JSONL so data isn't entirely lost

    # 5. Append to JSONL audit log (always attempt, even if DB failed)
    log_to_jsonl(record)

    # 6. Feature pipeline — runs AFTER raw telemetry is safely stored.
    #    A pipeline failure never prevents telemetry from being persisted.
    if _pipeline is not None and inserted:
        _pipeline.process_record(record)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global _pipeline
    logger.info("EdgePulse Consumer starting...")

    # Initialise database before accepting any messages
    try:
        init_db()
        logger.info("Database ready: %s", os.path.abspath("edgepulse.db"))
    except Exception as e:
        logger.critical("Cannot initialise database — aborting: %s", e)
        raise SystemExit(1)

    # Initialise feature pipeline
    _pipeline = FeaturePipeline()
    _pipeline.load_baselines_from_db()
    logger.info("Feature pipeline ready.")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # Configure automatic reconnect: wait 1s initially, up to 30s between attempts
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT, keepalive=60)
        logger.info("Connecting to broker at %s:%d...", MQTT_BROKER_HOST, MQTT_BROKER_PORT)
        client.loop_forever()
    except ConnectionRefusedError:
        logger.critical(
            "Connection refused by broker at %s:%d. Is Mosquitto running?",
            MQTT_BROKER_HOST, MQTT_BROKER_PORT,
        )
        raise SystemExit(1)
    except KeyboardInterrupt:
        logger.info("Consumer stopped by user.")
    finally:
        client.disconnect()
        logger.info("Disconnected from broker.")


if __name__ == "__main__":
    main()
