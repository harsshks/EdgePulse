# edge/consumer.py
# MQTT Edge Consumer — subscribes to the broker, validates incoming telemetry,
# timestamps it, persists it to SQLite, and appends it to the JSONL audit log.
#
# Week 3 changes vs Week 2:
# - Validation logic moved to edge/validation.py (imported here).
# - SQLite persistence added via database/db.py.
# - Configuration imported from config.py instead of hard-coded constants.
# - JSONL logging retained as an audit trail alongside SQLite.

import json
import sys
import os
from datetime import datetime

import paho.mqtt.client as mqtt

# Allow running from the project root or from the edge/ subdirectory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_TOPIC, LOG_FILE
from edge.validation import validate_telemetry
from database.db import init_db, insert_telemetry


def process_telemetry(data: dict) -> dict | None:
    """
    Validates and timestamps a raw telemetry dictionary.
    Returns the enriched dict on success, or None if validation fails.
    This function is kept here (not just in validation.py) so the existing
    tests that import process_telemetry from consumer still work.
    """
    is_valid, reason = validate_telemetry(data)
    if not is_valid:
        print(f"⚠️  Validation Error: {reason} — payload: {data}")
        return None

    data["timestamp"] = datetime.now().isoformat()
    return data


def log_to_jsonl(data: dict) -> None:
    """Appends one processed record to the JSONL audit log."""
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")


# --- MQTT Callbacks ---

def on_connect(client, userdata, flags, reason_code, properties):
    """Fires when the client successfully connects to the broker."""
    if reason_code == 0:
        print(f"✅ Connected to broker at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")
        client.subscribe(MQTT_TOPIC)
        print(f"📡 Subscribed to topic: {MQTT_TOPIC}")
    else:
        print(f"❌ Failed to connect, reason code: {reason_code}")


def on_message(client, userdata, message):
    """
    Fires for every MQTT message received.
    Pipeline: raw bytes → JSON → validate → timestamp → SQLite → JSONL
    A bad message at any step prints a warning but does NOT crash the consumer.
    """
    raw_payload = message.payload.decode("utf-8")

    try:
        data = json.loads(raw_payload)
    except json.JSONDecodeError:
        print(f"⚠️  Malformed JSON received: {raw_payload}")
        return

    processed = process_telemetry(data)
    if processed is None:
        return  # validation failed — already printed a warning

    # Persist to SQLite
    try:
        insert_telemetry(processed)
    except Exception as e:
        print(f"⚠️  Database write failed: {e}")
        # Continue — still attempt JSONL log so data isn't lost entirely

    # Append to JSONL audit log
    try:
        log_to_jsonl(processed)
    except Exception as e:
        print(f"⚠️  JSONL log write failed: {e}")

    print(f"📥 Stored: {processed}")


def main():
    print("🖥️  Starting Edge Consumer...")

    # Ensure the telemetry table exists before we start receiving messages
    init_db()
    print("🗄️  Database ready.")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT)
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n🛑 Consumer stopped by user.")
    finally:
        client.disconnect()
        print("Disconnected from broker.")


if __name__ == "__main__":
    main()
