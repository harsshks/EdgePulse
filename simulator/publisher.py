# simulator/publisher.py
# Vehicle Simulator for EdgePulse.
#
# Generates randomized vehicle telemetry and publishes it via MQTT every second.
# Phase 0 additions vs Week 3:
#   - event_timestamp: UTC ISO-8601 timestamp from the "vehicle" (simulator clock).
#   - message_id: UUID per message for deduplication at the consumer.
#   - Proper logging instead of print statements.
#   - ConnectionRefusedError handled clearly.

import json
import logging
import time
import random
import uuid
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

from config import MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_TOPIC, LOG_LEVEL

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("edgepulse.simulator")

VEHICLE_ID = "car_01"


def generate_telemetry() -> dict:
    """
    Generates one randomized vehicle telemetry reading.
    Includes event_timestamp and message_id so the consumer can
    distinguish delayed telemetry and deduplicate retried messages.
    """
    return {
        "message_id":      str(uuid.uuid4()),
        "event_timestamp": datetime.now(timezone.utc).isoformat(),
        "vehicle_id":      VEHICLE_ID,
        "speed":           random.randint(0, 140),        # km/h
        "rpm":             random.randint(0, 6000),       # engine RPM
        "engine_temp":     random.randint(60, 110),       # °C
    }


def main():
    logger.info("Vehicle Simulator starting (vehicle_id=%s)...", VEHICLE_ID)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT)
        logger.info("Connected to broker at %s:%d", MQTT_BROKER_HOST, MQTT_BROKER_PORT)

        while True:
            data = generate_telemetry()
            payload = json.dumps(data)
            client.publish(MQTT_TOPIC, payload, qos=1)
            logger.info(
                "Published | vehicle=%s speed=%d rpm=%d temp=%d",
                data["vehicle_id"], data["speed"], data["rpm"], data["engine_temp"],
            )
            time.sleep(1)

    except ConnectionRefusedError:
        logger.critical(
            "Connection refused at %s:%d. Is Mosquitto running?",
            MQTT_BROKER_HOST, MQTT_BROKER_PORT,
        )
        raise SystemExit(1)
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user.")
    finally:
        client.disconnect()
        logger.info("Disconnected from broker.")


if __name__ == "__main__":
    main()
