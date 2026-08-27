# simulator/publisher.py
# Vehicle Simulator — generates randomized telemetry and publishes it via MQTT.
# Week 3: configuration now imported from config.py instead of inline constants.

import json
import sys
import os
import time
import random

import paho.mqtt.client as mqtt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_TOPIC

VEHICLE_ID = "car_01"


def generate_telemetry() -> dict:
    """Generates one randomized vehicle telemetry reading."""
    return {
        "vehicle_id": VEHICLE_ID,
        "speed": random.randint(40, 140),       # km/h
        "rpm": random.randint(2000, 6000),      # engine revolutions per minute
        "engine_temp": random.randint(80, 110), # degrees Celsius
    }


def main():
    print(f"🚗 Starting Vehicle Simulator ({VEHICLE_ID})...")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT)
        print(f"✅ Connected to broker at {MQTT_BROKER_HOST}:{MQTT_BROKER_PORT}")

        while True:
            data = generate_telemetry()
            payload = json.dumps(data)
            client.publish(MQTT_TOPIC, payload)
            print(f"📤 Published to {MQTT_TOPIC}: {payload}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")
    finally:
        client.disconnect()
        print("Disconnected from broker.")


if __name__ == "__main__":
    main()
