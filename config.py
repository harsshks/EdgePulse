# config.py
# Central configuration for EdgePulse.
# Change values here instead of hunting through multiple files.
# In a future week, these will be loaded from environment variables (.env).

# --- MQTT ---
MQTT_BROKER_HOST = "localhost"
MQTT_BROKER_PORT = 1883
MQTT_TOPIC = "telemetry/car1"

# --- Storage ---
LOG_FILE = "logs/telemetry.jsonl"
DATABASE_PATH = "edgepulse.db"
