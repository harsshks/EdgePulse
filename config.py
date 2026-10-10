# config.py
# Central configuration for EdgePulse.
#
# Values are loaded from environment variables first.
# If a variable is not set, the default shown below is used.
#
# For local development:
#   1. Copy .env.example to .env
#   2. Adjust values as needed
#   3. Never commit .env to Git
#
# python-dotenv loads the .env file automatically if it exists.
# In production (Docker, cloud), set environment variables directly
# instead of using a .env file.

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Only load .env from the project root directory.
# We pass an explicit path so load_dotenv() never walks up to parent directories
# (which could pick up a user-level ~/.env with a different encoding or values).
_project_root = Path(__file__).parent
_env_file = _project_root / ".env"
if _env_file.exists():
    load_dotenv(dotenv_path=_env_file, encoding="utf-8-sig")

# --- MQTT ---
MQTT_BROKER_HOST: str = os.getenv("MQTT_BROKER_HOST", "localhost")
MQTT_BROKER_PORT: int = int(os.getenv("MQTT_BROKER_PORT", "1883"))
MQTT_TOPIC: str = os.getenv("MQTT_TOPIC", "telemetry/vehicle")

# --- Storage ---
DATABASE_PATH: str = os.getenv("DATABASE_PATH", "edgepulse.db")
LOG_FILE: str = os.getenv("LOG_FILE", "logs/telemetry.jsonl")

# --- API ---
API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# --- Logging ---
_LOG_LEVEL_STR: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL: int = getattr(logging, _LOG_LEVEL_STR, logging.INFO)

# --- Phase 2: Feature Engineering ---
# Rolling window length in number of observations (1 obs ≈ 1 second at default rate)
FEATURE_WINDOW_SIZE: int = int(os.getenv("FEATURE_WINDOW_SIZE", "60"))

# Minimum observations before rolling statistics are emitted
FEATURE_MIN_SAMPLES: int = int(os.getenv("FEATURE_MIN_SAMPLES", "3"))

# Minimum baseline samples before a baseline is considered ready
BASELINE_MIN_SAMPLES: int = int(os.getenv("BASELINE_MIN_SAMPLES", "30"))

# Engine temperature warning threshold (°C) — duration above this is tracked
TEMP_WARNING_THRESHOLD: float = float(os.getenv("TEMP_WARNING_THRESHOLD", "100.0"))

# Z-score thresholds for deviation levels
SCORE_WARNING_THRESHOLD: float = float(os.getenv("SCORE_WARNING_THRESHOLD", "2.0"))
SCORE_SEVERE_THRESHOLD: float  = float(os.getenv("SCORE_SEVERE_THRESHOLD", "3.5"))
