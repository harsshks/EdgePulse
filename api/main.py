# api/main.py
# FastAPI REST API for EdgePulse telemetry.
#
# Responsibilities:
#   - Expose stored telemetry over HTTP.
#   - Validate query parameters (FastAPI + Pydantic handles this automatically).
#   - Return structured JSON with correct HTTP status codes.
#   - Report database availability via /health.
#
# The API is read-only in Phase 0. Telemetry enters the system via MQTT,
# not via POST requests. The two paths are intentionally independent.

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from config import LOG_LEVEL
from database.db import (
    init_db,
    get_latest_telemetry,
    get_telemetry,
    get_telemetry_by_vehicle,
)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("edgepulse.api")


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------

class TelemetryRecord(BaseModel):
    """
    Defines the shape of every telemetry response.
    FastAPI uses this to:
      1. Serialise dicts to JSON.
      2. Validate that every response has the expected fields and types.
      3. Generate accurate /docs schema.

    Fields:
      id                  — internal database row ID
      message_id          — deduplication identifier (may be None for legacy records)
      event_timestamp     — UTC time the vehicle generated the reading
      ingestion_timestamp — UTC time EdgePulse stored the reading
      vehicle_id          — which vehicle sent this
      speed / rpm / engine_temp — sensor values
    """
    id: int
    message_id: str | None
    event_timestamp: str
    ingestion_timestamp: str
    vehicle_id: str
    speed: float
    rpm: float
    engine_temp: float


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialise the database before accepting requests."""
    logger.info("EdgePulse API starting — initialising database...")
    init_db()
    logger.info("Database ready.")
    yield
    logger.info("EdgePulse API shutting down.")


app = FastAPI(
    title="EdgePulse Telemetry API",
    description=(
        "REST API for querying vehicle telemetry collected by the EdgePulse "
        "edge consumer.\n\n"
        "Data flow: Vehicle → MQTT → Consumer → SQLite → this API."
    ),
    version="0.3.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Service identity")
def root():
    """Returns service name and version. Useful for a quick reachability check."""
    return {"service": "EdgePulse Telemetry API", "status": "running", "version": "0.3.0"}


@app.get("/health", summary="Health check")
def health():
    """
    Verifies the API is running and the database is reachable.
    Returns 200 if healthy, 503 if the database is unavailable.
    Suitable for use by a load balancer or monitoring system.
    """
    try:
        get_telemetry(limit=1)
        return {"status": "healthy", "database": "reachable"}
    except Exception as e:
        logger.error("Health check failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


@app.get(
    "/telemetry/latest",
    response_model=TelemetryRecord,
    summary="Most recent telemetry record",
)
def latest_telemetry():
    """
    Returns the single most recently stored telemetry reading.
    404 if the database is empty (consumer hasn't run yet).
    """
    try:
        record = get_latest_telemetry()
    except Exception as e:
        logger.error("Failed to fetch latest telemetry: %s", e)
        raise HTTPException(status_code=503, detail="Database error")

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="No telemetry records found. Is the consumer running?",
        )
    return record


@app.get(
    "/telemetry",
    response_model=list[TelemetryRecord],
    summary="List recent telemetry records",
)
def list_telemetry(
    limit: int = Query(
        default=20, ge=1, le=100,
        description="Number of records to return (1–100). Default: 20."
    )
):
    """
    Returns up to `limit` recent telemetry records, newest first.
    FastAPI returns 422 automatically if `limit` is not an integer or is out of range.
    """
    try:
        return get_telemetry(limit=limit)
    except Exception as e:
        logger.error("Failed to fetch telemetry list: %s", e)
        raise HTTPException(status_code=503, detail="Database error")


@app.get(
    "/telemetry/{vehicle_id}",
    response_model=list[TelemetryRecord],
    summary="Telemetry for a specific vehicle",
)
def vehicle_telemetry(
    vehicle_id: str,
    limit: int = Query(
        default=20, ge=1, le=100,
        description="Number of records to return (1–100). Default: 20."
    )
):
    """
    Returns recent telemetry for the given vehicle_id, newest first.
    404 if no records exist for that vehicle.
    """
    try:
        records = get_telemetry_by_vehicle(vehicle_id=vehicle_id, limit=limit)
    except Exception as e:
        logger.error("Failed to fetch telemetry for vehicle '%s': %s", vehicle_id, e)
        raise HTTPException(status_code=503, detail="Database error")

    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No telemetry found for vehicle '{vehicle_id}'.",
        )
    return records
