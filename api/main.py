# api/main.py
# FastAPI application for EdgePulse.
#
# === WHAT IS FastAPI? ===
# FastAPI is a Python web framework for building HTTP APIs. It uses Python type
# hints to automatically validate request/response data and auto-generates
# interactive documentation at /docs (Swagger UI).
#
# === WHAT IS AN API? ===
# An API (Application Programming Interface) is a defined way for one program
# to talk to another. Here, external clients (your browser, Postman, another
# service) ask EdgePulse questions over HTTP and get JSON answers back.
#
# === WHAT IS REST? ===
# REST (Representational State Transfer) is a style of designing APIs using
# standard HTTP methods:
#   GET    — retrieve data   (read-only, safe to repeat)
#   POST   — create data
#   PUT    — replace data
#   DELETE — remove data
# We only need GET for Week 3.
#
# === HTTP REQUEST / RESPONSE ===
# A request is: method + URL + optional body
# A response is: status code + optional body (JSON here)
#
# === HTTP STATUS CODES ===
# 200 OK           — success
# 404 Not Found    — requested resource doesn't exist
# 422 Unprocessable — request data failed validation
# 500 Internal Error — something crashed on the server
#
# === ROUTES / ENDPOINTS ===
# A route maps a URL path to a Python function.
# @app.get("/health") means: when someone sends GET /health, call this function.

import sys
import os
from contextlib import asynccontextmanager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from database.db import init_db, get_latest_telemetry, get_telemetry, get_telemetry_by_vehicle

# === PYDANTIC RESPONSE MODEL ===
# Pydantic models define the shape of our API responses.
# FastAPI uses them to:
#   1. Serialize Python dicts to JSON automatically.
#   2. Validate that every response contains the expected fields and types.
#   3. Generate accurate documentation in /docs.
# If a database row is missing a field, FastAPI raises an error rather than
# silently returning broken JSON to the client.

class TelemetryRecord(BaseModel):
    id: int
    timestamp: str
    vehicle_id: str
    speed: float
    rpm: float
    engine_temp: float


# === APPLICATION INSTANCE ===
# title and description appear at the top of /docs.

@asynccontextmanager
async def lifespan(application: FastAPI):
    """
    Runs setup code before the first request (startup) and teardown after
    the last request (shutdown). Using lifespan is the modern FastAPI pattern.
    """
    init_db()  # ensure the SQLite table exists
    yield       # application runs here


app = FastAPI(
    title="EdgePulse Telemetry API",
    description=(
        "REST API for querying vehicle telemetry collected by the EdgePulse "
        "edge consumer. Data flows: Vehicle → MQTT → Consumer → SQLite → this API."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --- Routes ---

@app.get("/", summary="Root — service identity")
def root():
    """
    Returns a simple JSON object identifying the service.
    Useful for quickly confirming the API is reachable.
    """
    return {"service": "EdgePulse Telemetry API", "status": "running", "version": "1.0.0"}


@app.get("/health", summary="Health check")
def health():
    """
    Health check endpoint.
    In production, this is what a load balancer pings to know the service is alive.
    We verify the database is reachable by attempting a lightweight query.
    Returns 200 if healthy, 503 if the database is unavailable.
    """
    try:
        # A cheap query — just checks DB connectivity, doesn't load all data
        get_telemetry(limit=1)
        return {"status": "healthy", "database": "reachable"}
    except Exception as e:
        # 503 Service Unavailable — server is up but a dependency is down
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


@app.get(
    "/telemetry/latest",
    response_model=TelemetryRecord,
    summary="Get the most recent telemetry record",
)
def latest_telemetry():
    """
    Returns the single most recently received telemetry reading.
    404 if the database is empty (consumer hasn't run yet).
    """
    record = get_latest_telemetry()
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
    limit: int = Query(default=20, ge=1, le=100, description="Number of records to return (1–100)")
):
    """
    Returns up to `limit` recent telemetry records, newest first.

    The `limit` query parameter is validated by FastAPI:
    - Must be an integer (422 if not).
    - Must be between 1 and 100 (422 if out of range).
    Example: GET /telemetry?limit=5
    """
    records = get_telemetry(limit=limit)
    return records


@app.get(
    "/telemetry/{vehicle_id}",
    response_model=list[TelemetryRecord],
    summary="Get telemetry for a specific vehicle",
)
def vehicle_telemetry(
    vehicle_id: str,
    limit: int = Query(default=20, ge=1, le=100, description="Number of records to return (1–100)")
):
    """
    Returns recent telemetry for the given vehicle_id.
    404 if no records exist for that vehicle.
    Example: GET /telemetry/car_01
    """
    records = get_telemetry_by_vehicle(vehicle_id=vehicle_id, limit=limit)
    if not records:
        raise HTTPException(
            status_code=404,
            detail=f"No telemetry found for vehicle '{vehicle_id}'.",
        )
    return records
