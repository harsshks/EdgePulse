# api/main.py
# FastAPI REST API for EdgePulse.
#
# Phase 0 endpoints: telemetry retrieval
# Phase 2 endpoints: feature vectors, baseline status, deviation scores
#
# All endpoints are read-only. Data enters via MQTT → consumer → SQLite.

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from config import (
    LOG_LEVEL,
    FEATURE_WINDOW_SIZE,
    FEATURE_MIN_SAMPLES,
    BASELINE_MIN_SAMPLES,
    TEMP_WARNING_THRESHOLD,
    SCORE_WARNING_THRESHOLD,
    SCORE_SEVERE_THRESHOLD,
)
from database.db import (
    init_db,
    get_latest_telemetry,
    get_telemetry,
    get_telemetry_by_vehicle,
    get_telemetry_by_vehicle_ordered,
    get_baselines_for_vehicle,
)
from edge.features import FeatureWindow, observation_from_record
from edge.baseline import BaselineRegistry, BaselineReadiness
from edge.scoring import score_all_features, DeviationLevel

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("edgepulse.api")


# ---------------------------------------------------------------------------
# Response models — Phase 0
# ---------------------------------------------------------------------------

class TelemetryRecord(BaseModel):
    id: int
    message_id: Optional[str] = None
    event_timestamp: str
    ingestion_timestamp: str
    vehicle_id: str
    speed: float
    rpm: float
    engine_temp: float
    battery_voltage: Optional[float] = None
    fuel_level: Optional[float] = None


# ---------------------------------------------------------------------------
# Response models — Phase 2
# ---------------------------------------------------------------------------

class FeatureResponse(BaseModel):
    """Feature vector for one vehicle at the most recent observation."""
    vehicle_id:      str
    event_timestamp: str
    engine_temp:             float
    temp_rolling_mean:       Optional[float] = None
    temp_rolling_min:        Optional[float] = None
    temp_rolling_max:        Optional[float] = None
    temp_rolling_std:        Optional[float] = None
    temp_slope:              Optional[float] = None
    temp_above_threshold_s:  Optional[float] = None
    rpm:                     float
    rpm_rolling_mean:        Optional[float] = None
    rpm_rolling_std:         Optional[float] = None
    rpm_variability:         Optional[float] = None
    rpm_change_rate:         Optional[float] = None
    speed:                   float
    speed_rolling_mean:      Optional[float] = None
    speed_rolling_std:       Optional[float] = None
    speed_change_rate:       Optional[float] = None
    acceleration:            Optional[float] = None
    battery_voltage:         Optional[float] = None
    voltage_rolling_mean:    Optional[float] = None
    voltage_rolling_std:     Optional[float] = None
    voltage_trend:           Optional[float] = None
    fuel_level:              Optional[float] = None
    fuel_rate:               Optional[float] = None
    rpm_variability_at_stable_speed: Optional[float] = None
    idle_duration_s:         Optional[float] = None
    window_size_used:        int = 0


class BaselineFeatureStatus(BaseModel):
    feature_name:  str
    readiness:     str
    sample_count:  int
    mean:          Optional[float] = None
    std:           Optional[float] = None


class BaselineStatusResponse(BaseModel):
    vehicle_id: str
    features:   list[BaselineFeatureStatus]


class DeviationScoreResponse(BaseModel):
    vehicle_id:      str
    feature_name:    str
    event_timestamp: str
    current_value:   Optional[float] = None
    baseline_mean:   Optional[float] = None
    baseline_std:    Optional[float] = None
    z_score:         Optional[float] = None
    level:           str
    direction:       int
    sample_count:    int
    readiness:       str


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("EdgePulse API starting — initialising database...")
    init_db()
    logger.info("Database ready.")
    yield
    logger.info("EdgePulse API shutting down.")


app = FastAPI(
    title="EdgePulse Telemetry API",
    description=(
        "REST API for EdgePulse vehicle telemetry, feature engineering, "
        "and statistical baseline deviation scoring.\n\n"
        "Data flow: Vehicle → MQTT → Consumer → SQLite → this API."
    ),
    version="0.4.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Phase 0 routes
# ---------------------------------------------------------------------------

@app.get("/", summary="Service identity")
def root():
    return {"service": "EdgePulse Telemetry API", "status": "running", "version": "0.4.0"}


@app.get("/health", summary="Health check")
def health():
    try:
        get_telemetry(limit=1)
        return {"status": "healthy", "database": "reachable"}
    except Exception as e:
        logger.error("Health check failed: %s", e)
        raise HTTPException(status_code=503, detail=f"Database unavailable: {e}")


@app.get("/telemetry/latest", response_model=TelemetryRecord, summary="Most recent telemetry")
def latest_telemetry():
    try:
        record = get_latest_telemetry()
    except Exception as e:
        logger.error("Failed to fetch latest telemetry: %s", e)
        raise HTTPException(status_code=503, detail="Database error")
    if record is None:
        raise HTTPException(status_code=404, detail="No telemetry records found.")
    return record


@app.get("/telemetry", response_model=list[TelemetryRecord], summary="List recent telemetry")
def list_telemetry(
    limit: int = Query(default=20, ge=1, le=100)
):
    try:
        return get_telemetry(limit=limit)
    except Exception as e:
        logger.error("Failed to fetch telemetry list: %s", e)
        raise HTTPException(status_code=503, detail="Database error")


@app.get("/telemetry/{vehicle_id}", response_model=list[TelemetryRecord],
         summary="Telemetry for a vehicle")
def vehicle_telemetry(
    vehicle_id: str,
    limit: int = Query(default=20, ge=1, le=100)
):
    try:
        records = get_telemetry_by_vehicle(vehicle_id=vehicle_id, limit=limit)
    except Exception as e:
        logger.error("Failed to fetch telemetry for '%s': %s", vehicle_id, e)
        raise HTTPException(status_code=503, detail="Database error")
    if not records:
        raise HTTPException(status_code=404, detail=f"No telemetry found for '{vehicle_id}'.")
    return records


# ---------------------------------------------------------------------------
# Phase 2 routes
# ---------------------------------------------------------------------------

def _build_feature_window_for_vehicle(vehicle_id: str, limit: int = 200) -> tuple:
    """
    Loads recent telemetry from the DB, replays it through a FeatureWindow,
    and returns (feature_vector_dict, window_size_used).
    Returns (None, 0) if no records found.
    """
    records = get_telemetry_by_vehicle_ordered(vehicle_id, limit=limit)
    if not records:
        return None, 0

    window = FeatureWindow(
        window_size=FEATURE_WINDOW_SIZE,
        min_samples=FEATURE_MIN_SAMPLES,
        temp_warn_thresh=TEMP_WARNING_THRESHOLD,
    )
    fv = None
    for rec in records:
        obs = observation_from_record(rec)
        if obs:
            fv = window.update(obs)

    return (fv.to_dict() if fv else None), window.sample_count


@app.get(
    "/features/{vehicle_id}",
    response_model=FeatureResponse,
    summary="Latest feature vector for a vehicle",
)
def vehicle_features(vehicle_id: str):
    """
    Computes the current feature vector for a vehicle by replaying its most
    recent telemetry through the rolling-window feature extractor.
    Returns 404 if no telemetry exists for this vehicle.
    Returns 422 if features cannot be computed (e.g. too few records).
    """
    try:
        fv_dict, n = _build_feature_window_for_vehicle(vehicle_id)
    except Exception as e:
        logger.error("Feature computation failed for '%s': %s", vehicle_id, e)
        raise HTTPException(status_code=503, detail="Feature computation error")

    if fv_dict is None:
        raise HTTPException(status_code=404,
                            detail=f"No telemetry found for vehicle '{vehicle_id}'.")
    fv_dict["window_size_used"] = n
    return fv_dict


@app.get(
    "/baseline/{vehicle_id}",
    response_model=BaselineStatusResponse,
    summary="Baseline status for a vehicle",
)
def vehicle_baseline_status(vehicle_id: str):
    """
    Returns the current baseline statistics for all tracked features of a vehicle.
    Baselines are loaded from the vehicle_baselines table.
    Returns 404 if no baseline data exists for this vehicle.
    """
    try:
        rows = get_baselines_for_vehicle(vehicle_id)
    except Exception as e:
        logger.error("Failed to fetch baseline for '%s': %s", vehicle_id, e)
        raise HTTPException(status_code=503, detail="Database error")

    if not rows:
        raise HTTPException(status_code=404,
                            detail=f"No baseline data found for vehicle '{vehicle_id}'.")

    registry = BaselineRegistry(min_samples=BASELINE_MIN_SAMPLES)
    registry.load_from_db(rows)
    vb = registry.get_or_create(vehicle_id)

    features = []
    for row in rows:
        fname = row["feature_name"]
        bl = vb.get(fname)
        features.append(BaselineFeatureStatus(
            feature_name=fname,
            readiness=bl.readiness.value if bl else BaselineReadiness.INSUFFICIENT.value,
            sample_count=bl.sample_count if bl else 0,
            mean=round(bl.mean, 4) if bl and bl.mean is not None else None,
            std=round(bl.std, 4) if bl and bl.std is not None else None,
        ))

    return BaselineStatusResponse(vehicle_id=vehicle_id, features=features)


@app.get(
    "/scores/{vehicle_id}",
    response_model=list[DeviationScoreResponse],
    summary="Statistical deviation scores for a vehicle",
)
def vehicle_scores(vehicle_id: str):
    """
    Computes statistical deviation scores for all tracked features by
    comparing current feature values against the stored vehicle baseline.
    Returns 404 if no telemetry or baseline exists.
    """
    try:
        fv_dict, _ = _build_feature_window_for_vehicle(vehicle_id)
        baseline_rows = get_baselines_for_vehicle(vehicle_id)
    except Exception as e:
        logger.error("Scoring failed for '%s': %s", vehicle_id, e)
        raise HTTPException(status_code=503, detail="Scoring error")

    if fv_dict is None:
        raise HTTPException(status_code=404,
                            detail=f"No telemetry found for vehicle '{vehicle_id}'.")

    registry = BaselineRegistry(min_samples=BASELINE_MIN_SAMPLES)
    if baseline_rows:
        registry.load_from_db(baseline_rows)
    vb = registry.get_or_create(vehicle_id)

    scores = score_all_features(
        vehicle_id=vehicle_id,
        fv_dict=fv_dict,
        vehicle_baseline=vb,
        event_timestamp=fv_dict.get("event_timestamp", ""),
        warning_threshold=SCORE_WARNING_THRESHOLD,
        severe_threshold=SCORE_SEVERE_THRESHOLD,
    )

    return [DeviationScoreResponse(**s.to_dict()) for s in scores]


# ---------------------------------------------------------------------------
# Phase 3 routes — LSTM anomaly scoring
# ---------------------------------------------------------------------------

class LSTMStatusResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    enabled:       bool
    model_loaded:  bool
    threshold_set: bool
    model_version: Optional[str] = None
    threshold:     Optional[float] = None
    vehicles_buffered: list = []


class LSTMAnomalyResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    vehicle_id:           str
    end_timestamp:        Optional[str] = None
    reconstruction_error: Optional[float] = None
    threshold:            Optional[float] = None
    is_anomalous:         Optional[bool] = None
    model_version:        Optional[str] = None
    status:               str


# Module-level pipeline reference for Phase 3 endpoints.
# The consumer creates its own FeaturePipeline instance; the API creates a
# read-only one here solely for serving LSTM status when the API is queried
# directly (without the consumer running).  In a live deployment the consumer
# pipeline instance holds the live LSTM state; the API reflects stored results.
_api_pipeline: Optional[object] = None


def _get_api_pipeline():
    """Return or create the API-side FeaturePipeline (lazy, no LSTM scorer)."""
    global _api_pipeline
    if _api_pipeline is None:
        from edge.pipeline import FeaturePipeline
        _api_pipeline = FeaturePipeline(lstm_enabled=False)
    return _api_pipeline


@app.get(
    "/lstm/status",
    response_model=LSTMStatusResponse,
    summary="LSTM model availability and threshold status",
)
def lstm_status():
    """
    Reports whether a trained LSTM model is available and threshold-calibrated.
    Does not require the consumer to be running.
    """
    from lstm.artifacts import artifact_exists, DEFAULT_ARTIFACT_DIR
    import json, os

    model_loaded  = artifact_exists()
    threshold_set = False
    threshold     = None
    model_version = None

    if model_loaded:
        try:
            meta_path = os.path.join(DEFAULT_ARTIFACT_DIR, "metadata.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            threshold     = meta.get("threshold")
            threshold_set = threshold is not None
            model_version = meta.get("model_version")
        except Exception as e:
            logger.warning("Could not read LSTM metadata: %s", e)

    return LSTMStatusResponse(
        enabled=model_loaded,
        model_loaded=model_loaded,
        threshold_set=threshold_set,
        model_version=model_version,
        threshold=threshold,
        vehicles_buffered=[],
    )


@app.get(
    "/lstm/anomaly/{vehicle_id}",
    response_model=LSTMAnomalyResponse,
    summary="Latest LSTM anomaly result for a vehicle",
)
def lstm_anomaly(vehicle_id: str):
    """
    Returns the latest LSTM reconstruction-error anomaly result for a vehicle.

    The result is produced by the running consumer process via AnomalyScorer.
    The API reads from the in-process scorer when available, otherwise reports
    the current model status.

    Status values:
      scored       — a complete sequence was scored
      warming_up   — buffer has not yet filled seq_len observations
      no_model     — no trained model loaded
      no_threshold — model loaded but threshold not yet calibrated
      error        — inference exception (Phase 2 unaffected)
    """
    from lstm.artifacts import artifact_exists

    if not artifact_exists():
        return LSTMAnomalyResponse(
            vehicle_id=vehicle_id,
            status="no_model",
        )

    # If the consumer pipeline is running in-process (e.g. tests), use it.
    # Otherwise fall back to a static no_model response since we don't share
    # the consumer's AnomalyScorer instance across processes.
    pipeline = _get_api_pipeline()
    result = pipeline.get_lstm_result(vehicle_id)

    if result is None:
        # Check threshold calibration status
        import json, os
        from lstm.artifacts import DEFAULT_ARTIFACT_DIR
        try:
            meta_path = os.path.join(DEFAULT_ARTIFACT_DIR, "metadata.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            threshold = meta.get("threshold")
            status = "warming_up" if threshold is not None else "no_threshold"
        except Exception:
            status = "no_model"

        return LSTMAnomalyResponse(vehicle_id=vehicle_id, status=status)

    return LSTMAnomalyResponse(**result)
