# edge/pipeline.py
# Feature-engineering pipeline integration for EdgePulse Phase 2 + 3.
#
# This module wires together:
#   1. FeatureWindow  — per-vehicle rolling feature extraction
#   2. VehicleBaseline — per-vehicle statistical baselines
#   3. score_all_features — z-score deviation scoring (Phase 2)
#   4. AnomalyScorer (optional) — LSTM reconstruction-error scoring (Phase 3)
#
# Phase 3 integration policy:
#   - AnomalyScorer is optional. It is disabled when no trained model artifact
#     exists or when lstm_model_dir is not provided.
#   - LSTM inference runs AFTER Phase 2 scoring is complete.
#   - Inference failures are caught and logged; they never interrupt
#     raw telemetry persistence or Phase 2 scoring.
#   - Model loading happens once at FeaturePipeline init, not per message.
#   - No training occurs at runtime.
#
# Persistence:
#   - Baselines are persisted to SQLite on every update so they survive restarts.
#   - Feature vectors are optionally persisted (controlled by persist_features flag).

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from config import (
    FEATURE_WINDOW_SIZE,
    FEATURE_MIN_SAMPLES,
    BASELINE_MIN_SAMPLES,
    TEMP_WARNING_THRESHOLD,
    SCORE_WARNING_THRESHOLD,
    SCORE_SEVERE_THRESHOLD,
)
from edge.features import FeatureWindow, FeatureVector, observation_from_record
from edge.baseline import BaselineRegistry, VehicleBaseline
from edge.scoring import score_all_features, DeviationScore
from database.db import upsert_baseline, get_baselines_for_vehicle

logger = logging.getLogger("edgepulse.pipeline")

# Lazy import for LSTM inference — keeps Phase 2 functional if lstm package
# is absent or artifacts are missing.
_AnomalyScorer = None


def _get_anomaly_scorer_class():
    global _AnomalyScorer
    if _AnomalyScorer is None:
        try:
            from lstm.inference import AnomalyScorer
            _AnomalyScorer = AnomalyScorer
        except ImportError:
            pass
    return _AnomalyScorer


class FeaturePipeline:
    """
    Per-process singleton that maintains:
      - one FeatureWindow per vehicle
      - one VehicleBaseline per vehicle (via BaselineRegistry)

    Intended usage in the consumer:
        pipeline = FeaturePipeline()
        pipeline.load_baselines_from_db()  # call once at startup

        # For each telemetry record after it is stored in SQLite:
        result = pipeline.process_record(record, exclude=False)
    """

    def __init__(
        self,
        window_size:        int   = FEATURE_WINDOW_SIZE,
        min_samples:        int   = FEATURE_MIN_SAMPLES,
        baseline_min:       int   = BASELINE_MIN_SAMPLES,
        temp_warn_thresh:   float = TEMP_WARNING_THRESHOLD,
        warn_z:             float = SCORE_WARNING_THRESHOLD,
        severe_z:           float = SCORE_SEVERE_THRESHOLD,
        persist_features:   bool  = False,
        lstm_model_dir:     Optional[str] = None,
        lstm_enabled:       bool  = False,
    ):
        self._window_size      = window_size
        self._min_samples      = min_samples
        self._temp_warn_thresh = temp_warn_thresh
        self._warn_z           = warn_z
        self._severe_z         = severe_z
        self._persist_features = persist_features

        self._windows:  dict[str, FeatureWindow]  = {}
        self._registry: BaselineRegistry = BaselineRegistry(min_samples=baseline_min)

        # Phase 3: optional LSTM anomaly scorer
        # Loaded once here; never reloaded per message.
        self._anomaly_scorer = None
        if lstm_enabled:
            ScorerClass = _get_anomaly_scorer_class()
            if ScorerClass is not None:
                try:
                    self._anomaly_scorer = ScorerClass(
                        model_dir=lstm_model_dir,
                        enabled=True,
                    )
                    logger.info(
                        "LSTM AnomalyScorer initialised (ready=%s)",
                        self._anomaly_scorer.is_ready,
                    )
                except Exception as e:
                    logger.warning("Failed to initialise AnomalyScorer: %s", e)

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def load_baselines_from_db(self, vehicle_ids: Optional[list[str]] = None) -> None:
        """
        Reload persisted baselines from SQLite so that a process restart
        doesn't throw away accumulated history.

        If vehicle_ids is None, no pre-loading is done (baselines start fresh).
        This is safe — baselines rebuild from new observations.
        """
        if vehicle_ids is None:
            return
        for vid in vehicle_ids:
            rows = get_baselines_for_vehicle(vid)
            if rows:
                self._registry.load_from_db(rows)
                logger.info("Loaded %d baseline rows for vehicle %s", len(rows), vid)

    # ------------------------------------------------------------------
    # Per-record processing
    # ------------------------------------------------------------------

    def process_record(
        self,
        record: dict,
        exclude: bool = False,
    ) -> Optional[tuple[FeatureVector, list[DeviationScore]]]:
        """
        Process one validated, persisted telemetry record through the pipeline.

        Parameters:
            record  — enriched telemetry dict (from consumer.process_telemetry)
            exclude — if True, update windows but NOT baselines (fault period)

        Returns (feature_vector, deviation_scores) or None if the observation
        cannot be parsed (e.g. missing required fields or bad timestamp).

        This method catches all internal exceptions so a pipeline failure
        never propagates to the consumer's MQTT handler.
        """
        try:
            return self._process(record, exclude)
        except Exception as e:
            logger.error(
                "Feature pipeline error for vehicle '%s': %s — raw telemetry preserved.",
                record.get("vehicle_id"), e,
                exc_info=True,
            )
            return None

    def _process(
        self,
        record: dict,
        exclude: bool,
    ) -> Optional[tuple[FeatureVector, list[DeviationScore]]]:
        obs = observation_from_record(record)
        if obs is None:
            logger.warning("Could not parse observation from record: %s", record)
            return None

        vid = obs.vehicle_id

        # --- Feature extraction ---
        window = self._get_window(vid)
        fv = window.update(obs)

        # --- Baseline update (skip if this is a known fault period) ---
        vb = self._registry.get_or_create(vid)
        accepted = vb.update_from_features(
            fv_dict=fv.to_dict(),
            timestamp=obs.event_timestamp,
            exclude=exclude,
        )

        # Persist baseline updates to SQLite
        self._persist_baselines(vb, record.get("event_timestamp", ""))

        # --- Scoring ---
        scores = score_all_features(
            vehicle_id=vid,
            fv_dict=fv.to_dict(),
            vehicle_baseline=vb,
            event_timestamp=record.get("event_timestamp", ""),
            warning_threshold=self._warn_z,
            severe_threshold=self._severe_z,
        )

        # Log any elevated/severe scores
        for s in scores:
            if s.level.value in ("ELEVATED", "SEVERE"):
                logger.warning(
                    "Deviation | vehicle=%s feature=%s z=%.2f level=%s",
                    s.vehicle_id, s.feature_name, s.z_score or 0.0, s.level.value,
                )

        # --- Phase 3: LSTM anomaly scoring (optional, after Phase 2) ---
        if self._anomaly_scorer is not None:
            try:
                self._anomaly_scorer.push(
                    vehicle_id=vid,
                    fv_dict=fv.to_dict(),
                    event_timestamp=record.get("event_timestamp", ""),
                )
            except Exception as e:
                logger.error(
                    "LSTM inference error for vehicle '%s': %s — Phase 2 unaffected.",
                    vid, e,
                )

        return fv, scores

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_window(self, vehicle_id: str) -> FeatureWindow:
        if vehicle_id not in self._windows:
            self._windows[vehicle_id] = FeatureWindow(
                window_size=self._window_size,
                min_samples=self._min_samples,
                temp_warn_thresh=self._temp_warn_thresh,
            )
        return self._windows[vehicle_id]

    def _persist_baselines(self, vb: VehicleBaseline, updated_at: str) -> None:
        """Write all feature baselines for a vehicle to SQLite."""
        if not updated_at:
            updated_at = datetime.now(timezone.utc).isoformat()
        for fname, bl in vb._baselines.items():
            if bl.sample_count == 0:
                continue
            try:
                upsert_baseline(
                    vehicle_id=vb.vehicle_id,
                    feature_name=fname,
                    sample_count=bl.sample_count,
                    mean=bl.mean or 0.0,
                    variance=bl.variance or 0.0,
                    updated_at=updated_at,
                )
            except Exception as e:
                logger.error("Failed to persist baseline %s/%s: %s", vb.vehicle_id, fname, e)

    def get_latest_features(self, vehicle_id: str) -> Optional[FeatureVector]:
        """Return the last computed feature vector for a vehicle (in-memory only)."""
        window = self._windows.get(vehicle_id)
        if window is None or window.sample_count == 0:
            return None
        # Re-compute on the last observation in the window
        if window._obs:
            return window._compute(window._obs[-1])
        return None

    def get_baseline_summary(self, vehicle_id: str) -> Optional[dict]:
        """Return baseline summary dict for a vehicle, or None if unknown."""
        vb = self._registry._vehicles.get(vehicle_id)
        if vb is None:
            return None
        return vb.summary()

    # ------------------------------------------------------------------
    # Phase 3 accessors
    # ------------------------------------------------------------------

    def get_lstm_result(self, vehicle_id: str) -> Optional[dict]:
        """
        Return the latest LSTM anomaly result for a vehicle, or None.
        Returns None when LSTM is disabled or the buffer is still warming up.
        """
        if self._anomaly_scorer is None:
            return None
        result = self._anomaly_scorer.get_latest(vehicle_id)
        return result.to_dict() if result is not None else None

    def get_lstm_status(self) -> dict:
        """Return LSTM scorer status summary for the API /lstm/status endpoint."""
        if self._anomaly_scorer is None:
            return {"enabled": False, "model_loaded": False, "threshold_set": False}
        return self._anomaly_scorer.status_summary()
