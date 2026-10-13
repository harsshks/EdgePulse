# lstm/inference.py
# LSTM anomaly scoring for EdgePulse Phase 3 — Stage D.
#
# Two responsibilities:
#
#   A. Threshold calibration
#      calibrate_threshold(val_split, model, imputer, scaler, ...)
#      — computes reconstruction errors on HEALTHY validation sequences only
#      — stores the calibrated threshold back into the artifact via save_model()
#      — NEVER touches test data or fault labels
#
#   B. AnomalyScorer
#      — loads saved artifact once at init (model, imputer, scaler, threshold)
#      — maintains a per-vehicle bounded sequence buffer
#      — scores complete sequences and returns LSTMAnomalyResult
#      — safe defaults when model is absent, buffer is warming up, or
#        inference fails (Phase 2 scoring is unaffected)
#
# Integration point:
#   FeaturePipeline.process_record() calls AnomalyScorer.push() after Phase 2
#   scoring is complete. If the scorer is None or raises, Phase 2 is
#   unaffected and raw telemetry is already persisted.

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch

from lstm.dataset import (
    LSTM_FEATURES,
    N_FEATURES,
    FeatureImputer,
    FeatureScaler,
    is_healthy,
)

logger = logging.getLogger(__name__)

# Default threshold calibration percentile
DEFAULT_THRESHOLD_PERCENTILE: float = 95.0

# Maximum observations to keep per vehicle buffer (seq_len is the trigger)
_BUFFER_MAX_MULTIPLIER: int = 3   # keep at most 3 × seq_len recent observations


# ---------------------------------------------------------------------------
# Anomaly result
# ---------------------------------------------------------------------------

@dataclass
class LSTMAnomalyResult:
    """
    Structured result for one scored sequence.

    Fields:
        vehicle_id        — which vehicle
        end_timestamp     — ISO-8601 string of last observation in the sequence
        reconstruction_error — MSE of the reconstructed vs. input sequence
        threshold         — calibrated anomaly threshold (None if not calibrated)
        is_anomalous      — True when reconstruction_error > threshold
                            (None when threshold is not available)
        model_version     — version string from artifact metadata
        status            — human-readable status: 'scored' | 'warming_up' |
                            'no_model' | 'no_threshold' | 'error'
    """
    vehicle_id:          str
    end_timestamp:       Optional[str]
    reconstruction_error: Optional[float]
    threshold:           Optional[float]
    is_anomalous:        Optional[bool]
    model_version:       Optional[str]
    status:              str

    def to_dict(self) -> dict:
        return {
            "vehicle_id":           self.vehicle_id,
            "end_timestamp":        self.end_timestamp,
            "reconstruction_error": self.reconstruction_error,
            "threshold":            self.threshold,
            "is_anomalous":         self.is_anomalous,
            "model_version":        self.model_version,
            "status":               self.status,
        }


# ---------------------------------------------------------------------------
# A. Threshold calibration
# ---------------------------------------------------------------------------

def calibrate_threshold(
    val_sequences: list,                 # list[LSTMSequence] — healthy only
    model,                               # LSTMAutoencoder
    imputer: FeatureImputer,
    scaler:  FeatureScaler,
    percentile: float = DEFAULT_THRESHOLD_PERCENTILE,
    model_dir: Optional[str] = None,
    model_version: str = "1.0",
    train_seed: Optional[int] = None,
    dataset_ids: Optional[dict] = None,
    best_val_loss: Optional[float] = None,
    train_epochs_run: Optional[int] = None,
) -> float:
    """
    Calibrate the anomaly threshold from healthy validation sequences.

    Uses the `percentile`-th percentile of reconstruction errors on validation
    data as the threshold.  Sequences containing anomalous steps are silently
    skipped — calibration uses healthy sequences only.

    The threshold is persisted back to the artifact via save_model().

    Parameters
    ----------
    val_sequences : list of LSTMSequence (from DatasetSplit.sequences)
    model         : trained LSTMAutoencoder (already in eval mode)
    imputer       : training-fitted FeatureImputer
    scaler        : training-fitted FeatureScaler
    percentile    : threshold percentile (0–100), default 95
    model_dir     : artifact directory to update (default: models/)
    ...           : remaining args forwarded to save_model for full metadata

    Returns
    -------
    threshold : float

    Raises
    ------
    ValueError if no valid healthy sequences are found
    ValueError if percentile is outside [0, 100]
    """
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"percentile must be in [0, 100], got {percentile}")

    # Keep only healthy sequences
    healthy = [s for s in val_sequences if not s.is_anomalous]
    if not healthy:
        raise ValueError(
            "No healthy validation sequences available for threshold calibration. "
            "Ensure val_split contains healthy-only sequences."
        )

    # Stack features into a tensor
    arrays = [s.features for s in healthy]
    X_val = torch.from_numpy(np.stack(arrays)).float()  # (N, seq_len, n_features)

    # Compute per-sequence reconstruction errors
    model.eval()
    with torch.no_grad():
        errors = model.reconstruction_error(X_val)  # (N,)

    errors_np = errors.numpy()

    # Validate — all should be finite
    if not np.all(np.isfinite(errors_np)):
        raise ValueError("Reconstruction errors contain non-finite values.")

    threshold = float(np.percentile(errors_np, percentile))
    logger.info(
        "Threshold calibrated | percentile=%.1f n_val=%d threshold=%.6f",
        percentile, len(healthy), threshold,
    )

    # Persist updated threshold into the artifact
    from lstm.artifacts import save_model
    save_model(
        model=model,
        imputer=imputer,
        scaler=scaler,
        feature_names=LSTM_FEATURES,
        seq_len=model.seq_len,
        model_dir=model_dir,
        threshold=threshold,
        threshold_percentile=percentile,
        threshold_method=f"{percentile:.0f}th-percentile of healthy val errors",
        train_seed=train_seed,
        train_epochs_run=train_epochs_run,
        best_val_loss=best_val_loss,
        dataset_ids=dataset_ids or {},
        model_version=model_version,
    )

    return threshold


# ---------------------------------------------------------------------------
# Per-vehicle observation buffer entry
# ---------------------------------------------------------------------------

@dataclass
class _BufferEntry:
    """One observation in the per-vehicle buffer."""
    event_timestamp: datetime
    features:        np.ndarray   # shape (N_FEATURES,) — imputed + scaled float32


# ---------------------------------------------------------------------------
# B. AnomalyScorer
# ---------------------------------------------------------------------------

class AnomalyScorer:
    """
    Runs LSTM inference on live telemetry using a pre-trained model artifact.

    Load once at startup via AnomalyScorer(model_dir='models/').
    Call push(vehicle_id, fv_dict, event_timestamp) for each telemetry record.

    Buffer behaviour:
    - Separate deque per vehicle, max length = _BUFFER_MAX_MULTIPLIER * seq_len.
    - Scoring fires when the buffer contains >= seq_len entries.
    - On duplicate/OOO timestamps or excessive gaps, the affected entry is
      dropped and the buffer is not scored.
    - Buffer is bounded; the oldest entries are evicted automatically.

    Thread safety: not thread-safe — use one instance per process/thread.
    """

    def __init__(
        self,
        model_dir:      Optional[str] = None,
        seq_len:        Optional[int] = None,
        max_gap_seconds: float = 30.0,
        enabled:        bool = True,
    ) -> None:
        """
        Parameters
        ----------
        model_dir       : artifact directory to load from (default: models/)
        seq_len         : override sequence length (default: from artifact)
        max_gap_seconds : gaps > this value break buffer continuity
        enabled         : if False, push() is a no-op (safe default)
        """
        self._model_dir      = model_dir
        self._max_gap_seconds = max_gap_seconds
        self._enabled        = enabled

        self._model          = None
        self._imputer:       Optional[FeatureImputer] = None
        self._scaler:        Optional[FeatureScaler]  = None
        self._threshold:     Optional[float]          = None
        self._model_version: Optional[str]            = None
        self._seq_len:       int                      = seq_len or 30
        self._feature_names: list                     = []
        self._ready:         bool                     = False

        # Per-vehicle buffers: vehicle_id -> deque[_BufferEntry]
        self._buffers: dict[str, deque] = {}

        # Cache latest result per vehicle
        self._latest: dict[str, LSTMAnomalyResult] = {}

        if enabled:
            self._try_load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _try_load(self) -> None:
        """Attempt to load the artifact. Logs and continues on failure."""
        from lstm.artifacts import load_model, artifact_exists
        if not artifact_exists(self._model_dir):
            logger.info(
                "AnomalyScorer: no model artifact found at '%s'. "
                "Inference disabled until a model is trained.",
                self._model_dir or "models/",
            )
            self._ready = False
            return
        try:
            artifact = load_model(
                self._model_dir,
                expected_features=LSTM_FEATURES,
            )
            self._model         = artifact["model"]
            self._imputer       = artifact["imputer"]
            self._scaler        = artifact["scaler"]
            self._threshold     = artifact["threshold"]
            self._model_version = artifact["metadata"].get("model_version")
            self._seq_len       = artifact["metadata"]["architecture"]["seq_len"]
            self._feature_names = artifact["metadata"]["feature_names"]
            self._model.eval()
            self._ready = True
            logger.info(
                "AnomalyScorer loaded | version=%s seq_len=%d threshold=%s",
                self._model_version, self._seq_len, self._threshold,
            )
        except Exception as e:
            logger.error("AnomalyScorer: failed to load artifact: %s", e)
            self._ready = False

    def reload(self) -> bool:
        """Explicit reload — clears all buffers and reloads from disk."""
        self._buffers.clear()
        self._latest.clear()
        self._try_load()
        return self._ready

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True if model loaded and threshold calibrated."""
        return self._ready and self._model is not None and self._threshold is not None

    @property
    def model_loaded(self) -> bool:
        return self._ready and self._model is not None

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold

    @property
    def model_version(self) -> Optional[str]:
        return self._model_version

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def push(
        self,
        vehicle_id: str,
        fv_dict:    dict,
        event_timestamp: str,
    ) -> Optional[LSTMAnomalyResult]:
        """
        Push one feature vector into the vehicle's buffer and score if ready.

        Parameters
        ----------
        vehicle_id      : vehicle identifier
        fv_dict         : feature vector dict (from FeatureVector.to_dict())
        event_timestamp : ISO-8601 string from the telemetry record

        Returns
        -------
        LSTMAnomalyResult if a complete sequence was scored, else None.
        Returns a warming-up / no_model result before the buffer fills.
        Never raises — exceptions are caught and logged.
        """
        if not self._enabled:
            return None

        try:
            return self._push(vehicle_id, fv_dict, event_timestamp)
        except Exception as e:
            logger.error(
                "AnomalyScorer.push failed for vehicle '%s': %s",
                vehicle_id, e,
            )
            return LSTMAnomalyResult(
                vehicle_id=vehicle_id,
                end_timestamp=event_timestamp,
                reconstruction_error=None,
                threshold=self._threshold,
                is_anomalous=None,
                model_version=self._model_version,
                status="error",
            )

    def _push(
        self,
        vehicle_id: str,
        fv_dict:    dict,
        event_timestamp: str,
    ) -> Optional[LSTMAnomalyResult]:

        if not self.model_loaded:
            return LSTMAnomalyResult(
                vehicle_id=vehicle_id,
                end_timestamp=event_timestamp,
                reconstruction_error=None,
                threshold=None,
                is_anomalous=None,
                model_version=None,
                status="no_model",
            )

        # Parse and validate the incoming feature vector
        raw_vals = [fv_dict.get(fname) for fname in LSTM_FEATURES]
        if any(v is None for v in raw_vals):
            # Not enough rolling history yet — not an error
            return None

        arr = np.array(raw_vals, dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, np.nan)

        # Impute + scale
        imputed = self._imputer.transform(arr)
        scaled  = self._scaler.transform(imputed).astype(np.float32)

        # Parse timestamp
        try:
            ts = datetime.fromisoformat(event_timestamp)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            logger.warning("AnomalyScorer: unparseable timestamp '%s'", event_timestamp)
            return None

        # Get or create buffer
        if vehicle_id not in self._buffers:
            max_len = _BUFFER_MAX_MULTIPLIER * self._seq_len
            self._buffers[vehicle_id] = deque(maxlen=max_len)

        buf = self._buffers[vehicle_id]

        # Gap / OOO guard
        if buf:
            last_ts = buf[-1].event_timestamp
            dt = (ts - last_ts).total_seconds()
            if dt <= 0:
                logger.debug(
                    "AnomalyScorer: duplicate/OOO timestamp for '%s' — skipping.",
                    vehicle_id,
                )
                return None
            if dt > self._max_gap_seconds:
                logger.debug(
                    "AnomalyScorer: gap %.1fs > %.1fs for '%s' — resetting buffer.",
                    dt, self._max_gap_seconds, vehicle_id,
                )
                buf.clear()

        buf.append(_BufferEntry(event_timestamp=ts, features=scaled))

        # Not enough observations yet
        if len(buf) < self._seq_len:
            return LSTMAnomalyResult(
                vehicle_id=vehicle_id,
                end_timestamp=event_timestamp,
                reconstruction_error=None,
                threshold=self._threshold,
                is_anomalous=None,
                model_version=self._model_version,
                status="warming_up",
            )

        # Score the most recent seq_len entries
        window = list(buf)[-self._seq_len:]
        feat_matrix = np.stack([e.features for e in window])  # (seq_len, n_features)
        x = torch.from_numpy(feat_matrix).unsqueeze(0).float()  # (1, seq_len, n_features)

        self._model.eval()
        with torch.no_grad():
            errors = self._model.reconstruction_error(x)  # (1,)

        recon_error = float(errors[0].item())

        # Determine anomaly status
        if self._threshold is None:
            is_anomalous = None
            status = "no_threshold"
        else:
            is_anomalous = recon_error > self._threshold
            status = "scored"

        result = LSTMAnomalyResult(
            vehicle_id=vehicle_id,
            end_timestamp=event_timestamp,
            reconstruction_error=round(recon_error, 6),
            threshold=self._threshold,
            is_anomalous=is_anomalous,
            model_version=self._model_version,
            status=status,
        )

        self._latest[vehicle_id] = result
        return result

    def get_latest(self, vehicle_id: str) -> Optional[LSTMAnomalyResult]:
        """Return the most recent scored result for a vehicle, or None."""
        return self._latest.get(vehicle_id)

    def buffer_size(self, vehicle_id: str) -> int:
        """Current buffer depth for a vehicle."""
        return len(self._buffers.get(vehicle_id, []))

    def status_summary(self) -> dict:
        """Summary dict for the API health / status endpoint."""
        return {
            "enabled":       self._enabled,
            "model_loaded":  self.model_loaded,
            "threshold_set": self._threshold is not None,
            "model_version": self._model_version,
            "threshold":     self._threshold,
            "vehicles_buffered": list(self._buffers.keys()),
        }
