# lstm/dataset.py
# Data preparation for EdgePulse Phase 3 LSTM Autoencoder.
#
# Key design decisions:
#   - Uses sim_time from GroundTruthRecord to build deterministic monotonic
#     timestamps. The simulator uses datetime.now() for event_timestamp, which
#     produces near-identical wall-clock times when run at full speed and
#     breaks sequence continuity detection. sim_time (0.0, 1.0, 2.0 ...) is
#     always strictly monotonic and is the authoritative simulation clock.
#   - Ground-truth labels are NEVER part of feature tensors.
#   - Scaler/imputer fitted on HEALTHY TRAINING DATA only.

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np

from edge.features import FeatureWindow, observation_from_record
from edge.baseline import VehicleBaseline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical feature order
# ---------------------------------------------------------------------------

LSTM_FEATURES: list = list(VehicleBaseline.DEFAULT_FEATURES)
N_FEATURES: int = len(LSTM_FEATURES)  # 12

# Epoch used to convert sim_time (float seconds) to timezone-aware datetimes
_SIM_EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def sim_time_to_datetime(sim_time: float) -> datetime:
    """Convert sim_time (seconds from simulation start) to a UTC datetime."""
    return _SIM_EPOCH + timedelta(seconds=sim_time)


# ---------------------------------------------------------------------------
# Healthy label definition
# ---------------------------------------------------------------------------

HEALTHY_FAULT_TYPE: str = "NONE"
HEALTHY_FAULT_STATE: str = "NORMAL"


def is_healthy(fault_type: str, fault_state: str) -> bool:
    """True only when ground-truth indicates no active fault."""
    return fault_type == HEALTHY_FAULT_TYPE and fault_state == HEALTHY_FAULT_STATE


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class LSTMObservation:
    """One preprocessed observation. features are raw (pre-scale). Labels separate."""
    vehicle_id:      str
    event_timestamp: datetime    # derived from sim_time - deterministic
    features:        np.ndarray  # shape (N_FEATURES,), float64, may contain NaN
    fault_type:      str         # ground-truth - NEVER model input
    fault_state:     str
    severity:        float
    is_healthy:      bool


@dataclass
class LSTMSequence:
    """One fixed-length input sequence. features is the model input only."""
    vehicle_id:    str
    end_timestamp: datetime
    features:      np.ndarray    # shape (seq_len, N_FEATURES), float32, scaled
    fault_types:   list          # per-step ground truth - NEVER model input
    fault_states:  list
    is_anomalous:  bool
    split:         str


@dataclass
class LabelledStep:
    record:      dict
    fault_type:  str
    fault_state: str
    severity:    float
    sim_time:    float   # deterministic simulation clock (seconds)


# ---------------------------------------------------------------------------
# Join records with ground-truth rows
# ---------------------------------------------------------------------------

def build_labelled_steps(records: list, ground_truth_rows: list) -> list:
    """
    Join raw telemetry records with ground-truth rows by message_id.
    Stores sim_time from ground_truth for deterministic timestamp generation.
    """
    if len(records) != len(ground_truth_rows):
        raise ValueError(
            f"Record count ({len(records)}) != ground_truth count "
            f"({len(ground_truth_rows)}). Cannot safely join."
        )
    steps = []
    for rec, gt in zip(records, ground_truth_rows):
        if rec.get("message_id") != gt.message_id:
            raise ValueError(
                f"message_id mismatch: record={rec.get('message_id')} gt={gt.message_id}"
            )
        steps.append(LabelledStep(
            record=rec,
            fault_type=gt.fault_type,
            fault_state=gt.fault_state,
            severity=gt.severity,
            sim_time=gt.sim_time,
        ))
    return steps


# ---------------------------------------------------------------------------
# Phase 2 feature extraction
# ---------------------------------------------------------------------------

def extract_observations(
    labelled_steps: list,
    feature_window_size: int = 60,
    feature_min_samples: int = 3,
    max_gap_seconds: float = 30.0,
) -> list:
    """
    Run each labelled step through per-vehicle Phase 2 FeatureWindow.

    Uses sim_time for timestamps (deterministic, strictly monotonic per vehicle).
    Wall-clock event_timestamp from simulator is NOT used because it reflects
    execution speed rather than simulation time, producing near-identical
    timestamps that would falsely trigger duplicate/OOO guards.

    Observations with any None in LSTM_FEATURES are discarded (insufficient history).
    Inf values replaced with NaN for downstream imputation.
    Labels carried through but never part of .features array.
    """
    from collections import defaultdict
    by_vehicle: dict = defaultdict(list)
    for step in labelled_steps:
        vid = step.record.get("vehicle_id", "UNKNOWN")
        by_vehicle[vid].append(step)

    observations = []
    for vid, steps in by_vehicle.items():
        window = FeatureWindow(
            window_size=feature_window_size,
            min_samples=feature_min_samples,
            max_gap_seconds=max_gap_seconds,
        )
        # Sort by sim_time to guarantee order (should already be ordered but be safe)
        steps_sorted = sorted(steps, key=lambda s: s.sim_time)

        for step in steps_sorted:
            # Build a modified record with deterministic sim_time-based timestamp
            rec = dict(step.record)
            det_ts = sim_time_to_datetime(step.sim_time)
            rec["event_timestamp"] = det_ts.isoformat()

            obs = observation_from_record(rec)
            if obs is None:
                continue

            fv = window.update(obs)
            fv_dict = fv.to_dict()

            raw_vals = [fv_dict.get(fname) for fname in LSTM_FEATURES]
            if any(v is None for v in raw_vals):
                continue

            arr = np.array(raw_vals, dtype=np.float64)
            arr = np.where(np.isfinite(arr), arr, np.nan)

            observations.append(LSTMObservation(
                vehicle_id=vid,
                event_timestamp=det_ts,
                features=arr,
                fault_type=step.fault_type,
                fault_state=step.fault_state,
                severity=step.severity,
                is_healthy=is_healthy(step.fault_type, step.fault_state),
            ))
    return observations


# ---------------------------------------------------------------------------
# Imputer - fitted on HEALTHY TRAINING DATA only
# ---------------------------------------------------------------------------

class FeatureImputer:
    """Per-feature median imputation. fit() on training data only."""

    def __init__(self):
        self.medians_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, observations: list) -> "FeatureImputer":
        if not observations:
            raise ValueError("Cannot fit imputer on empty observation list.")
        matrix = np.stack([o.features for o in observations])
        self.medians_ = np.nanmedian(matrix, axis=0)
        self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
        self._fitted = True
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureImputer must be fitted before transform.")
        out = arr.copy()
        nan_mask = ~np.isfinite(out)
        out[nan_mask] = np.broadcast_to(self.medians_, arr.shape)[nan_mask]
        return out

    def to_dict(self) -> dict:
        return {"medians": self.medians_.tolist() if self.medians_ is not None else None}

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureImputer":
        obj = cls()
        if d.get("medians") is not None:
            obj.medians_ = np.array(d["medians"], dtype=np.float64)
            obj._fitted = True
        return obj


# ---------------------------------------------------------------------------
# Scaler - MinMax fitted on HEALTHY TRAINING DATA only
# ---------------------------------------------------------------------------

class FeatureScaler:
    """
    Per-feature MinMax scaling to [0, 1].
    Constant features (zero range in training) → 0.5.
    Fitted on HEALTHY TRAINING DATA only.
    """

    def __init__(self):
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.range_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, observations: list, imputer: FeatureImputer) -> "FeatureScaler":
        if not observations:
            raise ValueError("Cannot fit scaler on empty observation list.")
        matrix = np.stack([imputer.transform(o.features) for o in observations])
        self.min_ = matrix.min(axis=0)
        self.max_ = matrix.max(axis=0)
        self.range_ = self.max_ - self.min_
        self._fitted = True
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform.")
        out = (arr - self.min_) / np.where(self.range_ > 0, self.range_, 1.0)
        out = np.clip(out, 0.0, 1.0)
        out[..., self.range_ == 0] = 0.5
        return out

    def to_dict(self) -> dict:
        return {
            "min":   self.min_.tolist()   if self.min_   is not None else None,
            "max":   self.max_.tolist()   if self.max_   is not None else None,
            "range": self.range_.tolist() if self.range_ is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureScaler":
        obj = cls()
        if d.get("min") is not None:
            obj.min_   = np.array(d["min"],   dtype=np.float64)
            obj.max_   = np.array(d["max"],   dtype=np.float64)
            obj.range_ = np.array(d["range"], dtype=np.float64)
            obj._fitted = True
        return obj


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------

def build_sequences(
    observations: list,
    seq_len: int,
    imputer: FeatureImputer,
    scaler: FeatureScaler,
    split: str,
    max_gap_seconds: float = 30.0,
    stride: int = 1,
) -> list:
    """
    Build fixed-length sequences with boundary guards:
    - No cross-vehicle sequences.
    - No sequences crossing duplicate or OOO timestamps.
    - No sequences crossing gaps > max_gap_seconds.
    """
    sequences = []
    if len(observations) < seq_len:
        logger.warning(
            "build_sequences: %d observations < seq_len=%d", len(observations), seq_len
        )
        return sequences

    break_before = [False] * len(observations)
    break_before[0] = True

    for i in range(1, len(observations)):
        prev = observations[i - 1]
        curr = observations[i]
        if curr.vehicle_id != prev.vehicle_id:
            break_before[i] = True
            continue
        dt = (curr.event_timestamp - prev.event_timestamp).total_seconds()
        if dt <= 0:
            break_before[i] = True
            continue
        if dt > max_gap_seconds:
            break_before[i] = True

    segments = []
    seg_start = 0
    for i in range(len(observations)):
        if break_before[i] and i > seg_start:
            segments.append((seg_start, i))
            seg_start = i
    segments.append((seg_start, len(observations)))

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        if seg_len < seq_len:
            continue
        for start in range(seg_start, seg_end - seq_len + 1, stride):
            window = observations[start: start + seq_len]
            scaled_steps = []
            for obs in window:
                imputed = imputer.transform(obs.features)
                scaled = scaler.transform(imputed).astype(np.float32)
                scaled_steps.append(scaled)

            feature_matrix = np.stack(scaled_steps)
            is_anomalous = any(not o.is_healthy for o in window)

            sequences.append(LSTMSequence(
                vehicle_id=window[-1].vehicle_id,
                end_timestamp=window[-1].event_timestamp,
                features=feature_matrix,
                fault_types=[o.fault_type for o in window],
                fault_states=[o.fault_state for o in window],
                is_anomalous=is_anomalous,
                split=split,
            ))

    return sequences


# ---------------------------------------------------------------------------
# DatasetSplit container
# ---------------------------------------------------------------------------

@dataclass
class DatasetSplit:
    split:         str
    sequences:     list
    imputer:       FeatureImputer
    scaler:        FeatureScaler
    feature_names: list
    seq_len:       int
    n_healthy:     int
    n_anomalous:   int


# ---------------------------------------------------------------------------
# High-level dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset(
    train_records:       list,
    train_gt:            list,
    val_records:         list,
    val_gt:              list,
    test_records:        list,
    test_gt:             list,
    seq_len:             int = 30,
    feature_window_size: int = 60,
    feature_min_samples: int = 3,
    max_gap_seconds:     float = 30.0,
    stride:              int = 1,
) -> tuple:
    """
    Full pipeline. Returns (train_split, val_split, test_split).

    Leakage guarantees:
    - Imputer and scaler fitted on HEALTHY TRAINING observations only.
    - Val and test receive frozen parameters.
    - Labels never enter feature tensors.
    - Sequences never cross vehicle boundaries or gaps > max_gap_seconds.
    - Test includes all observations (healthy + fault) for evaluation.
    """
    train_steps = build_labelled_steps(train_records, train_gt)
    val_steps   = build_labelled_steps(val_records,   val_gt)
    test_steps  = build_labelled_steps(test_records,  test_gt)

    train_obs = extract_observations(train_steps, feature_window_size, feature_min_samples, max_gap_seconds)
    val_obs   = extract_observations(val_steps,   feature_window_size, feature_min_samples, max_gap_seconds)
    test_obs  = extract_observations(test_steps,  feature_window_size, feature_min_samples, max_gap_seconds)

    train_healthy = [o for o in train_obs if o.is_healthy]
    val_healthy   = [o for o in val_obs   if o.is_healthy]

    if not train_healthy:
        raise ValueError("No healthy training observations found. Check split configuration.")
    if not val_healthy:
        raise ValueError("No healthy validation observations found. Check split configuration.")

    imputer = FeatureImputer().fit(train_healthy)
    scaler  = FeatureScaler().fit(train_healthy, imputer)

    train_seqs = build_sequences(train_healthy, seq_len, imputer, scaler, "train", max_gap_seconds, stride)
    val_seqs   = build_sequences(val_healthy,   seq_len, imputer, scaler, "val",   max_gap_seconds, stride)
    test_seqs  = build_sequences(test_obs,      seq_len, imputer, scaler, "test",  max_gap_seconds, stride)

    def _counts(seqs):
        h = sum(1 for s in seqs if not s.is_anomalous)
        a = sum(1 for s in seqs if s.is_anomalous)
        return h, a

    tr_h, tr_a = _counts(train_seqs)
    va_h, va_a = _counts(val_seqs)
    te_h, te_a = _counts(test_seqs)

    logger.info(
        "Dataset | train=%d(%dH) val=%d(%dH) test=%d(%dH/%dA)",
        len(train_seqs), tr_h, len(val_seqs), va_h, len(test_seqs), te_h, te_a,
    )

    return (
        DatasetSplit("train", train_seqs, imputer, scaler, LSTM_FEATURES, seq_len, tr_h, tr_a),
        DatasetSplit("val",   val_seqs,   imputer, scaler, LSTM_FEATURES, seq_len, va_h, va_a),
        DatasetSplit("test",  test_seqs,  imputer, scaler, LSTM_FEATURES, seq_len, te_h, te_a),
    )


# ---------------------------------------------------------------------------
# Standard three-split convenience function
# ---------------------------------------------------------------------------

def prepare_standard_splits(
    seq_len:             int = 30,
    train_steps:         int = 600,
    val_steps:           int = 600,
    test_steps:          int = 900,
    feature_window_size: int = 60,
    feature_min_samples: int = 3,
    max_gap_seconds:     float = 30.0,
    stride:              int = 1,
    train_seed:          int = 42,
    val_seed:            int = 43,
    test_seed:           int = 44,
) -> tuple:
    """
    Run the three standard simulations and return prepared splits.
    Train/val: normal mode (healthy only).
    Test: fault mode (healthy + faulty for evaluation).
    """
    from simulator.vehicle import DEFAULT_VEHICLES
    from simulator.faults import make_normal_scenarios, make_fault_scenarios
    from simulator.publisher import run_simulation

    tr_r, tr_fc = run_simulation(DEFAULT_VEHICLES, make_normal_scenarios(), train_steps, 1.0, train_seed)
    va_r, va_fc = run_simulation(DEFAULT_VEHICLES, make_normal_scenarios(), val_steps,   1.0, val_seed)
    te_r, te_fc = run_simulation(DEFAULT_VEHICLES, make_fault_scenarios(),  test_steps,  1.0, test_seed)

    return prepare_dataset(
        train_records=tr_r, train_gt=tr_fc.get_ground_truth_rows(),
        val_records=va_r,   val_gt=va_fc.get_ground_truth_rows(),
        test_records=te_r,  test_gt=te_fc.get_ground_truth_rows(),
        seq_len=seq_len,
        feature_window_size=feature_window_size,
        feature_min_samples=feature_min_samples,
        max_gap_seconds=max_gap_seconds,
        stride=stride,
    )
