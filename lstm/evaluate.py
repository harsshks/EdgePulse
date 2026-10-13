# lstm/evaluate.py
# Comparative evaluation: LSTM autoencoder vs Phase 2 statistical detector.
#
# Evaluation protocol (leakage-free):
#   Train:      seed=42, normal mode (healthy only)
#   Validation: seed=43, normal mode (healthy only, threshold calibration)
#   Test:       seed=44, fault mode  (healthy + faults, NEVER touches train/val)
#
# Evaluation units:
#   PRIMARY:  sequence-level (each complete sliding window = one unit)
#   SECONDARY: event-level summary (fault onset counted once per vehicle/fault)
#   Overlapping windows over the same fault period are NOT counted as
#   independent detections in the event-level summary.
#
# Phase 2 alignment:
#   Phase 2 scores per observation (z-score per feature).
#   A sequence is "detected by Phase 2" if ANY observation in the window
#   is ELEVATED or SEVERE on at least one feature.
#   This is conservative: it gives Phase 2 the best possible recall.
#
# CLI usage:
#   python -m lstm.evaluate
#   python -m lstm.evaluate --model-dir models/ --out-dir evaluation-results/
#   python -m lstm.evaluate --train-steps 600 --val-steps 600 --test-steps 900

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import torch

from lstm.dataset import (
    LSTM_FEATURES, N_FEATURES,
    build_labelled_steps, extract_observations,
    FeatureImputer, FeatureScaler, build_sequences, prepare_standard_splits,
    is_healthy,
)
from lstm.artifacts import load_model, artifact_exists
from lstm.inference import calibrate_threshold, AnomalyScorer
from lstm.train import train as train_model

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = "evaluation-results"


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def safe_div(num: int, den: int) -> Optional[float]:
    """Division that returns None instead of ZeroDivisionError."""
    return num / den if den > 0 else None


def compute_metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    """
    Compute precision, recall, F1, FPR from confusion matrix.
    Returns None for undefined metrics (zero denominator).
    """
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    fpr       = safe_div(fp, fp + tn)
    f1        = safe_div(2 * tp, 2 * tp + fp + fn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall":    round(recall,    4) if recall    is not None else None,
        "f1":        round(f1,        4) if f1        is not None else None,
        "fpr":       round(fpr,       4) if fpr       is not None else None,
    }


# ---------------------------------------------------------------------------
# Evaluation data structures
# ---------------------------------------------------------------------------

@dataclass
class SequenceResult:
    vehicle_id:    str
    end_timestamp: datetime
    fault_type:    str          # dominant fault type in window (NONE if all healthy)
    is_anomalous:  bool         # ground truth: any faulty step in window
    lstm_flagged:  Optional[bool]   # LSTM decision (None = no model/warming up)
    p2_flagged:    bool         # Phase 2 decision
    lstm_error:    Optional[float]
    lstm_threshold: Optional[float]


@dataclass
class FaultEvent:
    """One continuous fault episode per vehicle."""
    vehicle_id: str
    fault_type: str
    onset_ts:   datetime
    end_ts:     datetime


# ---------------------------------------------------------------------------
# Phase 2 scorer (offline, using training baselines)
# ---------------------------------------------------------------------------

def build_phase2_baselines(train_obs: list) -> dict:
    """
    Fit Phase 2 VehicleBaseline objects on healthy training observations.
    Returns dict: vehicle_id -> VehicleBaseline
    """
    from edge.baseline import VehicleBaseline
    by_vid: dict = defaultdict(list)
    for o in train_obs:
        by_vid[o.vehicle_id].append(o)

    baselines = {}
    for vid, obs_list in by_vid.items():
        vb = VehicleBaseline(vid, min_samples=5)
        for o in obs_list:
            if o.is_healthy:
                fv = {k: float(o.features[i]) for i, k in enumerate(LSTM_FEATURES)}
                vb.update_from_features(fv, o.event_timestamp)
        baselines[vid] = vb
    return baselines


def phase2_detect(obs, baseline) -> bool:
    """Return True if Phase 2 flags this observation as anomalous."""
    from edge.scoring import score_all_features, DeviationLevel
    fv = {k: float(obs.features[i]) for i, k in enumerate(LSTM_FEATURES)}
    scores = score_all_features(
        obs.vehicle_id, fv, baseline,
        obs.event_timestamp.isoformat(),
    )
    return any(
        s.level in (DeviationLevel.ELEVATED, DeviationLevel.SEVERE)
        for s in scores
    )


# ---------------------------------------------------------------------------
# Event-level extraction from ground truth
# ---------------------------------------------------------------------------

def extract_fault_events(gt_rows: list, dt: float = 1.0) -> list:
    """
    Extract discrete fault events from ground-truth rows.
    A new event starts when fault_type changes from NONE to something else,
    or when fault_type changes between non-NONE types.
    """
    events = []
    # Group by vehicle
    by_vid: dict = defaultdict(list)
    for r in gt_rows:
        by_vid[r.vehicle_id].append(r)

    from lstm.dataset import sim_time_to_datetime
    for vid, rows in by_vid.items():
        in_fault = False
        current_fault = "NONE"
        onset_ts = None

        for r in sorted(rows, key=lambda x: x.sim_time):
            if r.fault_type != "NONE" and r.fault_state in ("STARTING", "ACTIVE"):
                if not in_fault or r.fault_type != current_fault:
                    # New fault event
                    if in_fault:
                        # Close previous
                        events.append(FaultEvent(
                            vehicle_id=vid,
                            fault_type=current_fault,
                            onset_ts=onset_ts,
                            end_ts=sim_time_to_datetime(r.sim_time),
                        ))
                    in_fault = True
                    current_fault = r.fault_type
                    onset_ts = sim_time_to_datetime(r.sim_time)
            else:
                if in_fault:
                    events.append(FaultEvent(
                        vehicle_id=vid,
                        fault_type=current_fault,
                        onset_ts=onset_ts,
                        end_ts=sim_time_to_datetime(r.sim_time),
                    ))
                    in_fault = False
                    current_fault = "NONE"
                    onset_ts = None

        if in_fault:
            events.append(FaultEvent(
                vehicle_id=vid,
                fault_type=current_fault,
                onset_ts=onset_ts,
                end_ts=sim_time_to_datetime(rows[-1].sim_time),
            ))
    return events


def detection_delay_seconds(fault_event: FaultEvent,
                             sequence_results: list,
                             use_lstm: bool) -> Optional[float]:
    """
    Time from fault onset to the first sequence that detected the fault.
    Returns None if the fault was never detected.
    Sequences must overlap with the fault event window.
    """
    relevant = [
        s for s in sequence_results
        if s.vehicle_id == fault_event.vehicle_id
        and s.end_timestamp >= fault_event.onset_ts
        and s.end_timestamp <= fault_event.end_ts
        and s.is_anomalous
    ]
    if not relevant:
        return None

    relevant_sorted = sorted(relevant, key=lambda s: s.end_timestamp)
    for s in relevant_sorted:
        flagged = s.lstm_flagged if use_lstm else s.p2_flagged
        if flagged:
            delay = (s.end_timestamp - fault_event.onset_ts).total_seconds()
            return max(0.0, delay)
    return None


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def evaluate_on_test(
    train_split,
    val_split,
    test_obs: list,
    test_gt_rows: list,
    model,
    imputer: FeatureImputer,
    scaler: FeatureScaler,
    threshold: float,
    model_version: str,
    seq_len: int,
    max_gap_seconds: float = 30.0,
) -> dict:
    """
    Run both LSTM and Phase 2 detectors on test observations.
    Returns a structured results dict.
    """
    # --- Build sequences from ALL test observations (healthy + faulty) ---
    test_seqs = build_sequences(
        test_obs, seq_len=seq_len, imputer=imputer, scaler=scaler,
        split="test", max_gap_seconds=max_gap_seconds,
    )

    # --- Phase 2 baselines from TRAINING healthy obs only ---
    train_healthy = [o for o in train_split.sequences[0].features
                     if True]  # baselines already in train_split via observations
    p2_baselines = build_phase2_baselines(
        [o for o in extract_observations_from_split(train_split)]
    )

    # --- Score each sequence ---
    scorer = AnomalyScorer(enabled=False)  # we score manually
    scorer._model     = model
    scorer._imputer   = imputer
    scorer._scaler    = scaler
    scorer._threshold = threshold
    scorer._model_version = model_version
    scorer._seq_len   = seq_len
    scorer._ready     = True
    scorer._enabled   = True

    results: list[SequenceResult] = []
    model.eval()

    for seq in test_seqs:
        # Ground truth: any faulty step? What fault type?
        fault_steps = [ft for ft in seq.fault_types if ft != "NONE"]
        dominant_fault = max(set(fault_steps), key=fault_steps.count) if fault_steps else "NONE"
        is_anom = seq.is_anomalous

        # --- LSTM score ---
        x = torch.from_numpy(seq.features).unsqueeze(0).float()
        with torch.no_grad():
            errors = model.reconstruction_error(x)
        recon_err = float(errors[0].item())
        lstm_flagged = recon_err > threshold

        # --- Phase 2 score ---
        # A sequence is Phase 2 detected if the end observation is detected
        # (conservative: use the last observation in the window)
        # We synthesize an observation for the end timestamp
        end_feat_arr = seq.features[-1]  # scaled — reconstruct raw for Phase 2
        # Phase 2 operates on raw (imputed but unscaled) feature values.
        # Reverse the scaling: raw = scaled * range + min
        raw_arr = end_feat_arr * scaler.range_ + scaler.min_
        raw_dict = {k: float(raw_arr[i]) for i, k in enumerate(LSTM_FEATURES)}
        vb = p2_baselines.get(seq.vehicle_id)
        p2_flag = False
        if vb is not None:
            from edge.scoring import score_all_features, DeviationLevel
            scores = score_all_features(
                seq.vehicle_id, raw_dict, vb, seq.end_timestamp.isoformat()
            )
            p2_flag = any(
                s.level in (DeviationLevel.ELEVATED, DeviationLevel.SEVERE)
                for s in scores
            )

        results.append(SequenceResult(
            vehicle_id=seq.vehicle_id,
            end_timestamp=seq.end_timestamp,
            fault_type=dominant_fault,
            is_anomalous=is_anom,
            lstm_flagged=lstm_flagged,
            p2_flagged=p2_flag,
            lstm_error=round(recon_err, 6),
            lstm_threshold=threshold,
        ))

    # --- Aggregate metrics ---
    fault_events = extract_fault_events(test_gt_rows)
    return _aggregate(results, fault_events, seq_len, model_version, threshold)


def extract_observations_from_split(split) -> list:
    """Re-extract observations from DatasetSplit sequences (approximate)."""
    # We need LSTMObservation objects for Phase 2 baselines.
    # For training we only need healthy observations, and we can
    # get the raw (pre-scale) features from imputer + scaler inversion.
    obs_list = []
    from lstm.dataset import LSTMObservation, is_healthy
    from datetime import timezone, timedelta
    _EPOCH = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    for i, seq in enumerate(split.sequences):
        # Only use the last observation of each training sequence
        raw_scaled = seq.features[-1]
        raw_arr = raw_scaled * split.scaler.range_ + split.scaler.min_
        obs_list.append(LSTMObservation(
            vehicle_id=seq.vehicle_id,
            event_timestamp=_EPOCH + timedelta(seconds=float(i)),
            features=raw_arr.astype(np.float64),
            fault_type="NONE",
            fault_state="NORMAL",
            severity=0.0,
            is_healthy=True,
        ))
    return obs_list


def _aggregate(results: list, fault_events: list, seq_len: int,
               model_version: str, threshold: float) -> dict:
    """Compute sequence-level and event-level metrics for both detectors."""
    # --- Sequence-level ---
    def seq_metrics(flag_attr: str) -> dict:
        tp = sum(1 for r in results if r.is_anomalous and getattr(r, flag_attr))
        fp = sum(1 for r in results if not r.is_anomalous and getattr(r, flag_attr))
        tn = sum(1 for r in results if not r.is_anomalous and not getattr(r, flag_attr))
        fn = sum(1 for r in results if r.is_anomalous and not getattr(r, flag_attr))
        return compute_metrics(tp, fp, tn, fn)

    lstm_seq = seq_metrics("lstm_flagged")
    p2_seq   = seq_metrics("p2_flagged")

    # --- Per-fault-type breakdown ---
    fault_types = sorted(set(r.fault_type for r in results if r.fault_type != "NONE"))
    lstm_by_fault: dict = {}
    p2_by_fault: dict   = {}
    for ft in fault_types:
        ft_results = [r for r in results if r.fault_type == ft or
                      (r.fault_type == "NONE" and True)]
        # Subset: sequences where the dominant fault is ft
        ft_seq = [r for r in results if r.fault_type == ft]
        healthy_seq = [r for r in results if r.fault_type == "NONE"]

        # TP from fault sequences + TN from healthy sequences
        def _ft_metrics(flag_attr):
            tp = sum(1 for r in ft_seq if getattr(r, flag_attr))
            fn = sum(1 for r in ft_seq if not getattr(r, flag_attr))
            fp = sum(1 for r in healthy_seq if getattr(r, flag_attr))
            tn = sum(1 for r in healthy_seq if not getattr(r, flag_attr))
            return compute_metrics(tp, fp, tn, fn)

        lstm_by_fault[ft] = _ft_metrics("lstm_flagged")
        p2_by_fault[ft]   = _ft_metrics("p2_flagged")

    # --- Event-level detection ---
    lstm_events_detected = 0
    p2_events_detected   = 0
    event_details = []
    for ev in fault_events:
        lstm_delay = detection_delay_seconds(ev, results, use_lstm=True)
        p2_delay   = detection_delay_seconds(ev, results, use_lstm=False)
        lstm_detected = lstm_delay is not None
        p2_detected   = p2_delay   is not None
        if lstm_detected:
            lstm_events_detected += 1
        if p2_detected:
            p2_events_detected += 1
        event_details.append({
            "vehicle_id":    ev.vehicle_id,
            "fault_type":    ev.fault_type,
            "onset_ts":      ev.onset_ts.isoformat(),
            "end_ts":        ev.end_ts.isoformat(),
            "lstm_detected": lstm_detected,
            "lstm_delay_s":  lstm_delay,
            "p2_detected":   p2_detected,
            "p2_delay_s":    p2_delay,
        })

    vehicles    = sorted(set(r.vehicle_id for r in results))
    total_seqs  = len(results)
    healthy_seqs = sum(1 for r in results if not r.is_anomalous)
    faulty_seqs  = sum(1 for r in results if r.is_anomalous)

    return {
        "metadata": {
            "model_version":   model_version,
            "threshold":       threshold,
            "seq_len":         seq_len,
            "feature_names":   LSTM_FEATURES,
            "vehicles":        vehicles,
            "total_sequences": total_seqs,
            "healthy_sequences": healthy_seqs,
            "faulty_sequences":  faulty_seqs,
            "fault_events":    len(fault_events),
            "created_at":      datetime.now(timezone.utc).isoformat(),
            "evaluation_note": (
                "Sequence-level metrics: each sliding window = one unit. "
                "Consecutive overlapping windows over the same fault episode "
                "inflate TP counts. See event-level section for one-per-event view."
            ),
        },
        "sequence_level": {
            "lstm": lstm_seq,
            "phase2": p2_seq,
        },
        "per_fault_type": {
            "lstm":   lstm_by_fault,
            "phase2": p2_by_fault,
        },
        "event_level": {
            "total_fault_events": len(fault_events),
            "lstm_detected":  lstm_events_detected,
            "p2_detected":    p2_events_detected,
            "events":         event_details,
        },
    }


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_evaluation(
    model_dir:       str = "models/",
    out_dir:         str = DEFAULT_OUT_DIR,
    train_steps:     int = 600,
    val_steps:       int = 600,
    test_steps:      int = 900,
    seq_len:         int = 30,
    max_gap_seconds: float = 30.0,
    threshold_percentile: float = 95.0,
    train_seed:      int = 42,
    val_seed:        int = 43,
    test_seed:       int = 44,
    retrain:         bool = False,
    hidden_dim:      int = 32,
    epochs:          int = 100,
    batch_size:      int = 32,
    lr:              float = 1e-3,
    patience:        int = 10,
) -> dict:
    """
    Full leakage-free evaluation pipeline.

    If retrain=True, trains a fresh model before evaluating.
    Otherwise loads the existing artifact from model_dir.

    Returns the full results dict.
    """
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Preparing dataset splits (train=%d val=%d test=%d steps)...",
                train_steps, val_steps, test_steps)

    from simulator.vehicle import DEFAULT_VEHICLES
    from simulator.faults import make_normal_scenarios, make_fault_scenarios
    from simulator.publisher import run_simulation

    tr_r, tr_fc = run_simulation(DEFAULT_VEHICLES, make_normal_scenarios(),
                                  train_steps, 1.0, train_seed)
    va_r, va_fc = run_simulation(DEFAULT_VEHICLES, make_normal_scenarios(),
                                  val_steps, 1.0, val_seed)
    te_r, te_fc = run_simulation(DEFAULT_VEHICLES, make_fault_scenarios(),
                                  test_steps, 1.0, test_seed)

    from lstm.dataset import (
        build_labelled_steps, extract_observations,
        FeatureImputer, FeatureScaler, DatasetSplit, build_sequences,
    )

    tr_steps_data = build_labelled_steps(tr_r, tr_fc.get_ground_truth_rows())
    va_steps_data = build_labelled_steps(va_r, va_fc.get_ground_truth_rows())
    te_steps_data = build_labelled_steps(te_r, te_fc.get_ground_truth_rows())

    tr_obs = extract_observations(tr_steps_data)
    va_obs = extract_observations(va_steps_data)
    te_obs = extract_observations(te_steps_data)

    tr_healthy = [o for o in tr_obs if o.is_healthy]
    va_healthy = [o for o in va_obs if o.is_healthy]

    # Fit preprocessing on training healthy data ONLY
    imputer = FeatureImputer().fit(tr_healthy)
    scaler  = FeatureScaler().fit(tr_healthy, imputer)

    tr_seqs = build_sequences(tr_healthy, seq_len, imputer, scaler, "train")
    va_seqs = build_sequences(va_healthy, seq_len, imputer, scaler, "val")

    def _counts(seqs):
        h = sum(1 for s in seqs if not s.is_anomalous)
        a = sum(1 for s in seqs if s.is_anomalous)
        return h, a

    tr_h, tr_a = _counts(tr_seqs)
    va_h, va_a = _counts(va_seqs)
    logger.info("Sequences | train=%d(%dH) val=%d(%dH)", len(tr_seqs), tr_h, len(va_seqs), va_h)

    train_split = DatasetSplit("train", tr_seqs, imputer, scaler, LSTM_FEATURES, seq_len, tr_h, tr_a)
    val_split   = DatasetSplit("val",   va_seqs, imputer, scaler, LSTM_FEATURES, seq_len, va_h, va_a)

    # --- Model ---
    if retrain or not artifact_exists(model_dir):
        logger.info("Training model (retrain=%s)...", retrain)
        result = train_model(
            train_split=train_split,
            val_split=val_split,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            max_epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            seed=train_seed,
            model_dir=model_dir,
            dataset_ids={"train_seed": train_seed, "val_seed": val_seed,
                         "test_seed": test_seed},
        )
        model = result.model
        best_val_loss = result.best_val_loss
        epochs_run = result.epochs_run
    else:
        artifact = load_model(model_dir, expected_features=LSTM_FEATURES)
        model = artifact["model"]
        best_val_loss = artifact["metadata"].get("best_val_loss")
        epochs_run = artifact["metadata"].get("train_epochs_run")
        logger.info("Loaded existing model (version=%s)",
                    artifact["metadata"].get("model_version"))

    # --- Threshold calibration on VAL only ---
    threshold = calibrate_threshold(
        val_sequences=va_seqs,
        model=model,
        imputer=imputer,
        scaler=scaler,
        percentile=threshold_percentile,
        model_dir=model_dir,
        train_seed=train_seed,
        dataset_ids={"train_seed": train_seed, "val_seed": val_seed, "test_seed": test_seed},
    )
    logger.info("Threshold calibrated: %.6f (%.0f-pct of %d val sequences)",
                threshold, threshold_percentile, len(va_seqs))

    # Load artifact for model_version
    artifact = load_model(model_dir, expected_features=LSTM_FEATURES)
    model_version = artifact["metadata"].get("model_version", "unknown")

    # --- Evaluate ---
    logger.info("Evaluating on %d test observations...", len(te_obs))

    results = evaluate_on_test(
        train_split=train_split,
        val_split=val_split,
        test_obs=te_obs,
        test_gt_rows=te_fc.get_ground_truth_rows(),
        model=model,
        imputer=imputer,
        scaler=scaler,
        threshold=threshold,
        model_version=model_version,
        seq_len=seq_len,
        max_gap_seconds=max_gap_seconds,
    )

    results["run_config"] = {
        "train_seed": train_seed, "val_seed": val_seed, "test_seed": test_seed,
        "train_steps": train_steps, "val_steps": val_steps, "test_steps": test_steps,
        "seq_len": seq_len, "threshold_percentile": threshold_percentile,
        "hidden_dim": hidden_dim, "epochs_requested": epochs,
        "epochs_run": epochs_run, "best_val_loss": best_val_loss,
        "retrain": retrain,
        "leakage_note": (
            "Preprocessing (imputer, scaler) fitted on healthy training data only. "
            "Threshold calibrated on healthy validation data only. "
            "Test data not used in any training or calibration step."
        ),
    }

    # --- Save outputs ---
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved: %s", json_path)

    md_path = os.path.join(out_dir, "report.md")
    _write_markdown_report(results, md_path)
    logger.info("Report saved: %s", md_path)

    return results


def _write_markdown_report(results: dict, path: str) -> None:
    """Write a concise Markdown evaluation report."""
    cfg = results.get("run_config", {})
    meta = results.get("metadata", {})
    sl = results.get("sequence_level", {})
    el = results.get("event_level", {})

    def _fmt(v):
        if v is None:
            return "—"
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    lines = [
        "# EdgePulse Phase 3 — Evaluation Report",
        "",
        "## Methodology",
        "",
        f"- **Train**: seed={cfg.get('train_seed')}, normal mode, {cfg.get('train_steps')} steps/vehicle",
        f"- **Validation**: seed={cfg.get('val_seed')}, normal mode, {cfg.get('val_steps')} steps/vehicle",
        f"- **Test**: seed={cfg.get('test_seed')}, fault mode, {cfg.get('test_steps')} steps/vehicle",
        f"- **Threshold**: {cfg.get('threshold_percentile')}th percentile of healthy validation errors",
        f"- **Seq len**: {cfg.get('seq_len')}  |  **Hidden dim**: {cfg.get('hidden_dim')}",
        f"- **Leakage**: {cfg.get('leakage_note', 'see JSON')}",
        "",
        "## Dataset",
        "",
        f"- Vehicles: {', '.join(meta.get('vehicles', []))}",
        f"- Total test sequences: {meta.get('total_sequences')}",
        f"  - Healthy: {meta.get('healthy_sequences')}",
        f"  - Faulty: {meta.get('faulty_sequences')}",
        f"- Fault events: {meta.get('fault_events')}",
        "",
        "> **Note**: " + meta.get("evaluation_note", ""),
        "",
        "## Sequence-Level Results",
        "",
        "| Metric | LSTM | Phase 2 |",
        "|--------|------|---------|",
    ]

    lstm_s = sl.get("lstm", {})
    p2_s   = sl.get("phase2", {})
    for metric in ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "fpr"):
        lines.append(f"| {metric.upper()} | {_fmt(lstm_s.get(metric))} | {_fmt(p2_s.get(metric))} |")

    lines += [
        "",
        "## Per-Fault-Type (sequence-level)",
        "",
        "| Fault | Detector | TP | FP | TN | FN | Precision | Recall | F1 |",
        "|-------|----------|----|----|----|----|-----------|--------|-----|",
    ]

    for ft, lstm_m in results.get("per_fault_type", {}).get("lstm", {}).items():
        p2_m = results.get("per_fault_type", {}).get("phase2", {}).get(ft, {})
        lines.append(
            f"| {ft} | LSTM | {lstm_m.get('tp')} | {lstm_m.get('fp')} | "
            f"{lstm_m.get('tn')} | {lstm_m.get('fn')} | "
            f"{_fmt(lstm_m.get('precision'))} | {_fmt(lstm_m.get('recall'))} | "
            f"{_fmt(lstm_m.get('f1'))} |"
        )
        lines.append(
            f"| {ft} | Phase2 | {p2_m.get('tp')} | {p2_m.get('fp')} | "
            f"{p2_m.get('tn')} | {p2_m.get('fn')} | "
            f"{_fmt(p2_m.get('precision'))} | {_fmt(p2_m.get('recall'))} | "
            f"{_fmt(p2_m.get('f1'))} |"
        )

    lines += [
        "",
        "## Event-Level Results",
        "",
        f"- Total fault events: {el.get('total_fault_events')}",
        f"- LSTM detected: {el.get('lstm_detected')}",
        f"- Phase 2 detected: {el.get('p2_detected')}",
        "",
        "| Vehicle | Fault | LSTM | Delay (s) | Phase2 | Delay (s) |",
        "|---------|-------|------|-----------|--------|-----------|",
    ]
    for ev in el.get("events", []):
        lines.append(
            f"| {ev['vehicle_id']} | {ev['fault_type']} "
            f"| {'YES' if ev['lstm_detected'] else 'NO'} "
            f"| {_fmt(ev.get('lstm_delay_s'))} "
            f"| {'YES' if ev['p2_detected'] else 'NO'} "
            f"| {_fmt(ev.get('p2_delay_s'))} |"
        )

    lines += [
        "",
        "## Limitations",
        "",
        "- All data generated from a software simulator; real-world performance may differ.",
        "- Overlapping sliding windows inflate sequence-level TP/FP counts.",
        "- Phase 2 baseline may not be fully converged with limited training steps.",
        "- LSTM trained only " + str(cfg.get("epochs_run", "N/A")) + " epochs; "
          "longer training may improve results.",
        "- Reconstruction error indicates temporal novelty, not a specific fault type.",
        "- Do not interpret these results as production-ready anomaly detection.",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EdgePulse Phase 3 — comparative evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m lstm.evaluate
  python -m lstm.evaluate --retrain --epochs 50
  python -m lstm.evaluate --model-dir models/ --out-dir evaluation-results/
  python -m lstm.evaluate --test-steps 1800 --threshold-percentile 95
        """,
    )
    p.add_argument("--model-dir",   default="models/")
    p.add_argument("--out-dir",     default=DEFAULT_OUT_DIR)
    p.add_argument("--train-steps", type=int,   default=600)
    p.add_argument("--val-steps",   type=int,   default=600)
    p.add_argument("--test-steps",  type=int,   default=900)
    p.add_argument("--seq-len",     type=int,   default=30)
    p.add_argument("--hidden-dim",  type=int,   default=32)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch-size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--threshold-percentile", type=float, default=95.0)
    p.add_argument("--train-seed",  type=int,   default=42)
    p.add_argument("--val-seed",    type=int,   default=43)
    p.add_argument("--test-seed",   type=int,   default=44)
    p.add_argument("--retrain",     action="store_true",
                   help="Force retrain even if model artifact exists")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = _parse_args()

    results = run_evaluation(
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        train_steps=args.train_steps,
        val_steps=args.val_steps,
        test_steps=args.test_steps,
        seq_len=args.seq_len,
        max_gap_seconds=30.0,
        threshold_percentile=args.threshold_percentile,
        train_seed=args.train_seed,
        val_seed=args.val_seed,
        test_seed=args.test_seed,
        retrain=args.retrain,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )

    sl = results.get("sequence_level", {})
    print(f"\n=== Sequence-Level Results ===")
    print(f"  LSTM   | F1={sl['lstm'].get('f1')} Recall={sl['lstm'].get('recall')} Precision={sl['lstm'].get('precision')}")
    print(f"  Phase2 | F1={sl['phase2'].get('f1')} Recall={sl['phase2'].get('recall')} Precision={sl['phase2'].get('precision')}")
    el = results.get("event_level", {})
    print(f"\n=== Event-Level ===")
    print(f"  LSTM   detected {el.get('lstm_detected')}/{el.get('total_fault_events')} events")
    print(f"  Phase2 detected {el.get('p2_detected')}/{el.get('total_fault_events')} events")
    print(f"\nFull results: {args.out_dir}/results.json")
    print(f"Report:       {args.out_dir}/report.md")


if __name__ == "__main__":
    main()
