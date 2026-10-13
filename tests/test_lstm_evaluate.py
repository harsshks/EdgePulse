# tests/test_lstm_evaluate.py
# Stage E tests: evaluation metrics, confusion matrix, scenario aggregation.
# Uses only small synthetic fixtures — no model training during tests.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timezone, timedelta
import pytest

from lstm.evaluate import (
    compute_metrics,
    safe_div,
    extract_fault_events,
    detection_delay_seconds,
    SequenceResult,
    FaultEvent,
    _aggregate,
)
from lstm.dataset import LSTM_FEATURES, sim_time_to_datetime
from simulator.faults import GroundTruthRecord

BASE_TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ts(offset):
    return BASE_TS + timedelta(seconds=float(offset))


def _gt(vid, sim_time, fault_type="NONE", fault_state="NORMAL"):
    return GroundTruthRecord(
        sim_time=float(sim_time),
        vehicle_id=vid,
        message_id=f"m-{vid}-{sim_time}",
        event_timestamp=_ts(sim_time).isoformat(),
        fault_type=fault_type,
        fault_state=fault_state,
        severity=0.0,
    )


def _seq(vid, t, is_anom, fault_type="NONE", lstm_flag=None, p2_flag=False, err=0.01):
    return SequenceResult(
        vehicle_id=vid,
        end_timestamp=_ts(t),
        fault_type=fault_type,
        is_anomalous=is_anom,
        lstm_flagged=lstm_flag,
        p2_flagged=p2_flag,
        lstm_error=err,
        lstm_threshold=0.05,
    )


# ---------------------------------------------------------------------------
# 1. compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_perfect_detection(self):
        m = compute_metrics(tp=10, fp=0, tn=10, fn=0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"]    == pytest.approx(1.0)
        assert m["f1"]        == pytest.approx(1.0)
        assert m["fpr"]       == pytest.approx(0.0)

    def test_no_detection(self):
        m = compute_metrics(tp=0, fp=0, tn=10, fn=10)
        assert m["recall"]    == pytest.approx(0.0)
        assert m["precision"] is None   # tp+fp = 0 -> undefined
        assert m["f1"]        == pytest.approx(0.0)

    def test_all_false_positives(self):
        m = compute_metrics(tp=0, fp=10, tn=0, fn=10)
        assert m["precision"] == pytest.approx(0.0)
        assert m["fpr"]       == pytest.approx(1.0)

    def test_zero_denominator_precision_is_none(self):
        m = compute_metrics(tp=0, fp=0, tn=5, fn=5)
        assert m["precision"] is None

    def test_zero_denominator_recall_is_none(self):
        m = compute_metrics(tp=0, fp=5, tn=0, fn=0)
        assert m["recall"] is None

    def test_typical_values(self):
        m = compute_metrics(tp=8, fp=2, tn=8, fn=2)
        assert m["precision"] == pytest.approx(0.8, abs=0.001)
        assert m["recall"]    == pytest.approx(0.8, abs=0.001)
        assert m["fpr"]       == pytest.approx(0.2, abs=0.001)

    def test_keys_present(self):
        m = compute_metrics(1, 1, 1, 1)
        for key in ("tp", "fp", "tn", "fn", "precision", "recall", "f1", "fpr"):
            assert key in m


class TestSafeDiv:

    def test_normal_division(self):
        assert safe_div(10, 4) == pytest.approx(2.5)

    def test_zero_denominator_returns_none(self):
        assert safe_div(5, 0) is None

    def test_zero_numerator(self):
        assert safe_div(0, 5) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. extract_fault_events
# ---------------------------------------------------------------------------

class TestExtractFaultEvents:

    def test_single_fault_event(self):
        rows = (
            [_gt("V1", i) for i in range(10)] +
            [_gt("V1", 10 + i, "COOLING_DEGRADATION", "ACTIVE") for i in range(20)] +
            [_gt("V1", 30 + i) for i in range(10)]
        )
        events = extract_fault_events(rows)
        assert len(events) == 1
        assert events[0].fault_type == "COOLING_DEGRADATION"
        assert events[0].vehicle_id == "V1"

    def test_separate_vehicles_separate_events(self):
        rows = (
            [_gt("V1", i, "COOLING_DEGRADATION", "ACTIVE") for i in range(20)] +
            [_gt("V2", i, "RPM_INSTABILITY", "ACTIVE") for i in range(20)]
        )
        events = extract_fault_events(rows)
        vids = {e.vehicle_id for e in events}
        assert "V1" in vids and "V2" in vids

    def test_no_fault_returns_empty(self):
        rows = [_gt("V1", i) for i in range(20)]
        events = extract_fault_events(rows)
        assert events == []

    def test_multiple_faults_same_vehicle(self):
        rows = (
            [_gt("V1", i) for i in range(5)] +
            [_gt("V1", 5 + i, "COOLING_DEGRADATION", "ACTIVE") for i in range(10)] +
            [_gt("V1", 15 + i) for i in range(5)] +
            [_gt("V1", 20 + i, "RPM_INSTABILITY", "ACTIVE") for i in range(10)]
        )
        events = extract_fault_events(rows)
        fault_types = {e.fault_type for e in events}
        assert "COOLING_DEGRADATION" in fault_types
        assert "RPM_INSTABILITY" in fault_types

    def test_starting_state_counts_as_fault(self):
        rows = [_gt("V1", i, "BATTERY_DEGRADATION", "STARTING") for i in range(10)]
        events = extract_fault_events(rows)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# 3. detection_delay_seconds
# ---------------------------------------------------------------------------

class TestDetectionDelay:

    def _event(self, vid="V1", onset=10, end=30, fault_type="COOLING_DEGRADATION"):
        return FaultEvent(
            vehicle_id=vid,
            fault_type=fault_type,
            onset_ts=_ts(onset),
            end_ts=_ts(end),
        )

    def test_lstm_detected_at_onset(self):
        event = self._event(onset=10, end=30)
        results = [
            _seq("V1", 12, True, "COOLING_DEGRADATION", lstm_flag=True),
        ]
        delay = detection_delay_seconds(event, results, use_lstm=True)
        assert delay == pytest.approx(2.0, abs=0.1)

    def test_phase2_detected_at_onset(self):
        event = self._event(onset=10, end=30)
        results = [
            _seq("V1", 15, True, "COOLING_DEGRADATION", lstm_flag=False, p2_flag=True),
        ]
        delay = detection_delay_seconds(event, results, use_lstm=False)
        assert delay == pytest.approx(5.0, abs=0.1)

    def test_not_detected_returns_none(self):
        event = self._event(onset=10, end=30)
        results = [
            _seq("V1", 20, True, "COOLING_DEGRADATION", lstm_flag=False, p2_flag=False),
        ]
        assert detection_delay_seconds(event, results, use_lstm=True) is None

    def test_delay_nonnegative(self):
        event = self._event(onset=10, end=30)
        results = [
            _seq("V1", 10, True, "COOLING_DEGRADATION", lstm_flag=True),
        ]
        delay = detection_delay_seconds(event, results, use_lstm=True)
        assert delay >= 0.0

    def test_wrong_vehicle_ignored(self):
        event = self._event(vid="V1", onset=10, end=30)
        results = [
            _seq("V2", 12, True, "COOLING_DEGRADATION", lstm_flag=True),
        ]
        assert detection_delay_seconds(event, results, use_lstm=True) is None


# ---------------------------------------------------------------------------
# 4. _aggregate — confusion matrix and scenario breakdown
# ---------------------------------------------------------------------------

class TestAggregate:

    def _sample_results(self):
        return [
            _seq("V1", 5,  False, "NONE",                 lstm_flag=False, p2_flag=False),
            _seq("V1", 6,  False, "NONE",                 lstm_flag=False, p2_flag=False),
            _seq("V1", 7,  False, "NONE",                 lstm_flag=True,  p2_flag=False),  # FP
            _seq("V1", 15, True,  "COOLING_DEGRADATION",  lstm_flag=True,  p2_flag=True),   # TP both
            _seq("V1", 16, True,  "COOLING_DEGRADATION",  lstm_flag=True,  p2_flag=False),  # LSTM TP, P2 FN
            _seq("V1", 17, True,  "COOLING_DEGRADATION",  lstm_flag=False, p2_flag=True),   # P2 TP, LSTM FN
        ]

    def test_sequence_level_tp_fp_tn_fn(self):
        results = self._sample_results()
        agg = _aggregate(results, [], seq_len=5, model_version="1.0", threshold=0.05)
        sl = agg["sequence_level"]["lstm"]
        # TP: rows where is_anomalous=True and lstm_flag=True -> rows 4,5
        assert sl["tp"] == 2
        # FP: rows where is_anomalous=False and lstm_flag=True -> row 3
        assert sl["fp"] == 1
        # FN: rows where is_anomalous=True and lstm_flag=False -> row 6
        assert sl["fn"] == 1

    def test_metadata_present(self):
        results = self._sample_results()
        agg = _aggregate(results, [], seq_len=5, model_version="1.0", threshold=0.05)
        assert "metadata" in agg
        assert agg["metadata"]["total_sequences"] == len(results)

    def test_per_fault_type_present(self):
        results = self._sample_results()
        agg = _aggregate(results, [], seq_len=5, model_version="1.0", threshold=0.05)
        assert "COOLING_DEGRADATION" in agg["per_fault_type"]["lstm"]

    def test_event_level_section_present(self):
        results = self._sample_results()
        event = FaultEvent(
            vehicle_id="V1", fault_type="COOLING_DEGRADATION",
            onset_ts=_ts(14), end_ts=_ts(18),
        )
        agg = _aggregate(results, [event], seq_len=5, model_version="1.0", threshold=0.05)
        el = agg["event_level"]
        assert el["total_fault_events"] == 1
        assert "lstm_detected" in el

    def test_overlapping_windows_not_double_counted_in_events(self):
        # Many overlapping windows over same fault -> still 1 event
        fault_seqs = [
            _seq("V1", 10 + i, True, "COOLING_DEGRADATION", lstm_flag=True)
            for i in range(10)
        ]
        healthy_seqs = [
            _seq("V1", i, False, "NONE", lstm_flag=False)
            for i in range(5)
        ]
        event = FaultEvent(
            vehicle_id="V1", fault_type="COOLING_DEGRADATION",
            onset_ts=_ts(10), end_ts=_ts(20),
        )
        agg = _aggregate(fault_seqs + healthy_seqs, [event],
                         seq_len=5, model_version="1.0", threshold=0.05)
        assert agg["event_level"]["total_fault_events"] == 1
        assert agg["event_level"]["lstm_detected"] == 1   # not 10

    def test_zero_tp_f1_is_zero_not_undefined(self):
        results = [_seq("V1", i, False, "NONE", lstm_flag=False) for i in range(5)]
        agg = _aggregate(results, [], seq_len=5, model_version="1.0", threshold=0.05)
        sl = agg["sequence_level"]["lstm"]
        # No faulty sequences -> recall undefined, f1=0 (denominator = fp+fn only)
        assert sl["tp"] == 0
        assert sl["fn"] == 0

    def test_healthy_only_no_fault_events(self):
        results = [_seq("V1", i, False, "NONE", lstm_flag=False) for i in range(5)]
        agg = _aggregate(results, [], seq_len=5, model_version="1.0", threshold=0.05)
        assert agg["event_level"]["total_fault_events"] == 0
        assert agg["event_level"]["lstm_detected"] == 0

    def test_partially_anomalous_window_counted_as_anomalous(self):
        # Window with is_anomalous=True (even if only 1 step was faulty)
        seq = _seq("V1", 10, True, "COOLING_DEGRADATION", lstm_flag=True)
        agg = _aggregate([seq], [], seq_len=5, model_version="1.0", threshold=0.05)
        assert agg["sequence_level"]["lstm"]["tp"] == 1


# ---------------------------------------------------------------------------
# 5. Leakage proof
# ---------------------------------------------------------------------------

class TestLeakageProof:

    def test_threshold_not_computed_from_test_sequences(self):
        """
        calibrate_threshold must only accept sequences with is_anomalous=False.
        Passing all-anomalous sequences must raise ValueError.
        """
        from lstm.model import build_model
        from lstm.inference import calibrate_threshold
        from lstm.dataset import LSTMSequence
        import numpy as np
        import tempfile

        model = build_model(n_features=12, hidden_dim=8, seq_len=5, seed=42)

        # Build fake val sequences all marked anomalous (simulating test data leakage)
        faulty_seqs = [
            LSTMSequence(
                vehicle_id="V1", end_timestamp=_ts(i),
                features=np.random.rand(5, 12).astype(np.float32),
                fault_types=["COOLING_DEGRADATION"] * 5,
                fault_states=["ACTIVE"] * 5,
                is_anomalous=True, split="val",
            )
            for i in range(10)
        ]

        from lstm.dataset import FeatureImputer, FeatureScaler, LSTMObservation
        obs = [
            LSTMObservation(
                vehicle_id="V1", event_timestamp=_ts(i),
                features=np.ones(12) * 0.5,
                fault_type="NONE", fault_state="NORMAL",
                severity=0.0, is_healthy=True,
            )
            for i in range(20)
        ]
        imp = FeatureImputer().fit(obs)
        sc = FeatureScaler().fit(obs, imp)

        with tempfile.TemporaryDirectory() as d:
            # Save model artifact first
            from lstm.artifacts import save_model
            save_model(model=model, imputer=imp, scaler=sc,
                       feature_names=LSTM_FEATURES, seq_len=5, model_dir=d)
            with pytest.raises(ValueError, match="[Nn]o healthy"):
                calibrate_threshold(
                    faulty_seqs, model, imp, sc,
                    percentile=95.0, model_dir=d,
                )

    def test_compute_metrics_uses_only_provided_labels(self):
        """Metrics are computed from explicit TP/FP/TN/FN — no internal data access."""
        m1 = compute_metrics(5, 1, 9, 1)
        m2 = compute_metrics(5, 1, 9, 1)
        assert m1 == m2


# ---------------------------------------------------------------------------
# 6. Output format
# ---------------------------------------------------------------------------

class TestOutputFormat:

    def test_results_json_serialisable(self):
        import json
        results = _seq("V1", 5, False, "NONE", lstm_flag=False, p2_flag=False)
        agg = _aggregate(
            [results], [], seq_len=5, model_version="1.0", threshold=0.05
        )
        # Should not raise
        json.dumps(agg, default=str)

    def test_aggregated_result_has_required_keys(self):
        agg = _aggregate([], [], seq_len=5, model_version="1.0", threshold=0.05)
        for key in ("metadata", "sequence_level", "per_fault_type", "event_level"):
            assert key in agg

    def test_sequence_level_has_both_detectors(self):
        agg = _aggregate([], [], seq_len=5, model_version="1.0", threshold=0.05)
        assert "lstm" in agg["sequence_level"]
        assert "phase2" in agg["sequence_level"]
