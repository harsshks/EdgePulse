# tests/test_lstm_inference.py
# Stage D tests: threshold calibration, AnomalyScorer, pipeline integration.
# Uses tiny datasets and temp directories. No large model training.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import tempfile
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest
import torch

from lstm.dataset import (
    LSTM_FEATURES, N_FEATURES, LSTMObservation, LSTMSequence,
    DatasetSplit, FeatureImputer, FeatureScaler, build_sequences, is_healthy,
)
from lstm.model import build_model
from lstm.artifacts import save_model, load_model, artifact_exists
from lstm.train import train
from lstm.inference import (
    AnomalyScorer, LSTMAnomalyResult, calibrate_threshold,
    DEFAULT_THRESHOLD_PERCENTILE,
)

BASE_TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ts(i):
    return BASE_TS + timedelta(seconds=float(i))


def _obs(i, vid="V1", fault_type="NONE", fault_state="NORMAL"):
    arr = np.ones(N_FEATURES, dtype=np.float64) * (float(i) * 0.05 + 0.5)
    return LSTMObservation(
        vehicle_id=vid, event_timestamp=_ts(i),
        features=arr, fault_type=fault_type, fault_state=fault_state,
        severity=0.0, is_healthy=is_healthy(fault_type, fault_state),
    )


def _make_split(n=80, seq_len=5, split="train", faulty=False):
    obs = [_obs(i) for i in range(n)]
    if faulty:
        for i in range(n - 15, n):
            obs[i] = _obs(i, fault_type="COOLING_DEGRADATION", fault_state="ACTIVE")
    imp = FeatureImputer().fit(obs)
    sc = FeatureScaler().fit(obs, imp)
    seqs = build_sequences(obs, seq_len=seq_len, imputer=imp, scaler=sc, split=split)
    n_h = sum(1 for s in seqs if not s.is_anomalous)
    n_a = sum(1 for s in seqs if s.is_anomalous)
    return DatasetSplit(split, seqs, imp, sc, LSTM_FEATURES, seq_len, n_h, n_a)


def _train_tiny(tmpdir, seq_len=5, hidden_dim=8, epochs=3):
    tr = _make_split(n=80, seq_len=seq_len, split="train")
    va = _make_split(n=50, seq_len=seq_len, split="val")
    result = train(tr, va, seq_len=seq_len, hidden_dim=hidden_dim,
                   max_epochs=epochs, batch_size=8, seed=42, model_dir=tmpdir)
    return result, tr, va


class TestThresholdCalibration:

    def test_calibration_returns_positive_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            result, tr, va = _train_tiny(d)
            threshold = calibrate_threshold(
                val_sequences=va.sequences,
                model=result.model,
                imputer=tr.imputer,
                scaler=tr.scaler,
                percentile=95.0,
                model_dir=d,
            )
            assert threshold > 0.0

    def test_calibration_persisted_to_artifact(self):
        with tempfile.TemporaryDirectory() as d:
            result, tr, va = _train_tiny(d)
            threshold = calibrate_threshold(
                va.sequences, result.model, tr.imputer, tr.scaler,
                percentile=95.0, model_dir=d,
            )
            artifact = load_model(d)
            assert artifact["threshold"] == pytest.approx(threshold, rel=1e-5)

    def test_higher_percentile_gives_higher_threshold(self):
        with tempfile.TemporaryDirectory() as d:
            result, tr, va = _train_tiny(d)
            t50 = calibrate_threshold(
                va.sequences, result.model, tr.imputer, tr.scaler,
                percentile=50.0, model_dir=d,
            )
            t95 = calibrate_threshold(
                va.sequences, result.model, tr.imputer, tr.scaler,
                percentile=95.0, model_dir=d,
            )
            assert t95 >= t50

    def test_calibration_skips_anomalous_sequences(self):
        with tempfile.TemporaryDirectory() as d:
            result, tr, va = _train_tiny(d)
            faulty_split = _make_split(n=80, seq_len=5, split="val", faulty=True)
            mixed = va.sequences + faulty_split.sequences
            threshold = calibrate_threshold(
                mixed, result.model, tr.imputer, tr.scaler,
                percentile=95.0, model_dir=d,
            )
            assert threshold > 0.0

    def test_test_labels_cannot_influence_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            result, tr, va = _train_tiny(d)
            all_faulty = [
                LSTMSequence(
                    vehicle_id="V1", end_timestamp=_ts(i),
                    features=s.features,
                    fault_types=["COOLING_DEGRADATION"] * 5,
                    fault_states=["ACTIVE"] * 5,
                    is_anomalous=True, split="val",
                )
                for i, s in enumerate(va.sequences)
            ]
            with pytest.raises(ValueError, match="[Nn]o healthy"):
                calibrate_threshold(
                    all_faulty, result.model, tr.imputer, tr.scaler,
                    percentile=95.0, model_dir=d,
                )

    def test_empty_val_sequences_raises(self):
        with tempfile.TemporaryDirectory() as d:
            result, tr, va = _train_tiny(d)
            with pytest.raises(ValueError):
                calibrate_threshold(
                    [], result.model, tr.imputer, tr.scaler,
                    percentile=95.0, model_dir=d,
                )

    def test_invalid_percentile_raises(self):
        with tempfile.TemporaryDirectory() as d:
            result, tr, va = _train_tiny(d)
            with pytest.raises(ValueError, match="percentile"):
                calibrate_threshold(
                    va.sequences, result.model, tr.imputer, tr.scaler,
                    percentile=110.0, model_dir=d,
                )


class TestAnomalyScorerLoading:

    def test_disabled_scorer_is_not_ready(self):
        s = AnomalyScorer(enabled=False)
        assert not s.is_ready
        assert not s.model_loaded

    def test_enabled_scorer_no_artifact_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            s = AnomalyScorer(model_dir=d, enabled=True)
            assert not s.model_loaded

    def test_scorer_loads_model_on_init(self):
        with tempfile.TemporaryDirectory() as d:
            result, tr, va = _train_tiny(d)
            calibrate_threshold(
                va.sequences, result.model, tr.imputer, tr.scaler,
                percentile=95.0, model_dir=d,
            )
            s = AnomalyScorer(model_dir=d, enabled=True)
            assert s.model_loaded
            assert s.is_ready
            assert s.threshold is not None

    def test_model_not_loaded_per_message(self):
        with tempfile.TemporaryDirectory() as d:
            result, tr, va = _train_tiny(d)
            calibrate_threshold(
                va.sequences, result.model, tr.imputer, tr.scaler,
                percentile=95.0, model_dir=d,
            )
            s = AnomalyScorer(model_dir=d, enabled=True)
            model_id_before = id(s._model)
            fv = {k: 0.5 for k in LSTM_FEATURES}
            for i in range(20):
                s.push("V1", fv, _ts(i).isoformat())
            assert id(s._model) == model_id_before

    def test_wrong_feature_order_not_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            result, tr, va = _train_tiny(d)
            save_model(
                model=result.model,
                imputer=tr.imputer,
                scaler=tr.scaler,
                feature_names=list(reversed(LSTM_FEATURES)),
                seq_len=5,
                model_dir=d,
            )
            s = AnomalyScorer(model_dir=d, enabled=True)
            assert not s.model_loaded

    def test_missing_artifact_not_ready(self):
        with tempfile.TemporaryDirectory() as d:
            s = AnomalyScorer(model_dir=os.path.join(d, "nonexistent"), enabled=True)
            assert not s.model_loaded

    def test_reload_picks_up_new_model(self):
        with tempfile.TemporaryDirectory() as d:
            s = AnomalyScorer(model_dir=d, enabled=True)
            assert not s.model_loaded
            result, tr, va = _train_tiny(d)
            calibrate_threshold(
                va.sequences, result.model, tr.imputer, tr.scaler,
                percentile=95.0, model_dir=d,
            )
            s.reload()
            assert s.model_loaded


class TestAnomalyScorerScoring:

    def _ready_scorer(self, tmpdir, seq_len=5):
        result, tr, va = _train_tiny(tmpdir, seq_len=seq_len)
        calibrate_threshold(
            va.sequences, result.model, tr.imputer, tr.scaler,
            percentile=95.0, model_dir=tmpdir,
        )
        return AnomalyScorer(model_dir=tmpdir, enabled=True, seq_len=seq_len)

    def test_warming_up_before_seq_len(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._ready_scorer(d, seq_len=5)
            fv = {k: 0.5 for k in LSTM_FEATURES}
            for i in range(4):
                result = s.push("V1", fv, _ts(i).isoformat())
            assert result is not None
            assert result.status == "warming_up"

    def test_scores_after_seq_len_observations(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._ready_scorer(d, seq_len=5)
            fv = {k: 0.5 for k in LSTM_FEATURES}
            result = None
            for i in range(6):
                result = s.push("V1", fv, _ts(i).isoformat())
            assert result is not None
            assert result.status == "scored"
            assert result.reconstruction_error is not None
            assert result.is_anomalous is not None

    def test_reconstruction_error_nonnegative(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._ready_scorer(d, seq_len=5)
            fv = {k: 0.5 for k in LSTM_FEATURES}
            result = None
            for i in range(10):
                result = s.push("V1", fv, _ts(i).isoformat())
            assert result.reconstruction_error >= 0.0

    def test_result_has_all_fields(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._ready_scorer(d, seq_len=5)
            fv = {k: 0.5 for k in LSTM_FEATURES}
            for i in range(10):
                result = s.push("V1", fv, _ts(i).isoformat())
            assert result.vehicle_id == "V1"
            assert result.reconstruction_error is not None
            assert result.threshold is not None
            assert result.is_anomalous is not None
            assert result.model_version is not None
            assert result.status == "scored"

    def test_no_model_status(self):
        with tempfile.TemporaryDirectory() as d:
            s = AnomalyScorer(model_dir=d, enabled=True)
            fv = {k: 0.5 for k in LSTM_FEATURES}
            result = s.push("V1", fv, _ts(0).isoformat())
            assert result.status == "no_model"

    def test_disabled_scorer_returns_none(self):
        s = AnomalyScorer(enabled=False)
        result = s.push("V1", {k: 0.5 for k in LSTM_FEATURES}, _ts(0).isoformat())
        assert result is None


class TestPerVehicleBuffers:

    def _ready_scorer(self, tmpdir, seq_len=5):
        result, tr, va = _train_tiny(tmpdir, seq_len=seq_len)
        calibrate_threshold(
            va.sequences, result.model, tr.imputer, tr.scaler,
            percentile=95.0, model_dir=tmpdir,
        )
        return AnomalyScorer(model_dir=tmpdir, enabled=True, seq_len=seq_len)

    def test_independent_buffers_per_vehicle(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._ready_scorer(d, seq_len=5)
            fv = {k: 0.5 for k in LSTM_FEATURES}
            for i in range(3):
                s.push("V1", fv, _ts(i).isoformat())
                s.push("V2", fv, _ts(i).isoformat())
            assert s.buffer_size("V1") == 3
            assert s.buffer_size("V2") == 3

    def test_v1_buffer_does_not_affect_v2(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._ready_scorer(d, seq_len=5)
            fv = {k: 0.5 for k in LSTM_FEATURES}
            for i in range(6):
                s.push("V1", fv, _ts(i).isoformat())
            assert s.buffer_size("V2") == 0

    def test_duplicate_timestamp_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._ready_scorer(d, seq_len=5)
            fv = {k: 0.5 for k in LSTM_FEATURES}
            s.push("V1", fv, _ts(0).isoformat())
            s.push("V1", fv, _ts(1).isoformat())
            s.push("V1", fv, _ts(1).isoformat())
            assert s.buffer_size("V1") == 2

    def test_out_of_order_timestamp_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._ready_scorer(d, seq_len=5)
            fv = {k: 0.5 for k in LSTM_FEATURES}
            s.push("V1", fv, _ts(5).isoformat())
            s.push("V1", fv, _ts(3).isoformat())
            assert s.buffer_size("V1") == 1

    def test_excessive_gap_resets_buffer(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._ready_scorer(d, seq_len=5)
            s._max_gap_seconds = 5.0
            fv = {k: 0.5 for k in LSTM_FEATURES}
            for i in range(3):
                s.push("V1", fv, _ts(i).isoformat())
            assert s.buffer_size("V1") == 3
            s.push("V1", fv, _ts(103).isoformat())
            assert s.buffer_size("V1") == 1

    def test_buffer_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._ready_scorer(d, seq_len=5)
            fv = {k: 0.5 for k in LSTM_FEATURES}
            for i in range(50):
                s.push("V1", fv, _ts(i).isoformat())
            from lstm.inference import _BUFFER_MAX_MULTIPLIER
            assert s.buffer_size("V1") <= _BUFFER_MAX_MULTIPLIER * 5

    def test_none_features_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            s = self._ready_scorer(d, seq_len=5)
            fv_missing = {k: None for k in LSTM_FEATURES}
            result = s.push("V1", fv_missing, _ts(0).isoformat())
            assert result is None
            assert s.buffer_size("V1") == 0


class TestInferenceIsolation:

    def test_inference_exception_caught(self):
        with tempfile.TemporaryDirectory() as d:
            result, tr, va = _train_tiny(d)
            calibrate_threshold(
                va.sequences, result.model, tr.imputer, tr.scaler,
                percentile=95.0, model_dir=d,
            )
            s = AnomalyScorer(model_dir=d, enabled=True)
            # Patch scaler to raise on transform — triggers error during inference
            original_transform = s._scaler.transform
            def _broken_transform(arr):
                raise RuntimeError("injected scaler failure")
            s._scaler.transform = _broken_transform
            # Fill buffer to seq_len so scoring is attempted
            fv = {k: 0.5 for k in LSTM_FEATURES}
            result_obj = None
            for i in range(10):
                result_obj = s.push("V1", fv, _ts(i).isoformat())
            # At least one push after seq_len should return error
            assert result_obj is not None
            assert result_obj.status == "error"

    def test_pipeline_phase2_unaffected_when_lstm_disabled(self):
        from edge.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(lstm_enabled=False)
        record = {
            "message_id": "test-001",
            "vehicle_id": "CAR-01",
            "event_timestamp": "2026-01-01T12:00:00+00:00",
            "ingestion_timestamp": "2026-01-01T12:00:00+00:00",
            "speed": 60.0, "rpm": 3000.0, "engine_temp": 90.0,
            "battery_voltage": 12.6, "fuel_level": 80.0,
        }
        result = pipeline.process_record(record)
        assert result is not None
        fv, scores = result
        assert fv is not None
        assert isinstance(scores, list)

    def test_pipeline_with_lstm_enabled_no_artifact(self):
        from edge.pipeline import FeaturePipeline
        with tempfile.TemporaryDirectory() as d:
            pipeline = FeaturePipeline(lstm_enabled=True, lstm_model_dir=d)
            record = {
                "message_id": "test-002",
                "vehicle_id": "CAR-01",
                "event_timestamp": "2026-01-01T12:00:01+00:00",
                "ingestion_timestamp": "2026-01-01T12:00:01+00:00",
                "speed": 60.0, "rpm": 3000.0, "engine_temp": 90.0,
                "battery_voltage": 12.6, "fuel_level": 80.0,
            }
            result = pipeline.process_record(record)
            assert result is not None
            fv, scores = result
            assert fv is not None


class TestLSTMAPIEndpoints:

    def test_lstm_status_returns_200(self):
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        r = client.get("/lstm/status")
        assert r.status_code == 200

    def test_lstm_status_no_model(self):
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        r = client.get("/lstm/status")
        data = r.json()
        assert "model_loaded" in data
        assert "threshold_set" in data

    def test_lstm_anomaly_no_model_returns_status(self):
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        r = client.get("/lstm/anomaly/UNKNOWN-V1")
        assert r.status_code == 200
        assert r.json()["status"] in ("no_model", "warming_up", "no_threshold", "error")

    def test_lstm_anomaly_response_has_vehicle_id(self):
        from fastapi.testclient import TestClient
        from api.main import app
        client = TestClient(app)
        r = client.get("/lstm/anomaly/CAR-TEST")
        data = r.json()
        assert data["vehicle_id"] == "CAR-TEST"


class TestLSTMAnomalyResult:

    def test_to_dict_has_all_keys(self):
        r = LSTMAnomalyResult(
            vehicle_id="V1", end_timestamp="2026-01-01T00:00:00+00:00",
            reconstruction_error=0.05, threshold=0.1, is_anomalous=False,
            model_version="1.0", status="scored",
        )
        d = r.to_dict()
        for key in ["vehicle_id", "end_timestamp", "reconstruction_error",
                    "threshold", "is_anomalous", "model_version", "status"]:
            assert key in d

    def test_anomalous_when_error_exceeds_threshold(self):
        r = LSTMAnomalyResult(
            vehicle_id="V1", end_timestamp=None,
            reconstruction_error=0.2, threshold=0.1, is_anomalous=True,
            model_version="1.0", status="scored",
        )
        assert r.is_anomalous is True

    def test_normal_when_error_below_threshold(self):
        r = LSTMAnomalyResult(
            vehicle_id="V1", end_timestamp=None,
            reconstruction_error=0.05, threshold=0.1, is_anomalous=False,
            model_version="1.0", status="scored",
        )
        assert r.is_anomalous is False
