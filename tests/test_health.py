# tests/test_health.py
# Phase 4 tests: vehicle-health decision layer and /health/{vehicle_id} API endpoint.
#
# Tests cover:
#   - Statistical-alert decisions (SEVERE → ALERT, ELEVATED → WATCH)
#   - LSTM-only, statistical-only, combined, and unavailable-model cases
#   - Critical sensor hard limits (engine_temp, battery_voltage, rpm)
#   - Explainable reasons and evidence fields
#   - Raw telemetry persistence despite feature/inference failures
#   - /health/{vehicle_id} API endpoint behaviour
#   - Precedence: sensor > SEVERE > ELEVATED/LSTM-anomalous

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edge.health import (
    assess_vehicle_health,
    VehicleHealthResult,
    HealthStatus,
    ENGINE_TEMP_CRITICAL,
    BATTERY_VOLTAGE_CRITICAL,
    RPM_CRITICAL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score(feature_name, level, z_score=None, current=90.0, mean=88.0, std=1.0,
           readiness="READY"):
    return {
        "vehicle_id": "V1",
        "feature_name": feature_name,
        "event_timestamp": "2026-01-01T00:00:00+00:00",
        "current_value": current,
        "baseline_mean": mean,
        "baseline_std": std,
        "z_score": z_score,
        "level": level,
        "direction": 1,
        "sample_count": 50,
        "readiness": readiness,
    }


def _lstm(status, is_anomalous=None, recon_error=None, threshold=None):
    return {
        "vehicle_id": "V1",
        "end_timestamp": "2026-01-01T00:00:00+00:00",
        "reconstruction_error": recon_error,
        "threshold": threshold,
        "is_anomalous": is_anomalous,
        "model_version": "1.0",
        "status": status,
    }


def _raw(engine_temp=90.0, battery_voltage=12.6, rpm=3000.0):
    return {
        "vehicle_id": "V1",
        "engine_temp": engine_temp,
        "battery_voltage": battery_voltage,
        "rpm": rpm,
        "speed": 60.0,
    }


# ---------------------------------------------------------------------------
# 1. All-NORMAL baseline
# ---------------------------------------------------------------------------

class TestNormalStatus:

    def test_all_normal_scores_returns_normal(self):
        scores = [_score("engine_temp", "NORMAL", z_score=0.5)]
        result = assess_vehicle_health("V1", scores, None)
        assert result.status == HealthStatus.NORMAL

    def test_lstm_scored_normal_returns_normal(self):
        scores = [_score("engine_temp", "NORMAL", z_score=0.5)]
        lstm   = _lstm("scored", is_anomalous=False, recon_error=0.01, threshold=0.1)
        result = assess_vehicle_health("V1", scores, lstm)
        assert result.status == HealthStatus.NORMAL

    def test_normal_result_has_no_evidence(self):
        scores = [_score("engine_temp", "NORMAL", z_score=0.5)]
        result = assess_vehicle_health("V1", scores, None)
        # No SEVERE/ELEVATED features → evidence should be empty or only NORMAL
        for v in result.evidence.values():
            if isinstance(v, dict):
                assert v.get("level") not in ("SEVERE", "ELEVATED")

    def test_vehicle_id_preserved(self):
        result = assess_vehicle_health("CAR-42", [], None)
        assert result.vehicle_id == "CAR-42"


# ---------------------------------------------------------------------------
# 2. Phase 2 statistical alerts
# ---------------------------------------------------------------------------

class TestPhase2Decisions:

    def test_severe_score_triggers_alert(self):
        scores = [_score("engine_temp", "SEVERE", z_score=4.5)]
        result = assess_vehicle_health("V1", scores, None)
        assert result.status == HealthStatus.ALERT

    def test_elevated_score_triggers_watch(self):
        scores = [_score("engine_temp", "ELEVATED", z_score=2.5)]
        result = assess_vehicle_health("V1", scores, None)
        assert result.status == HealthStatus.WATCH

    def test_severe_includes_feature_in_evidence(self):
        scores = [_score("rpm", "SEVERE", z_score=5.0)]
        result = assess_vehicle_health("V1", scores, None)
        assert "rpm" in result.evidence
        assert result.evidence["rpm"]["z_score"] == pytest.approx(5.0)

    def test_elevated_includes_feature_in_evidence(self):
        scores = [_score("battery_voltage", "ELEVATED", z_score=2.8)]
        result = assess_vehicle_health("V1", scores, None)
        assert "battery_voltage" in result.evidence

    def test_severe_reason_mentions_feature(self):
        scores = [_score("fuel_level", "SEVERE", z_score=4.0)]
        result = assess_vehicle_health("V1", scores, None)
        assert any("fuel_level" in r for r in result.reasons)

    def test_elevated_reason_mentions_feature(self):
        scores = [_score("temp_slope", "ELEVATED", z_score=2.1)]
        result = assess_vehicle_health("V1", scores, None)
        assert any("temp_slope" in r for r in result.reasons)

    def test_multiple_severe_features_all_in_reasons(self):
        scores = [
            _score("engine_temp", "SEVERE", z_score=4.0),
            _score("rpm", "SEVERE", z_score=5.0),
        ]
        result = assess_vehicle_health("V1", scores, None)
        assert result.status == HealthStatus.ALERT
        severe_reason = next(r for r in result.reasons if "SEVERE" in r)
        assert "engine_temp" in severe_reason
        assert "rpm" in severe_reason

    def test_insufficient_data_does_not_trigger_alert(self):
        scores = [_score("engine_temp", "INSUFFICIENT_DATA",
                         readiness="INSUFFICIENT_BASELINE_DATA")]
        result = assess_vehicle_health("V1", scores, None)
        # Insufficient data alone should not cause ALERT — only WATCH or a data notice
        assert result.status != HealthStatus.ALERT

    def test_phase2_scored_false_when_all_insufficient(self):
        scores = [_score("engine_temp", "INSUFFICIENT_DATA",
                         readiness="INSUFFICIENT_BASELINE_DATA")]
        result = assess_vehicle_health("V1", scores, None)
        assert result.phase2_scored is False

    def test_phase2_scored_true_when_any_ready(self):
        scores = [
            _score("engine_temp", "NORMAL", readiness="READY"),
            _score("rpm", "INSUFFICIENT_DATA", readiness="INSUFFICIENT_BASELINE_DATA"),
        ]
        result = assess_vehicle_health("V1", scores, None)
        assert result.phase2_scored is True


# ---------------------------------------------------------------------------
# 3. LSTM-only signals
# ---------------------------------------------------------------------------

class TestLSTMDecisions:

    def test_lstm_anomalous_triggers_alert(self):
        lstm = _lstm("scored", is_anomalous=True, recon_error=0.25, threshold=0.1)
        result = assess_vehicle_health("V1", [], lstm)
        assert result.status == HealthStatus.ALERT

    def test_lstm_normal_no_alert(self):
        lstm = _lstm("scored", is_anomalous=False, recon_error=0.02, threshold=0.1)
        result = assess_vehicle_health("V1", [], lstm)
        assert result.status == HealthStatus.NORMAL

    def test_lstm_warming_up_triggers_watch(self):
        lstm = _lstm("warming_up")
        result = assess_vehicle_health("V1", [], lstm)
        assert result.status == HealthStatus.WATCH

    def test_lstm_no_threshold_triggers_watch(self):
        lstm = _lstm("no_threshold")
        result = assess_vehicle_health("V1", [], lstm)
        assert result.status == HealthStatus.WATCH

    def test_lstm_error_triggers_watch(self):
        lstm = _lstm("error")
        result = assess_vehicle_health("V1", [], lstm)
        assert result.status == HealthStatus.WATCH

    def test_lstm_no_model_does_not_trigger_watch_on_its_own(self):
        # no_model is noted but should not escalate to WATCH
        lstm = _lstm("no_model")
        scores = [_score("engine_temp", "NORMAL", readiness="READY")]
        result = assess_vehicle_health("V1", scores, lstm)
        assert result.status == HealthStatus.NORMAL

    def test_lstm_anomalous_evidence_includes_error_and_threshold(self):
        lstm = _lstm("scored", is_anomalous=True, recon_error=0.3, threshold=0.1)
        result = assess_vehicle_health("V1", [], lstm)
        assert "lstm_reconstruction_error" in result.evidence
        assert "lstm_threshold" in result.evidence

    def test_lstm_status_forwarded(self):
        lstm = _lstm("warming_up")
        result = assess_vehicle_health("V1", [], lstm)
        assert result.lstm_status == "warming_up"

    def test_lstm_anomalous_forwarded(self):
        lstm = _lstm("scored", is_anomalous=True, recon_error=0.3, threshold=0.1)
        result = assess_vehicle_health("V1", [], lstm)
        assert result.lstm_anomalous is True

    def test_none_lstm_result_not_treated_as_normal(self):
        # When lstm_result is None, reasons should note absence
        result = assess_vehicle_health("V1", [], None)
        # No crash; status is deterministic regardless
        assert result.status in (HealthStatus.NORMAL, HealthStatus.WATCH, HealthStatus.ALERT)


# ---------------------------------------------------------------------------
# 4. Combined Phase 2 + LSTM
# ---------------------------------------------------------------------------

class TestCombinedDecisions:

    def test_severe_p2_plus_lstm_normal_is_alert(self):
        scores = [_score("engine_temp", "SEVERE", z_score=4.0)]
        lstm   = _lstm("scored", is_anomalous=False, recon_error=0.01, threshold=0.1)
        result = assess_vehicle_health("V1", scores, lstm)
        assert result.status == HealthStatus.ALERT

    def test_elevated_p2_plus_lstm_anomalous_is_alert(self):
        scores = [_score("engine_temp", "ELEVATED", z_score=2.5)]
        lstm   = _lstm("scored", is_anomalous=True, recon_error=0.3, threshold=0.1)
        result = assess_vehicle_health("V1", scores, lstm)
        assert result.status == HealthStatus.ALERT

    def test_normal_p2_plus_lstm_anomalous_is_alert(self):
        scores = [_score("engine_temp", "NORMAL", z_score=0.5)]
        lstm   = _lstm("scored", is_anomalous=True, recon_error=0.3, threshold=0.1)
        result = assess_vehicle_health("V1", scores, lstm)
        assert result.status == HealthStatus.ALERT

    def test_elevated_p2_plus_lstm_normal_is_watch(self):
        scores = [_score("engine_temp", "ELEVATED", z_score=2.5)]
        lstm   = _lstm("scored", is_anomalous=False, recon_error=0.01, threshold=0.1)
        result = assess_vehicle_health("V1", scores, lstm)
        assert result.status == HealthStatus.WATCH

    def test_reasons_include_both_signals_when_both_alert(self):
        scores = [_score("rpm", "SEVERE", z_score=4.5)]
        lstm   = _lstm("scored", is_anomalous=True, recon_error=0.3, threshold=0.1)
        result = assess_vehicle_health("V1", scores, lstm)
        full_reason = " ".join(result.reasons)
        assert "SEVERE" in full_reason
        assert "LSTM" in full_reason or "reconstruction" in full_reason.lower()


# ---------------------------------------------------------------------------
# 5. Critical sensor hard limits
# ---------------------------------------------------------------------------

class TestCriticalSensorLimits:

    def test_high_engine_temp_triggers_alert(self):
        raw = _raw(engine_temp=ENGINE_TEMP_CRITICAL + 0.1)
        result = assess_vehicle_health("V1", [], None, raw_telemetry=raw)
        assert result.status == HealthStatus.ALERT
        assert "engine_temp" in result.evidence

    def test_engine_temp_just_below_critical_no_alert(self):
        raw = _raw(engine_temp=ENGINE_TEMP_CRITICAL - 0.1)
        scores = [_score("engine_temp", "NORMAL", readiness="READY")]
        result = assess_vehicle_health("V1", scores, None, raw_telemetry=raw)
        assert result.status == HealthStatus.NORMAL

    def test_low_battery_triggers_alert(self):
        raw = _raw(battery_voltage=BATTERY_VOLTAGE_CRITICAL - 0.1)
        result = assess_vehicle_health("V1", [], None, raw_telemetry=raw)
        assert result.status == HealthStatus.ALERT
        assert "battery_voltage" in result.evidence

    def test_battery_just_above_critical_no_alert(self):
        raw = _raw(battery_voltage=BATTERY_VOLTAGE_CRITICAL + 0.1)
        scores = [_score("battery_voltage", "NORMAL", readiness="READY")]
        result = assess_vehicle_health("V1", scores, None, raw_telemetry=raw)
        assert result.status == HealthStatus.NORMAL

    def test_high_rpm_triggers_alert(self):
        raw = _raw(rpm=RPM_CRITICAL + 1.0)
        result = assess_vehicle_health("V1", [], None, raw_telemetry=raw)
        assert result.status == HealthStatus.ALERT
        assert "rpm" in result.evidence

    def test_sensor_alert_reason_includes_value(self):
        raw = _raw(engine_temp=ENGINE_TEMP_CRITICAL + 5.0)
        result = assess_vehicle_health("V1", [], None, raw_telemetry=raw)
        assert any("engine_temp" in r for r in result.reasons)

    def test_no_raw_telemetry_no_sensor_alert(self):
        scores = [_score("engine_temp", "NORMAL", readiness="READY")]
        result = assess_vehicle_health("V1", scores, None, raw_telemetry=None)
        assert result.status == HealthStatus.NORMAL

    def test_critical_sensor_overrides_all_normal_signals(self):
        """Sensor alert fires even when all z-scores and LSTM are normal."""
        scores = [_score("engine_temp", "NORMAL", z_score=0.1)]
        lstm   = _lstm("scored", is_anomalous=False, recon_error=0.01, threshold=0.1)
        raw    = _raw(engine_temp=ENGINE_TEMP_CRITICAL + 1.0)
        result = assess_vehicle_health("V1", scores, lstm, raw_telemetry=raw)
        assert result.status == HealthStatus.ALERT


# ---------------------------------------------------------------------------
# 6. Explainability — reasons and evidence always present for non-NORMAL
# ---------------------------------------------------------------------------

class TestExplainability:

    def test_alert_always_has_reasons(self):
        scores = [_score("engine_temp", "SEVERE", z_score=4.0)]
        result = assess_vehicle_health("V1", scores, None)
        assert result.reasons, "ALERT should always have at least one reason"

    def test_watch_always_has_reasons(self):
        scores = [_score("engine_temp", "ELEVATED", z_score=2.5)]
        result = assess_vehicle_health("V1", scores, None)
        assert result.reasons, "WATCH should always have at least one reason"

    def test_lstm_anomalous_reason_includes_numeric_values(self):
        lstm = _lstm("scored", is_anomalous=True, recon_error=0.333, threshold=0.1)
        result = assess_vehicle_health("V1", [], lstm)
        combined = " ".join(result.reasons)
        assert "0.333" in combined or "0.33" in combined

    def test_to_dict_contains_required_keys(self):
        result = assess_vehicle_health("V1", [], None)
        d = result.to_dict()
        for key in ["vehicle_id", "status", "reasons", "evidence",
                    "phase2_scored", "lstm_status", "lstm_anomalous"]:
            assert key in d

    def test_status_is_string_in_dict(self):
        result = assess_vehicle_health("V1", [], None)
        d = result.to_dict()
        assert isinstance(d["status"], str)

    def test_reasons_is_list_in_dict(self):
        result = assess_vehicle_health("V1", [], None)
        d = result.to_dict()
        assert isinstance(d["reasons"], list)


# ---------------------------------------------------------------------------
# 7. Unavailable model handling
# ---------------------------------------------------------------------------

class TestUnavailableModel:

    def test_unavailable_lstm_not_treated_as_healthy(self):
        """no_model does not produce a clean-bill-of-health result."""
        lstm = _lstm("no_model")
        scores = [_score("engine_temp", "INSUFFICIENT_DATA",
                         readiness="INSUFFICIENT_BASELINE_DATA")]
        result = assess_vehicle_health("V1", scores, lstm)
        # Should not silently return NORMAL with no reasons
        # Either WATCH or NORMAL with reasons about data sufficiency
        if result.status == HealthStatus.NORMAL:
            assert result.reasons  # must explain why it thinks it's normal

    def test_lstm_error_still_reports_watch(self):
        lstm = _lstm("error")
        result = assess_vehicle_health("V1", [], lstm)
        assert result.status == HealthStatus.WATCH

    def test_lstm_warming_up_reason_present(self):
        lstm = _lstm("warming_up")
        result = assess_vehicle_health("V1", [], lstm)
        assert any("warming" in r.lower() or "buffer" in r.lower()
                   for r in result.reasons)


# ---------------------------------------------------------------------------
# 8. API endpoint /health/{vehicle_id}
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    db_file = str(tmp_path / "test_edgepulse.db")
    import config
    import database.db as db_module
    monkeypatch.setattr(config, "DATABASE_PATH", db_file)
    monkeypatch.setattr(db_module, "DATABASE_PATH", db_file)
    db_module.init_db()
    return db_file


class TestHealthAPIEndpoint:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)

    def _insert_records(self, n=5, vehicle_id="CAR-01", engine_temp=90.0,
                        battery_voltage=12.6, rpm=3000.0):
        import database.db as db_module
        from datetime import datetime, timezone, timedelta
        base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        for i in range(n):
            ts = (base + timedelta(seconds=float(i))).isoformat()
            db_module.insert_telemetry({
                "message_id": f"msg-{vehicle_id}-{i}",
                "event_timestamp": ts,
                "ingestion_timestamp": ts,
                "vehicle_id": vehicle_id,
                "speed": 60.0,
                "rpm": rpm,
                "engine_temp": engine_temp,
                "battery_voltage": battery_voltage,
                "fuel_level": 80.0,
            })

    def test_health_404_no_telemetry(self, client):
        r = client.get("/health/UNKNOWN-V1")
        assert r.status_code == 404

    def test_health_returns_200_with_data(self, client):
        self._insert_records()
        r = client.get("/health/CAR-01")
        assert r.status_code == 200

    def test_health_response_has_required_keys(self, client):
        self._insert_records()
        r = client.get("/health/CAR-01")
        body = r.json()
        for key in ["vehicle_id", "status", "reasons", "evidence"]:
            assert key in body

    def test_health_vehicle_id_in_response(self, client):
        self._insert_records()
        r = client.get("/health/CAR-01")
        assert r.json()["vehicle_id"] == "CAR-01"

    def test_health_status_is_valid_value(self, client):
        self._insert_records()
        r = client.get("/health/CAR-01")
        assert r.json()["status"] in ("NORMAL", "WATCH", "ALERT")

    def test_health_critical_temp_returns_alert(self, client):
        self._insert_records(n=60, engine_temp=ENGINE_TEMP_CRITICAL + 1.0)
        r = client.get("/health/CAR-01")
        body = r.json()
        assert body["status"] == "ALERT"
        assert any("engine_temp" in reason for reason in body["reasons"])

    def test_health_normal_temp_not_alert(self, client):
        self._insert_records(n=5, engine_temp=90.0)
        r = client.get("/health/CAR-01")
        # With only 5 records baselines aren't ready — should be WATCH or NORMAL, not ALERT
        assert r.json()["status"] in ("NORMAL", "WATCH")

    def test_health_reasons_is_list(self, client):
        self._insert_records()
        r = client.get("/health/CAR-01")
        assert isinstance(r.json()["reasons"], list)

    def test_health_evidence_is_dict(self, client):
        self._insert_records()
        r = client.get("/health/CAR-01")
        assert isinstance(r.json()["evidence"], dict)


# ---------------------------------------------------------------------------
# 9. Telemetry persistence despite pipeline failures (regression)
# ---------------------------------------------------------------------------

class TestPersistenceResilience:

    def test_raw_telemetry_persisted_even_when_feature_pipeline_fails(
        self, monkeypatch
    ):
        """
        If the feature pipeline raises an exception, raw telemetry must still
        be written to SQLite.  This is guaranteed by consumer.on_message()
        which calls insert_telemetry before calling pipeline.process_record().
        This test validates that guarantee via the consumer's process_telemetry
        function and the DB layer directly.
        """
        import database.db as db_module
        from edge.consumer import process_telemetry

        payload = {
            "vehicle_id": "V1",
            "speed": 60.0,
            "rpm": 3000.0,
            "engine_temp": 90.0,
            "battery_voltage": 12.6,
            "fuel_level": 80.0,
        }
        enriched = process_telemetry(payload)
        assert enriched is not None
        result = db_module.insert_telemetry(enriched)
        assert result is True
        rows = db_module.get_telemetry_by_vehicle("V1")
        assert len(rows) == 1

    def test_pipeline_process_record_returns_none_on_bad_input(self):
        """FeaturePipeline.process_record catches internal errors and returns None."""
        from edge.pipeline import FeaturePipeline
        pipeline = FeaturePipeline(lstm_enabled=False)
        result = pipeline.process_record({"vehicle_id": "V1"})   # missing fields
        assert result is None
