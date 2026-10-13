# edge/health.py
# Vehicle-health decision layer for EdgePulse Phase 4.
#
# Combines Phase 2 statistical scores and Phase 3 LSTM anomaly results
# into a single, deterministic health status for a vehicle.
#
# Design principles:
#   - Pure function — no I/O, no MQTT, no database, no ML inference.
#   - Deterministic: same inputs always produce the same output.
#   - Explicit precedence: critical sensor conditions > statistical SEVERE >
#     statistical ELEVATED / LSTM anomalous > LSTM-only signals > NORMAL.
#   - Unavailable LSTM output is never treated as healthy.
#   - No fault-type diagnosis and no failure-time prediction.
#   - The caller (API layer) is responsible for providing input data;
#     this module only aggregates it.
#
# Status levels (ordered by severity, high to low):
#   ALERT  — at least one critical condition: SEVERE z-score, LSTM anomalous,
#             or engine_temp in the critical zone.
#   WATCH  — ELEVATED z-scores or LSTM not yet scored but model loaded.
#   NORMAL — all scored signals within normal range and LSTM normal (or absent).
#
# Unavailability handling:
#   - LSTM unavailable (no_model / no_threshold / error):
#       → does NOT contribute to NORMAL; status is noted in reasons.
#   - Phase 2 INSUFFICIENT_DATA on some features:
#       → those features do not contribute; noted in reasons if all features
#         are insufficient.
#   - Phase 2 fully insufficient + LSTM unavailable:
#       → returns WATCH with reason "insufficient data to assess".

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Critical sensor hard limits (distinct from z-score baselines)
# ---------------------------------------------------------------------------

# Engine temperature hard ceiling — above this is a physical danger.
# (Baseline z-score approach may not catch a cold-start vehicle at 130 °C
#  because its baseline mean could still be low after just a few samples.)
ENGINE_TEMP_CRITICAL: float = 115.0   # °C

# Battery voltage hard floor — below this risks imminent shutdown.
BATTERY_VOLTAGE_CRITICAL: float = 11.0   # V

# RPM hard ceiling — structural damage risk above this.
RPM_CRITICAL: float = 8000.0   # rpm


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class HealthStatus(str, Enum):
    NORMAL = "NORMAL"
    WATCH  = "WATCH"
    ALERT  = "ALERT"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VehicleHealthResult:
    """
    Aggregated vehicle health assessment.

    Fields:
        vehicle_id      — which vehicle
        status          — NORMAL | WATCH | ALERT
        reasons         — list of human-readable strings explaining non-NORMAL status
        evidence        — dict of signal-name → value for every signal that
                          contributed to the status (empty when status is NORMAL
                          and no anomalies were found)
        phase2_scored   — True if at least one Phase 2 score was at READY baseline
        lstm_status     — forwarded LSTM status string (scored/warming_up/no_model/…)
        lstm_anomalous  — True/False/None from the LSTM result
    """
    vehicle_id:    str
    status:        HealthStatus
    reasons:       list[str] = field(default_factory=list)
    evidence:      dict      = field(default_factory=dict)
    phase2_scored: bool      = False
    lstm_status:   Optional[str]  = None
    lstm_anomalous: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "vehicle_id":    self.vehicle_id,
            "status":        self.status.value,
            "reasons":       self.reasons,
            "evidence":      self.evidence,
            "phase2_scored": self.phase2_scored,
            "lstm_status":   self.lstm_status,
            "lstm_anomalous": self.lstm_anomalous,
        }


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def assess_vehicle_health(
    vehicle_id:     str,
    deviation_scores: list[dict],
    lstm_result:    Optional[dict],
    raw_telemetry:  Optional[dict] = None,
) -> VehicleHealthResult:
    """
    Compute a health assessment for a single vehicle.

    Parameters
    ----------
    vehicle_id
        Vehicle identifier.
    deviation_scores
        List of DeviationScore.to_dict() entries from Phase 2 scoring.
        May be empty or all INSUFFICIENT_DATA when the baseline is not ready.
    lstm_result
        LSTMAnomalyResult.to_dict() from Phase 3, or None when LSTM is
        disabled / consumer not running.
    raw_telemetry
        Most recent raw telemetry dict (for critical-sensor hard limits).
        Pass None if not available — hard-limit checks are skipped.

    Returns
    -------
    VehicleHealthResult

    Notes
    -----
    Precedence (highest to lowest):
      1. Critical sensor hard limits (engine_temp, battery_voltage, rpm)
      2. Phase 2 SEVERE z-scores
      3. Phase 2 ELEVATED z-scores  |  LSTM is_anomalous = True
      4. LSTM warming_up / no_threshold (model present but not yet decisive)
      5. All signals NORMAL
    """
    reasons:  list[str] = []
    evidence: dict      = {}

    # --- Step 1: Critical sensor hard limits ---
    alert_from_sensor = False
    if raw_telemetry is not None:
        temp = _safe_float(raw_telemetry.get("engine_temp"))
        if temp is not None and temp >= ENGINE_TEMP_CRITICAL:
            alert_from_sensor = True
            reasons.append(
                f"engine_temp {temp:.1f} °C ≥ critical threshold "
                f"{ENGINE_TEMP_CRITICAL:.0f} °C"
            )
            evidence["engine_temp"] = temp

        volt = _safe_float(raw_telemetry.get("battery_voltage"))
        if volt is not None and volt <= BATTERY_VOLTAGE_CRITICAL:
            alert_from_sensor = True
            reasons.append(
                f"battery_voltage {volt:.2f} V ≤ critical floor "
                f"{BATTERY_VOLTAGE_CRITICAL:.1f} V"
            )
            evidence["battery_voltage"] = volt

        rpm_val = _safe_float(raw_telemetry.get("rpm"))
        if rpm_val is not None and rpm_val >= RPM_CRITICAL:
            alert_from_sensor = True
            reasons.append(
                f"rpm {rpm_val:.0f} ≥ critical ceiling {RPM_CRITICAL:.0f}"
            )
            evidence["rpm"] = rpm_val

    # --- Step 2: Phase 2 z-score deviation signals ---
    severe_features: list[str]   = []
    elevated_features: list[str] = []
    phase2_scored = False

    for s in deviation_scores:
        level = s.get("level", "")
        readiness = s.get("readiness", "")
        if readiness == "READY":
            phase2_scored = True
        if level == "SEVERE":
            fname = s.get("feature_name", "?")
            z     = s.get("z_score")
            severe_features.append(fname)
            evidence[fname] = {
                "z_score": round(z, 3) if z is not None else None,
                "level": "SEVERE",
                "current_value": s.get("current_value"),
                "baseline_mean": s.get("baseline_mean"),
            }
        elif level == "ELEVATED":
            fname = s.get("feature_name", "?")
            z     = s.get("z_score")
            elevated_features.append(fname)
            evidence[fname] = {
                "z_score": round(z, 3) if z is not None else None,
                "level": "ELEVATED",
                "current_value": s.get("current_value"),
                "baseline_mean": s.get("baseline_mean"),
            }

    if severe_features:
        reasons.append(
            f"SEVERE deviation on: {', '.join(severe_features)}"
        )

    if elevated_features:
        reasons.append(
            f"ELEVATED deviation on: {', '.join(elevated_features)}"
        )

    # --- Step 3: LSTM anomaly signal ---
    lstm_status:    Optional[str]  = None
    lstm_anomalous: Optional[bool] = None
    lstm_is_alert   = False
    lstm_is_watch   = False

    if lstm_result is not None:
        lstm_status    = lstm_result.get("status")
        lstm_anomalous = lstm_result.get("is_anomalous")
        recon_error    = lstm_result.get("reconstruction_error")
        threshold      = lstm_result.get("threshold")

        if lstm_status == "scored":
            if lstm_anomalous is True:
                lstm_is_alert = True
                reasons.append(
                    f"LSTM reconstruction error {recon_error:.4f} "
                    f"exceeds threshold {threshold:.4f}"
                )
                evidence["lstm_reconstruction_error"] = recon_error
                evidence["lstm_threshold"] = threshold
            # lstm_anomalous is False → no reason added, no watch
        elif lstm_status == "warming_up":
            lstm_is_watch = True
            reasons.append(
                "LSTM model loaded but sequence buffer still filling "
                "(insufficient observations)"
            )
        elif lstm_status == "no_threshold":
            lstm_is_watch = True
            reasons.append(
                "LSTM model loaded but threshold not yet calibrated — "
                "run lstm.evaluate to calibrate"
            )
        elif lstm_status == "no_model":
            # Absence of model is noted but does not elevate status on its own
            reasons.append("LSTM model not available — temporal anomaly detection inactive")
        elif lstm_status == "error":
            lstm_is_watch = True
            reasons.append("LSTM inference error — temporal signal unavailable")

    # --- Step 4: Data sufficiency check ---
    # If no Phase 2 signal was scored AND LSTM gave nothing decisive,
    # warn the caller rather than silently returning NORMAL.
    no_decisive_p2 = not phase2_scored
    lstm_decisive  = lstm_status == "scored" and lstm_anomalous is not None

    if no_decisive_p2 and not lstm_decisive and not alert_from_sensor:
        reasons.append(
            "Insufficient scored data — baselines still building "
            "(Phase 2 requires ≥30 samples per feature)"
        )

    # --- Step 5: Aggregate status ---
    alert = alert_from_sensor or bool(severe_features) or lstm_is_alert
    watch = bool(elevated_features) or lstm_is_watch

    if alert:
        status = HealthStatus.ALERT
    elif watch:
        status = HealthStatus.WATCH
    elif not reasons:
        # Everything scored and all signals normal
        status = HealthStatus.NORMAL
    else:
        # Has reasons (e.g. "LSTM not available") but nothing ALERT/WATCH-worthy
        status = HealthStatus.NORMAL

    return VehicleHealthResult(
        vehicle_id=vehicle_id,
        status=status,
        reasons=reasons,
        evidence=evidence,
        phase2_scored=phase2_scored,
        lstm_status=lstm_status,
        lstm_anomalous=lstm_anomalous,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
