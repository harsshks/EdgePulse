# simulator/faults.py
# Fault injection framework for EdgePulse Phase 1.
#
# Design principles:
#   - Faults produce VALID but ABNORMAL telemetry — not corrupt data.
#   - Faults have a full lifecycle: NORMAL → STARTING → ACTIVE → RECOVERING → NORMAL
#   - Ground truth is recorded per-step so ML models can be evaluated precisely.
#   - Faults modify sensor readings AFTER the normal vehicle state machine runs.
#   - Multiple faults can be configured per simulation run.
#
# The consumer and validation layer never see fault metadata.
# Ground truth is written to a separate CSV file for ML experiments.

from __future__ import annotations

import csv
import io
import math
import os
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Fault type enumeration
# ---------------------------------------------------------------------------

class FaultType(Enum):
    NONE                = "NONE"
    COOLING_DEGRADATION = "COOLING_DEGRADATION"
    RPM_INSTABILITY     = "RPM_INSTABILITY"
    BATTERY_DEGRADATION = "BATTERY_DEGRADATION"
    SENSOR_DRIFT        = "SENSOR_DRIFT"
    EXCESSIVE_IDLING    = "EXCESSIVE_IDLING"


class FaultState(Enum):
    NORMAL     = "NORMAL"
    STARTING   = "STARTING"    # fault ramping up
    ACTIVE     = "ACTIVE"      # fault fully active
    RECOVERING = "RECOVERING"  # fault ramping down
    RESOLVED   = "RESOLVED"    # fault gone


# ---------------------------------------------------------------------------
# Fault scenario definition
# ---------------------------------------------------------------------------

@dataclass
class FaultScenario:
    """
    Configures one fault injection episode.

    Attributes:
        fault_type   — which fault to inject
        vehicle_id   — which vehicle is affected
        start_time   — simulation time (seconds) when fault begins
        duration     — how long the fault is ACTIVE (seconds)
        severity     — 0.0 (minimal) to 1.0 (severe); scales effect magnitude
        ramp_time    — seconds to ramp from NORMAL to ACTIVE (default 30s)
        recovery_time— seconds to ramp from ACTIVE back to NORMAL (default 60s)
    """
    fault_type:    FaultType
    vehicle_id:    str
    start_time:    float
    duration:      float
    severity:      float = 0.5
    ramp_time:     float = 30.0
    recovery_time: float = 60.0

    def __post_init__(self):
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")


# ---------------------------------------------------------------------------
# Ground-truth record
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthRecord:
    """One row in the ground-truth log."""
    sim_time:    float
    vehicle_id:  str
    message_id:  str
    event_timestamp: str
    fault_type:  str
    fault_state: str
    severity:    float


# ---------------------------------------------------------------------------
# Fault controller
# ---------------------------------------------------------------------------

class FaultController:
    """
    Applies fault effects to a sensor reading dict and records ground truth.

    Usage:
        fc = FaultController(scenarios, rng=random.Random(42))
        reading = vehicle_sm.step()              # normal reading
        reading = fc.apply(vehicle_id, sim_time, reading)   # fault-modified
        gt = fc.ground_truth_for(vehicle_id)     # current fault label
    """

    def __init__(
        self,
        scenarios: list[FaultScenario],
        rng: Optional[random.Random] = None,
    ):
        self._scenarios = scenarios
        self._rng = rng or random.Random()
        # Accumulated sensor drift offsets per vehicle (for SENSOR_DRIFT fault)
        self._drift: dict[str, float] = {}
        self._ground_truth_rows: list[GroundTruthRecord] = []

    def apply(
        self,
        vehicle_id: str,
        sim_time: float,
        reading: dict,
        message_id: str,
        event_timestamp: str,
    ) -> dict:
        """
        Apply any active faults for this vehicle at this simulation time.
        Modifies a COPY of the reading dict — never mutates the original.
        Records ground truth regardless of whether a fault is active.
        """
        reading = dict(reading)  # defensive copy

        active_fault_type = FaultType.NONE
        active_fault_state = FaultState.NORMAL
        active_severity = 0.0

        for scenario in self._scenarios:
            if scenario.vehicle_id != vehicle_id:
                continue

            state, progress = self._fault_state(scenario, sim_time)

            if state in (FaultState.STARTING, FaultState.ACTIVE, FaultState.RECOVERING, FaultState.RESOLVED):
                active_fault_type = scenario.fault_type
                active_fault_state = state
                active_severity = scenario.severity * progress

                if state in (FaultState.STARTING, FaultState.ACTIVE, FaultState.RECOVERING):
                    reading = self._apply_fault(
                        scenario.fault_type, vehicle_id, reading, active_severity, sim_time
                    )
                break  # one fault at a time per vehicle for now

        self._ground_truth_rows.append(GroundTruthRecord(
            sim_time=sim_time,
            vehicle_id=vehicle_id,
            message_id=message_id,
            event_timestamp=event_timestamp,
            fault_type=active_fault_type.value,
            fault_state=active_fault_state.value,
            severity=round(active_severity, 4),
        ))

        return reading

    def ground_truth_for(self, vehicle_id: str) -> tuple[FaultType, FaultState, float]:
        """Returns (fault_type, fault_state, severity) for the most recent record."""
        for row in reversed(self._ground_truth_rows):
            if row.vehicle_id == vehicle_id:
                return (
                    FaultType(row.fault_type),
                    FaultState(row.fault_state),
                    row.severity,
                )
        return FaultType.NONE, FaultState.NORMAL, 0.0

    def export_ground_truth_csv(self, path: str) -> None:
        """
        Writes all ground-truth rows to a CSV file.
        This is a development-only export for future ML experiments.
        It is NOT part of the production telemetry pipeline.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["sim_time", "vehicle_id", "message_id",
                            "event_timestamp", "fault_type", "fault_state", "severity"]
            )
            writer.writeheader()
            for row in self._ground_truth_rows:
                writer.writerow({
                    "sim_time":        row.sim_time,
                    "vehicle_id":      row.vehicle_id,
                    "message_id":      row.message_id,
                    "event_timestamp": row.event_timestamp,
                    "fault_type":      row.fault_type,
                    "fault_state":     row.fault_state,
                    "severity":        row.severity,
                })

    def get_ground_truth_rows(self) -> list[GroundTruthRecord]:
        """Returns all ground-truth records accumulated so far."""
        return list(self._ground_truth_rows)

    # ------------------------------------------------------------------
    # Fault lifecycle math
    # ------------------------------------------------------------------

    def _fault_state(
        self, scenario: FaultScenario, sim_time: float
    ) -> tuple[FaultState, float]:
        """
        Calculates the fault state and progress (0.0–1.0) at `sim_time`.

        Timeline:
          [0, start)                           → NORMAL,     progress=0
          [start, start+ramp)                  → STARTING,   progress ramps 0→1
          [start+ramp, start+ramp+duration)    → ACTIVE,     progress=1
          [start+ramp+duration, ...+recovery)  → RECOVERING, progress ramps 1→0
          [start+ramp+duration+recovery, ...)  → RESOLVED,   progress=0
        """
        t0 = scenario.start_time
        t_ramp_end = t0 + scenario.ramp_time
        t_active_end = t_ramp_end + scenario.duration
        t_recover_end = t_active_end + scenario.recovery_time

        if sim_time < t0:
            return FaultState.NORMAL, 0.0
        elif sim_time < t_ramp_end:
            progress = (sim_time - t0) / max(scenario.ramp_time, 1e-9)
            return FaultState.STARTING, progress
        elif sim_time < t_active_end:
            return FaultState.ACTIVE, 1.0
        elif sim_time < t_recover_end:
            progress = 1.0 - (sim_time - t_active_end) / max(scenario.recovery_time, 1e-9)
            return FaultState.RECOVERING, max(0.0, progress)
        else:
            return FaultState.RESOLVED, 0.0

    # ------------------------------------------------------------------
    # Fault effect implementations
    # ------------------------------------------------------------------

    def _apply_fault(
        self,
        fault_type: FaultType,
        vehicle_id: str,
        reading: dict,
        severity: float,
        sim_time: float,
    ) -> dict:
        if fault_type == FaultType.COOLING_DEGRADATION:
            return self._cooling_degradation(reading, severity)
        elif fault_type == FaultType.RPM_INSTABILITY:
            return self._rpm_instability(reading, severity)
        elif fault_type == FaultType.BATTERY_DEGRADATION:
            return self._battery_degradation(reading, severity)
        elif fault_type == FaultType.SENSOR_DRIFT:
            return self._sensor_drift(vehicle_id, reading, severity)
        elif fault_type == FaultType.EXCESSIVE_IDLING:
            # EXCESSIVE_IDLING is implemented at the state machine level
            # by the publisher; no sensor modification needed here
            return reading
        return reading

    def _cooling_degradation(self, reading: dict, severity: float) -> dict:
        """
        Temperature rises above normal and recovers more slowly.
        Max extra rise: +20°C at severity=1.0.
        Values stay within validation range (max 130°C).
        """
        max_extra = 20.0 * severity
        reading["engine_temp"] = min(130.0, reading["engine_temp"] + max_extra)
        return reading

    def _rpm_instability(self, reading: dict, severity: float) -> dict:
        """
        RPM oscillates sinusoidally around its normal value.
        Amplitude: up to ±600 RPM at severity=1.0.
        Frequency: ~0.1 Hz (one cycle per ~10 seconds).
        """
        amplitude = 600.0 * severity
        # Use reading's rpm as phase proxy — not time-aware here,
        # so we add a noise-based oscillation
        oscillation = self._rng.gauss(0, amplitude * 0.4)
        reading["rpm"] = max(0.0, min(9000.0, reading["rpm"] + oscillation))
        return reading

    def _battery_degradation(self, reading: dict, severity: float) -> dict:
        """
        Battery voltage drops gradually and becomes more unstable.
        Max drop: -1.5V at severity=1.0. Noise increases with severity.
        """
        voltage_drop = 1.5 * severity
        extra_noise = self._rng.gauss(0, 0.15 * severity)
        new_v = reading["battery_voltage"] - voltage_drop + extra_noise
        reading["battery_voltage"] = round(max(10.0, min(15.0, new_v)), 3)
        return reading

    def _sensor_drift(self, vehicle_id: str, reading: dict, severity: float) -> dict:
        """
        Engine temperature sensor gradually reads higher than actual.
        The drift accumulates over time, simulating bias growth.
        Max accumulated drift: +8°C at severity=1.0.
        """
        key = vehicle_id
        if key not in self._drift:
            self._drift[key] = 0.0

        # Drift grows slowly (0.02°C/step × severity)
        self._drift[key] += 0.02 * severity
        self._drift[key] = min(self._drift[key], 8.0 * severity)

        reading["engine_temp"] = min(
            130.0, reading["engine_temp"] + self._drift[key]
        )
        return reading


# ---------------------------------------------------------------------------
# Predefined scenario sets
# ---------------------------------------------------------------------------

def make_normal_scenarios() -> list[FaultScenario]:
    """No faults — used in 'normal' mode."""
    return []


def make_fault_scenarios(duration: float = 600.0) -> list[FaultScenario]:
    """
    One fault per vehicle, staggered start times.
    Total simulation should be at least duration + 300 seconds.
    """
    return [
        FaultScenario(
            fault_type=FaultType.COOLING_DEGRADATION,
            vehicle_id="CAR-01",
            start_time=60.0,
            duration=duration,
            severity=0.6,
            ramp_time=30.0,
            recovery_time=90.0,
        ),
        FaultScenario(
            fault_type=FaultType.RPM_INSTABILITY,
            vehicle_id="CAR-02",
            start_time=120.0,
            duration=duration * 0.8,
            severity=0.5,
            ramp_time=20.0,
            recovery_time=60.0,
        ),
        FaultScenario(
            fault_type=FaultType.BATTERY_DEGRADATION,
            vehicle_id="CAR-03",
            start_time=90.0,
            duration=duration * 0.9,
            severity=0.7,
            ramp_time=40.0,
            recovery_time=80.0,
        ),
    ]


def make_mixed_scenarios(duration: float = 300.0) -> list[FaultScenario]:
    """Mixed faults across all vehicles for a rich dataset."""
    return [
        FaultScenario(
            fault_type=FaultType.COOLING_DEGRADATION,
            vehicle_id="CAR-01",
            start_time=120.0,
            duration=duration,
            severity=0.5,
            ramp_time=30.0,
            recovery_time=60.0,
        ),
        FaultScenario(
            fault_type=FaultType.SENSOR_DRIFT,
            vehicle_id="CAR-01",
            start_time=120.0 + duration + 120.0,
            duration=duration * 0.7,
            severity=0.6,
            ramp_time=60.0,
            recovery_time=90.0,
        ),
        FaultScenario(
            fault_type=FaultType.RPM_INSTABILITY,
            vehicle_id="CAR-02",
            start_time=90.0,
            duration=duration,
            severity=0.4,
            ramp_time=20.0,
            recovery_time=45.0,
        ),
        FaultScenario(
            fault_type=FaultType.BATTERY_DEGRADATION,
            vehicle_id="CAR-02",
            start_time=90.0 + duration + 90.0,
            duration=duration * 0.5,
            severity=0.8,
            ramp_time=30.0,
            recovery_time=60.0,
        ),
        FaultScenario(
            fault_type=FaultType.EXCESSIVE_IDLING,
            vehicle_id="CAR-03",
            start_time=60.0,
            duration=duration * 1.2,
            severity=0.5,
            ramp_time=0.0,
            recovery_time=0.0,
        ),
        FaultScenario(
            fault_type=FaultType.SENSOR_DRIFT,
            vehicle_id="CAR-03",
            start_time=60.0 + duration * 1.2 + 60.0,
            duration=duration * 0.8,
            severity=0.7,
            ramp_time=45.0,
            recovery_time=75.0,
        ),
    ]
