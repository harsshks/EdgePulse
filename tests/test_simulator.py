# tests/test_simulator.py
# Phase 1 simulator tests.
# Covers: vehicle state machine, sensor behaviour, fault injection,
#         fault lifecycle, ground-truth labeling, reproducibility,
#         MQTT payload compatibility with Phase 0 validation.
#
# No MQTT broker required — all tests are pure unit/integration tests
# against the simulation logic.

import random
import sys
import os
import csv
import io
import math

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.vehicle import (
    DEFAULT_VEHICLES,
    VehicleProfile,
    VehicleState,
    VehicleStateMachine,
)
from simulator.faults import (
    FaultController,
    FaultScenario,
    FaultState,
    FaultType,
    make_fault_scenarios,
    make_mixed_scenarios,
    make_normal_scenarios,
)
from simulator.publisher import run_simulation
from edge.validation import validate_telemetry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_vsm(seed: int = 0, profile: VehicleProfile = None) -> VehicleStateMachine:
    p = profile or DEFAULT_VEHICLES[0]
    return VehicleStateMachine(p, rng=random.Random(seed))


def run_steps(vsm: VehicleStateMachine, n: int, dt: float = 1.0) -> list[dict]:
    return [vsm.step(dt) for _ in range(n)]


# ---------------------------------------------------------------------------
# 1. Normal vehicle behaviour — sensor evolution
# ---------------------------------------------------------------------------

class TestNormalBehaviour:

    def test_speed_starts_at_zero(self):
        vsm = make_vsm()
        reading = vsm.step()
        # First step: STOPPED → IDLE, speed still 0
        assert reading["speed"] == 0.0

    def test_speed_increases_during_acceleration(self):
        vsm = make_vsm()
        # Advance past idle into ACCELERATING
        readings = run_steps(vsm, 120)
        # Find readings in ACCELERATING state and check speed trend
        accel_readings = [r for r in readings if vsm.state == VehicleState.ACCELERATING]
        speeds = [r["speed"] for r in readings]
        # At least some speed should have been reached
        assert max(speeds) > 10.0

    def test_speed_does_not_jump(self):
        """Speed changes should be gradual — no single-step jumps > accel_rate."""
        vsm = make_vsm()
        readings = run_steps(vsm, 300)
        speeds = [r["speed"] for r in readings]
        for i in range(1, len(speeds)):
            delta = abs(speeds[i] - speeds[i - 1])
            # Max change per step is accel_rate or decel_rate (~10-12 km/h/s)
            assert delta <= 15.0, f"Speed jumped {delta:.1f} km/h at step {i}"

    def test_speed_zero_during_idle(self):
        vsm = make_vsm()
        for _ in range(5):
            reading = vsm.step()
            if vsm.state == VehicleState.IDLE:
                assert reading["speed"] == 0.0

    def test_rpm_responds_to_vehicle_state(self):
        """RPM at cruise should be higher than RPM at idle."""
        vsm = make_vsm()
        idle_rpms = []
        cruise_rpms = []
        prev_state = vsm.state
        for _ in range(600):
            reading = vsm.step()
            if vsm.state == VehicleState.IDLE:
                idle_rpms.append(reading["rpm"])
            elif vsm.state == VehicleState.CRUISING:
                cruise_rpms.append(reading["rpm"])

        if idle_rpms and cruise_rpms:
            assert sum(idle_rpms) / len(idle_rpms) < sum(cruise_rpms) / len(cruise_rpms)

    def test_rpm_never_negative(self):
        vsm = make_vsm()
        for reading in run_steps(vsm, 300):
            assert reading["rpm"] >= 0.0

    def test_rpm_within_validation_range(self):
        vsm = make_vsm()
        for reading in run_steps(vsm, 300):
            assert 0 <= reading["rpm"] <= 9000

    def test_engine_temp_evolves_gradually(self):
        """Temperature changes should be small per step."""
        vsm = make_vsm()
        readings = run_steps(vsm, 300)
        temps = [r["engine_temp"] for r in readings]
        for i in range(1, len(temps)):
            delta = abs(temps[i] - temps[i - 1])
            assert delta <= 5.0, f"Temp jumped {delta:.2f}°C at step {i}"

    def test_engine_temp_within_validation_range(self):
        vsm = make_vsm()
        for reading in run_steps(vsm, 600):
            assert -10.0 <= reading["engine_temp"] <= 130.0

    def test_fuel_decreases_monotonically(self):
        """Fuel should only go down, never up."""
        vsm = make_vsm()
        readings = run_steps(vsm, 600)
        fuels = [r["fuel_level"] for r in readings]
        for i in range(1, len(fuels)):
            assert fuels[i] <= fuels[i - 1] + 1e-9, \
                f"Fuel increased at step {i}: {fuels[i-1]:.3f} → {fuels[i]:.3f}"

    def test_fuel_starts_at_100(self):
        vsm = make_vsm()
        assert vsm.sensors.fuel_level == 100.0

    def test_battery_voltage_stays_near_nominal(self):
        vsm = make_vsm()
        nominal = vsm.profile.battery_voltage_nominal
        for reading in run_steps(vsm, 300):
            assert abs(reading["battery_voltage"] - nominal) < 1.5

    def test_all_required_fields_present(self):
        vsm = make_vsm()
        required = ["vehicle_id", "speed", "rpm", "engine_temp"]
        for reading in run_steps(vsm, 10):
            for field in required:
                assert field in reading


# ---------------------------------------------------------------------------
# 2. State machine transitions
# ---------------------------------------------------------------------------

class TestStateTransitions:

    def test_starts_in_stopped(self):
        vsm = make_vsm()
        assert vsm.state == VehicleState.STOPPED

    def test_stopped_transitions_to_idle(self):
        vsm = make_vsm()
        vsm.step()
        assert vsm.state == VehicleState.IDLE

    def test_idle_transitions_to_accelerating(self):
        vsm = make_vsm()
        states_seen = set()
        for _ in range(200):
            vsm.step()
            states_seen.add(vsm.state)
        assert VehicleState.ACCELERATING in states_seen

    def test_accelerating_transitions_to_cruising(self):
        vsm = make_vsm()
        states_seen = set()
        for _ in range(400):
            vsm.step()
            states_seen.add(vsm.state)
        assert VehicleState.CRUISING in states_seen

    def test_cruising_transitions_to_decelerating(self):
        vsm = make_vsm()
        states_seen = set()
        for _ in range(600):
            vsm.step()
            states_seen.add(vsm.state)
        assert VehicleState.DECELERATING in states_seen

    def test_full_cycle_completes(self):
        """All states should be visited in a long enough simulation."""
        vsm = make_vsm()
        states_seen = set()
        for _ in range(800):
            vsm.step()
            states_seen.add(vsm.state)
        expected = {
            VehicleState.STOPPED, VehicleState.IDLE,
            VehicleState.ACCELERATING, VehicleState.CRUISING,
            VehicleState.DECELERATING,
        }
        # STOPPED only appears at t=0, may not be seen after first step
        assert expected - {VehicleState.STOPPED} <= states_seen


# ---------------------------------------------------------------------------
# 3. Vehicle diversity
# ---------------------------------------------------------------------------

class TestVehicleDiversity:

    def test_different_vehicles_have_different_baseline_temp(self):
        assert DEFAULT_VEHICLES[0].baseline_temp != DEFAULT_VEHICLES[1].baseline_temp
        assert DEFAULT_VEHICLES[1].baseline_temp != DEFAULT_VEHICLES[2].baseline_temp

    def test_cruising_temperatures_differ_between_vehicles(self):
        """Different vehicles should produce different temperatures at cruise."""
        avg_temps = {}
        for profile in DEFAULT_VEHICLES:
            vsm = VehicleStateMachine(profile, rng=random.Random(42))
            cruise_temps = []
            for _ in range(800):
                reading = vsm.step()
                if vsm.state == VehicleState.CRUISING:
                    cruise_temps.append(reading["engine_temp"])
            if cruise_temps:
                avg_temps[profile.vehicle_id] = sum(cruise_temps) / len(cruise_temps)

        if len(avg_temps) >= 2:
            temps = list(avg_temps.values())
            # At least two vehicles should differ by more than 1°C
            differences = [abs(temps[i] - temps[j]) for i in range(len(temps))
                           for j in range(i + 1, len(temps))]
            assert max(differences) > 1.0


# ---------------------------------------------------------------------------
# 4. Fault injection
# ---------------------------------------------------------------------------

class TestFaultInjection:

    def _make_fc(self, scenarios: list[FaultScenario], seed: int = 0) -> FaultController:
        return FaultController(scenarios, rng=random.Random(seed))

    def _base_reading(self) -> dict:
        return {
            "vehicle_id": "CAR-01",
            "speed": 80.0,
            "rpm": 3000.0,
            "engine_temp": 88.0,
            "battery_voltage": 12.6,
            "fuel_level": 80.0,
        }

    def test_cooling_degradation_raises_temperature(self):
        scenario = FaultScenario(
            fault_type=FaultType.COOLING_DEGRADATION,
            vehicle_id="CAR-01",
            start_time=0.0,
            duration=100.0,
            severity=1.0,
            ramp_time=0.0,
        )
        fc = self._make_fc([scenario])
        reading = fc.apply("CAR-01", 10.0, self._base_reading(), "m1", "ts1")
        assert reading["engine_temp"] > 88.0

    def test_cooling_degradation_temp_within_valid_range(self):
        scenario = FaultScenario(
            fault_type=FaultType.COOLING_DEGRADATION,
            vehicle_id="CAR-01",
            start_time=0.0,
            duration=1000.0,
            severity=1.0,
            ramp_time=0.0,
        )
        fc = self._make_fc([scenario])
        for t in range(200):
            r = fc.apply("CAR-01", float(t), self._base_reading(), f"m{t}", "ts")
            assert r["engine_temp"] <= 130.0

    def test_rpm_instability_adds_oscillation(self):
        scenario = FaultScenario(
            fault_type=FaultType.RPM_INSTABILITY,
            vehicle_id="CAR-01",
            start_time=0.0,
            duration=200.0,
            severity=1.0,
            ramp_time=0.0,
        )
        fc = self._make_fc([scenario], seed=1)
        rpms = []
        for t in range(50):
            r = fc.apply("CAR-01", float(t), self._base_reading(), f"m{t}", "ts")
            rpms.append(r["rpm"])
        # RPM should vary — std dev should be non-trivial with high severity
        mean = sum(rpms) / len(rpms)
        std = math.sqrt(sum((x - mean) ** 2 for x in rpms) / len(rpms))
        assert std > 50.0

    def test_rpm_instability_stays_within_valid_range(self):
        scenario = FaultScenario(
            fault_type=FaultType.RPM_INSTABILITY,
            vehicle_id="CAR-01",
            start_time=0.0,
            duration=200.0,
            severity=1.0,
            ramp_time=0.0,
        )
        fc = self._make_fc([scenario], seed=2)
        for t in range(100):
            r = fc.apply("CAR-01", float(t), self._base_reading(), f"m{t}", "ts")
            assert 0.0 <= r["rpm"] <= 9000.0

    def test_battery_degradation_lowers_voltage(self):
        scenario = FaultScenario(
            fault_type=FaultType.BATTERY_DEGRADATION,
            vehicle_id="CAR-01",
            start_time=0.0,
            duration=200.0,
            severity=1.0,
            ramp_time=0.0,
        )
        fc = self._make_fc([scenario])
        voltages = []
        for t in range(20):
            r = fc.apply("CAR-01", float(t), self._base_reading(), f"m{t}", "ts")
            voltages.append(r["battery_voltage"])
        # Average should be below nominal
        assert sum(voltages) / len(voltages) < 12.6

    def test_sensor_drift_accumulates(self):
        scenario = FaultScenario(
            fault_type=FaultType.SENSOR_DRIFT,
            vehicle_id="CAR-01",
            start_time=0.0,
            duration=500.0,
            severity=1.0,
            ramp_time=0.0,
        )
        fc = self._make_fc([scenario])
        temps = []
        base_temp = 88.0
        for t in range(100):
            r = fc.apply("CAR-01", float(t), {**self._base_reading(), "engine_temp": base_temp},
                         f"m{t}", "ts")
            temps.append(r["engine_temp"])
        # Drift should grow — later temps should be higher than earlier
        assert temps[-1] > temps[0]

    def test_fault_only_affects_target_vehicle(self):
        scenario = FaultScenario(
            fault_type=FaultType.COOLING_DEGRADATION,
            vehicle_id="CAR-01",
            start_time=0.0,
            duration=100.0,
            severity=1.0,
            ramp_time=0.0,
        )
        fc = self._make_fc([scenario])
        reading_car02 = {**self._base_reading(), "vehicle_id": "CAR-02"}
        r = fc.apply("CAR-02", 10.0, reading_car02, "m1", "ts")
        # CAR-02 should be unaffected
        assert r["engine_temp"] == 88.0

    def test_no_fault_in_normal_mode(self):
        fc = FaultController(make_normal_scenarios())
        for t in range(100):
            r = fc.apply("CAR-01", float(t), self._base_reading(), f"m{t}", "ts")
            assert r["engine_temp"] == 88.0


# ---------------------------------------------------------------------------
# 5. Fault lifecycle
# ---------------------------------------------------------------------------

class TestFaultLifecycle:

    def _lifecycle_states(
        self, fault_type: FaultType, steps: int = 400
    ) -> list[FaultState]:
        scenario = FaultScenario(
            fault_type=fault_type,
            vehicle_id="CAR-01",
            start_time=50.0,
            duration=100.0,
            severity=0.8,
            ramp_time=30.0,
            recovery_time=60.0,
        )
        fc = FaultController([scenario], rng=random.Random(0))
        states = []
        for t in range(steps):
            fc.apply("CAR-01", float(t), {
                "vehicle_id": "CAR-01", "speed": 80.0, "rpm": 3000.0,
                "engine_temp": 88.0, "battery_voltage": 12.6, "fuel_level": 80.0,
            }, f"m{t}", "ts")
            _, fs, _ = fc.ground_truth_for("CAR-01")
            states.append(fs)
        return states

    def test_lifecycle_includes_normal_before_fault(self):
        states = self._lifecycle_states(FaultType.COOLING_DEGRADATION)
        assert states[0] == FaultState.NORMAL

    def test_lifecycle_includes_starting(self):
        states = self._lifecycle_states(FaultType.COOLING_DEGRADATION)
        assert FaultState.STARTING in states

    def test_lifecycle_includes_active(self):
        states = self._lifecycle_states(FaultType.COOLING_DEGRADATION)
        assert FaultState.ACTIVE in states

    def test_lifecycle_includes_recovering(self):
        states = self._lifecycle_states(FaultType.COOLING_DEGRADATION)
        assert FaultState.RECOVERING in states

    def test_lifecycle_includes_resolved(self):
        states = self._lifecycle_states(FaultType.COOLING_DEGRADATION, steps=300)
        assert FaultState.RESOLVED in states

    def test_lifecycle_order_is_correct(self):
        """States must follow: NORMAL → STARTING → ACTIVE → RECOVERING → RESOLVED (then NORMAL again)"""
        states = self._lifecycle_states(FaultType.COOLING_DEGRADATION, steps=300)
        # Compress consecutive duplicates to get the transition sequence
        compressed = [states[0]]
        for s in states[1:]:
            if s != compressed[-1]:
                compressed.append(s)
        # Expected order (NORMAL may appear at start and after RESOLVED)
        valid_order = [
            FaultState.NORMAL, FaultState.STARTING, FaultState.ACTIVE,
            FaultState.RECOVERING, FaultState.RESOLVED
        ]
        # Walk compressed states and verify they only move forward in valid_order
        # (NORMAL is allowed to appear at the start and the very end)
        seen_max_idx = -1
        for s in compressed:
            if s == FaultState.NORMAL and seen_max_idx >= valid_order.index(FaultState.RESOLVED):
                continue  # NORMAL after RESOLVED is fine
            if s not in valid_order:
                continue
            idx = valid_order.index(s)
            assert idx >= seen_max_idx, f"Unexpected state regression: {s}"
            seen_max_idx = max(seen_max_idx, idx)


# ---------------------------------------------------------------------------
# 6. Ground-truth labeling
# ---------------------------------------------------------------------------

class TestGroundTruth:

    def test_ground_truth_recorded_for_every_step(self):
        fc = FaultController([], rng=random.Random(0))
        for t in range(10):
            fc.apply("CAR-01", float(t), {
                "vehicle_id": "CAR-01", "speed": 0.0, "rpm": 800.0,
                "engine_temp": 85.0, "battery_voltage": 12.6, "fuel_level": 90.0,
            }, f"m{t}", "ts")
        rows = fc.get_ground_truth_rows()
        assert len(rows) == 10

    def test_ground_truth_fault_type_none_when_no_fault(self):
        fc = FaultController([], rng=random.Random(0))
        fc.apply("CAR-01", 5.0, {
            "vehicle_id": "CAR-01", "speed": 0.0, "rpm": 800.0,
            "engine_temp": 85.0, "battery_voltage": 12.6, "fuel_level": 90.0,
        }, "m1", "ts")
        ft, _, _ = fc.ground_truth_for("CAR-01")
        assert ft == FaultType.NONE

    def test_ground_truth_correct_during_active_fault(self):
        scenario = FaultScenario(
            fault_type=FaultType.COOLING_DEGRADATION,
            vehicle_id="CAR-01",
            start_time=0.0,
            duration=100.0,
            severity=0.5,
            ramp_time=0.0,
        )
        fc = FaultController([scenario], rng=random.Random(0))
        fc.apply("CAR-01", 10.0, {
            "vehicle_id": "CAR-01", "speed": 80.0, "rpm": 3000.0,
            "engine_temp": 88.0, "battery_voltage": 12.6, "fuel_level": 80.0,
        }, "m1", "ts")
        ft, fs, sev = fc.ground_truth_for("CAR-01")
        assert ft == FaultType.COOLING_DEGRADATION
        assert fs == FaultState.ACTIVE
        assert sev > 0.0

    def test_ground_truth_csv_export(self, tmp_path):
        scenario = FaultScenario(
            fault_type=FaultType.RPM_INSTABILITY,
            vehicle_id="CAR-01",
            start_time=0.0,
            duration=10.0,
            severity=0.5,
            ramp_time=0.0,
        )
        fc = FaultController([scenario], rng=random.Random(0))
        for t in range(15):
            fc.apply("CAR-01", float(t), {
                "vehicle_id": "CAR-01", "speed": 80.0, "rpm": 3000.0,
                "engine_temp": 88.0, "battery_voltage": 12.6, "fuel_level": 80.0,
            }, f"m{t}", f"ts{t}")
        csv_path = str(tmp_path / "gt.csv")
        fc.export_ground_truth_csv(csv_path)
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 15
        assert "fault_type" in rows[0]
        assert "fault_state" in rows[0]
        assert "severity" in rows[0]


# ---------------------------------------------------------------------------
# 7. Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:

    def test_same_seed_produces_identical_speeds(self):
        vsm1 = VehicleStateMachine(DEFAULT_VEHICLES[0], rng=random.Random(42))
        vsm2 = VehicleStateMachine(DEFAULT_VEHICLES[0], rng=random.Random(42))
        for _ in range(200):
            r1 = vsm1.step()
            r2 = vsm2.step()
            assert r1["speed"] == r2["speed"]
            assert r1["rpm"] == r2["rpm"]
            assert r1["engine_temp"] == r2["engine_temp"]

    def test_different_seeds_produce_different_results(self):
        vsm1 = VehicleStateMachine(DEFAULT_VEHICLES[0], rng=random.Random(1))
        vsm2 = VehicleStateMachine(DEFAULT_VEHICLES[0], rng=random.Random(99))
        readings1 = run_steps(vsm1, 200)
        readings2 = run_steps(vsm2, 200)
        speeds1 = [r["speed"] for r in readings1]
        speeds2 = [r["speed"] for r in readings2]
        # Very unlikely to be identical with different seeds
        assert speeds1 != speeds2

    def test_run_simulation_reproducible(self):
        records1, _ = run_simulation(DEFAULT_VEHICLES, [], 50, 1.0, seed=42)
        records2, _ = run_simulation(DEFAULT_VEHICLES, [], 50, 1.0, seed=42)
        assert len(records1) == len(records2)
        for r1, r2 in zip(records1, records2):
            assert r1["vehicle_id"] == r2["vehicle_id"]


# ---------------------------------------------------------------------------
# 8. MQTT payload compatibility (Phase 0 validation contract)
# ---------------------------------------------------------------------------

class TestMQTTCompatibility:

    def test_all_generated_payloads_pass_phase0_validation(self):
        """Every payload produced by run_simulation must pass Phase 0 validation."""
        records, _ = run_simulation(DEFAULT_VEHICLES, [], 200, 1.0, seed=42)
        failures = []
        for record in records:
            ok, reason = validate_telemetry(record)
            if not ok:
                failures.append((record, reason))
        assert failures == [], f"{len(failures)} payloads failed validation:\n{failures[:3]}"

    def test_payloads_contain_required_fields(self):
        records, _ = run_simulation(DEFAULT_VEHICLES[:1], [], 30, 1.0, seed=1)
        required = ["vehicle_id", "speed", "rpm", "engine_temp"]
        for r in records:
            for field in required:
                assert field in r, f"Missing field '{field}' in {r}"

    def test_fault_payloads_still_pass_validation(self):
        """Fault-affected payloads must still pass Phase 0 validation."""
        scenarios = make_fault_scenarios(duration=100.0)
        records, _ = run_simulation(DEFAULT_VEHICLES, scenarios, 200, 1.0, seed=42)
        failures = []
        for record in records:
            ok, reason = validate_telemetry(record)
            if not ok:
                failures.append((record, reason))
        assert failures == [], f"{len(failures)} fault payloads failed validation"

    def test_vehicle_ids_match_profiles(self):
        records, _ = run_simulation(DEFAULT_VEHICLES, [], 30, 1.0, seed=0)
        expected_ids = {p.vehicle_id for p in DEFAULT_VEHICLES}
        actual_ids = {r["vehicle_id"] for r in records}
        assert actual_ids == expected_ids

    def test_message_id_present_and_unique(self):
        records, _ = run_simulation(DEFAULT_VEHICLES[:1], [], 30, 1.0, seed=0)
        ids = [r["message_id"] for r in records]
        assert len(ids) == len(set(ids)), "Duplicate message_ids found"

    def test_event_timestamp_present_and_valid_iso8601(self):
        from datetime import datetime
        records, _ = run_simulation(DEFAULT_VEHICLES[:1], [], 20, 1.0, seed=0)
        for r in records:
            assert "event_timestamp" in r
            datetime.fromisoformat(r["event_timestamp"])  # raises if invalid


# ---------------------------------------------------------------------------
# 9. Multi-vehicle simulation
# ---------------------------------------------------------------------------

class TestMultiVehicle:

    def test_all_vehicles_produce_records(self):
        records, _ = run_simulation(DEFAULT_VEHICLES, [], 10, 1.0, seed=0)
        vehicle_ids = {r["vehicle_id"] for r in records}
        assert vehicle_ids == {"CAR-01", "CAR-02", "CAR-03"}

    def test_record_count_equals_vehicles_times_steps(self):
        steps = 20
        records, _ = run_simulation(DEFAULT_VEHICLES, [], steps, 1.0, seed=0)
        assert len(records) == steps * len(DEFAULT_VEHICLES)


# ---------------------------------------------------------------------------
# 10. Fault scenario helpers
# ---------------------------------------------------------------------------

class TestFaultScenarioHelpers:

    def test_normal_scenarios_is_empty(self):
        assert make_normal_scenarios() == []

    def test_fault_scenarios_covers_expected_faults(self):
        scenarios = make_fault_scenarios()
        fault_types = {s.fault_type for s in scenarios}
        assert FaultType.COOLING_DEGRADATION in fault_types
        assert FaultType.RPM_INSTABILITY in fault_types
        assert FaultType.BATTERY_DEGRADATION in fault_types

    def test_mixed_scenarios_covers_all_fault_types(self):
        scenarios = make_mixed_scenarios()
        fault_types = {s.fault_type for s in scenarios}
        expected = {
            FaultType.COOLING_DEGRADATION,
            FaultType.RPM_INSTABILITY,
            FaultType.BATTERY_DEGRADATION,
            FaultType.SENSOR_DRIFT,
            FaultType.EXCESSIVE_IDLING,
        }
        assert expected <= fault_types

    def test_fault_scenario_severity_validation(self):
        with pytest.raises(ValueError):
            FaultScenario(
                fault_type=FaultType.COOLING_DEGRADATION,
                vehicle_id="CAR-01",
                start_time=0.0,
                duration=100.0,
                severity=1.5,  # invalid
            )
