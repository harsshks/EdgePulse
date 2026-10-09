# simulator/vehicle.py
# Stateful vehicle model for EdgePulse Phase 1.
#
# Design principles:
#   - Every sensor value evolves from its previous value — no independent random jumps.
#   - Vehicle state drives behaviour (IDLE, ACCELERATING, CRUISING, etc.).
#   - Per-vehicle characteristics create realistic fleet diversity.
#   - All randomness is seeded so simulations are reproducible.
#   - Pure data + logic — no MQTT, no I/O.  Easily unit-testable.
#
# Sensor relationships (simplified but internally consistent):
#   speed    → driven by state + throttle; changes gradually
#   rpm      → correlates with speed and engine load; higher under acceleration
#   engine_temp → rises under sustained load, falls slowly when load drops
#   battery_voltage → stable with small realistic fluctuations
#   fuel_level → decreases proportionally to engine load; never refuelled in sim
#
# All values stay within Phase 0 validation ranges:
#   speed: 0–300 km/h  |  rpm: 0–9000  |  engine_temp: -10–130°C

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Vehicle state enum
# ---------------------------------------------------------------------------

class VehicleState(Enum):
    STOPPED      = auto()   # engine off / key off
    IDLE         = auto()   # engine on, vehicle stationary
    ACCELERATING = auto()   # increasing speed
    CRUISING     = auto()   # steady speed
    DECELERATING = auto()   # decreasing speed


# ---------------------------------------------------------------------------
# Per-vehicle characteristics
# ---------------------------------------------------------------------------

@dataclass
class VehicleProfile:
    """
    Defines the 'personality' of a vehicle.
    These values represent the vehicle's normal operating baseline.
    They are NOT anomaly-detection thresholds — they are simulation parameters.
    """
    vehicle_id: str

    # Normal operating temperature at cruise (°C)
    baseline_temp: float = 88.0

    # Temperature at idle (slightly below cruise because of lower load)
    idle_temp: float = 82.0

    # How fast temp rises per second under load (°C/s)
    temp_rise_rate: float = 0.15

    # How fast temp falls per second when load drops (°C/s)
    temp_fall_rate: float = 0.05

    # RPM at idle (engine running, not moving)
    idle_rpm: float = 850.0

    # RPM per km/h — approximates gear ratio effect
    rpm_per_kmh: float = 30.0

    # Extra RPM added during acceleration (load spike)
    accel_rpm_boost: float = 400.0

    # Max cruise speed for this vehicle (km/h)
    cruise_speed: float = 100.0

    # Acceleration rate (km/h per second)
    accel_rate: float = 8.0

    # Deceleration rate (km/h per second) — positive value means slowing
    decel_rate: float = 10.0

    # Fuel consumption rate at cruise (% per second)
    fuel_consumption_cruise: float = 0.003

    # Extra fuel consumption multiplier during acceleration
    fuel_consumption_accel_mult: float = 2.0

    # Normal battery voltage
    battery_voltage_nominal: float = 12.6

    # Small random fluctuation in battery voltage (std dev)
    battery_voltage_noise: float = 0.05

    # How long (seconds) this vehicle typically idles before starting to move
    normal_idle_duration: float = 45.0


# Default fleet of 3 vehicles with distinct characteristics
DEFAULT_VEHICLES = [
    VehicleProfile(
        vehicle_id="CAR-01",
        baseline_temp=88.0,
        idle_temp=82.0,
        temp_rise_rate=0.14,
        temp_fall_rate=0.05,
        idle_rpm=820.0,
        rpm_per_kmh=28.0,
        accel_rpm_boost=380.0,
        cruise_speed=100.0,
        accel_rate=9.0,
        decel_rate=11.0,
        fuel_consumption_cruise=0.003,
        battery_voltage_nominal=12.6,
        battery_voltage_noise=0.04,
        normal_idle_duration=40.0,
    ),
    VehicleProfile(
        vehicle_id="CAR-02",
        baseline_temp=93.0,    # runs slightly hotter
        idle_temp=86.0,
        temp_rise_rate=0.18,   # heats up faster
        temp_fall_rate=0.04,   # cools slower
        idle_rpm=900.0,
        rpm_per_kmh=32.0,
        accel_rpm_boost=450.0,
        cruise_speed=95.0,
        accel_rate=7.5,
        decel_rate=9.0,
        fuel_consumption_cruise=0.0035,
        battery_voltage_nominal=12.4,   # slightly lower baseline
        battery_voltage_noise=0.06,
        normal_idle_duration=50.0,
    ),
    VehicleProfile(
        vehicle_id="CAR-03",
        baseline_temp=86.0,    # runs cooler
        idle_temp=80.0,
        temp_rise_rate=0.12,
        temp_fall_rate=0.06,   # cools faster
        idle_rpm=800.0,
        rpm_per_kmh=26.0,
        accel_rpm_boost=350.0,
        cruise_speed=110.0,    # faster cruiser
        accel_rate=10.0,
        decel_rate=12.0,
        fuel_consumption_cruise=0.0025,
        battery_voltage_nominal=12.7,
        battery_voltage_noise=0.03,
        normal_idle_duration=35.0,
    ),
]


# ---------------------------------------------------------------------------
# Vehicle sensor state
# ---------------------------------------------------------------------------

@dataclass
class VehicleSensorState:
    """
    Mutable sensor values that evolve over time.
    Separated from VehicleProfile so profile stays immutable.
    """
    speed: float = 0.0           # km/h
    rpm: float = 0.0             # engine RPM
    engine_temp: float = 20.0   # °C — starts at ambient
    battery_voltage: float = 12.6
    fuel_level: float = 100.0    # percent, 0–100

    # Current operational state
    state: VehicleState = VehicleState.STOPPED

    # Seconds spent in current state
    state_elapsed: float = 0.0


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class VehicleStateMachine:
    """
    Drives a single vehicle's operational state and sensor values forward
    by one time step.

    Usage:
        vsm = VehicleStateMachine(profile, rng=random.Random(42))
        for _ in range(3600):
            reading = vsm.step(dt=1.0)
            # reading is a dict ready to publish

    The state machine follows a simple loop:
        STOPPED → IDLE → ACCELERATING → CRUISING → DECELERATING → IDLE → ...

    Fault injection is handled by the caller (FaultController) by modifying
    sensor values AFTER a normal step is computed.
    """

    def __init__(
        self,
        profile: VehicleProfile,
        rng: Optional[random.Random] = None,
        initial_state: VehicleState = VehicleState.STOPPED,
    ):
        self.profile = profile
        self.rng = rng or random.Random()
        self.sensors = VehicleSensorState(
            state=initial_state,
            engine_temp=20.0,
            battery_voltage=profile.battery_voltage_nominal,
        )
        # How long to stay in IDLE before accelerating (randomised per trip)
        self._idle_target = self._random_idle_duration()
        # Cruise speed target for this trip (slight variation per trip)
        self._cruise_target = self._random_cruise_speed()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(self, dt: float = 1.0) -> dict:
        """
        Advance simulation by `dt` seconds.
        Returns sensor readings as a dict (without timestamps — caller adds those).
        """
        self.sensors.state_elapsed += dt
        self._transition_state()
        self._update_sensors(dt)
        return self._reading()

    @property
    def state(self) -> VehicleState:
        return self.sensors.state

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def _transition_state(self) -> None:
        s = self.sensors
        p = self.profile

        if s.state == VehicleState.STOPPED:
            # Start engine → go to IDLE
            s.state = VehicleState.IDLE
            s.state_elapsed = 0.0
            self._idle_target = self._random_idle_duration()

        elif s.state == VehicleState.IDLE:
            if s.state_elapsed >= self._idle_target:
                s.state = VehicleState.ACCELERATING
                s.state_elapsed = 0.0
                self._cruise_target = self._random_cruise_speed()

        elif s.state == VehicleState.ACCELERATING:
            if s.speed >= self._cruise_target * 0.95:
                s.state = VehicleState.CRUISING
                s.state_elapsed = 0.0
                # Cruise for a random duration (30–120s)
                self._cruise_duration = self.rng.uniform(30, 120)

        elif s.state == VehicleState.CRUISING:
            if s.state_elapsed >= self._cruise_duration:
                s.state = VehicleState.DECELERATING
                s.state_elapsed = 0.0

        elif s.state == VehicleState.DECELERATING:
            if s.speed <= 2.0:
                s.speed = 0.0
                s.state = VehicleState.IDLE
                s.state_elapsed = 0.0
                self._idle_target = self._random_idle_duration()

    # ------------------------------------------------------------------
    # Sensor update
    # ------------------------------------------------------------------

    def _update_sensors(self, dt: float) -> None:
        s = self.sensors
        p = self.profile

        # --- Speed ---
        if s.state == VehicleState.ACCELERATING:
            s.speed = min(s.speed + p.accel_rate * dt, self._cruise_target)
        elif s.state == VehicleState.DECELERATING:
            s.speed = max(s.speed - p.decel_rate * dt, 0.0)
        elif s.state == VehicleState.CRUISING:
            # Small speed variation around target (±2 km/h)
            noise = self.rng.gauss(0, 0.5)
            s.speed = max(0.0, min(self._cruise_target + noise, 300.0))
        elif s.state in (VehicleState.IDLE, VehicleState.STOPPED):
            s.speed = 0.0

        # --- RPM ---
        if s.state == VehicleState.IDLE:
            target_rpm = p.idle_rpm + self.rng.gauss(0, 20)
        elif s.state == VehicleState.ACCELERATING:
            target_rpm = p.idle_rpm + s.speed * p.rpm_per_kmh + p.accel_rpm_boost
        elif s.state == VehicleState.CRUISING:
            target_rpm = p.idle_rpm + s.speed * p.rpm_per_kmh + self.rng.gauss(0, 30)
        elif s.state == VehicleState.DECELERATING:
            target_rpm = max(p.idle_rpm, p.idle_rpm + s.speed * p.rpm_per_kmh * 0.6)
        else:  # STOPPED
            target_rpm = 0.0

        target_rpm = max(0.0, min(target_rpm, 9000.0))
        # RPM moves toward target at 30% per second (gradual response)
        s.rpm += (target_rpm - s.rpm) * min(0.3 * dt, 1.0)
        s.rpm = max(0.0, min(s.rpm, 9000.0))

        # --- Engine temperature ---
        if s.state in (VehicleState.IDLE, VehicleState.STOPPED):
            target_temp = p.idle_temp
            rate = p.temp_fall_rate
        elif s.state == VehicleState.ACCELERATING:
            # High load: aim above baseline
            target_temp = p.baseline_temp + 8.0
            rate = p.temp_rise_rate
        elif s.state == VehicleState.CRUISING:
            target_temp = p.baseline_temp
            rate = p.temp_rise_rate if s.engine_temp < p.baseline_temp else p.temp_fall_rate
        else:  # DECELERATING
            target_temp = p.baseline_temp - 2.0
            rate = p.temp_fall_rate

        # Gradual approach to target
        diff = target_temp - s.engine_temp
        change = min(abs(diff), rate * dt) * (1 if diff > 0 else -1)
        s.engine_temp += change + self.rng.gauss(0, 0.1)
        s.engine_temp = max(-10.0, min(s.engine_temp, 130.0))

        # --- Battery voltage ---
        noise = self.rng.gauss(0, p.battery_voltage_noise)
        s.battery_voltage = round(
            max(10.0, min(15.0, s.battery_voltage + noise * 0.1)), 3
        )
        # Slowly drift back toward nominal
        s.battery_voltage += (p.battery_voltage_nominal - s.battery_voltage) * 0.01

        # --- Fuel level ---
        if s.state == VehicleState.ACCELERATING:
            consumption = p.fuel_consumption_cruise * p.fuel_consumption_accel_mult * dt
        elif s.state == VehicleState.CRUISING:
            consumption = p.fuel_consumption_cruise * dt
        elif s.state == VehicleState.DECELERATING:
            consumption = p.fuel_consumption_cruise * 0.4 * dt
        else:
            consumption = p.fuel_consumption_cruise * 0.1 * dt  # idle consumption

        s.fuel_level = max(0.0, s.fuel_level - consumption)

    # ------------------------------------------------------------------
    # Reading dict
    # ------------------------------------------------------------------

    def _reading(self) -> dict:
        s = self.sensors
        return {
            "vehicle_id":      self.profile.vehicle_id,
            "speed":           round(s.speed, 2),
            "rpm":             round(s.rpm, 1),
            "engine_temp":     round(s.engine_temp, 2),
            "battery_voltage": round(s.battery_voltage, 3),
            "fuel_level":      round(s.fuel_level, 3),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_idle_duration(self) -> float:
        """Randomised idle duration around the vehicle's normal idle time."""
        base = self.profile.normal_idle_duration
        return self.rng.uniform(base * 0.5, base * 1.5)

    def _random_cruise_speed(self) -> float:
        """Randomised cruise speed around the vehicle's cruise target."""
        base = self.profile.cruise_speed
        return self.rng.uniform(base * 0.8, base * 1.15)
