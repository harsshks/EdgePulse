# simulator/publisher.py
# EdgePulse Phase 1 — Stateful Vehicle Simulator
#
# Replaces the Phase 0 random simulator with a stateful, temporally correlated
# multi-vehicle simulation supporting fault injection and ground-truth labeling.
#
# Usage:
#   python simulator/publisher.py [options]
#
# Options:
#   --mode    normal | fault | mixed  (default: normal)
#   --seed    integer seed for reproducibility (default: random)
#   --steps   number of telemetry steps per vehicle (default: 600)
#   --dt      seconds between steps (default: 1.0)
#   --export  path to write ground-truth CSV (default: none)
#   --vehicles vehicle IDs to simulate (default: CAR-01 CAR-02 CAR-03)
#
# Examples:
#   python simulator/publisher.py --mode normal --seed 42 --steps 300
#   python simulator/publisher.py --mode fault --seed 42 --export data/gt.csv
#   python simulator/publisher.py --mode mixed --seed 7 --steps 1800

import argparse
import json
import logging
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import paho.mqtt.client as mqtt

from config import MQTT_BROKER_HOST, MQTT_BROKER_PORT, MQTT_TOPIC, LOG_LEVEL
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

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("edgepulse.simulator")


# ---------------------------------------------------------------------------
# EXCESSIVE_IDLING fault — implemented at state machine level
# ---------------------------------------------------------------------------

def _apply_excessive_idling(
    vsm: VehicleStateMachine,
    fault_state: FaultState,
    severity: float,
) -> None:
    """
    Excessive idling: when the fault is active, dramatically extend the idle
    duration target so the vehicle stays in IDLE far longer than normal.
    This modifies the VSM's internal idle target rather than sensor values.
    """
    if fault_state in (FaultState.STARTING, FaultState.ACTIVE):
        # Extend idle to 5–10 minutes (severity-scaled)
        base_normal = vsm.profile.normal_idle_duration
        extended = base_normal + (300.0 + 300.0 * severity)
        if vsm.sensors.state == VehicleState.IDLE:
            vsm._idle_target = max(vsm._idle_target, extended)


# ---------------------------------------------------------------------------
# Simulation runner (without MQTT — for testing and CSV export)
# ---------------------------------------------------------------------------

def run_simulation(
    profiles: list[VehicleProfile],
    scenarios: list[FaultScenario],
    steps: int,
    dt: float,
    seed: Optional[int],
) -> tuple[list[dict], FaultController]:
    """
    Runs the simulation and returns (telemetry_records, fault_controller).
    Does NOT publish to MQTT — use this for testing and CSV export.
    """
    rng = random.Random(seed)
    fault_controller = FaultController(scenarios, rng=random.Random(seed))

    # One state machine per vehicle
    state_machines: dict[str, VehicleStateMachine] = {
        p.vehicle_id: VehicleStateMachine(p, rng=random.Random(rng.randint(0, 2**32)))
        for p in profiles
    }

    all_records: list[dict] = []
    sim_time = 0.0

    for _ in range(steps):
        for vehicle_id, vsm in state_machines.items():
            # --- Apply EXCESSIVE_IDLING state influence before stepping ---
            ft, fs, sev = fault_controller.ground_truth_for(vehicle_id)
            if ft == FaultType.EXCESSIVE_IDLING:
                _apply_excessive_idling(vsm, fs, sev)

            # --- Normal vehicle step ---
            reading = vsm.step(dt)

            # --- Build telemetry payload ---
            msg_id = str(uuid.uuid4())
            event_ts = datetime.now(timezone.utc).isoformat()

            # --- Apply fault effects ---
            reading = fault_controller.apply(
                vehicle_id=vehicle_id,
                sim_time=sim_time,
                reading=reading,
                message_id=msg_id,
                event_timestamp=event_ts,
            )

            payload = {
                "message_id":      msg_id,
                "event_timestamp": event_ts,
                "vehicle_id":      reading["vehicle_id"],
                "speed":           reading["speed"],
                "rpm":             reading["rpm"],
                "engine_temp":     reading["engine_temp"],
                "battery_voltage": reading.get("battery_voltage"),
                "fuel_level":      reading.get("fuel_level"),
            }
            all_records.append(payload)

        sim_time += dt

    return all_records, fault_controller


# ---------------------------------------------------------------------------
# MQTT publisher
# ---------------------------------------------------------------------------

def publish_simulation(
    profiles: list[VehicleProfile],
    scenarios: list[FaultScenario],
    steps: int,
    dt: float,
    seed: Optional[int],
    export_path: Optional[str],
) -> None:
    """
    Runs simulation and publishes each telemetry record to MQTT in real time.
    """
    rng = random.Random(seed)
    fault_controller = FaultController(scenarios, rng=random.Random(seed))

    state_machines: dict[str, VehicleStateMachine] = {
        p.vehicle_id: VehicleStateMachine(p, rng=random.Random(rng.randint(0, 2**32)))
        for p in profiles
    }

    # MQTT setup
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    try:
        client.connect(MQTT_BROKER_HOST, MQTT_BROKER_PORT)
        logger.info(
            "Connected to MQTT broker at %s:%d — starting simulation.",
            MQTT_BROKER_HOST, MQTT_BROKER_PORT,
        )
    except ConnectionRefusedError:
        logger.critical(
            "Connection refused at %s:%d. Is Mosquitto running?",
            MQTT_BROKER_HOST, MQTT_BROKER_PORT,
        )
        raise SystemExit(1)

    sim_time = 0.0
    step_start = time.monotonic()

    try:
        for step in range(steps):
            for vehicle_id, vsm in state_machines.items():
                # --- Excessive idling: state machine influence ---
                ft, fs, sev = fault_controller.ground_truth_for(vehicle_id)
                if ft == FaultType.EXCESSIVE_IDLING:
                    _apply_excessive_idling(vsm, fs, sev)

                # --- Normal vehicle step ---
                reading = vsm.step(dt)

                # --- Build payload ---
                msg_id = str(uuid.uuid4())
                event_ts = datetime.now(timezone.utc).isoformat()

                # --- Apply fault effects ---
                reading = fault_controller.apply(
                    vehicle_id=vehicle_id,
                    sim_time=sim_time,
                    reading=reading,
                    message_id=msg_id,
                    event_timestamp=event_ts,
                )

                payload = {
                    "message_id":      msg_id,
                    "event_timestamp": event_ts,
                    "vehicle_id":      reading["vehicle_id"],
                    "speed":           reading["speed"],
                    "rpm":             reading["rpm"],
                    "engine_temp":     reading["engine_temp"],
                    "battery_voltage": reading.get("battery_voltage"),
                    "fuel_level":      reading.get("fuel_level"),
                }

                client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)

                ft_now, fs_now, sev_now = fault_controller.ground_truth_for(vehicle_id)
                logger.info(
                    "step=%d vehicle=%s state=%-12s speed=%5.1f rpm=%5.0f "
                    "temp=%5.1f fault=%s(%s)",
                    step, vehicle_id, vsm.state.name,
                    reading["speed"], reading["rpm"], reading["engine_temp"],
                    ft_now.value, fs_now.value,
                )

            sim_time += dt

            # Real-time pacing: sleep remainder of dt
            elapsed = time.monotonic() - step_start
            sleep_time = (step + 1) * dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Simulation stopped by user at step %d.", step)
    finally:
        client.disconnect()
        logger.info("Disconnected from MQTT broker.")

        if export_path:
            fault_controller.export_ground_truth_csv(export_path)
            logger.info("Ground-truth CSV written to: %s", export_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EdgePulse Phase 1 Vehicle Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  normal  — healthy vehicle behaviour only (no faults)
  fault   — one fault per vehicle at staggered times
  mixed   — multiple faults per vehicle including recovery

Examples:
  python simulator/publisher.py --mode normal --seed 42 --steps 600
  python simulator/publisher.py --mode fault  --seed 42 --export data/ground_truth.csv
  python simulator/publisher.py --mode mixed  --seed 7  --steps 1800 --dt 1.0
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "fault", "mixed"],
        default="normal",
        help="Simulation mode (default: normal)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: random)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=600,
        help="Number of time steps per vehicle (default: 600)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Time step in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Path to write ground-truth CSV (optional)",
    )
    parser.add_argument(
        "--vehicles",
        nargs="+",
        default=None,
        help="Vehicle IDs to simulate (default: CAR-01 CAR-02 CAR-03)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Select vehicle profiles
    if args.vehicles:
        profiles = [p for p in DEFAULT_VEHICLES if p.vehicle_id in args.vehicles]
        if not profiles:
            logger.error("No matching vehicle profiles for: %s", args.vehicles)
            raise SystemExit(1)
    else:
        profiles = DEFAULT_VEHICLES

    # Select fault scenarios
    if args.mode == "normal":
        scenarios = make_normal_scenarios()
    elif args.mode == "fault":
        scenarios = make_fault_scenarios()
    else:
        scenarios = make_mixed_scenarios()

    logger.info(
        "Starting simulation | mode=%s seed=%s steps=%d dt=%.1fs vehicles=%s",
        args.mode, args.seed, args.steps, args.dt,
        [p.vehicle_id for p in profiles],
    )

    publish_simulation(
        profiles=profiles,
        scenarios=scenarios,
        steps=args.steps,
        dt=args.dt,
        seed=args.seed,
        export_path=args.export,
    )


if __name__ == "__main__":
    main()
