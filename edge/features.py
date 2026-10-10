# edge/features.py
# Temporal feature extraction for EdgePulse Phase 2.
#
# Design:
#   - Pure functions — no I/O, no database, no MQTT.
#   - All temporal calculations use actual elapsed time from event_timestamp.
#   - Features are UNAVAILABLE (None) rather than fabricated when there is
#     insufficient history or an irregular time gap.
#   - Units are documented on every feature.
#   - Handles duplicate timestamps, out-of-order observations, and gaps.
#
# Feature availability rules:
#   - Rolling statistics require at least `min_samples` observations in the window.
#   - Rate-of-change features require at least 2 observations with dt > 0.
#   - Observations with dt > MAX_GAP_SECONDS are treated as a new sequence;
#     rates are not computed across the gap.
#   - Duplicate timestamps (dt == 0) are ignored for rate calculations.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# Maximum gap in seconds before rate calculations are suppressed.
# A gap larger than this suggests buffered/delayed telemetry, not continuity.
MAX_GAP_SECONDS: float = 30.0

# Minimum number of samples required before rolling statistics are emitted.
MIN_ROLLING_SAMPLES: int = 3


@dataclass
class TelemetryObservation:
    """
    One normalised observation fed into the feature window.
    All numeric fields are float or None.
    """
    vehicle_id:      str
    event_timestamp: datetime          # timezone-aware UTC
    speed:           float             # km/h
    rpm:             float             # rev/min
    engine_temp:     float             # °C
    battery_voltage: Optional[float]   # V
    fuel_level:      Optional[float]   # %


@dataclass
class FeatureVector:
    """
    All features computed for one telemetry observation.
    None means the feature could not be computed at this step.

    Units:
        speed_*         km/h
        rpm_*           rev/min
        engine_temp_*   °C
        temp_slope      °C/min
        battery_*       V
        voltage_trend   V/min
        fuel_*          %
        fuel_rate       %/min  (negative = consumption)
        speed_change_rate   km/h per second
        acceleration        km/h per second  (positive = acceleration)
        rpm_change_rate     rev/min per second
        idle_duration_s     seconds
    """
    vehicle_id:      str
    event_timestamp: str    # ISO-8601 string, for serialisation

    # --- Engine temperature ---
    engine_temp:            float
    temp_rolling_mean:      Optional[float] = None   # °C
    temp_rolling_min:       Optional[float] = None   # °C
    temp_rolling_max:       Optional[float] = None   # °C
    temp_rolling_std:       Optional[float] = None   # °C
    temp_slope:             Optional[float] = None   # °C/min
    temp_above_threshold_s: Optional[float] = None   # seconds above warning threshold

    # --- RPM ---
    rpm:                float = 0.0
    rpm_rolling_mean:   Optional[float] = None   # rev/min
    rpm_rolling_std:    Optional[float] = None   # rev/min
    rpm_variability:    Optional[float] = None   # coefficient of variation (std/mean)
    rpm_change_rate:    Optional[float] = None   # rev/min per second

    # --- Speed ---
    speed:              float = 0.0
    speed_rolling_mean: Optional[float] = None   # km/h
    speed_rolling_std:  Optional[float] = None   # km/h
    speed_change_rate:  Optional[float] = None   # km/h per second
    acceleration:       Optional[float] = None   # km/h per second

    # --- Battery ---
    battery_voltage:        Optional[float] = None   # V
    voltage_rolling_mean:   Optional[float] = None   # V
    voltage_rolling_std:    Optional[float] = None   # V
    voltage_trend:          Optional[float] = None   # V/min

    # --- Fuel ---
    fuel_level:    Optional[float] = None   # %
    fuel_rate:     Optional[float] = None   # %/min (negative = consuming)

    # --- Cross-signal / context ---
    rpm_variability_at_stable_speed: Optional[float] = None
    idle_duration_s:                 Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


class FeatureWindow:
    """
    Maintains a sliding window of observations for one vehicle and computes
    features when a new observation is added.

    Parameters:
        window_size       — maximum number of observations to retain
        min_samples       — minimum samples before rolling stats are emitted
        temp_warn_thresh  — °C above which engine_temp is considered elevated
        max_gap_seconds   — gaps larger than this suppress rate calculations
    """

    def __init__(
        self,
        window_size: int = 60,
        min_samples: int = MIN_ROLLING_SAMPLES,
        temp_warn_thresh: float = 100.0,
        max_gap_seconds: float = MAX_GAP_SECONDS,
    ):
        self._window_size = window_size
        self._min_samples = min_samples
        self._temp_warn_thresh = temp_warn_thresh
        self._max_gap_seconds = max_gap_seconds
        self._obs: list[TelemetryObservation] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, obs: TelemetryObservation) -> FeatureVector:
        """
        Add one observation and return the current feature vector.
        Out-of-order observations (timestamp <= last seen) are silently ignored.
        """
        if self._obs and obs.event_timestamp <= self._obs[-1].event_timestamp:
            # Duplicate or out-of-order — do not corrupt the window
            # Still return features based on current window without adding obs
            return self._compute(self._obs[-1])

        self._obs.append(obs)
        # Trim to window size
        if len(self._obs) > self._window_size:
            self._obs = self._obs[-self._window_size:]

        return self._compute(obs)

    def clear(self) -> None:
        """Reset the window (e.g. when restarting a vehicle simulation)."""
        self._obs.clear()

    @property
    def sample_count(self) -> int:
        return len(self._obs)

    # ------------------------------------------------------------------
    # Feature computation
    # ------------------------------------------------------------------

    def _compute(self, current: TelemetryObservation) -> FeatureVector:
        fv = FeatureVector(
            vehicle_id=current.vehicle_id,
            event_timestamp=current.event_timestamp.isoformat(),
            engine_temp=current.engine_temp,
            rpm=current.rpm,
            speed=current.speed,
            battery_voltage=current.battery_voltage,
            fuel_level=current.fuel_level,
        )

        n = len(self._obs)
        if n < self._min_samples:
            return fv  # not enough history yet

        window = self._obs

        # --- Rolling temperature stats ---
        temps = [o.engine_temp for o in window]
        fv.temp_rolling_mean = _mean(temps)
        fv.temp_rolling_min  = min(temps)
        fv.temp_rolling_max  = max(temps)
        fv.temp_rolling_std  = _std(temps)

        # --- Temperature slope (°C/min) ---
        fv.temp_slope = _slope_per_minute(
            window, lambda o: o.engine_temp, self._max_gap_seconds
        )

        # --- Duration above warning threshold ---
        fv.temp_above_threshold_s = self._duration_above(
            window, lambda o: o.engine_temp, self._temp_warn_thresh
        )

        # --- Rolling RPM stats ---
        rpms = [o.rpm for o in window]
        fv.rpm_rolling_mean = _mean(rpms)
        fv.rpm_rolling_std  = _std(rpms)
        mean_rpm = fv.rpm_rolling_mean
        if mean_rpm and mean_rpm > 1.0:
            fv.rpm_variability = fv.rpm_rolling_std / mean_rpm
        else:
            fv.rpm_variability = None

        # --- RPM change rate (rev/min per second) ---
        fv.rpm_change_rate = _rate_per_second(
            window, lambda o: o.rpm, self._max_gap_seconds
        )

        # --- Rolling speed stats ---
        speeds = [o.speed for o in window]
        fv.speed_rolling_mean = _mean(speeds)
        fv.speed_rolling_std  = _std(speeds)
        fv.speed_change_rate  = _rate_per_second(
            window, lambda o: o.speed, self._max_gap_seconds
        )
        fv.acceleration = fv.speed_change_rate  # same value, clearer semantic name

        # --- Battery voltage ---
        volts = [o.battery_voltage for o in window if o.battery_voltage is not None]
        if len(volts) >= self._min_samples:
            fv.voltage_rolling_mean = _mean(volts)
            fv.voltage_rolling_std  = _std(volts)
            fv.voltage_trend = _slope_per_minute(
                [o for o in window if o.battery_voltage is not None],
                lambda o: o.battery_voltage,
                self._max_gap_seconds,
            )

        # --- Fuel level ---
        fuels = [o.fuel_level for o in window if o.fuel_level is not None]
        if len(fuels) >= 2:
            fv.fuel_rate = _slope_per_minute(
                [o for o in window if o.fuel_level is not None],
                lambda o: o.fuel_level,
                self._max_gap_seconds,
            )

        # --- Cross-signal: RPM variability at stable speed ---
        # "Stable" = speed_rolling_std < 5 km/h and mean speed > 20 km/h
        if (fv.speed_rolling_std is not None and fv.speed_rolling_std < 5.0
                and fv.speed_rolling_mean is not None and fv.speed_rolling_mean > 20.0):
            fv.rpm_variability_at_stable_speed = fv.rpm_variability

        # --- Idle duration ---
        fv.idle_duration_s = self._idle_duration(window)

        return fv

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _duration_above(self, window, value_fn, threshold: float) -> float:
        """
        Counts seconds where value_fn(obs) > threshold using actual timestamps.
        Uses trapezoidal approximation of time above threshold.
        """
        total = 0.0
        for i in range(1, len(window)):
            prev, curr = window[i - 1], window[i]
            dt = _elapsed_seconds(prev.event_timestamp, curr.event_timestamp)
            if 0 < dt <= self._max_gap_seconds:
                if value_fn(curr) > threshold:
                    total += dt
        return total

    def _idle_duration(self, window) -> float:
        """Counts consecutive seconds at the end of the window where speed == 0."""
        total = 0.0
        for i in range(len(window) - 1, 0, -1):
            obs = window[i]
            if obs.speed > 0.5:
                break
            dt = _elapsed_seconds(window[i - 1].event_timestamp, obs.event_timestamp)
            if 0 < dt <= self._max_gap_seconds:
                total += dt
        return total


# ---------------------------------------------------------------------------
# Pure numeric helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> Optional[float]:
    if len(values) < 2:
        return None
    m = sum(values) / len(values)
    variance = sum((x - m) ** 2 for x in values) / len(values)
    return math.sqrt(variance)


def _elapsed_seconds(t1: datetime, t2: datetime) -> float:
    """Returns elapsed time in seconds between two timezone-aware datetimes."""
    return (t2 - t1).total_seconds()


def _rate_per_second(
    window,
    value_fn,
    max_gap: float,
) -> Optional[float]:
    """
    Computes the rate of change of value_fn per second between the last two
    observations. Returns None if there are fewer than 2 observations, dt <= 0,
    or the gap exceeds max_gap.
    """
    if len(window) < 2:
        return None
    prev, curr = window[-2], window[-1]
    dt = _elapsed_seconds(prev.event_timestamp, curr.event_timestamp)
    if dt <= 0 or dt > max_gap:
        return None
    return (value_fn(curr) - value_fn(prev)) / dt


def _slope_per_minute(
    window,
    value_fn,
    max_gap: float,
) -> Optional[float]:
    """
    Computes rate of change per MINUTE using the last two observations.
    Returns None on gap or insufficient data.
    """
    rate = _rate_per_second(window, value_fn, max_gap)
    if rate is None:
        return None
    return rate * 60.0


def parse_event_timestamp(ts_str: str) -> Optional[datetime]:
    """
    Parses an ISO-8601 timestamp string to a timezone-aware UTC datetime.
    Returns None if parsing fails.
    """
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            # Treat naive timestamps as UTC
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def observation_from_record(record: dict) -> Optional[TelemetryObservation]:
    """
    Converts a raw telemetry dict (from DB or consumer) to a TelemetryObservation.
    Returns None if required fields are missing or unparseable.
    """
    try:
        ts = parse_event_timestamp(record.get("event_timestamp", ""))
        if ts is None:
            return None
        return TelemetryObservation(
            vehicle_id=record["vehicle_id"],
            event_timestamp=ts,
            speed=float(record["speed"]),
            rpm=float(record["rpm"]),
            engine_temp=float(record["engine_temp"]),
            battery_voltage=_safe_float(record.get("battery_voltage")),
            fuel_level=_safe_float(record.get("fuel_level")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
