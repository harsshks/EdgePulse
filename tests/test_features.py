# tests/test_features.py
# Unit tests for edge/features.py — rolling-window feature extraction.
# All tests use deterministic, hand-crafted data. No sleeps, no randomness.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from datetime import datetime, timezone, timedelta
import pytest

from edge.features import (
    FeatureWindow,
    TelemetryObservation,
    observation_from_record,
    parse_event_timestamp,
    _mean, _std, _elapsed_seconds, _rate_per_second,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts(offset_seconds: float = 0.0) -> datetime:
    """Returns a UTC datetime offset from a fixed base time."""
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_seconds)


def obs(t_offset: float, speed=60.0, rpm=3000.0, temp=90.0,
        voltage=12.6, fuel=80.0, vehicle_id="CAR-01") -> TelemetryObservation:
    return TelemetryObservation(
        vehicle_id=vehicle_id,
        event_timestamp=ts(t_offset),
        speed=speed, rpm=rpm, engine_temp=temp,
        battery_voltage=voltage, fuel_level=fuel,
    )


def make_window(window_size=60, min_samples=3, max_gap_seconds=30.0) -> FeatureWindow:
    return FeatureWindow(window_size=window_size, min_samples=min_samples,
                         temp_warn_thresh=100.0, max_gap_seconds=max_gap_seconds)


# ---------------------------------------------------------------------------
# 1. Insufficient history — no rolling stats until min_samples
# ---------------------------------------------------------------------------

class TestInsufficientHistory:

    def test_no_rolling_stats_with_one_sample(self):
        w = make_window(min_samples=3)
        fv = w.update(obs(0))
        assert fv.temp_rolling_mean is None
        assert fv.rpm_rolling_std is None

    def test_no_rolling_stats_with_two_samples(self):
        w = make_window(min_samples=3)
        w.update(obs(0))
        fv = w.update(obs(1))
        assert fv.temp_rolling_mean is None

    def test_rolling_stats_available_at_min_samples(self):
        w = make_window(min_samples=3)
        w.update(obs(0))
        w.update(obs(1))
        fv = w.update(obs(2))
        assert fv.temp_rolling_mean is not None

    def test_required_fields_always_present(self):
        """speed, rpm, engine_temp are always in the feature vector."""
        w = make_window(min_samples=3)
        fv = w.update(obs(0, speed=70, rpm=3500, temp=88))
        assert fv.speed == 70.0
        assert fv.rpm == 3500.0
        assert fv.engine_temp == 88.0


# ---------------------------------------------------------------------------
# 2. Rolling mean and std
# ---------------------------------------------------------------------------

class TestRollingStats:

    def test_rolling_mean_correct(self):
        w = make_window(min_samples=3)
        for i, temp in enumerate([80.0, 90.0, 100.0]):
            fv = w.update(obs(float(i), temp=temp))
        assert fv.temp_rolling_mean == pytest.approx(90.0)

    def test_rolling_std_correct(self):
        w = make_window(min_samples=3)
        for i, temp in enumerate([90.0, 90.0, 90.0]):
            fv = w.update(obs(float(i), temp=temp))
        assert fv.temp_rolling_std == pytest.approx(0.0)

    def test_rolling_std_nonzero_for_varying_data(self):
        w = make_window(min_samples=3)
        for i, temp in enumerate([80.0, 90.0, 100.0]):
            fv = w.update(obs(float(i), temp=temp))
        assert fv.temp_rolling_std > 0.0

    def test_rolling_min_max(self):
        w = make_window(min_samples=3)
        for i, temp in enumerate([80.0, 90.0, 100.0]):
            fv = w.update(obs(float(i), temp=temp))
        assert fv.temp_rolling_min == pytest.approx(80.0)
        assert fv.temp_rolling_max == pytest.approx(100.0)

    def test_window_trims_oldest(self):
        """With window_size=3, adding a 4th observation drops the first."""
        w = make_window(window_size=3, min_samples=3)
        for i, temp in enumerate([80.0, 90.0, 100.0, 110.0]):
            fv = w.update(obs(float(i), temp=temp))
        # Window is now [90, 100, 110], mean = 100
        assert fv.temp_rolling_mean == pytest.approx(100.0)
        assert fv.temp_rolling_min == pytest.approx(90.0)

    def test_rpm_rolling_stats(self):
        w = make_window(min_samples=3)
        for i, rpm in enumerate([2000.0, 3000.0, 4000.0]):
            fv = w.update(obs(float(i), rpm=rpm))
        assert fv.rpm_rolling_mean == pytest.approx(3000.0)
        assert fv.rpm_rolling_std > 0

    def test_speed_rolling_stats(self):
        w = make_window(min_samples=3)
        for i, speed in enumerate([60.0, 70.0, 80.0]):
            fv = w.update(obs(float(i), speed=speed))
        assert fv.speed_rolling_mean == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# 3. Temperature slope (°C/min)
# ---------------------------------------------------------------------------

class TestTemperatureSlope:

    def test_temp_slope_rising(self):
        """2°C rise over 2 seconds = 1°C/s = 60°C/min."""
        w = make_window(min_samples=3)
        for i, temp in enumerate([90.0, 91.0, 92.0]):
            fv = w.update(obs(float(i), temp=temp))
        # slope = 1°C/s = 60°C/min
        assert fv.temp_slope == pytest.approx(60.0)

    def test_temp_slope_stable(self):
        """Stable temperature → slope near zero."""
        w = make_window(min_samples=3)
        for i in range(5):
            fv = w.update(obs(float(i), temp=90.0))
        assert fv.temp_slope == pytest.approx(0.0, abs=0.1)

    def test_temp_slope_none_on_large_gap(self):
        """Gap larger than max_gap_seconds suppresses slope."""
        w = make_window(min_samples=3, max_gap_seconds=30.0)
        # Build enough history with small gaps
        for i in range(3):
            w.update(obs(float(i), temp=90.0))
        # Now add an observation with a 60-second gap
        fv = w.update(obs(100.0, temp=95.0))
        assert fv.temp_slope is None


# ---------------------------------------------------------------------------
# 4. RPM variability
# ---------------------------------------------------------------------------

class TestRPMVariability:

    def test_rpm_variability_zero_for_constant_rpm(self):
        w = make_window(min_samples=3)
        for i in range(5):
            fv = w.update(obs(float(i), rpm=3000.0))
        assert fv.rpm_variability == pytest.approx(0.0)

    def test_rpm_variability_positive_for_varying_rpm(self):
        w = make_window(min_samples=3)
        for i, rpm in enumerate([2000.0, 3000.0, 4000.0]):
            fv = w.update(obs(float(i), rpm=rpm))
        assert fv.rpm_variability is not None
        assert fv.rpm_variability > 0

    def test_rpm_variability_none_when_mean_zero(self):
        """RPM = 0 → variability undefined (avoid division by zero)."""
        w = make_window(min_samples=3)
        for i in range(5):
            fv = w.update(obs(float(i), rpm=0.0))
        assert fv.rpm_variability is None


# ---------------------------------------------------------------------------
# 5. Speed change rate / acceleration
# ---------------------------------------------------------------------------

class TestSpeedChangeRate:

    def test_acceleration_positive_on_increase(self):
        """Speed increases 10 km/h over 1 second = 10 km/h/s."""
        w = make_window(min_samples=3)
        for i, speed in enumerate([60.0, 70.0, 80.0]):
            fv = w.update(obs(float(i), speed=speed))
        assert fv.speed_change_rate == pytest.approx(10.0)
        assert fv.acceleration == pytest.approx(10.0)

    def test_deceleration_negative_on_decrease(self):
        w = make_window(min_samples=3)
        for i, speed in enumerate([80.0, 70.0, 60.0]):
            fv = w.update(obs(float(i), speed=speed))
        assert fv.speed_change_rate == pytest.approx(-10.0)

    def test_speed_change_rate_none_on_large_gap(self):
        w = make_window(min_samples=3, max_gap_seconds=5.0)
        for i in range(3):
            w.update(obs(float(i), speed=60.0))
        fv = w.update(obs(100.0, speed=80.0))   # 100s gap > 5s max
        assert fv.speed_change_rate is None


# ---------------------------------------------------------------------------
# 6. Battery voltage trend
# ---------------------------------------------------------------------------

class TestBatteryVoltageTrend:

    def test_voltage_trend_negative_on_drop(self):
        """Voltage drops 0.06V over 3 seconds → -0.02V/s = -1.2V/min."""
        w = make_window(min_samples=3)
        for i, v in enumerate([12.6, 12.58, 12.56]):
            fv = w.update(obs(float(i), voltage=v))
        assert fv.voltage_trend is not None
        assert fv.voltage_trend < 0.0

    def test_voltage_trend_positive_on_rise(self):
        w = make_window(min_samples=3)
        for i, v in enumerate([12.4, 12.5, 12.6]):
            fv = w.update(obs(float(i), voltage=v))
        assert fv.voltage_trend is not None
        assert fv.voltage_trend > 0.0

    def test_voltage_stats_none_when_missing(self):
        """Observations without battery_voltage → voltage features are None."""
        w = make_window(min_samples=3)
        for i in range(5):
            fv = w.update(obs(float(i), voltage=None))
        assert fv.voltage_rolling_mean is None


# ---------------------------------------------------------------------------
# 7. Fuel consumption rate
# ---------------------------------------------------------------------------

class TestFuelRate:

    def test_fuel_rate_negative_on_consumption(self):
        """Fuel drops 0.3% over 3 seconds → -0.1%/s = -6%/min."""
        w = make_window(min_samples=3)
        for i, f in enumerate([80.0, 79.9, 79.8]):
            fv = w.update(obs(float(i), fuel=f))
        assert fv.fuel_rate is not None
        assert fv.fuel_rate < 0.0

    def test_fuel_rate_none_when_missing(self):
        w = make_window(min_samples=3)
        for i in range(5):
            fv = w.update(obs(float(i), fuel=None))
        assert fv.fuel_rate is None


# ---------------------------------------------------------------------------
# 8. Duplicate timestamps — out-of-order handling
# ---------------------------------------------------------------------------

class TestDuplicateAndOutOfOrder:

    def test_duplicate_timestamp_not_added(self):
        w = make_window(min_samples=3)
        for i in range(4):
            w.update(obs(float(i), temp=90.0))
        count_before = w.sample_count
        # Duplicate timestamp
        w.update(obs(3.0, temp=95.0))  # same timestamp as last
        assert w.sample_count == count_before

    def test_out_of_order_observation_rejected(self):
        w = make_window(min_samples=3)
        for i in range(4):
            w.update(obs(float(i), temp=90.0))
        count_before = w.sample_count
        # Earlier timestamp
        w.update(obs(1.0, temp=95.0))
        assert w.sample_count == count_before

    def test_features_still_valid_after_rejected_obs(self):
        w = make_window(min_samples=3)
        for i in range(5):
            w.update(obs(float(i), temp=90.0))
        # Reject an out-of-order obs
        fv_bad = w.update(obs(1.0, temp=999.0))
        # Features should reflect the last valid window, not the rejected one
        assert fv_bad.engine_temp != 999.0 or fv_bad.temp_rolling_mean is None or \
               fv_bad.temp_rolling_max <= 90.0 + 1e-6


# ---------------------------------------------------------------------------
# 9. Long gap handling
# ---------------------------------------------------------------------------

class TestLongGaps:

    def test_slope_suppressed_after_long_gap(self):
        w = make_window(min_samples=3, max_gap_seconds=10.0)
        for i in range(5):
            w.update(obs(float(i), temp=90.0))
        fv = w.update(obs(500.0, temp=95.0))   # 495s gap
        assert fv.temp_slope is None

    def test_speed_rate_suppressed_after_gap(self):
        w = make_window(min_samples=3, max_gap_seconds=10.0)
        for i in range(5):
            w.update(obs(float(i), speed=60.0))
        fv = w.update(obs(500.0, speed=80.0))
        assert fv.speed_change_rate is None


# ---------------------------------------------------------------------------
# 10. Idle duration
# ---------------------------------------------------------------------------

class TestIdleDuration:

    def test_idle_duration_zero_when_moving(self):
        w = make_window(min_samples=3)
        for i in range(5):
            fv = w.update(obs(float(i), speed=60.0))
        assert fv.idle_duration_s == pytest.approx(0.0)

    def test_idle_duration_counts_stationary_period(self):
        w = make_window(min_samples=3)
        # 3 moving, then 3 stationary
        for i in range(3):
            w.update(obs(float(i), speed=60.0))
        for i in range(3, 6):
            fv = w.update(obs(float(i), speed=0.0))
        # 3 gaps of 1s each = 3s idle (indices 3→4, 4→5, 5→6)
        assert fv.idle_duration_s == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 11. observation_from_record
# ---------------------------------------------------------------------------

class TestObservationFromRecord:

    def test_valid_record_converts(self):
        rec = {
            "vehicle_id": "CAR-01",
            "event_timestamp": "2026-01-01T12:00:00+00:00",
            "speed": 80.0, "rpm": 3000.0, "engine_temp": 90.0,
            "battery_voltage": 12.6, "fuel_level": 80.0,
        }
        o = observation_from_record(rec)
        assert o is not None
        assert o.vehicle_id == "CAR-01"
        assert o.speed == 80.0
        assert o.battery_voltage == 12.6

    def test_missing_required_field_returns_none(self):
        rec = {"vehicle_id": "CAR-01", "speed": 80.0}
        assert observation_from_record(rec) is None

    def test_bad_timestamp_returns_none(self):
        rec = {
            "vehicle_id": "CAR-01", "event_timestamp": "not-a-date",
            "speed": 80.0, "rpm": 3000.0, "engine_temp": 90.0,
        }
        assert observation_from_record(rec) is None

    def test_missing_battery_voltage_ok(self):
        rec = {
            "vehicle_id": "CAR-01",
            "event_timestamp": "2026-01-01T12:00:00+00:00",
            "speed": 80.0, "rpm": 3000.0, "engine_temp": 90.0,
        }
        o = observation_from_record(rec)
        assert o is not None
        assert o.battery_voltage is None


# ---------------------------------------------------------------------------
# 12. Cross-signal: RPM variability at stable speed
# ---------------------------------------------------------------------------

class TestCrossSignal:

    def test_rpm_variability_at_stable_speed_populated(self):
        """With stable speed and varying RPM, cross-signal feature is populated."""
        w = make_window(min_samples=3)
        # Stable speed (~60 km/h), varying RPM
        for i in range(10):
            fv = w.update(obs(float(i), speed=60.0 + (0.5 if i % 2 else -0.5),
                              rpm=3000.0 + (100 * (i % 3))))
        # speed_std should be small → cross-signal should be set
        if fv.speed_rolling_std is not None and fv.speed_rolling_std < 5.0:
            assert fv.rpm_variability_at_stable_speed is not None

    def test_rpm_variability_at_stable_speed_none_when_speed_varies(self):
        """With highly variable speed, cross-signal is None."""
        w = make_window(min_samples=3)
        for i, speed in enumerate([0.0, 50.0, 100.0, 50.0, 0.0]):
            fv = w.update(obs(float(i), speed=speed, rpm=3000.0))
        # speed_std should be large → cross-signal should be None
        if fv.speed_rolling_std is not None and fv.speed_rolling_std >= 5.0:
            assert fv.rpm_variability_at_stable_speed is None
