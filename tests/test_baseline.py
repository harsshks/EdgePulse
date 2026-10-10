# tests/test_baseline.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timezone, timedelta
import pytest
from edge.baseline import FeatureBaseline, VehicleBaseline, BaselineRegistry, BaselineReadiness

def ts(offset=0.0):
    base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset)

def make_bl(min_samples=5):
    return FeatureBaseline('CAR-01', 'engine_temp', min_samples=min_samples)

class TestReadiness:
    def test_insufficient_before_min_samples(self):
        bl = make_bl(min_samples=5)
        for i in range(4): bl.update(90.0, ts(float(i)))
        assert bl.readiness == BaselineReadiness.INSUFFICIENT

    def test_ready_at_min_samples(self):
        bl = make_bl(min_samples=5)
        for i in range(5): bl.update(90.0, ts(float(i)))
        assert bl.readiness == BaselineReadiness.READY

    def test_ready_above_min_samples(self):
        bl = make_bl(min_samples=5)
        for i in range(20): bl.update(90.0, ts(float(i)))
        assert bl.readiness == BaselineReadiness.READY

class TestRunningStats:
    def test_mean_of_constant_series(self):
        bl = make_bl(min_samples=3)
        for i in range(10): bl.update(90.0, ts(float(i)))
        assert bl.mean == pytest.approx(90.0)

    def test_mean_of_known_values(self):
        bl = make_bl(min_samples=3)
        for i, v in enumerate([80.0, 90.0, 100.0]): bl.update(v, ts(float(i)))
        assert bl.mean == pytest.approx(90.0)

    def test_variance_of_constant_series_is_zero(self):
        bl = make_bl(min_samples=3)
        for i in range(10): bl.update(90.0, ts(float(i)))
        assert bl.variance == pytest.approx(0.0)

    def test_variance_nonzero_for_varying_data(self):
        bl = make_bl(min_samples=3)
        for i, v in enumerate([80.0, 90.0, 100.0]): bl.update(v, ts(float(i)))
        assert bl.variance is not None and bl.variance > 0.0

class TestDuplicateAndOutOfOrder:
    def test_duplicate_timestamp_rejected(self):
        bl = make_bl(min_samples=3)
        for i in range(5): bl.update(90.0, ts(float(i)))
        count_before = bl.sample_count
        assert not bl.update(95.0, ts(4.0))
        assert bl.sample_count == count_before

    def test_out_of_order_timestamp_rejected(self):
        bl = make_bl(min_samples=3)
        for i in range(5): bl.update(90.0, ts(float(i)))
        count_before = bl.sample_count
        assert not bl.update(95.0, ts(2.0))
        assert bl.sample_count == count_before

class TestNonFiniteRejection:
    def test_reject_nan(self):
        bl = make_bl(min_samples=3)
        for i in range(3): bl.update(90.0, ts(float(i)))
        count_before = bl.sample_count
        import math
        assert not bl.update(math.nan, ts(10.0))
        assert bl.sample_count == count_before

    def test_reject_inf(self):
        bl = make_bl(min_samples=3)
        for i in range(3): bl.update(90.0, ts(float(i)))
        count_before = bl.sample_count
        import math
        assert not bl.update(math.inf, ts(10.0))
        assert bl.sample_count == count_before

class TestOutlierRejection:
    def test_accept_within_5_sigma(self):
        bl = make_bl(min_samples=5)
        for i in range(10): bl.update(90.0, ts(float(i)))
        assert bl.update(92.0, ts(20.0))

    def test_reject_beyond_5_sigma(self):
        bl = make_bl(min_samples=5)
        for i in range(10): bl.update(90.0 + (0.1 if i % 2 else -0.1), ts(float(i)))
        count_before = bl.sample_count
        assert not bl.update(200.0, ts(20.0))
        assert bl.sample_count == count_before

class TestExcludeFaultPeriod:
    def test_exclude_flag_prevents_update(self):
        bl = make_bl(min_samples=3)
        for i in range(5): bl.update(90.0, ts(float(i)))
        count_before = bl.sample_count
        assert not bl.update(95.0, ts(10.0), exclude=True)
        assert bl.sample_count == count_before

class TestSerialization:
    def test_to_dict_contains_all_fields(self):
        bl = make_bl(min_samples=3)
        for i in range(5): bl.update(90.0, ts(float(i)))
        d = bl.to_dict()
        assert d['vehicle_id'] == 'CAR-01'
        assert d['feature_name'] == 'engine_temp'
        assert d['sample_count'] == 5
        assert d['mean'] is not None and d['variance'] is not None
        assert 'readiness' in d

    def test_from_dict_reconstructs_baseline(self):
        bl = make_bl(min_samples=5)
        for i, v in enumerate([80.0, 85.0, 90.0, 95.0, 100.0]): bl.update(v, ts(float(i)))
        d = bl.to_dict()
        bl2 = FeatureBaseline.from_dict(d, min_samples=5)
        assert bl2.sample_count == bl.sample_count
        assert bl2.mean == pytest.approx(bl.mean)
        assert bl2.variance == pytest.approx(bl.variance)

class TestVehicleBaseline:
    def test_updates_multiple_features(self):
        vb = VehicleBaseline('CAR-01', min_samples=3)
        r = vb.update_from_features({'engine_temp': 90.0, 'rpm': 3000.0, 'speed': 60.0}, ts(0))
        assert r['engine_temp'] is True and r['rpm'] is True and r['speed'] is True

    def test_ignores_none_values(self):
        vb = VehicleBaseline('CAR-01', min_samples=3)
        r = vb.update_from_features({'engine_temp': 90.0, 'battery_voltage': None}, ts(0))
        assert r['engine_temp'] is True and r['battery_voltage'] is False

    def test_readiness_per_feature(self):
        vb = VehicleBaseline('CAR-01', min_samples=3)
        for i in range(2): vb.update_from_features({'engine_temp': 90.0}, ts(float(i)))
        assert vb.readiness('engine_temp') == BaselineReadiness.INSUFFICIENT
        vb.update_from_features({'engine_temp': 90.0}, ts(2.0))
        assert vb.readiness('engine_temp') == BaselineReadiness.READY

    def test_get_returns_baseline(self):
        vb = VehicleBaseline('CAR-01', min_samples=3)
        vb.update_from_features({'engine_temp': 90.0}, ts(0))
        bl = vb.get('engine_temp')
        assert bl is not None and bl.vehicle_id == 'CAR-01'

class TestBaselineRegistry:
    def test_creates_vehicle_on_demand(self):
        reg = BaselineRegistry(min_samples=5)
        assert reg.get_or_create('CAR-01').vehicle_id == 'CAR-01'

    def test_vehicle_ids_returns_list(self):
        reg = BaselineRegistry()
        reg.get_or_create('CAR-01'); reg.get_or_create('CAR-02')
        assert 'CAR-01' in reg.vehicle_ids() and 'CAR-02' in reg.vehicle_ids()

    def test_load_from_db(self):
        reg = BaselineRegistry(min_samples=5)
        reg.load_from_db([{'vehicle_id': 'CAR-01', 'feature_name': 'engine_temp',
                           'sample_count': 10, 'mean': 90.0, 'variance': 4.0}])
        bl = reg.get_or_create('CAR-01').get('engine_temp')
        assert bl is not None and bl.sample_count == 10
        assert bl.mean == pytest.approx(90.0)
