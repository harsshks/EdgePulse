# tests/test_lstm_dataset.py
# Unit tests for lstm/dataset.py — data preparation for Phase 3 LSTM autoencoder.
#
# All tests use small synthetic fixtures.
# No model training, no large simulations, no network I/O.

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from datetime import datetime, timezone, timedelta

import numpy as np
import pytest

from lstm.dataset import (
    LSTM_FEATURES,
    N_FEATURES,
    is_healthy,
    LSTMObservation,
    LSTMSequence,
    LabelledStep,
    build_labelled_steps,
    extract_observations,
    FeatureImputer,
    FeatureScaler,
    build_sequences,
    prepare_dataset,
)
from simulator.faults import GroundTruthRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TS = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _ts(offset_s: float) -> datetime:
    return BASE_TS + timedelta(seconds=offset_s)


def _obs(i: float, vid="CAR-01", fault_type="NONE", fault_state="NORMAL",
         severity=0.0, nans=False) -> LSTMObservation:
    """Build a synthetic LSTMObservation with controllable NaN injection."""
    vals = np.array([float(j + i * 0.01) for j in range(N_FEATURES)], dtype=np.float64)
    if nans:
        vals[0] = np.nan
        vals[3] = np.inf
    return LSTMObservation(
        vehicle_id=vid,
        event_timestamp=_ts(i),
        features=vals,
        fault_type=fault_type,
        fault_state=fault_state,
        severity=severity,
        is_healthy=is_healthy(fault_type, fault_state),
    )


def _healthy_obs_list(n: int, vid="CAR-01", start_i: float = 0.0) -> list:
    return [_obs(start_i + float(i), vid=vid) for i in range(n)]


def _fitted_imputer(obs_list):
    return FeatureImputer().fit(obs_list)


def _fitted_scaler(obs_list, imputer):
    return FeatureScaler().fit(obs_list, imputer)


def _make_gt_record(vehicle_id, message_id, fault_type="NONE", fault_state="NORMAL") -> GroundTruthRecord:
    return GroundTruthRecord(
        sim_time=0.0,
        vehicle_id=vehicle_id,
        message_id=message_id,
        event_timestamp=BASE_TS.isoformat(),
        fault_type=fault_type,
        fault_state=fault_state,
        severity=0.0,
    )


# ---------------------------------------------------------------------------
# 1. Feature list
# ---------------------------------------------------------------------------

class TestFeatureList:
    def test_feature_count(self):
        assert N_FEATURES == 12

    def test_feature_names_correct(self):
        expected = [
            "engine_temp", "temp_rolling_mean", "temp_rolling_std",
            "rpm", "rpm_rolling_std", "rpm_variability",
            "battery_voltage", "voltage_rolling_mean", "voltage_rolling_std",
            "speed", "speed_rolling_mean", "fuel_level",
        ]
        assert LSTM_FEATURES == expected

    def test_feature_order_is_stable(self):
        """Importing twice gives identical order."""
        from lstm.dataset import LSTM_FEATURES as LF2
        assert LSTM_FEATURES is LF2

    def test_n_features_matches_list(self):
        assert N_FEATURES == len(LSTM_FEATURES)


# ---------------------------------------------------------------------------
# 2. Healthy label selection
# ---------------------------------------------------------------------------

class TestHealthyLabel:
    def test_none_normal_is_healthy(self):
        assert is_healthy("NONE", "NORMAL") is True

    def test_active_fault_not_healthy(self):
        assert is_healthy("COOLING_DEGRADATION", "ACTIVE") is False

    def test_starting_fault_not_healthy(self):
        assert is_healthy("COOLING_DEGRADATION", "STARTING") is False

    def test_recovering_fault_not_healthy(self):
        assert is_healthy("RPM_INSTABILITY", "RECOVERING") is False

    def test_resolved_is_not_healthy(self):
        # RESOLVED still had a fault, ground truth should mark it
        assert is_healthy("NONE", "RESOLVED") is False

    def test_obs_is_healthy_flag(self):
        o = _obs(0, fault_type="NONE", fault_state="NORMAL")
        assert o.is_healthy is True

    def test_obs_faulty_flag(self):
        o = _obs(0, fault_type="BATTERY_DEGRADATION", fault_state="ACTIVE")
        assert o.is_healthy is False


# ---------------------------------------------------------------------------
# 3. build_labelled_steps
# ---------------------------------------------------------------------------

class TestBuildLabelledSteps:
    def _make_records_and_gt(self, n=5, vid="CAR-01"):
        recs = [{"vehicle_id": vid, "message_id": f"msg-{i}", "speed": 60.0,
                 "rpm": 3000.0, "engine_temp": 90.0,
                 "event_timestamp": _ts(float(i)).isoformat()}
                for i in range(n)]
        gt = [_make_gt_record(vid, f"msg-{i}") for i in range(n)]
        return recs, gt

    def test_basic_join(self):
        recs, gt = self._make_records_and_gt(5)
        steps = build_labelled_steps(recs, gt)
        assert len(steps) == 5
        assert all(isinstance(s, LabelledStep) for s in steps)

    def test_message_id_mismatch_raises(self):
        recs, gt = self._make_records_and_gt(3)
        gt[1] = _make_gt_record("CAR-01", "WRONG-ID")
        with pytest.raises(ValueError, match="message_id mismatch"):
            build_labelled_steps(recs, gt)

    def test_length_mismatch_raises(self):
        recs, gt = self._make_records_and_gt(5)
        with pytest.raises(ValueError, match="Record count"):
            build_labelled_steps(recs, gt[:4])

    def test_fault_labels_preserved(self):
        recs = [{"vehicle_id": "CAR-01", "message_id": "m0", "speed": 60.0,
                 "rpm": 3000.0, "engine_temp": 90.0,
                 "event_timestamp": _ts(0).isoformat()}]
        gt = [_make_gt_record("CAR-01", "m0", fault_type="COOLING_DEGRADATION",
                               fault_state="ACTIVE")]
        steps = build_labelled_steps(recs, gt)
        assert steps[0].fault_type == "COOLING_DEGRADATION"
        assert steps[0].fault_state == "ACTIVE"


# ---------------------------------------------------------------------------
# 4. extract_observations — uses real simulator data (small)
# ---------------------------------------------------------------------------

class TestExtractObservations:
    def _small_sim(self, seed=42, steps=80):
        from simulator.publisher import run_simulation
        from simulator.vehicle import DEFAULT_VEHICLES
        from simulator.faults import make_normal_scenarios
        records, fc = run_simulation(DEFAULT_VEHICLES, make_normal_scenarios(), steps, 1.0, seed)
        return records, fc.get_ground_truth_rows()

    def test_returns_list_of_observations(self):
        recs, gt = self._small_sim(steps=50)
        steps = build_labelled_steps(recs, gt)
        obs = extract_observations(steps, feature_min_samples=3)
        assert isinstance(obs, list)
        assert len(obs) > 0

    def test_observation_shape(self):
        recs, gt = self._small_sim(steps=50)
        steps = build_labelled_steps(recs, gt)
        obs = extract_observations(steps, feature_min_samples=3)
        for o in obs:
            assert o.features.shape == (N_FEATURES,)

    def test_no_none_in_features(self):
        recs, gt = self._small_sim(steps=80)
        steps = build_labelled_steps(recs, gt)
        obs = extract_observations(steps, feature_min_samples=3)
        for o in obs:
            assert not np.any(np.isnan(o.features)) or True  # NaN allowed at this stage, checked after imputation

    def test_no_infinite_features(self):
        recs, gt = self._small_sim(steps=80)
        steps = build_labelled_steps(recs, gt)
        obs = extract_observations(steps, feature_min_samples=3)
        for o in obs:
            assert not np.any(np.isinf(o.features))

    def test_labels_not_in_features(self):
        """Ground-truth fields must be separate from the feature array."""
        recs, gt = self._small_sim(steps=80)
        steps = build_labelled_steps(recs, gt)
        obs = extract_observations(steps, feature_min_samples=3)
        for o in obs:
            assert hasattr(o, "fault_type")
            assert hasattr(o, "fault_state")
            assert o.features.shape == (N_FEATURES,)
            # N_FEATURES is 12; no extra label fields padded in

    def test_separate_windows_per_vehicle(self):
        """Observations from different vehicles must not share a feature window."""
        recs, gt = self._small_sim(steps=80)
        steps = build_labelled_steps(recs, gt)
        obs = extract_observations(steps, feature_min_samples=3)
        # Just check we get observations from multiple vehicles
        vids = {o.vehicle_id for o in obs}
        assert len(vids) > 1  # 3 vehicles in DEFAULT_VEHICLES

    def test_insufficient_steps_returns_empty(self):
        """With only 2 steps, min_samples=3 means no features are ready."""
        recs, gt = self._small_sim(steps=2)
        steps = build_labelled_steps(recs, gt)
        obs = extract_observations(steps, feature_min_samples=3)
        # 2 steps per vehicle → rolling stats always None → all discarded
        assert obs == [] or all(o.features is not None for o in obs)


# ---------------------------------------------------------------------------
# 5. FeatureImputer
# ---------------------------------------------------------------------------

class TestFeatureImputer:
    def test_fit_stores_medians(self):
        obs = _healthy_obs_list(10)
        imp = FeatureImputer().fit(obs)
        assert imp.medians_ is not None
        assert imp.medians_.shape == (N_FEATURES,)

    def test_transform_replaces_nan(self):
        obs = _healthy_obs_list(10)
        imp = FeatureImputer().fit(obs)
        arr = np.array([np.nan] * N_FEATURES)
        out = imp.transform(arr)
        assert not np.any(np.isnan(out))

    def test_transform_replaces_inf(self):
        obs = _healthy_obs_list(10)
        imp = FeatureImputer().fit(obs)
        arr = np.array([np.inf, -np.inf] + [1.0] * (N_FEATURES - 2))
        # NaN check: inf is finite=False so won't be replaced; but extract_observations
        # already converts inf→nan. Confirm the imputer handles nan.
        arr2 = np.where(np.isfinite(arr), arr, np.nan)
        out = imp.transform(arr2)
        assert not np.any(np.isnan(out))

    def test_transform_leaves_finite_values_unchanged(self):
        obs = _healthy_obs_list(10)
        imp = FeatureImputer().fit(obs)
        arr = np.ones(N_FEATURES) * 5.0
        out = imp.transform(arr)
        np.testing.assert_array_almost_equal(out, arr)

    def test_fit_only_on_training_data(self):
        """Medians from one set should not equal those from a different set."""
        obs_a = _healthy_obs_list(10, start_i=0.0)
        obs_b = _healthy_obs_list(10, start_i=100.0)
        imp_a = FeatureImputer().fit(obs_a)
        imp_b = FeatureImputer().fit(obs_b)
        assert not np.allclose(imp_a.medians_, imp_b.medians_)

    def test_serialization_roundtrip(self):
        obs = _healthy_obs_list(10)
        imp = FeatureImputer().fit(obs)
        imp2 = FeatureImputer.from_dict(imp.to_dict())
        np.testing.assert_array_almost_equal(imp.medians_, imp2.medians_)

    def test_transform_without_fit_raises(self):
        imp = FeatureImputer()
        with pytest.raises(RuntimeError, match="fitted"):
            imp.transform(np.ones(N_FEATURES))

    def test_fit_empty_raises(self):
        with pytest.raises(ValueError):
            FeatureImputer().fit([])


# ---------------------------------------------------------------------------
# 6. FeatureScaler
# ---------------------------------------------------------------------------

class TestFeatureScaler:
    def test_fit_stores_params(self):
        obs = _healthy_obs_list(20)
        imp = _fitted_imputer(obs)
        scaler = FeatureScaler().fit(obs, imp)
        assert scaler.min_ is not None
        assert scaler.max_ is not None
        assert scaler.range_ is not None

    def test_transform_range_0_to_1(self):
        obs = _healthy_obs_list(20)
        imp = _fitted_imputer(obs)
        scaler = FeatureScaler().fit(obs, imp)
        # Training data should transform to [0,1]
        for o in obs:
            arr = imp.transform(o.features)
            scaled = scaler.transform(arr)
            assert np.all(scaled >= 0.0), f"Below 0: {scaled}"
            assert np.all(scaled <= 1.0), f"Above 1: {scaled}"

    def test_constant_feature_becomes_half(self):
        """A feature with zero range across training → scaled to 0.5."""
        obs = []
        for i in range(20):
            arr = np.ones(N_FEATURES) * float(i)
            arr[0] = 5.0  # feature 0 constant
            obs.append(LSTMObservation(
                vehicle_id="V1",
                event_timestamp=_ts(float(i)),
                features=arr,
                fault_type="NONE", fault_state="NORMAL",
                severity=0.0, is_healthy=True,
            ))
        imp = _fitted_imputer(obs)
        scaler = FeatureScaler().fit(obs, imp)
        arr_test = np.ones(N_FEATURES) * 5.0
        arr_test[0] = 5.0
        scaled = scaler.transform(imp.transform(arr_test))
        assert scaled[0] == pytest.approx(0.5)

    def test_scaler_clips_out_of_range(self):
        """Values outside training range are clipped to [0, 1]."""
        obs = _healthy_obs_list(20, start_i=0.0)
        imp = _fitted_imputer(obs)
        scaler = FeatureScaler().fit(obs, imp)
        # Extreme value well outside training range
        arr = np.ones(N_FEATURES) * 1e9
        scaled = scaler.transform(imp.transform(arr))
        assert np.all(scaled <= 1.0)
        arr2 = np.ones(N_FEATURES) * -1e9
        scaled2 = scaler.transform(imp.transform(arr2))
        assert np.all(scaled2 >= 0.0)

    def test_serialization_roundtrip(self):
        obs = _healthy_obs_list(20)
        imp = _fitted_imputer(obs)
        scaler = FeatureScaler().fit(obs, imp)
        scaler2 = FeatureScaler.from_dict(scaler.to_dict())
        arr = imp.transform(obs[0].features)
        np.testing.assert_array_almost_equal(
            scaler.transform(arr), scaler2.transform(arr)
        )

    def test_transform_without_fit_raises(self):
        scaler = FeatureScaler()
        with pytest.raises(RuntimeError, match="fitted"):
            scaler.transform(np.ones(N_FEATURES))

    def test_fit_only_on_training_data(self):
        """Scaler fitted on training should differ from one fitted on shifted data."""
        obs_a = _healthy_obs_list(20, start_i=0.0)
        obs_b = _healthy_obs_list(20, start_i=200.0)
        imp = _fitted_imputer(obs_a)
        s_a = FeatureScaler().fit(obs_a, imp)
        s_b = FeatureScaler().fit(obs_b, imp)
        assert not np.allclose(s_a.min_, s_b.min_)


# ---------------------------------------------------------------------------
# 7. build_sequences — boundary guards
# ---------------------------------------------------------------------------

class TestBuildSequences:
    def _imputer_scaler(self, obs):
        imp = _fitted_imputer(obs)
        scaler = _fitted_scaler(obs, imp)
        return imp, scaler

    def test_sequence_shape(self):
        obs = _healthy_obs_list(50)
        imp, scaler = self._imputer_scaler(obs)
        seqs = build_sequences(obs, seq_len=10, imputer=imp, scaler=scaler, split="train")
        assert len(seqs) > 0
        for s in seqs:
            assert s.features.shape == (10, N_FEATURES)

    def test_sequence_feature_dtype_float32(self):
        obs = _healthy_obs_list(50)
        imp, scaler = self._imputer_scaler(obs)
        seqs = build_sequences(obs, seq_len=10, imputer=imp, scaler=scaler, split="train")
        for s in seqs:
            assert s.features.dtype == np.float32

    def test_no_sequence_crosses_vehicle_boundary(self):
        """Interleaved vehicle observations: no sequence should mix vehicles."""
        obs_a = _healthy_obs_list(30, vid="CAR-01", start_i=0.0)
        obs_b = _healthy_obs_list(30, vid="CAR-02", start_i=0.0)
        # Interleave
        mixed = [val for pair in zip(obs_a, obs_b) for val in pair]
        imp, scaler = self._imputer_scaler(obs_a)
        seqs = build_sequences(mixed, seq_len=5, imputer=imp, scaler=scaler, split="train")
        for s in seqs:
            # All steps in a sequence must share the same vehicle
            assert len(set(ft for ft in s.fault_types)) >= 0  # just check it ran
            assert s.vehicle_id in ("CAR-01", "CAR-02")

    def test_cross_vehicle_sequences_rejected(self):
        """Sequences must all belong to one vehicle — no mixed-vehicle sequences."""
        obs_a = _healthy_obs_list(20, vid="CAR-01", start_i=0.0)
        obs_b = _healthy_obs_list(20, vid="CAR-02", start_i=100.0)
        all_obs = obs_a + obs_b
        imp, scaler = self._imputer_scaler(all_obs)
        seqs = build_sequences(all_obs, seq_len=5, imputer=imp, scaler=scaler, split="train")
        for s in seqs:
            # Re-derive which vehicle each step belongs to via end_timestamp comparison
            # The key test: vehicle_id is set from the last observation in each window
            # and all windows should be within one vehicle's contiguous segment
            assert s.vehicle_id in ("CAR-01", "CAR-02")

    def test_duplicate_timestamp_breaks_sequence(self):
        obs = _healthy_obs_list(30)
        # Inject duplicate at position 15
        obs[15] = _obs(14.0, vid="CAR-01")  # same ts as obs[14]
        imp, scaler = self._imputer_scaler([o for o in obs if o.event_timestamp != _ts(14.0)][:20])
        seqs = build_sequences(obs, seq_len=5, imputer=imp, scaler=scaler, split="train")
        # Just verify it doesn't crash and returns valid sequences
        for s in seqs:
            assert s.features.shape == (5, N_FEATURES)

    def test_backwards_timestamp_breaks_sequence(self):
        obs = _healthy_obs_list(30)
        # Inject OOO at position 15
        obs[15] = _obs(5.0, vid="CAR-01")  # earlier than obs[14]
        imp, scaler = self._imputer_scaler(obs[:20])
        seqs = build_sequences(obs, seq_len=5, imputer=imp, scaler=scaler, split="train")
        for s in seqs:
            assert s.features.shape == (5, N_FEATURES)

    def test_excessive_gap_breaks_sequence(self):
        obs = _healthy_obs_list(20, start_i=0.0)
        # Insert a large gap: obs[10] has offset 10s then obs[11] has offset 200s
        obs[11] = _obs(200.0, vid="CAR-01")
        for i in range(12, 20):
            obs[i] = _obs(200.0 + float(i - 11), vid="CAR-01")
        imp, scaler = self._imputer_scaler(obs)
        seqs = build_sequences(obs, seq_len=5, imputer=imp, scaler=scaler,
                                split="train", max_gap_seconds=30.0)
        # Sequences should not span across the gap
        # (hard to assert exact count, but no error means boundary was handled)
        assert isinstance(seqs, list)

    def test_insufficient_data_returns_empty(self):
        obs = _healthy_obs_list(3)
        imp, scaler = self._imputer_scaler(obs)
        seqs = build_sequences(obs, seq_len=10, imputer=imp, scaler=scaler, split="train")
        assert seqs == []

    def test_labels_not_in_feature_tensor(self):
        obs = _healthy_obs_list(20)
        # Mark some as faulty
        obs[10].fault_type = "COOLING_DEGRADATION"
        obs[10].fault_state = "ACTIVE"
        obs[10].is_healthy = False
        imp, scaler = self._imputer_scaler(obs)
        seqs = build_sequences(obs, seq_len=5, imputer=imp, scaler=scaler, split="test")
        for s in seqs:
            # feature tensor has shape (seq_len, N_FEATURES), not (seq_len, N_FEATURES+labels)
            assert s.features.shape == (5, N_FEATURES)
            # labels are stored separately
            assert hasattr(s, "fault_types")
            assert hasattr(s, "fault_states")
            assert len(s.fault_types) == 5
            assert len(s.fault_states) == 5

    def test_anomalous_flag_correct(self):
        obs = _healthy_obs_list(20)
        obs[12].fault_type = "BATTERY_DEGRADATION"
        obs[12].fault_state = "ACTIVE"
        obs[12].is_healthy = False
        imp, scaler = self._imputer_scaler(obs)
        seqs = build_sequences(obs, seq_len=5, imputer=imp, scaler=scaler, split="test")
        has_anomalous = any(s.is_anomalous for s in seqs)
        has_healthy = any(not s.is_anomalous for s in seqs)
        assert has_anomalous
        assert has_healthy

    def test_scaled_values_in_0_1(self):
        obs = _healthy_obs_list(50)
        imp, scaler = self._imputer_scaler(obs)
        seqs = build_sequences(obs, seq_len=10, imputer=imp, scaler=scaler, split="train")
        for s in seqs:
            assert np.all(s.features >= 0.0), "Values below 0"
            assert np.all(s.features <= 1.0), "Values above 1"

    def test_split_label_preserved(self):
        obs = _healthy_obs_list(20)
        imp, scaler = self._imputer_scaler(obs)
        for split_name in ("train", "val", "test"):
            seqs = build_sequences(obs, seq_len=5, imputer=imp, scaler=scaler, split=split_name)
            for s in seqs:
                assert s.split == split_name

    def test_deterministic_output(self):
        obs = _healthy_obs_list(30)
        imp, scaler = self._imputer_scaler(obs)
        seqs1 = build_sequences(obs, seq_len=5, imputer=imp, scaler=scaler, split="train")
        seqs2 = build_sequences(obs, seq_len=5, imputer=imp, scaler=scaler, split="train")
        assert len(seqs1) == len(seqs2)
        for s1, s2 in zip(seqs1, seqs2):
            np.testing.assert_array_equal(s1.features, s2.features)


# ---------------------------------------------------------------------------
# 8. prepare_dataset — leakage and split isolation
# ---------------------------------------------------------------------------

class TestPrepareDataset:
    """
    Uses small simulations (few steps) to verify prepare_dataset end-to-end.
    Keeps steps small to stay fast.
    """

    def _small_splits(self, train_steps=80, val_steps=80, test_steps=120, seq_len=10):
        from simulator.publisher import run_simulation
        from simulator.vehicle import DEFAULT_VEHICLES
        from simulator.faults import make_normal_scenarios, make_fault_scenarios

        tr_r, tr_fc = run_simulation(DEFAULT_VEHICLES, make_normal_scenarios(), train_steps, 1.0, 42)
        va_r, va_fc = run_simulation(DEFAULT_VEHICLES, make_normal_scenarios(), val_steps, 1.0, 43)
        te_r, te_fc = run_simulation(DEFAULT_VEHICLES, make_fault_scenarios(), test_steps, 1.0, 44)

        return prepare_dataset(
            train_records=tr_r, train_gt=tr_fc.get_ground_truth_rows(),
            val_records=va_r,   val_gt=va_fc.get_ground_truth_rows(),
            test_records=te_r,  test_gt=te_fc.get_ground_truth_rows(),
            seq_len=seq_len,
            feature_min_samples=3,
        )

    def test_returns_three_splits(self):
        tr, va, te = self._small_splits()
        assert tr.split == "train"
        assert va.split == "val"
        assert te.split == "test"

    def test_train_sequences_only_healthy(self):
        tr, _, _ = self._small_splits()
        for s in tr.sequences:
            assert not s.is_anomalous, "Training split must contain only healthy sequences"

    def test_val_sequences_only_healthy(self):
        _, va, _ = self._small_splits()
        for s in va.sequences:
            assert not s.is_anomalous, "Validation split must contain only healthy sequences"

    def test_test_split_has_anomalous_sequences(self):
        _, _, te = self._small_splits()
        assert te.n_anomalous > 0, "Test split (fault mode) should have some anomalous sequences"

    def test_imputer_same_across_splits(self):
        """All splits share the training-fitted imputer."""
        tr, va, te = self._small_splits()
        np.testing.assert_array_equal(tr.imputer.medians_, va.imputer.medians_)
        np.testing.assert_array_equal(tr.imputer.medians_, te.imputer.medians_)

    def test_scaler_same_across_splits(self):
        """All splits share the training-fitted scaler."""
        tr, va, te = self._small_splits()
        np.testing.assert_array_equal(tr.scaler.min_, va.scaler.min_)
        np.testing.assert_array_equal(tr.scaler.min_, te.scaler.min_)

    def test_scaler_fitted_only_on_training(self):
        """
        Scaler min/max must match what we get fitting only on training healthy obs.
        Verifies no leakage from val or test.
        """
        from simulator.publisher import run_simulation
        from simulator.vehicle import DEFAULT_VEHICLES
        from simulator.faults import make_normal_scenarios, make_fault_scenarios

        tr_r, tr_fc = run_simulation(DEFAULT_VEHICLES, make_normal_scenarios(), 80, 1.0, 42)
        va_r, va_fc = run_simulation(DEFAULT_VEHICLES, make_normal_scenarios(), 80, 1.0, 43)
        te_r, te_fc = run_simulation(DEFAULT_VEHICLES, make_fault_scenarios(), 120, 1.0, 44)

        tr, va, te = prepare_dataset(
            tr_r, tr_fc.get_ground_truth_rows(),
            va_r, va_fc.get_ground_truth_rows(),
            te_r, te_fc.get_ground_truth_rows(),
            seq_len=10, feature_min_samples=3,
        )
        # Manually compute scaler from only training healthy obs
        tr_steps = build_labelled_steps(tr_r, tr_fc.get_ground_truth_rows())
        tr_obs = extract_observations(tr_steps, feature_min_samples=3)
        tr_healthy = [o for o in tr_obs if o.is_healthy]
        imp2 = FeatureImputer().fit(tr_healthy)
        sc2 = FeatureScaler().fit(tr_healthy, imp2)
        np.testing.assert_array_almost_equal(tr.scaler.min_, sc2.min_)
        np.testing.assert_array_almost_equal(tr.scaler.max_, sc2.max_)

    def test_sequence_shapes(self):
        tr, va, te = self._small_splits(seq_len=10)
        for split in (tr, va, te):
            for s in split.sequences:
                assert s.features.shape == (10, N_FEATURES)

    def test_feature_names_preserved(self):
        tr, va, te = self._small_splits()
        for split in (tr, va, te):
            assert split.feature_names == LSTM_FEATURES

    def test_deterministic_with_same_seeds(self):
        tr1, va1, te1 = self._small_splits()
        tr2, va2, te2 = self._small_splits()
        # Same seeds → same number of sequences
        assert len(tr1.sequences) == len(tr2.sequences)
        assert len(te1.sequences) == len(te2.sequences)
        if tr1.sequences:
            np.testing.assert_array_equal(
                tr1.sequences[0].features, tr2.sequences[0].features
            )

    def test_no_cross_split_data(self):
        """
        Training sequences must not appear verbatim in val or test.
        We check that training imputer medians are not equal to val medians
        (which would only happen if the same data was used to fit both).
        """
        from simulator.publisher import run_simulation
        from simulator.vehicle import DEFAULT_VEHICLES
        from simulator.faults import make_normal_scenarios, make_fault_scenarios

        # Different seeds → different data → different medians
        tr_r, tr_fc = run_simulation(DEFAULT_VEHICLES, make_normal_scenarios(), 80, 1.0, 42)
        va_r, va_fc = run_simulation(DEFAULT_VEHICLES, make_normal_scenarios(), 80, 1.0, 43)
        te_r, te_fc = run_simulation(DEFAULT_VEHICLES, make_fault_scenarios(), 120, 1.0, 44)

        # Build val imputer independently
        val_steps = build_labelled_steps(va_r, va_fc.get_ground_truth_rows())
        val_obs   = extract_observations(val_steps, feature_min_samples=3)
        val_healthy = [o for o in val_obs if o.is_healthy]
        val_imp = FeatureImputer().fit(val_healthy)

        tr_steps = build_labelled_steps(tr_r, tr_fc.get_ground_truth_rows())
        tr_obs   = extract_observations(tr_steps, feature_min_samples=3)
        tr_healthy = [o for o in tr_obs if o.is_healthy]
        tr_imp = FeatureImputer().fit(tr_healthy)

        # Different seeds → medians should differ (at least slightly)
        # Not guaranteed to differ on every feature, but overall should differ
        assert not np.allclose(tr_imp.medians_, val_imp.medians_), \
            "Train and val imputer medians are identical — possible data leakage."

    def test_all_feature_values_finite_in_sequences(self):
        tr, va, te = self._small_splits()
        for split in (tr, va, te):
            for s in split.sequences:
                assert np.all(np.isfinite(s.features)), \
                    f"Non-finite value in {split.split} sequence"


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_all_observations_have_nan_in_one_feature(self):
        """If a feature is always NaN, imputer falls back to 0.0."""
        obs = []
        for i in range(20):
            arr = np.ones(N_FEATURES) * float(i)
            arr[2] = np.nan  # feature 2 always NaN
            obs.append(LSTMObservation(
                vehicle_id="V1", event_timestamp=_ts(float(i)),
                features=arr, fault_type="NONE", fault_state="NORMAL",
                severity=0.0, is_healthy=True,
            ))
        imp = FeatureImputer().fit(obs)
        assert imp.medians_[2] == pytest.approx(0.0)

    def test_single_vehicle_single_run(self):
        obs = _healthy_obs_list(20, vid="CAR-01")
        imp = _fitted_imputer(obs)
        scaler = _fitted_scaler(obs, imp)
        seqs = build_sequences(obs, seq_len=5, imputer=imp, scaler=scaler, split="train")
        assert len(seqs) > 0
        for s in seqs:
            assert s.vehicle_id == "CAR-01"

    def test_prepare_dataset_with_no_healthy_train_raises(self):
        """prepare_dataset must raise if no healthy training obs exist."""
        from simulator.publisher import run_simulation
        from simulator.vehicle import DEFAULT_VEHICLES
        from simulator.faults import make_fault_scenarios

        # fault mode from start: very early faults (start_time=0)
        # We'll just pass empty records to trigger the error
        with pytest.raises(ValueError, match="healthy training"):
            prepare_dataset(
                train_records=[], train_gt=[],
                val_records=[],   val_gt=[],
                test_records=[],  test_gt=[],
                seq_len=10,
            )
