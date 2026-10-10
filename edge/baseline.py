# edge/baseline.py
# Vehicle-specific statistical baseline engine for EdgePulse Phase 2.
#
# Design:
#   - Each vehicle has an independent baseline — no fleet-wide pooling.
#   - Uses Welford's online algorithm for numerically stable running mean/variance.
#   - Baseline is INSUFFICIENT until min_samples observations have been seen.
#   - Conservative update policy: reject duplicate timestamps, out-of-order,
#     non-finite values, and known fault periods.
#   - Baselines can be persisted to / loaded from SQLite.
#   - No ML, no clustering, no anomaly detection — just statistics.
#
# Welford's algorithm:
#   Running mean M_n and variance S_n updated for each new sample x:
#     n     = n + 1
#     delta = x - M_{n-1}
#     M_n   = M_{n-1} + delta / n
#     delta2= x - M_n
#     S_n   = S_{n-1} + delta * delta2
#   Population variance  = S_n / n
#   Sample variance      = S_n / (n - 1)  (used here for unbiased estimate)
#
# Baseline contamination warning:
#   When ground truth is unavailable (runtime), there is no way to guarantee
#   that a fault period is excluded from baseline training. The conservative
#   update policy (reject obvious outliers beyond 5σ) provides partial
#   protection, but a prolonged, gradual fault may still drift the baseline.
#   The correct mitigation is to train baselines exclusively on known-good data
#   (Phase 1 ground truth available in offline experiments) and freeze the
#   baseline for production use.

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum samples before the baseline is considered reliable
DEFAULT_MIN_SAMPLES: int = 30

# Outlier rejection: updates more than this many σ from the current mean are
# rejected to protect against fault contamination when ground truth is absent.
# Only applies after the baseline is ready (min_samples reached).
OUTLIER_REJECTION_SIGMA: float = 5.0


class BaselineReadiness(Enum):
    INSUFFICIENT = "INSUFFICIENT_BASELINE_DATA"
    READY        = "READY"


@dataclass
class FeatureBaseline:
    """
    Running baseline for a single feature of a single vehicle.
    Stores Welford accumulators rather than raw samples — O(1) memory.
    """
    vehicle_id:   str
    feature_name: str
    min_samples:  int = DEFAULT_MIN_SAMPLES

    _n:           int   = field(default=0, repr=False)
    _mean:        float = field(default=0.0, repr=False)
    _S:           float = field(default=0.0, repr=False)   # Welford S accumulator
    _last_ts:     Optional[datetime] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        value: float,
        timestamp: datetime,
        exclude: bool = False,
    ) -> bool:
        """
        Incorporate one new observation.

        Parameters:
            value     — the feature value to add
            timestamp — event timestamp (timezone-aware)
            exclude   — if True, skip this update (e.g. known fault period)

        Returns True if the update was applied, False if rejected.

        Rejection conditions:
            - exclude=True (caller marks known fault period)
            - non-finite value
            - timestamp <= last seen (duplicate or out-of-order)
            - value is an outlier beyond OUTLIER_REJECTION_SIGMA × σ
              (only when baseline is already READY)
        """
        if exclude:
            return False

        if not math.isfinite(value):
            logger.debug("Baseline %s/%s: rejected non-finite value %s",
                         self.vehicle_id, self.feature_name, value)
            return False

        if self._last_ts is not None and timestamp <= self._last_ts:
            logger.debug("Baseline %s/%s: rejected out-of-order/duplicate timestamp",
                         self.vehicle_id, self.feature_name)
            return False

        # Outlier guard — only when ready (to avoid rejecting valid early samples)
        if self.readiness == BaselineReadiness.READY:
            std = self.std
            if std is not None and std > 1e-9:
                if abs(value - self._mean) > OUTLIER_REJECTION_SIGMA * std:
                    logger.debug(
                        "Baseline %s/%s: rejected outlier %.3f (mean=%.3f σ=%.3f)",
                        self.vehicle_id, self.feature_name, value, self._mean, std
                    )
                    return False

        # Welford update
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._S += delta * delta2
        self._last_ts = timestamp
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def readiness(self) -> BaselineReadiness:
        return BaselineReadiness.READY if self._n >= self.min_samples else BaselineReadiness.INSUFFICIENT

    @property
    def mean(self) -> Optional[float]:
        return self._mean if self._n > 0 else None

    @property
    def variance(self) -> Optional[float]:
        """Sample variance (unbiased)."""
        if self._n < 2:
            return None
        return self._S / (self._n - 1)

    @property
    def std(self) -> Optional[float]:
        v = self.variance
        return math.sqrt(v) if v is not None else None

    @property
    def sample_count(self) -> int:
        return self._n

    def to_dict(self) -> dict:
        return {
            "vehicle_id":   self.vehicle_id,
            "feature_name": self.feature_name,
            "sample_count": self._n,
            "mean":         self._mean,
            "variance":     self.variance,
            "readiness":    self.readiness.value,
        }

    @classmethod
    def from_dict(cls, data: dict, min_samples: int = DEFAULT_MIN_SAMPLES) -> "FeatureBaseline":
        """Reconstruct a baseline from a persisted dict (e.g. loaded from SQLite)."""
        obj = cls(
            vehicle_id=data["vehicle_id"],
            feature_name=data["feature_name"],
            min_samples=min_samples,
        )
        obj._n    = data.get("sample_count", 0)
        obj._mean = data.get("mean") or 0.0
        # Reconstruct S from stored variance: S = variance * (n-1)
        v = data.get("variance")
        obj._S = (v * (obj._n - 1)) if (v is not None and obj._n >= 2) else 0.0
        return obj


class VehicleBaseline:
    """
    Manages all feature baselines for a single vehicle.
    Acts as a registry — creates FeatureBaseline objects on demand.
    """

    # Features to track by default
    DEFAULT_FEATURES = [
        "engine_temp",
        "temp_rolling_mean",
        "temp_rolling_std",
        "rpm",
        "rpm_rolling_std",
        "rpm_variability",
        "battery_voltage",
        "voltage_rolling_mean",
        "voltage_rolling_std",
        "speed",
        "speed_rolling_mean",
        "fuel_level",
    ]

    def __init__(self, vehicle_id: str, min_samples: int = DEFAULT_MIN_SAMPLES):
        self.vehicle_id = vehicle_id
        self.min_samples = min_samples
        self._baselines: dict[str, FeatureBaseline] = {}

    def _get_or_create(self, feature_name: str) -> FeatureBaseline:
        if feature_name not in self._baselines:
            self._baselines[feature_name] = FeatureBaseline(
                vehicle_id=self.vehicle_id,
                feature_name=feature_name,
                min_samples=self.min_samples,
            )
        return self._baselines[feature_name]

    def update_from_features(
        self,
        fv_dict: dict,
        timestamp: datetime,
        exclude: bool = False,
        features: Optional[list[str]] = None,
    ) -> dict[str, bool]:
        """
        Update baselines from a feature vector dict for all tracked features.

        Parameters:
            fv_dict   — dict of feature_name → value (None means unavailable)
            timestamp — event timestamp
            exclude   — mark this step as a known fault period (offline only)
            features  — which features to update; defaults to DEFAULT_FEATURES

        Returns a dict of feature_name → whether update was accepted.
        """
        target = features or self.DEFAULT_FEATURES
        results: dict[str, bool] = {}
        for fname in target:
            val = fv_dict.get(fname)
            if val is None:
                results[fname] = False
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                results[fname] = False
                continue
            bl = self._get_or_create(fname)
            results[fname] = bl.update(v, timestamp, exclude=exclude)
        return results

    def get(self, feature_name: str) -> Optional[FeatureBaseline]:
        return self._baselines.get(feature_name)

    def readiness(self, feature_name: str) -> BaselineReadiness:
        bl = self._baselines.get(feature_name)
        if bl is None:
            return BaselineReadiness.INSUFFICIENT
        return bl.readiness

    def summary(self) -> dict:
        return {fname: bl.to_dict() for fname, bl in self._baselines.items()}


class BaselineRegistry:
    """
    Fleet-level registry: one VehicleBaseline per vehicle.
    Entry point for the consumer pipeline.
    """

    def __init__(self, min_samples: int = DEFAULT_MIN_SAMPLES):
        self._min_samples = min_samples
        self._vehicles: dict[str, VehicleBaseline] = {}

    def get_or_create(self, vehicle_id: str) -> VehicleBaseline:
        if vehicle_id not in self._vehicles:
            self._vehicles[vehicle_id] = VehicleBaseline(vehicle_id, self._min_samples)
        return self._vehicles[vehicle_id]

    def vehicle_ids(self) -> list[str]:
        return list(self._vehicles.keys())

    def load_from_db(self, rows: list[dict]) -> None:
        """
        Populate baselines from persisted rows (loaded from vehicle_baselines table).
        Allows baselines to survive a process restart.
        """
        for row in rows:
            vid  = row["vehicle_id"]
            bl   = FeatureBaseline.from_dict(row, self._min_samples)
            vb   = self.get_or_create(vid)
            vb._baselines[bl.feature_name] = bl
            logger.debug("Loaded baseline %s/%s n=%d", vid, bl.feature_name, bl._n)
