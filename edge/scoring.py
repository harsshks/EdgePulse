# edge/scoring.py
# Statistical deviation scoring for EdgePulse Phase 2.
#
# Computes z-score based deviation of current feature values from vehicle baselines.
# Returns structured results with direction, severity, and readiness status.
#
# Important:
#   A deviation score is NOT a failure probability.
#   A z-score of 3.0 means "3 standard deviations from baseline",
#   not "30% probability of failure".
#   Physical safety limits remain a separate concern.
#
# Variance floor:
#   Features with near-zero variance (e.g. battery voltage is very stable)
#   would produce enormous z-scores for tiny absolute deviations.
#   A minimum variance floor prevents division by near-zero standard deviation.

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from edge.baseline import BaselineReadiness, FeatureBaseline

logger = logging.getLogger(__name__)

# Minimum standard deviation used as denominator in z-score calculation.
# Prevents inflation of scores when a feature has very little natural variance.
# Set conservatively — 0.1 means deviations smaller than 0.1σ-equivalent are
# treated as noise.
MIN_STD_FLOOR: float = 0.1


class DeviationLevel(Enum):
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"   # baseline not ready
    NORMAL            = "NORMAL"              # |z| < warning_threshold
    ELEVATED          = "ELEVATED"            # warning_threshold <= |z| < severe_threshold
    SEVERE            = "SEVERE"              # |z| >= severe_threshold


@dataclass
class DeviationScore:
    """
    Structured result for one feature at one point in time.

    Fields:
        vehicle_id      — which vehicle
        feature_name    — which feature
        event_timestamp — ISO-8601 string
        current_value   — the raw feature value
        baseline_mean   — mean from the vehicle baseline
        baseline_std    — std from the vehicle baseline
        z_score         — (current - mean) / std   (None if baseline not ready)
        level           — DeviationLevel enum
        direction       — +1 above mean, -1 below mean, 0 if z=0 or unavailable
        sample_count    — how many samples the baseline was built on
        readiness       — BaselineReadiness
    """
    vehicle_id:      str
    feature_name:    str
    event_timestamp: str
    current_value:   Optional[float]
    baseline_mean:   Optional[float]
    baseline_std:    Optional[float]
    z_score:         Optional[float]
    level:           DeviationLevel
    direction:       int                    # +1, -1, or 0
    sample_count:    int
    readiness:       str                    # BaselineReadiness.value

    def to_dict(self) -> dict:
        return {
            "vehicle_id":      self.vehicle_id,
            "feature_name":    self.feature_name,
            "event_timestamp": self.event_timestamp,
            "current_value":   self.current_value,
            "baseline_mean":   self.baseline_mean,
            "baseline_std":    self.baseline_std,
            "z_score":         self.z_score,
            "level":           self.level.value,
            "direction":       self.direction,
            "sample_count":    self.sample_count,
            "readiness":       self.readiness,
        }


def score_feature(
    vehicle_id:          str,
    feature_name:        str,
    current_value:       Optional[float],
    baseline:            Optional[FeatureBaseline],
    event_timestamp:     str,
    warning_threshold:   float = 2.0,
    severe_threshold:    float = 3.5,
    min_std_floor:       float = MIN_STD_FLOOR,
) -> DeviationScore:
    """
    Compute a deviation score for one feature value against one baseline.

    Parameters:
        warning_threshold  — z-score at which level becomes ELEVATED
        severe_threshold   — z-score at which level becomes SEVERE
        min_std_floor      — minimum σ to use as denominator (prevents division
                             by near-zero variance)

    Returns a DeviationScore regardless of input quality.
    Returns INSUFFICIENT_DATA level when:
        - baseline is None
        - baseline is not ready
        - current_value is None or non-finite
    """
    # No baseline at all
    if baseline is None:
        return DeviationScore(
            vehicle_id=vehicle_id, feature_name=feature_name,
            event_timestamp=event_timestamp,
            current_value=current_value, baseline_mean=None, baseline_std=None,
            z_score=None, level=DeviationLevel.INSUFFICIENT_DATA,
            direction=0, sample_count=0,
            readiness=BaselineReadiness.INSUFFICIENT.value,
        )

    # Baseline not yet ready
    if baseline.readiness != BaselineReadiness.READY:
        return DeviationScore(
            vehicle_id=vehicle_id, feature_name=feature_name,
            event_timestamp=event_timestamp,
            current_value=current_value, baseline_mean=baseline.mean,
            baseline_std=baseline.std,
            z_score=None, level=DeviationLevel.INSUFFICIENT_DATA,
            direction=0, sample_count=baseline.sample_count,
            readiness=baseline.readiness.value,
        )

    # Current value unavailable or non-finite
    if current_value is None or not math.isfinite(current_value):
        return DeviationScore(
            vehicle_id=vehicle_id, feature_name=feature_name,
            event_timestamp=event_timestamp,
            current_value=current_value, baseline_mean=baseline.mean,
            baseline_std=baseline.std,
            z_score=None, level=DeviationLevel.INSUFFICIENT_DATA,
            direction=0, sample_count=baseline.sample_count,
            readiness=baseline.readiness.value,
        )

    mean = baseline.mean
    std  = baseline.std or 0.0
    # Apply variance floor
    effective_std = max(std, min_std_floor)

    z = (current_value - mean) / effective_std

    abs_z = abs(z)
    if abs_z >= severe_threshold:
        level = DeviationLevel.SEVERE
    elif abs_z >= warning_threshold:
        level = DeviationLevel.ELEVATED
    else:
        level = DeviationLevel.NORMAL

    direction = 1 if z > 0 else (-1 if z < 0 else 0)

    return DeviationScore(
        vehicle_id=vehicle_id, feature_name=feature_name,
        event_timestamp=event_timestamp,
        current_value=round(current_value, 4),
        baseline_mean=round(mean, 4),
        baseline_std=round(std, 4),
        z_score=round(z, 4),
        level=level,
        direction=direction,
        sample_count=baseline.sample_count,
        readiness=baseline.readiness.value,
    )


def score_all_features(
    vehicle_id:      str,
    fv_dict:         dict,
    vehicle_baseline,           # VehicleBaseline
    event_timestamp: str,
    features:        Optional[list[str]] = None,
    warning_threshold: float = 2.0,
    severe_threshold:  float = 3.5,
) -> list[DeviationScore]:
    """
    Score all requested features for one vehicle at one timestep.
    Features with no baseline entry return INSUFFICIENT_DATA.
    """
    from edge.baseline import VehicleBaseline
    target = features or VehicleBaseline.DEFAULT_FEATURES
    scores: list[DeviationScore] = []
    for fname in target:
        val = fv_dict.get(fname)
        bl  = vehicle_baseline.get(fname)
        scores.append(score_feature(
            vehicle_id=vehicle_id,
            feature_name=fname,
            current_value=float(val) if val is not None else None,
            baseline=bl,
            event_timestamp=event_timestamp,
            warning_threshold=warning_threshold,
            severe_threshold=severe_threshold,
        ))
    return scores
