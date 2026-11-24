from __future__ import annotations
import numpy as np
from .channel import ReceptionTrace


def estimate_from_receptions(trace: ReceptionTrace, estimator: str, drift=0.0, beta=1.0, slope_clip=0.0):
    if estimator not in {'hold', 'full_drift', 'weighted_drift', 'local_linear'}:
        raise ValueError(f'Unknown estimator: {estimator}')
    if not 0 <= beta <= 1:
        raise ValueError('beta must be in [0, 1]')
    elapsed = trace.times - trace.latest_generation_time
    if estimator == 'local_linear':
        if not np.isfinite(slope_clip) or slope_clip < 0:
            raise ValueError('slope_clip must be nonnegative and finite')
        slopes = np.zeros(len(elapsed))
        previous_time, previous_value, slope = trace.times[0], trace.latest_value[0], 0.0
        for i in range(1, len(elapsed)):
            if trace.latest_generation_time[i] > previous_time:
                current_time, current_value = trace.latest_generation_time[i], trace.latest_value[i]
                slope = np.clip((current_value-previous_value)/(current_time-previous_time), -slope_clip, slope_clip)
                previous_time, previous_value = current_time, current_value
            slopes[i] = slope
        return trace.latest_value + beta * slopes * elapsed
    weight = 0.0 if estimator == 'hold' else 1.0 if estimator == 'full_drift' else beta
    return trace.latest_value + weight * drift * elapsed
