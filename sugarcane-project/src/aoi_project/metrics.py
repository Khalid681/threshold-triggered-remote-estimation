from __future__ import annotations
import numpy as np
from .channel import ReceptionTrace


def _safe_duration(times):
    times = np.asarray(times, dtype=float)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Require at least two finite, strictly increasing times')
    return float(times[-1] - times[0])


def estimation_metrics(actual, estimate):
    actual, estimate = np.asarray(actual, float), np.asarray(estimate, float)
    if actual.shape != estimate.shape or actual.size == 0 or not np.isfinite(actual).all() or not np.isfinite(estimate).all():
        raise ValueError('Require equal nonempty finite arrays')
    error = estimate - actual
    mse = float(np.mean(error**2))
    variance = float(np.var(actual))
    # Normalized errors are undefined for (near-)constant test traces, not zero.
    valid = variance > 1e-12
    return dict(mae=float(np.mean(abs(error))), mse=mse, rmse=float(np.sqrt(mse)),
                bias=float(np.mean(error)), test_variance=variance,
                nmse=mse / variance if valid else np.nan,
                nrmse=float(np.sqrt(mse / variance)) if valid else np.nan,
                constant_trace=not valid)


def time_weighted_mean(times, values):
    times, values = np.asarray(times, float), np.asarray(values, float)
    if len(times) != len(values):
        raise ValueError('times and values must have equal length')
    return float(np.dot(values[:-1], np.diff(times)) / _safe_duration(times))


def communication_metrics(trace):
    duration = _safe_duration(trace.times)
    attempted, delivered = int(trace.attempted.sum()), int(trace.delivered.sum())
    if np.any(trace.delivered & ~trace.attempted) or not 0 <= delivered <= attempted:
        raise ValueError('Deliveries must be a subset of attempts')
    # A retry is an attempt while a prior failed threshold update remains unacknowledged.
    outstanding, retries = False, 0
    for attempt, delivery in zip(trace.attempted, trace.delivered):
        if attempt:
            retries += int(outstanding)
            outstanding = not delivery
    return dict(attempted_packets=attempted, delivered_packets=delivered,
                attempted_update_rate_per_minute=attempted / duration,
                delivered_update_rate_per_minute=delivered / duration,
                attempted_update_rate_per_observation=attempted / (len(trace.times)-1),
                delivered_update_rate_per_observation=delivered / (len(trace.times)-1),
                packet_delivery_ratio=delivered / attempted if attempted else np.nan,
                retry_attempts=retries, retry_fraction=retries / attempted if attempted else np.nan)


def aggregate_pdr(frame):
    attempts, delivered = frame.attempted_packets.sum(), frame.delivered_packets.sum()
    if ((frame.delivered_packets < 0) | (frame.delivered_packets > frame.attempted_packets)).any():
        raise ValueError('Invalid packet totals')
    return float(delivered / attempts) if attempts else np.nan


def freshness_metrics(trace, deadline_minutes):
    if not np.isfinite(deadline_minutes) or deadline_minutes <= 0:
        raise ValueError('deadline_minutes must be positive and finite')
    duration = _safe_duration(trace.times)
    age = trace.times - trace.latest_generation_time
    if not np.isfinite(age).all() or np.any(age < 0):
        raise ValueError('Generation times cannot be in the future')
    widths = np.diff(trace.times)
    start = age[:-1]
    return dict(mean_aoi_minutes=float(np.sum(start*widths + 0.5*widths**2)/duration),
                sample_mean_aoi_minutes=float(np.mean(age)),
                peak_aoi_minutes=float(max(np.max(start+widths), age[-1])),
                sample_aoi_p95_minutes=float(np.percentile(age, 95)),
                freshness_deadline_minutes=float(deadline_minutes),
                freshness_violation_rate=float(np.clip(widths-(deadline_minutes-start), 0, widths).sum()/duration))
