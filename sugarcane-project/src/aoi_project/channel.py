from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class ReceptionTrace:
    times: np.ndarray
    values: np.ndarray
    latest_value: np.ndarray
    latest_generation_time: np.ndarray
    attempted: np.ndarray
    delivered: np.ndarray


def derive_seed(base_seed, phase, node_id, signal, segment_id, threshold_multiplier,
                loss_probability, feedback_model, channel_model):
    identity = [int(base_seed), str(phase), int(node_id), str(signal), int(segment_id),
                float(threshold_multiplier), float(loss_probability), feedback_model, channel_model]
    return int.from_bytes(hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).digest()[:8], 'big')


def burst_transition_probabilities(loss_probability, mean_bad_burst_length):
    if not 0 <= loss_probability < 1 or not np.isfinite(mean_bad_burst_length) or mean_bad_burst_length < 1:
        raise ValueError('Require loss in [0,1) and mean burst length >= 1 observation')
    b = 1.0 / mean_bad_burst_length
    a = loss_probability*b/(1-loss_probability)
    if a > 1:
        raise ValueError('Requested stationary loss and burst length imply an invalid transition probability')
    return a, b


def loss_stream(n, loss_probability, seed, channel_model='bernoulli', mean_bad_burst_length=5.0):
    rng = np.random.default_rng(seed)
    if channel_model == 'bernoulli':
        return rng.random(n) < loss_probability
    if channel_model != 'gilbert_elliott':
        raise ValueError('Unknown channel model')
    a, b = burst_transition_probabilities(loss_probability, mean_bad_burst_length)
    bad = rng.random() < loss_probability  # Stationary initialization.
    output = np.empty(n, dtype=bool)
    for i, u in enumerate(rng.random(n)):
        output[i] = bad
        bad = u >= b if bad else u < a
    return output


def simulate_threshold_channel(times, values, threshold, loss_probability, seed,
                               feedback_model='ack', delivery_delay_minutes=0.0,
                               channel_model='bernoulli', mean_bad_burst_length=5.0):
    times, values = np.asarray(times, float), np.asarray(values, float)
    if times.ndim != 1 or times.shape != values.shape or len(times) < 2:
        raise ValueError('Require matching one-dimensional traces of length >= 2')
    if not np.isfinite(times).all() or not np.isfinite(values).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Require finite values and strictly increasing times')
    if not np.isfinite(threshold) or threshold <= 0 or not 0 <= loss_probability < 1:
        raise ValueError('Require positive threshold and loss probability in [0,1)')
    if feedback_model not in {'ack', 'no_ack'}:
        raise ValueError('feedback_model must be ack or no_ack')
    if delivery_delay_minutes != 0:
        raise ValueError('Nonzero delivery delay is unsupported: exact arrival events are required')
    n = len(times)
    attempted, delivered = np.zeros(n, bool), np.zeros(n, bool)
    latest_value, latest_generation = np.empty(n), np.empty(n)
    sender = receiver = float(values[0])
    generation = float(times[0])
    # Channel advances at every observation, including observations without an attempt.
    lost = loss_stream(n, loss_probability, seed, channel_model, mean_bad_burst_length)
    for i, (time, value) in enumerate(zip(times, values)):
        if i and abs(value-sender) >= threshold:
            attempted[i] = True
            if feedback_model == 'no_ack':
                sender = float(value)
            if not lost[i]:
                delivered[i] = True
                receiver, generation = float(value), float(time)
                if feedback_model == 'ack':
                    sender = float(value)
        latest_value[i], latest_generation[i] = receiver, generation
    return ReceptionTrace(times, values, latest_value, latest_generation, attempted, delivered)


def trace_digest(trace):
    digest = hashlib.sha256()
    for array in (trace.times, trace.latest_value, trace.latest_generation_time, trace.attempted, trace.delivered):
        digest.update(np.asarray(array).tobytes())
    return digest.hexdigest()


def packet_event_rows(trace):
    return [dict(sample_index=int(i), time_minutes=float(trace.times[i]), value=float(trace.values[i]),
                 attempted=bool(trace.attempted[i]), delivered=bool(trace.delivered[i]))
            for i in np.flatnonzero(trace.attempted)]
