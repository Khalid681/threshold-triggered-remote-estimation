"""Normalized paired experiment tables with calibration separate from evaluation."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from .channel import simulate_threshold_channel, derive_seed, trace_digest
from .data import estimate_drift, increment_scale, trace_arrays
from .estimators import estimate_from_receptions
from .metrics import communication_metrics, estimation_metrics, freshness_metrics

ESTIMATORS = ('hold', 'full_drift', 'weighted_drift', 'local_linear')


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    threshold_multiplier: float
    drift: float
    beta: float
    validation_mse: float
    local_beta: float
    local_validation_mse: float
    slope_clip: float


def validate_config(config):
    a, b = config['calibration_seeds'], config['evaluation_seeds']
    if not a or not b or set(a) & set(b):
        raise ValueError('Calibration and evaluation seeds must be nonempty and disjoint')
    for name in ['calibration_seeds','evaluation_seeds','threshold_multipliers','loss_probabilities','freshness_deadline_multipliers','beta_grid','signals','nodes']:
        values = config[name]
        if not values or len(values) != len(set(values)):
            raise ValueError(f'{name} must be nonempty and unique')
    if any(not isinstance(s, int) or s < 0 for s in a+b):
        raise ValueError('Seeds must be nonnegative integers')
    if any(not 0 <= x <= 1 for x in config['beta_grid']):
        raise ValueError('Beta grid must lie in [0,1]')
    if any(not np.isfinite(x) or x <= 0 for x in config['threshold_multipliers']+config['freshness_deadline_multipliers']):
        raise ValueError('Threshold and deadline multipliers must be positive')
    if any(not 0 <= x < 1 for x in config['loss_probabilities']):
        raise ValueError('Loss probabilities must lie in [0,1)')
    if config.get('delivery_delay_minutes', 0) != 0:
        raise ValueError('Nonzero delay requires exact arrival events and is unsupported')
    if config['feedback_model'] not in {'ack','no_ack'} or config['channel_model'] not in {'bernoulli','gilbert_elliott'}:
        raise ValueError('Unknown feedback or channel model')
    if config['channel_model'] == 'gilbert_elliott':
        from .channel import burst_transition_probabilities
        for p in config['loss_probabilities']:
            burst_transition_probabilities(p, config['mean_bad_burst_length'])


def reception(split, frame, multiplier, threshold, loss, seed, phase, config):
    channel_seed = derive_seed(seed, phase, split.node_id, split.signal, split.segment_id,
                               multiplier, loss, config['feedback_model'], config['channel_model'])
    t, x = trace_arrays(frame)
    trace = simulate_threshold_channel(t, x, threshold, loss, channel_seed,
                                      config['feedback_model'], config.get('delivery_delay_minutes',0),
                                      config['channel_model'], config.get('mean_bad_burst_length',5.0))
    return trace, channel_seed


def calibrate_split(split, multiplier, loss, config):
    drift = estimate_drift(split.train)
    threshold = multiplier * increment_scale(split.train)
    t, x = trace_arrays(split.train)
    slope_clip = float(np.quantile(np.abs(np.diff(x)/np.diff(t)), .95))
    grid = sorted(config['beta_grid'])
    scores = {name: {beta: [] for beta in grid} for name in ['weighted_drift','local_linear']}
    seeds = config['calibration_seeds'][:1] if loss == 0 else config['calibration_seeds']
    for seed in seeds:
        trace, _ = reception(split, split.validation, multiplier, threshold, loss, seed, 'calibration', config)
        for name in scores:
            for beta in grid:
                estimate = estimate_from_receptions(trace, name, drift, beta, slope_clip)
                scores[name][beta].append(estimation_metrics(trace.values, estimate)['mse'])
    means = {name: {b: float(np.mean(v)) for b,v in bs.items()} for name,bs in scores.items()}
    selected = {name: min(bs, key=lambda b: (bs[b], b)) for name,bs in means.items()}
    beta, local = selected['weighted_drift'], selected['local_linear']
    return CalibrationResult(threshold, multiplier, drift, beta, means['weighted_drift'][beta],
                             local, means['local_linear'][local], slope_clip)


def run_agriculture_sweep(splits, config):
    validate_config(config)
    rows = {name: [] for name in ['estimation','channel','freshness','calibration']}
    if not splits:
        raise ValueError('No eligible trace segments')
    for split in splits:
        if not (split.train.timestamp.max() < split.validation.timestamp.min() and split.validation.timestamp.max() < split.test.timestamp.min()):
            raise ValueError('Chronological split leakage')
        for multiplier in config['threshold_multipliers']:
            for loss in config['loss_probabilities']:
                identity = dict(node_id=split.node_id, signal=split.signal, segment_id=split.segment_id,
                                threshold_multiplier=multiplier, loss_probability=loss,
                                feedback_model=config['feedback_model'], channel_model=config['channel_model'])
                calibration = calibrate_split(split, multiplier, loss, config)
                rows['calibration'].append({**identity, **asdict(calibration),
                                            'calibration_seed_count': 1 if loss == 0 else len(config['calibration_seeds'])})
                seeds = config['evaluation_seeds'][:1] if loss == 0 else config['evaluation_seeds']
                for seed in seeds:
                    trace, channel_seed = reception(split, split.test, multiplier, calibration.threshold, loss, seed, 'evaluation', config)
                    common = dict(**identity, seed=seed, channel_seed=str(channel_seed), trace_sha256=trace_digest(trace))
                    comm = communication_metrics(trace)
                    if config['feedback_model'] == 'no_ack':
                        comm.update(retry_attempts=0, retry_fraction=np.nan)
                    fresh = freshness_metrics(trace, split.nominal_interval_minutes)
                    aoi = {k:v for k,v in fresh.items() if k not in ['freshness_deadline_minutes','freshness_violation_rate']}
                    rows['channel'].append(dict(**common, threshold=calibration.threshold, records=len(trace.times),
                                               duration_minutes=float(trace.times[-1]), nominal_interval_minutes=split.nominal_interval_minutes,
                                               **comm, **aoi))
                    for name in ESTIMATORS:
                        beta = {'hold':0., 'full_drift':1., 'weighted_drift':calibration.beta, 'local_linear':calibration.local_beta}[name]
                        estimate = estimate_from_receptions(trace, name, calibration.drift, beta, calibration.slope_clip)
                        rows['estimation'].append(dict(**common, estimator=name, beta=beta,
                                                      **estimation_metrics(trace.values, estimate)))
                    for deadline in config['freshness_deadline_multipliers']:
                        fm = freshness_metrics(trace, deadline*split.nominal_interval_minutes)
                        rows['freshness'].append(dict(**common, freshness_deadline_multiplier=deadline,
                                                      freshness_deadline_minutes=fm['freshness_deadline_minutes'],
                                                      freshness_violation_rate=fm['freshness_violation_rate']))
    return {k: pd.DataFrame(v) for k,v in rows.items()}


def save_results(tables, output_directory):
    from pathlib import Path
    from .analysis import write_summaries
    output = Path(output_directory)
    (output/'raw').mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output/'raw'/f'{name}_results.csv', index=False)
    write_summaries(tables, output/'summaries')
