"""Equal-node inference: seeds -> segments -> nodes; no deadline in accuracy keys."""
from functools import lru_cache
from itertools import product
import numpy as np
import pandas as pd
from .metrics import aggregate_pdr

CONDITION = ['signal', 'threshold_multiplier', 'loss_probability']
TRACE = ['node_id', 'signal', 'segment_id']
KEY = TRACE + ['threshold_multiplier', 'loss_probability', 'seed', 'feedback_model', 'channel_model']
BOOTSTRAP_SEED = 8675309
BOOTSTRAP_RESAMPLES = 10000


@lru_cache(maxsize=32)
def bootstrap_indices(n, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    return np.random.default_rng(seed).integers(0, n, (resamples, n))


def cluster_interval(node_values, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    x = np.asarray(node_values, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    if resamples < 10000:
        raise ValueError('At least 10000 node bootstrap resamples required')
    samples = x[bootstrap_indices(len(x), resamples, seed)].mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]))


def node_means(frame, metrics):
    segments = frame.groupby(['node_id', 'segment_id'], sort=True)[metrics].mean()
    return segments.groupby('node_id', sort=True).mean()


def fdr_bh(pvalues):
    p = np.asarray(pvalues, float)
    result = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    if len(order):
        result[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(order)/np.arange(1, len(order)+1))[::-1])[::-1])
    return result


def sign_flip_pvalue(differences):
    d = np.asarray(differences, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan
    if len(d) > 20:
        raise ValueError('Exact sign-flip inference limited to 20 independent nodes')
    signs = np.asarray(list(product([-1., 1.], repeat=len(d))))
    return float(np.mean(np.abs(signs @ d / len(d)) >= abs(d.mean()) - 1e-14))


def unique_accuracy(frame):
    frame = frame.copy()
    for column, default in [('feedback_model', 'ack'), ('channel_model', 'bernoulli')]:
        if column not in frame:
            frame[column] = default
    keys = KEY + ['estimator']
    if frame.duplicated(keys).any():
        # Accept only exact legacy deadline copies; never silently average inconsistent pairs.
        metrics = [c for c in ['mse', 'mae', 'rmse', 'nmse', 'nrmse', 'trace_sha256'] if c in frame]
        if (frame.groupby(keys, dropna=False)[metrics].nunique(dropna=False) > 1).any().any():
            raise ValueError('Conflicting duplicate accuracy keys')
        frame = frame.drop_duplicates(keys)
    # Legacy loss-free seed copies carry no Monte Carlo evidence.
    deterministic = frame.loss_probability.eq(0)
    no_seed_keys = [c for c in keys if c != 'seed']
    if (frame[deterministic].groupby(no_seed_keys).mse.nunique() > 1).any():
        raise ValueError('Inconsistent loss-free outcomes')
    return pd.concat([frame[~deterministic], frame[deterministic].drop_duplicates(no_seed_keys)], ignore_index=True)


def summarize(frame, groups, metrics, channel=False):
    rows = []
    for values, g in frame.groupby(groups, sort=True, dropna=False):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(groups, values))
        nodes = node_means(g, metrics)
        row.update(runs=len(g), nodes=g.node_id.nunique(), segments=len(g[['node_id', 'segment_id']].drop_duplicates()), seeds=g.seed.nunique())
        for metric in metrics:
            x = nodes[metric]
            lo, hi = cluster_interval(x)
            row.update({metric+'_mean': x.mean(), metric+'_ci_low': lo, metric+'_ci_high': hi,
                        metric+'_valid_nodes': x.notna().sum(), metric+'_undefined_runs': g[metric].isna().sum()})
        if channel:
            row.update(total_attempted_packets=int(g.attempted_packets.sum()), total_delivered_packets=int(g.delivered_packets.sum()),
                       zero_attempt_runs=int(g.attempted_packets.eq(0).sum()), aggregate_pdr=aggregate_pdr(g),
                       total_retry_attempts=int(g.retry_attempts.sum()),
                       aggregate_retry_fraction=g.retry_attempts.sum()/g.attempted_packets.sum() if g.attempted_packets.sum() else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(frame):
    frame = unique_accuracy(frame)
    pivot = frame.pivot(index=KEY, columns='estimator', values='mse').reset_index()
    required = ['hold', 'weighted_drift', 'full_drift']
    if any(c not in pivot for c in required) or pivot[required].isna().any().any():
        raise ValueError('Missing paired estimators')
    candidates = [c for c in ['weighted_drift', 'full_drift', 'local_linear'] if c in pivot]
    rows, effects = [], []
    for values, g in pivot.groupby(CONDITION + ['feedback_model', 'channel_model'], sort=True):
        identity = dict(zip(CONDITION + ['feedback_model', 'channel_model'], values))
        nodes = node_means(g, ['hold'] + candidates)
        for candidate in candidates:
            if g[candidate].isna().any():
                raise ValueError('Incomplete candidate pairs')
            difference = g[candidate] - g.hold
            node_difference = nodes[candidate] - nodes.hold
            lo, hi = cluster_interval(node_difference)
            hold_mean, candidate_mean = nodes.hold.mean(), nodes[candidate].mean()
            row = dict(**identity, candidate=candidate, paired_runs=len(g), nodes=len(nodes),
                       segments=len(g[['node_id','segment_id']].drop_duplicates()), seeds=g.seed.nunique(),
                       mean_hold_mse=hold_mean, mean_candidate_mse=candidate_mean,
                       mean_weighted_drift_mse=nodes.weighted_drift.mean(), mean_full_drift_mse=nodes.full_drift.mean(),
                       mean_mse_difference=node_difference.mean(), median_mse_difference=float(np.median(difference)),
                       median_node_mse_difference=float(np.median(node_difference)),
                       aggregate_relative_improvement_percent=100*(hold_mean-candidate_mean)/hold_mean if hold_mean > 1e-12 else np.nan,
                       difference_ci_low=lo, difference_ci_high=hi, raw_p_value=sign_flip_pvalue(node_difference),
                       pooled_pair_mean_difference=difference.mean(),
                       descriptive_pair_sd=difference.std(ddof=1))
            tie = np.isclose(difference, 0, atol=1e-12, rtol=0)
            row.update(win_fraction=float(((difference < 0) & ~tie).mean()), tie_fraction=float(tie.mean()),
                       loss_fraction=float(((difference > 0) & ~tie).mean()))
            # Within-segment seed variability is descriptive, not the inferential sample size.
            row['mean_within_segment_seed_sd'] = g.assign(difference=difference).groupby(['node_id','segment_id']).difference.std(ddof=1).mean()
            rows.append(row)
            for node, d in node_difference.items():
                effects.append(dict(**identity, candidate=candidate, node_id=node, mse_difference=d,
                                    hold_mse=nodes.loc[node,'hold'], candidate_mse=nodes.loc[node,candidate]))
    result = pd.DataFrame(rows)
    # One conservative FDR family across all signals, candidates, thresholds and losses in a run.
    result['q_value'] = fdr_bh(result.raw_p_value)
    return result, pd.DataFrame(effects)


def build_summaries(tables):
    common = CONDITION + ['feedback_model', 'channel_model']
    accuracy = summarize(tables['estimation'], common+['estimator'], ['mse','mae','rmse','nmse','nrmse'])
    channel = summarize(tables['channel'], common,
                        ['mean_aoi_minutes','peak_aoi_minutes','sample_mean_aoi_minutes',
                         'attempted_update_rate_per_minute','delivered_update_rate_per_minute',
                         'attempted_update_rate_per_observation','delivered_update_rate_per_observation'], channel=True)
    fresh = summarize(tables['freshness'], common+['freshness_deadline_multiplier'], ['freshness_violation_rate'])
    paired, effects = paired_comparison(tables['estimation'])
    outputs = dict(accuracy_summary=accuracy, channel_summary=channel, freshness_summary=fresh,
                   paired_comparison=paired, node_level_effects=effects)
    return outputs


def write_summaries(tables, output):
    from pathlib import Path
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    outputs = build_summaries(tables)
    for name, frame in outputs.items():
        frame.to_csv(output / (name+'.csv'), index=False)
    return outputs
