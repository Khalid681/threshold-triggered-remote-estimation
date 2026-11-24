"""Machine-readable checks of raw tables, derived tables and run provenance."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .analysis import KEY, TRACE, CONDITION, build_summaries
from .experiments import ESTIMATORS, validate_config
from .provenance import sha256
from .channel import derive_seed
from .data import load_sensor_data, build_trace_splits


def check_run(run_dir):
    root = Path(run_dir)
    config = json.loads((root/'config_used.json').read_text())
    metadata = json.loads((root/'run_metadata.json').read_text())
    validate_config(config)
    tables = {name:pd.read_csv(root/'raw'/f'{name}_results.csv', dtype={'channel_seed':str}) for name in ['estimation','channel','freshness','calibration']}
    e,c,f,k = (tables[x] for x in ['estimation','channel','freshness','calibration'])
    manifest = pd.read_csv(root/'reports'/'split_manifest.csv')
    checks, details = {}, {}
    expected_channel = len(manifest)*len(config['threshold_multipliers'])*sum(1 if p == 0 else len(config['evaluation_seeds']) for p in config['loss_probabilities'])
    expected = dict(estimation=expected_channel*len(ESTIMATORS), channel=expected_channel,
                    freshness=expected_channel*len(config['freshness_deadline_multipliers']),
                    calibration=len(manifest)*len(config['threshold_multipliers'])*len(config['loss_probabilities']))
    keys = dict(estimation=KEY+['estimator'], channel=KEY, freshness=KEY+['freshness_deadline_multiplier'], calibration=[x for x in KEY if x != 'seed'])
    for name, df in tables.items():
        checks[name+'_unique_keys'] = not df.duplicated(keys[name]).any()
        checks[name+'_expected_rows'] = len(df) == expected[name]
        checks[name+'_no_infinity'] = not np.isinf(df.select_dtypes(include='number')).any().any()
        allowed = {'estimation':['nmse','nrmse'], 'channel':['packet_delivery_ratio','retry_fraction']}.get(name,[])
        checks[name+'_no_unexpected_missing'] = not df.drop(columns=allowed).isna().any().any()
        details[name] = dict(actual_rows=len(df), expected_rows=expected[name], missing_values={col:int(n) for col,n in df.isna().sum().items() if n})
    checks['normalized_missing_only_constant'] = bool((e.nmse.isna() == e.constant_trace).all() and (e.nrmse.isna() == e.constant_trace).all())
    checks['delivery_bounds'] = bool(((c.delivered_packets >= 0) & (c.delivered_packets <= c.attempted_packets)).all())
    checks['zero_loss_delivery'] = bool((c.loc[c.loss_probability.eq(0),'delivered_packets'] == c.loc[c.loss_probability.eq(0),'attempted_packets']).all())
    checks['zero_attempt_pdr'] = bool((c.packet_delivery_ratio.isna() == c.attempted_packets.eq(0)).all())
    has_attempt = c.attempted_packets.gt(0)
    checks['per_run_pdr'] = bool(np.allclose(c.loc[has_attempt,'packet_delivery_ratio'], c.loc[has_attempt,'delivered_packets']/c.loc[has_attempt,'attempted_packets']))
    checks['zero_loss_single_seed'] = not c[c.loss_probability.eq(0)].duplicated([x for x in KEY if x!='seed']).any()
    keys_join = KEY
    joined = e.merge(c[KEY+['trace_sha256','channel_seed']], on=keys_join, suffixes=('_est','_channel'), validate='many_to_one')
    checks['shared_channel_traces'] = bool(len(joined)==len(e) and (joined.trace_sha256_est==joined.trace_sha256_channel).all() and (joined.channel_seed_est==joined.channel_seed_channel).all() and e.groupby(KEY).estimator.nunique().eq(len(ESTIMATORS)).all())
    joined_f = f.merge(c[KEY+['trace_sha256','nominal_interval_minutes']], on=KEY, suffixes=('_fresh','_channel'), validate='many_to_one')
    checks['freshness_channel_link'] = bool(len(joined_f)==len(f) and (joined_f.trace_sha256_fresh==joined_f.trace_sha256_channel).all())
    checks['deadline_minutes_from_multiplier'] = bool(np.allclose(joined_f.freshness_deadline_minutes, joined_f.freshness_deadline_multiplier*joined_f.nominal_interval_minutes))
    checks['deadline_multiplier_coverage'] = bool(f.groupby(KEY).freshness_deadline_multiplier.apply(lambda x:set(x)==set(config['freshness_deadline_multipliers'])).all())
    checks['freshness_bounds'] = bool(f.freshness_violation_rate.between(0,1).all())
    checks['aoi_bounds'] = bool(((c.mean_aoi_minutes >= 0) & (c.peak_aoi_minutes >= c.mean_aoi_minutes)).all())
    expected_seeds = [str(derive_seed(r.seed,'evaluation',r.node_id,r.signal,r.segment_id,r.threshold_multiplier,r.loss_probability,r.feedback_model,r.channel_model)) for r in c.itertuples()]
    checks['stable_channel_seeds'] = c.channel_seed.tolist() == expected_seeds
    checks['independent_channel_seeds'] = c.channel_seed.nunique() == len(c)
    checks['disjoint_seed_sets'] = not set(config['calibration_seeds']) & set(config['evaluation_seeds'])
    checks['evaluation_seed_coverage'] = all(set(g.seed)==set(config['evaluation_seeds'][:1] if identity[4]==0 else config['evaluation_seeds']) for identity,g in c.groupby([x for x in KEY if x!='seed']))
    checks['chronological_split_integrity'] = bool(((pd.to_datetime(manifest.train_end)<pd.to_datetime(manifest.validation_start)) & (pd.to_datetime(manifest.validation_end)<pd.to_datetime(manifest.test_start))).all())
    data_splits = build_trace_splits(load_sensor_data(config['data_path']), signals=config['signals'], nodes=config['nodes'], gap_multiplier=config['split_gap_multiplier'], minimum_segment_records=config['minimum_segment_records'], train_fraction=config['train_fraction'], validation_fraction=config['validation_fraction'])
    if 'max_splits' in config:
        data_splits = data_splits[:config['max_splits']]
    expected_traces = {(s.node_id,s.signal,s.segment_id) for s in data_splits}
    checks['node_segment_coverage'] = set(c[TRACE].itertuples(index=False,name=None)) == expected_traces == set(manifest[TRACE].itertuples(index=False,name=None))
    checks['record_counts_match_source'] = all(int(manifest.loc[(manifest.node_id==s.node_id)&(manifest.signal==s.signal)&(manifest.segment_id==s.segment_id),'test_records'].iloc[0]) == len(s.test) for s in data_splits)
    checks['per_segment_condition_coverage'] = all(set(g[['threshold_multiplier','loss_probability']].itertuples(index=False,name=None)) == {(t,p) for t in config['threshold_multipliers'] for p in config['loss_probabilities']} for _,g in c.groupby(TRACE))
    checks['scenario_identity'] = all(df.feedback_model.eq(config['feedback_model']).all() and df.channel_model.eq(config['channel_model']).all() for df in tables.values())
    checks['condition_coverage'] = set(c[CONDITION].itertuples(index=False,name=None)) == {(s,t,p) for s in {split.signal for split in data_splits} for t in config['threshold_multipliers'] for p in config['loss_probabilities']}
    checks['config_hash'] = sha256(root/'config_used.json') == metadata['config_sha256']
    checks['dataset_hash'] = sha256(config['data_path']) == metadata['dataset_sha256']
    common = CONDITION+['feedback_model','channel_model']
    summary_keys = {'accuracy_summary':common+['estimator'], 'channel_summary':common,
                    'freshness_summary':common+['freshness_deadline_multiplier'], 'paired_comparison':common+['candidate'],
                    'node_level_effects':common+['candidate','node_id']}
    n_conditions = len(c[common].drop_duplicates())
    expected_summary = dict(accuracy_summary=n_conditions*len(ESTIMATORS), channel_summary=n_conditions,
                            freshness_summary=n_conditions*len(config['freshness_deadline_multipliers']), paired_comparison=n_conditions*(len(ESTIMATORS)-1),
                            node_level_effects=len(c[common+['node_id']].drop_duplicates())*(len(ESTIMATORS)-1))
    raw_finite = all(checks[name+'_no_infinity'] for name in tables)
    recomputed = build_summaries(tables) if raw_finite else {}
    for name, logical_key in summary_keys.items():
        df = pd.read_csv(root/'summaries'/f'{name}.csv')
        checks[name+'_unique_keys'] = not df.duplicated(logical_key).any()
        try:
            if name not in recomputed:
                raise AssertionError('Raw nonfinite metrics prevent statistical recomputation')
            pd.testing.assert_frame_equal(df.sort_values(logical_key).reset_index(drop=True),
                                          recomputed[name].sort_values(logical_key).reset_index(drop=True),
                                          check_dtype=False, check_exact=False, rtol=1e-10, atol=1e-12)
            checks[name+'_matches_recomputed_statistics'] = True
        except AssertionError:
            checks[name+'_matches_recomputed_statistics'] = False
        checks[name+'_expected_rows'] = len(df) == expected_summary[name]
        checks[name+'_no_infinity'] = not np.isinf(df.select_dtypes(include='number')).any().any()
        # Undefined CIs require <2 valid nodes; normalized scores require nonconstant traces.
        allowed = [x for x in df if x.endswith(('_ci_low','_ci_high')) or x.startswith(('nmse_','nrmse_'))]
        allowed += ['aggregate_pdr','aggregate_retry_fraction','aggregate_relative_improvement_percent', 'mean_within_segment_seed_sd','raw_p_value','q_value','descriptive_pair_sd']
        checks[name+'_no_unexpected_missing'] = not df.drop(columns=[x for x in allowed if x in df]).isna().any().any()
        details[name] = dict(actual_rows=len(df), expected_rows=expected_summary[name], missing_values={col:int(n) for col,n in df.isna().sum().items() if n})
        if name == 'channel_summary':
            totals = c.groupby(common)[['attempted_packets','delivered_packets']].sum().reset_index()
            joined_totals = df.merge(totals, on=common, validate='one_to_one')
            checks['aggregate_pdr_packet_totals'] = bool(np.allclose(joined_totals.aggregate_pdr, joined_totals.delivered_packets/joined_totals.attempted_packets.replace(0,np.nan), equal_nan=True))
        if name == 'paired_comparison':
            counts = c.groupby(common).size().rename('expected_pairs').reset_index()
            paired = df.merge(counts,on=common,validate='many_to_one')
            checks['unique_paired_counts'] = bool((paired.paired_runs==paired.expected_pairs).all())
            checks['fdr_bounds'] = bool(df.q_value.dropna().between(0,1).all() and (df.q_value.dropna() >= df.raw_p_value.dropna()-1e-12).all())
    figure_input = root/'reports'/'figure_inputs.json'
    if figure_input.exists():
        fig = json.loads(figure_input.read_text())
        checks['all_nodes_in_figures'] = fig['node_ids'] == sorted(c.node_id.unique().tolist())
        checks['figure_thresholds_separate'] = fig['thresholds'] == sorted(c.threshold_multiplier.unique().tolist())
    checks = {key:bool(value) for key,value in checks.items()}
    report = dict(passed=all(checks.values()), checks=checks, tables=details, node_ids=sorted(c.node_id.unique().tolist()),
                  segments=len(expected_traces), undefined_policy='Constant-trace normalization, zero-attempt PDR, loss-free seed SD and <2-node CIs are undefined, never imputed.')
    (root/'reports'/'quality_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
