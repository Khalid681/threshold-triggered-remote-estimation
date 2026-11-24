from __future__ import annotations
import argparse
import json
from pathlib import Path
from aoi_project.data import build_trace_splits, load_sensor_data
from aoi_project.experiments import run_agriculture_sweep, save_results, validate_config
from aoi_project.provenance import prepare_run, write_metadata, finish_run


def main():
    parser = argparse.ArgumentParser(description='Run normalized agricultural experiments in a unique directory')
    parser.add_argument('--config', default='configs/experiment.json')
    parser.add_argument('--run-name')
    parser.add_argument('--output', default='results/runs', help='Parent directory for named runs')
    parser.add_argument('--overwrite', action='store_true', help='Archive existing named run before replacement')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--max-splits', type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    validate_config(config)
    if args.quick:
        config.update(threshold_multipliers=[config['threshold_multipliers'][0], config['threshold_multipliers'][-1]],
                      loss_probabilities=[0., .1, .3], evaluation_seeds=config['evaluation_seeds'][:2],
                      calibration_seeds=config['calibration_seeds'][:2], beta_grid=[0.,.5,1.], quick=True)
    data = load_sensor_data(config['data_path'])
    splits = build_trace_splits(data, signals=config['signals'], nodes=config['nodes'],
                               gap_multiplier=config['split_gap_multiplier'], minimum_segment_records=config['minimum_segment_records'],
                               train_fraction=config['train_fraction'], validation_fraction=config['validation_fraction'])
    if args.max_splits is not None:
        if args.max_splits < 1:
            raise ValueError('--max-splits must be positive')
        splits = splits[:args.max_splits]
        config['max_splits'] = args.max_splits
    if not splits:
        raise ValueError('No eligible splits')
    output = prepare_run(args.output, args.run_name, args.overwrite)
    metadata = write_metadata(output, config, args.config, splits)
    print(f'Running {len(splits)} node-signal segments from {len(set(s.node_id for s in splits))} nodes', flush=True)
    try:
        tables = run_agriculture_sweep(splits, config)
        save_results(tables, output)
        metadata['row_counts'] = {k:len(v) for k,v in tables.items()}
        finish_run(output, metadata)
        from aoi_project.quality import check_run
        report = check_run(output)
        if not report['passed']:
            raise RuntimeError('Quality checks failed; see reports/quality_report.json')
    except Exception as error:
        metadata['error'] = str(error)
        finish_run(output, metadata, 'failed')
        raise
    (output/'logs'/'experiment.txt').write_text(json.dumps(metadata['row_counts'], indent=2)+'\n', encoding='utf-8')
    print(f'Run: {output}\nRows: {metadata["row_counts"]}\nQuality: PASS')


if __name__ == '__main__':
    main()
