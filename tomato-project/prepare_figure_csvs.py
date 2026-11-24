"""Export saved figure inputs without recalculating or rounding statistics."""
import csv
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results-final2'
KEY = ['signal', 'threshold_multiplier', 'loss_probability', 'feedback_model', 'channel_model']


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=True)
    sources = {}
    exports = {}

    def read(path):
        sources[str(path.relative_to(ROOT))] = checksum(path)
        with path.open(newline='', encoding='utf-8-sig') as stream:
            return list(csv.DictReader(stream))

    def write(name, rows):
        if not rows:
            raise ValueError(f'Empty figure input: {name}')
        fields = list(dict.fromkeys(k for row in rows for k in row))
        path = OUTPUT / name
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        # Assert exact cell strings, including confidence limits and adjusted p-values.
        with path.open(newline='', encoding='utf-8') as stream:
            saved = list(csv.DictReader(stream))
        assert len(saved) == len(rows)
        for original, exported in zip(rows, saved):
            assert all(exported[k] == v for k, v in original.items())
        exports[name] = dict(rows=len(rows), sha256=checksum(path))

    accuracy, improvement, tradeoff, freshness = [], [], [], []
    for run in ('tomato2025-ack', 'tomato2025-no-ack'):
        folder = ROOT / 'Results2' / run
        metadata = json.loads((folder/'run_metadata.json').read_text())
        quality = json.loads((folder/'reports/quality_report.json').read_text())
        if metadata['status'] != 'complete' or not quality['passed']:
            raise ValueError(f'Incomplete or failed source run: {run}')
        sources[str((folder/'run_metadata.json').relative_to(ROOT))] = checksum(folder/'run_metadata.json')
        sources[str((folder/'reports/quality_report.json').relative_to(ROOT))] = checksum(folder/'reports/quality_report.json')
        a = read(folder/'summaries/accuracy_summary.csv')
        p = read(folder/'summaries/paired_comparison.csv')
        c = read(folder/'summaries/channel_summary.csv')
        f = read(folder/'summaries/freshness_summary.csv')
        channel = {tuple(row[k] for k in KEY): row for row in c}
        assert len(channel) == len(c), 'Channel join keys must be unique'
        for row in a:
            assert row['channel_model'] == 'bernoulli'
            accuracy.append({**row, 'source_run': run})
            if row['estimator'] in ('hold', 'weighted_drift'):
                other = channel[tuple(row[k] for k in KEY)]
                joined = {**row, 'source_run': run}
                for k, v in other.items():
                    if k in KEY:
                        assert joined[k] == v
                    elif k in joined:
                        joined['channel_' + k] = v
                    else:
                        joined[k] = v
                tradeoff.append(joined)
        improvement.extend({**row, 'source_run': run} for row in p if row['candidate'] == 'weighted_drift')
        freshness.extend({**row, 'source_run': run} for row in f)
    write('Graph_2_Accuracy_vs_Packet_Loss.csv', accuracy)
    write('Graph_3_Weighted_Drift_Improvement.csv', improvement)
    write('Graph_4_Accuracy_Freshness_Communication.csv', tradeoff)
    write('Graph_5_Deadline_Freshness_Violations.csv', freshness)

    baseline_folder = ROOT/'Results2/tomato2025-baseline'
    baseline = baseline_folder/'raw/baseline_validation.csv'
    check = json.loads((baseline_folder/'reports/baseline_quality.json').read_text())
    if not check['passed']:
        raise ValueError('Baseline quality check failed')
    rows = read(baseline)
    destination = OUTPUT/'Graph_1_Baseline_Validation.csv'
    shutil.copyfile(baseline, destination)
    assert checksum(baseline) == checksum(destination)
    exports[destination.name] = dict(rows=len(rows), sha256=checksum(destination))
    for name, before in sources.items():
        assert checksum(ROOT/name) == before, f'Source changed: {name}'
    manifest = dict(method='CSV string copy/filter/join only; no statistical recomputation',
                    sources=sources, exports=exports, source_cells_preserved=True)
    (OUTPUT/'export_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(exports, indent=2))


if __name__ == '__main__':
    main()
