"""Prepare a separate, auditable 2025 soil input without modifying raw CSVs."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from aoi_project.data import build_trace_splits, load_sensor_data


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preprocess(source='2025/soil2025.csv', output='data/processed/tomato_soil_2025.csv',
               reports='reports/preprocessing_2025'):
    source, output, reports = Path(source), Path(output), Path(reports)
    if source.resolve() == output.resolve():
        raise ValueError('Processed output must differ from the raw source')
    before = digest(source)
    raw = pd.read_csv(source)
    required = {'ts_generation', 'generated_by', 'line', 'temperature', 'humidity'}
    if required.difference(raw.columns):
        raise ValueError(f'Missing columns: {sorted(required.difference(raw.columns))}')
    if raw.generated_by.isna().any():
        raise ValueError('Missing device identifier')
    timestamp = pd.to_datetime(raw.ts_generation, unit='ms', utc=True, errors='raise')
    if timestamp.isna().any():
        raise ValueError('Missing timestamp')
    if raw.duplicated(['generated_by', 'ts_generation']).any():
        raise ValueError('Duplicate device timestamps require explicit review')
    mapping = {device: i + 1 for i, device in enumerate(sorted(raw.generated_by.unique()))}
    clean = pd.DataFrame({'time stamp': timestamp, 'Node_ID': raw.generated_by.map(mapping),
                          'Temp': pd.to_numeric(raw.temperature, errors='coerce'),
                          'Moist': pd.to_numeric(raw.humidity, errors='coerce'),
                          'source_device': raw.generated_by, 'source_line': raw.line,
                          'source_row': np.arange(len(raw)), 'source_file': source.as_posix()})
    invalid_temp = ~np.isfinite(clean.Temp) | ~clean.Temp.between(-50, 80)
    invalid_moist = ~np.isfinite(clean.Moist) | ~clean.Moist.between(0, 100)
    clean.loc[invalid_temp, 'Temp'] = np.nan
    clean.loc[invalid_moist, 'Moist'] = np.nan
    clean['zero_moisture_flag'] = clean.Moist.eq(0)
    clean = clean.sort_values(['Node_ID', 'time stamp']).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False)
    validated = load_sensor_data(output)
    splits = build_trace_splits(validated, minimum_segment_records=100)
    from aoi_project.data import infer_nominal_interval_minutes
    gap_count = 0
    maximum_gap = 0.0
    for _, frame in validated.groupby('Node_ID'):
        if len(frame) < 2:
            continue
        delta = frame.timestamp.diff().dt.total_seconds().div(60)
        gap_count += int((delta > 3 * infer_nominal_interval_minutes(frame)).sum())
        maximum_gap = max(maximum_gap, float(delta.max()))
    manifest = []
    for s in splits:
        row = dict(node_id=s.node_id, signal=s.signal, segment_id=s.segment_id,
                   nominal_interval_minutes=s.nominal_interval_minutes)
        for part in ('train', 'validation', 'test'):
            f = getattr(s, part)
            row.update({part+'_records': len(f), part+'_start': str(f.timestamp.min()),
                        part+'_end': str(f.timestamp.max())})
        manifest.append(row)
    pd.DataFrame(manifest).to_csv(reports/'split_manifest.csv', index=False)
    nodes = clean.groupby(['Node_ID', 'source_device', 'source_line']).agg(
        records=('Temp', 'size'), start=('time stamp', 'min'), end=('time stamp', 'max'),
        temperature_min=('Temp', 'min'), temperature_max=('Temp', 'max'),
        moisture_min=('Moist', 'min'), moisture_max=('Moist', 'max'),
        zero_moisture_records=('zero_moisture_flag', 'sum')).reset_index()
    nodes.to_csv(reports/'node_manifest.csv', index=False)
    after = digest(source)
    if after != before:
        raise RuntimeError('Raw source checksum changed')
    summary = dict(source=str(source), source_sha256=before, raw_unchanged=before == after,
                   processed=str(output), processed_sha256=digest(output), records=len(raw),
                   devices=len(mapping), timestamp_start=str(timestamp.min()), timestamp_end=str(timestamp.max()),
                   invalid_or_missing_temperature=int(invalid_temp.sum()),
                   invalid_or_missing_moisture=int(invalid_moist.sum()),
                   zero_moisture_retained=int(clean.zero_moisture_flag.sum()),
                   gaps_over_three_nominal_intervals=gap_count, maximum_gap_minutes=maximum_gap,
                   eligible_node_signal_segments=len(splits),
                   split_records={p: sum(len(getattr(s, p)) for s in splits) for p in ('train','validation','test')})
    (reports/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    text = f'''# Tomato soil dataset: preprocessing report

## Scope and provenance
The initial experiment uses only the 2025 soil sensor file, `{source}`. The 2023,
2024 and environmental files are not pooled into this analysis. The raw CSV is
preserved; its SHA-256 before and after preprocessing is `{before}`.
The processed file is `{output}` (SHA-256 `{summary['processed_sha256']}`).

## Reproducible procedure
Run `python preprocess_tomato.py` after installing this project and its dependencies.
Unix millisecond timestamps are converted to UTC, preserving subsecond precision.
Device identifiers, sorted lexicographically, map to integer Node_ID values;
irrigation lines are retained as metadata, not used as device IDs. Temperature maps
to Temp and soil humidity maps to Moist. No unit conversion is applied. The source
README labels soil moisture as %RH; its physical calibration should be confirmed
before describing it as volumetric water content in a paper.

Records are ordered by device and time. Missing device IDs, missing timestamps,
and duplicate device/timestamp pairs stop preprocessing for review. Nonfinite,
nonnumeric or out-of-range temperature (outside -50 to 80 degrees C) and moisture
(outside 0 to 100) are marked missing independently per signal. These are broad
screening bounds, not sensor-calibrated anomaly detection. Zero moisture values
are retained and flagged because their validity cannot be inferred from this file.
No interpolation, resampling, smoothing, clipping, or statistical outlier removal
is performed. The original source row, device, line and filename remain traceable.

## Observed preprocessing results
- Source records: {len(raw):,}; devices: {len(mapping)}.
- UTC coverage: {timestamp.min()} to {timestamp.max()}.
- Missing/invalid temperature: {invalid_temp.sum():,}; moisture: {invalid_moist.sum():,}.
- Zero-moisture readings retained: {clean.zero_moisture_flag.sum():,}.
- Retained rows: {len(clean):,}; removed rows: 0.
- Temperature range: {clean.Temp.min()} to {clean.Temp.max()} degrees C.
- Moisture range: {clean.Moist.min()} to {clean.Moist.max()} percent as supplied.
- Recording gaps above three nominal intervals: {gap_count}; largest gap: {maximum_gap:.3f} minutes.

## Experiment preparation
The existing splitter infers each device's modal positive sampling interval,
rounded to 0.001 minute, and separates recordings at gaps greater than three
times that interval. Segments require at least 100 valid readings per signal.
Each eligible signal segment is divided chronologically into 60% training,
20% validation and the remaining approximately 20% testing, using integer
boundaries. This produces {len(splits)} eligible node-signal segments.
Counts across the two signals (not unique source rows) are
{summary['split_records']['train']:,} training,
{summary['split_records']['validation']:,} validation and
{summary['split_records']['test']:,} test observations.
Exact boundaries appear in split_manifest.csv; device metadata in node_manifest.csv.
The segment-length requirement excludes {2 * len(clean) - sum(summary['split_records'].values()):,}
signal observations from experiment splits; these remain in the processed CSV.
Training-only drift and increment-scale estimation in the existing pipeline
are retained. Random seeds, channel conditions and estimator settings are unchanged.

## Suggested methods paragraph
We evaluated temperature and soil-moisture traces from ten devices in the 2025
tomato-cultivation dataset. The dataset contained {len(raw):,} timestamped records.
Raw measurements were preserved in a separate source file, while a reproducible
conversion mapped Unix-millisecond timestamps to UTC and device identifiers to
stable numeric identifiers. No missing or out-of-range temperature or moisture
values were detected under the stated screening bounds. Zero-moisture readings
were retained with quality flags. We performed no interpolation or resampling.
Recordings were segmented at gaps exceeding three times each device's inferred
nominal sampling interval, and eligible segments were partitioned chronologically
into training, validation and test subsets in approximately 60:20:20 proportions.

## Limitations and next stage
This is a preprocessing and input-compatibility report, not an experimental-results
report. Zero readings and sensor calibration remain interpretation limitations.
Segments from one device are not independent devices. The timestamp jitter and
recording gaps are retained; simulated packet loss is additional to those gaps.
Cross-year generalization has not been evaluated. Full experiment sweeps have
not been run by preprocessing.
'''
    (reports/'report.md').write_text(text, encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    preprocess()
