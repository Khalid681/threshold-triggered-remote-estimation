# Tomato soil dataset: preprocessing report

## Scope and provenance
The initial experiment uses only the 2025 soil sensor file, `2025/soil2025.csv`. The 2023,
2024 and environmental files are not pooled into this analysis. The raw CSV is
preserved; its SHA-256 before and after preprocessing is `22e52db17116feb8ba5736466d4ef04f3ca0848d8ddc0ac0833267d21b988c86`.
The processed file is `data/processed/tomato_soil_2025.csv` (SHA-256 `1d1ded2b9608681bacb8cad8af1ab1a96cd5f6f89162c24074c5babb3e2c600b`).

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
- Source records: 193,066; devices: 10.
- UTC coverage: 2025-05-31 22:00:07.298000+00:00 to 2025-11-03 10:03:52.195000+00:00.
- Missing/invalid temperature: 0; moisture: 0.
- Zero-moisture readings retained: 2,283.
- Retained rows: 193,066; removed rows: 0.
- Temperature range: 16.0 to 36.7 degrees C.
- Moisture range: 0.0 to 58.76 percent as supplied.
- Recording gaps above three nominal intervals: 101; largest gap: 11867.099 minutes.

## Experiment preparation
The existing splitter infers each device's modal positive sampling interval,
rounded to 0.001 minute, and separates recordings at gaps greater than three
times that interval. Segments require at least 100 valid readings per signal.
Each eligible signal segment is divided chronologically into 60% training,
20% validation and the remaining approximately 20% testing, using integer
boundaries. This produces 188 eligible node-signal segments.
Counts across the two signals (not unique source rows) are
231,060 training,
77,038 validation and
77,130 test observations.
Exact boundaries appear in split_manifest.csv; device metadata in node_manifest.csv.
The segment-length requirement excludes 904
signal observations from experiment splits; these remain in the processed CSV.
Training-only drift and increment-scale estimation in the existing pipeline
are retained. Random seeds, channel conditions and estimator settings are unchanged.

## Suggested methods paragraph
We evaluated temperature and soil-moisture traces from ten devices in the 2025
tomato-cultivation dataset. The dataset contained 193,066 timestamped records.
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
