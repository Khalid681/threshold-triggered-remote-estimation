# Revision audit and validation report

Completed 2026-09-28. All 19 identified problems were corrected; the broader experiment requirements in the attached request were implemented. Full publication sweeps were intentionally not executed.

## Changed and added project files

| File | Change |
|---|---|
| `README.md` | Removed conflict markers; documented provenance, assumptions, exact metrics, inference, scenarios, commands and limits. |
| `configs/experiment.json` | Disjoint calibration/evaluation seeds, loss levels through 0.5, channel model; removed unused gap_path. |
| `configs/publication_ack_bernoulli.json` | Ready-to-run primary ACK scenario. |
| `configs/publication_no_ack_bernoulli.json` | Separate no-ACK scenario. |
| `configs/publication_ack_burst.json` | Separate Gilbert-Elliott scenario. |
| `run_agriculture.py` | Unique runs, normalized output, all-node smoke mode, metadata, automatic quality checks. |
| `run_baseline.py` | Protected baseline runs, million-slot validation, PNG/PDF and Monte Carlo intervals. |
| `paired_analysis.py` | Unique pairing, legacy deadline deduplication, node inference and FDR. |
| `plot_results.py` | Entry point for all-node publication figures. |
| `quality_checks.py` | Machine-readable quality-check entry point. |
| `src/aoi_project/channel.py` | Stable independent channel seeds, shared trace digests, burst loss, explicit delay rejection. |
| `src/aoi_project/data.py` | Reject infinite measurements and invalid chronological splits. |
| `src/aoi_project/estimators.py` | Receiver-only, clipped, shrinkage-tuned local linear comparator. |
| `src/aoi_project/experiments.py` | Separate result tables and leakage-safe calibration/evaluation. |
| `src/aoi_project/metrics.py` | Exact continuous AoI, crossings, peak, NaN PDR, packet totals, normalized error. |
| `src/aoi_project/analysis.py` | Seed-to-segment-to-node aggregation, 10,000 bootstrap draws, exact sign flips, BH FDR. |
| `src/aoi_project/plotting.py` | Threshold panels, multiplier deadlines, clustered intervals, trade-offs and descriptive plots. |
| `src/aoi_project/provenance.py` | Protected directories, hashes, environment and split manifests. |
| `src/aoi_project/quality.py` | 70 checks including recomputation of summaries from raw tables. |
| `tests/test_metrics.py` | Kept original test, corrected obsolete discrete expectations and added sample-mean check. |
| `tests/test_revision.py` | Regression, channel, hierarchy and all-node end-to-end tests. |
| `previous_results_manifest.json` | SHA-256 manifest of all 32 pre-existing result files. |
| `REVISION_REPORT.md` | This file inventory, acceptance mapping and validation evidence. |

The original baseline, channel and data tests remain in place. A repository-local `.venv` was created for validation; dependency installation refreshed generated editable-package metadata. No source measurements were modified. Generated result artifacts are listed under their run directories, rather than individually in this source-file inventory.

## Resolution of the 19 issues

1-3. Estimation, channel, freshness and calibration tables have separate logical keys. Deadlines never enter accuracy pairing. A regression with 40 segments x 30 seeds x 3 deadlines asserts exactly 1,200 pairs, not 3,600. Conflicting duplicate legacy scores are rejected.

4. Seeds are averaged within segment, segments within node; nodes are equally weighted. Fixed-seed 10,000-resample percentile intervals resample nodes. Exact two-sided node sign flips and BH q-values are reported.

5. Zero-loss conditions run one seed, including calibration. Legacy zero-loss copies are collapsed.

6. Relative improvement uses the ratio of node-balanced mean MSEs, not the mean of individual percentages. Zero/near-zero denominator yields NaN. NMSE/NRMSE and constant-trace safeguards complement raw MSE.

7-8. Figures retain all nodes and group freshness by nominal-interval multiplier. Both multiplier and absolute minutes remain in raw freshness rows.

9-11. AoI uses exact linear integration and interval-crossing durations, with pre-boundary peaks. The specified example gives 25/3, 20, 2/3 and sample mean 5.

12-13. No-attempt PDR is NaN; aggregate PDR uses packet totals. Summaries disclose zero-attempt counts, communication rates and ACK retry burden.

14-15. SHA-256-derived streams distinguish phase, node, signal, segment, threshold, loss, feedback and channel. Calibration seeds 1001-1005 and evaluation seeds 101-130 are disjoint. Estimators share one reception trace.

16. Each threshold gets separate panels with node-cluster intervals. Additional distribution and individual-node figures are explicitly descriptive. Figures are 300 dpi PNG plus vector PDF.

17-18. README conflicts are removed. Unused gap_path was removed from active configurations; the gap CSV is preserved and its historical-only role explained.

19. All original tests were retained (the incorrect freshness expectations were corrected), and regression/integration coverage was expanded to 45 tests.

## Validation evidence

Final test command: `.venv/Scripts/python.exe -m pytest -v`. Result: **45 passed in 20.69 seconds**, no warnings. Full log: `results/runs/revision_final_20260928/logs/pytest.txt`.

Final ACK Bernoulli smoke run: `results/runs/revision_final_20260928`.

- Coverage: 12 nodes, 80 node-signal segments, both signals, two thresholds, three loss levels and all deadline multipliers (2, 3, 6).
- Raw rows: 3,200 estimation; 800 channel; 2,400 freshness; 480 calibration.
- Summary rows: 48 accuracy; 12 channel; 36 freshness; 36 paired comparisons; 432 node effects.
- Per signal/threshold/loss: 40 unique zero-loss pairs; 80 pairs at each nonzero loss (two evaluation seeds).
- All 70 quality checks passed, including recomputed summaries, packet totals, split/seed/trace integrity, table keys/counts and all-node figure coverage.
- 22 agricultural figures generated in both PNG and PDF: 44 files.
- Visual inspection covered accuracy, freshness and revised per-node effects.

Million-slot baseline: `results/runs/revision_baseline_validated_20260928`. Ten thresholds, finite metrics, analytical-agreement check passed. Maximum relative errors: update rate 0.6899%, mean AoI 1.1213%, MSE 0.6974%. Both PNG and PDF are saved. Baseline intervals describe block Monte Carlo variation, not sensor uncertainty.

Preservation: SHA-256 rechecked all **32 previous result files**; none changed or disappeared. Source dataset files have no Git diff. Details: `results/runs/revision_final_20260928/reports/previous_results_preservation.json`.

Intermediate uniquely named validation runs were retained too. The final agricultural and baseline directories above are the handoff artifacts.

## Interpretation and limitations

In the smoke run, full drift increased node-balanced MSE in every examined condition. Weighted drift had small temperature improvements at threshold multiplier 3, but none survived FDR correction; these limited smoke results do not establish superiority. Local prediction also helps and hurts depending on the condition.

The full ACK, no-ACK and burst sweeps remain unrun. No-ACK and burst implementations were covered by synthetic unit tests, including long-run stationary-loss/burst checks. No nonzero-delay experiment is supported. Twelve spatially related sensors at one site limit generalization; node clustering does not remove all spatial dependence. Constant traces yield undefined normalized metrics. The original cleaning script/version mapping is unavailable. See README for the verified source citation/license and complete assumptions.

## Full reproduction commands (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1

python run_agriculture.py --config configs/publication_ack_bernoulli.json --run-name publication_ack_bernoulli
python run_agriculture.py --config configs/publication_no_ack_bernoulli.json --run-name publication_no_ack_bernoulli
python run_agriculture.py --config configs/publication_ack_burst.json --run-name publication_ack_burst

foreach ($run in @('publication_ack_bernoulli', 'publication_no_ack_bernoulli', 'publication_ack_burst')) {
    python paired_analysis.py --run-dir "results/runs/$run"
    python plot_results.py --run-dir "results/runs/$run"
    python quality_checks.py --run-dir "results/runs/$run"
}
```

Existing names are refused; choose a new name for another execution. Explicit `--overwrite` archives the prior run before replacement. With the current 80 eligible segments, each full scenario is expected to produce 241,600 accuracy rows, 60,400 channel rows, 181,200 freshness rows and 2,400 calibration rows. Deterministic zero-loss seeds account for the reduced counts.
