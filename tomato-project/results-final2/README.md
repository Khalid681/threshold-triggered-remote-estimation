# R results and publication figures

- [R Markdown code](code/publication-figures.Rmd) reads the five CSVs and generates the figures.
- [Publication figures](Figures_Publication/) contains the five graphs in PNG, PDF, and TIFF.
- [Python tests](tests/) cover the experiment and preprocessing code. Run them from the full repository checkout using the [root README](../README.md#run-the-tests).

## View the five graphs

| Graph | PNG | Results CSV |
| --- | --- | --- |
| 1. Baseline validation | [View](Figures_Publication/Graph_1_Baseline_Validation.png) | [CSV](Graph_1_Baseline_Validation.csv) |
| 2. Accuracy versus packet loss | [View](Figures_Publication/Graph_2_Accuracy_vs_Packet_Loss.png) | [CSV](Graph_2_Accuracy_vs_Packet_Loss.csv) |
| 3. Weighted-drift improvement | [View](Figures_Publication/Graph_3_Weighted_Drift_Improvement.png) | [CSV](Graph_3_Weighted_Drift_Improvement.csv) |
| 4. Accuracy, freshness and communication | [View](Figures_Publication/Graph_4_Accuracy_Freshness_Communication.png) | [CSV](Graph_4_Accuracy_Freshness_Communication.csv) |
| 5. Deadline freshness violations | [View](Figures_Publication/Graph_5_Deadline_Freshness_Violations.png) | [CSV](Graph_5_Deadline_Freshness_Violations.csv) |

The [publication figures directory](Figures_Publication/) also contains PDF
and TIFF versions. The [export manifest](export_manifest.json) records row
counts and SHA-256 hashes for the five CSVs.

## Reproduce the figures in R

Open [publication-figures.Rmd](code/publication-figures.Rmd) and knit it in RStudio, or run the following
in an R session started at the repository root. HTML rendering requires Pandoc,
and the PDF export uses Cairo support in R.

```r
install.packages(c("rmarkdown", "readr", "dplyr", "tidyr", "ggplot2", "patchwork", "scales"))
rmarkdown::render("results-final2/code/publication-figures.Rmd")
```

The code reads the five checked-in CSVs and writes new exports to
`results-final2/generated-figures/`.

## Figure data

One CSV is provided per figure. Graphs 2–5 use the complete 2025 soil ACK and
no-ACK Bernoulli-loss runs. Quick and burst-loss runs are excluded, so channel
models and reduced test grids are not mixed. Temp means soil temperature;
Moist means soil moisture. Keep these labels distinct from air measurements.

All original numerical cell strings, confidence bounds and adjusted p-values are
preserved. Exports only copy, filter, concatenate or join saved summary rows;
no averaging, rounding, confidence-interval estimation or p-value adjustment is
performed. Percentage conversions belong in R, not in these CSVs. The export
manifest records source and output hashes. Original results are unchanged.

## Graph 1 — Baseline validation

Use `Graph_1_Baseline_Validation.csv` for three panels, with threshold on x and
update_rate, mean_aoi, and mse on y. Show simulation points and corresponding
theoretical_* values as dashed curves. The baseline uses 1,000,000 slots,
seed 2026, thresholds 1–10. Its AoI is in discrete slots; do not label it minutes.
Its saved *_mc_ci_low/high are descriptive block Monte Carlo intervals, not the
device-clustered confidence intervals used for the agricultural results.

## Graph 2 — Accuracy versus packet loss

Use `Graph_2_Accuracy_vs_Packet_Loss.csv`. Plot loss_probability * 100 versus
mse_mean with mse_ci_low/high error bars, faceted by signal and feedback_model.
Keep estimator colors identical across all panels and figures. Four estimators
are hold, full_drift, weighted_drift, and local_linear.

All five threshold multipliers are retained. For the requested four-panel figure,
choose and explicitly state ONE threshold_multiplier (e.g. 1.0) before plotting.
Alternatively create a separate four-panel figure for each threshold. Never
average these saved summaries or join different thresholds into one curve.

## Graph 3 — Weighted-drift improvement

Use `Graph_3_Weighted_Drift_Improvement.csv`. It contains candidate weighted_drift
only, compared with hold. Map loss_probability to x, threshold_multiplier to y,
aggregate_relative_improvement_percent to fill, centered at zero. Positive
values indicate lower MSE than hold. Add an asterisk where q_value < 0.05.
Facet by signal and feedback_model so each heatmap cell is unique. q_value
retains its original within-run adjustment family; it was not readjusted across
the two combined runs.

## Graph 4 — Accuracy, freshness and communication

Use `Graph_4_Accuracy_Freshness_Communication.csv`. This is a many-to-one join
of the hold/weighted_drift accuracy summaries to channel summaries using signal,
threshold_multiplier, loss_probability, feedback_model and channel_model.
Shared channel counts have the channel_ prefix; metric names are unchanged.

Use attempted_update_rate_per_minute_mean on x; mse_mean in upper panels and
mean_aoi_minutes_mean in lower panels. Distinguish feedback_model by color and
estimator by line style. Use soil temperature and moisture as columns. In lower
panels retain ONLY estimator == "hold" because the channel values repeat for
both estimators. AoI and rates are in minutes and updates/minute, respectively.

Choose ONE loss_probability for the four-panel figure, or generate a separate
figure per loss probability. If displaying several probabilities together,
explicitly distinguish them and include loss_probability in line groups.
Within each group sort numerically by threshold_multiplier and use geom_path,
which follows row order; geom_line instead reorders by x. Retain saved confidence
bounds if displaying uncertainty; joined accuracy/channel intervals do not
constitute a joint confidence region.

## Graph 5 — Deadline freshness

Use `Graph_5_Deadline_Freshness_Violations.csv`. Plot loss_probability * 100
against freshness_violation_rate_mean * 100; multiply both corresponding
confidence bounds by 100 as well. Use separate curves for
freshness_deadline_multiplier values 2, 3 and 6 and panels for signal and
feedback_model. Choose and state ONE threshold_multiplier, consistently with
Graph 2, or create a separate four-panel figure per threshold.

## Read the files in R

```r
folder <- "results-final2" # From the repository root
accuracy <- read.csv(file.path(folder, "Graph_2_Accuracy_vs_Packet_Loss.csv"))
accuracy <- subset(accuracy, threshold_multiplier == 1.0)
```

## Re-export CSVs from the source runs

From the project root, with the project installed and the completed source runs
available, run `python prepare_figure_csvs.py`. If the baseline is absent, run
`python run_baseline.py --output Results2 --run-name tomato2025-baseline --csv-only`
first. The exporter verifies saved completion/quality flags and exact source-cell
preservation. This verifies export integrity; it does not independently validate
the scientific assumptions of the experiments.
