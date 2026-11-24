# IoT Dataset from an Evolving Tomato Cultivation

This repository contains tomato cultivation sensor data from 2023–2025 and
experiments on Age of Information (AoI) and estimation accuracy under packet
loss. The experiments use 2025 soil temperature and moisture readings from ten
devices.

## Repository guide

| What to find | Where to start |
| --- | --- |
| Original sensor datasets and file descriptions | [2023](2023/README.md), [2024](2024/README.md), [2025](2025/README.md) |
| Processed 2025 soil dataset used by the experiments | [CSV](data/processed/tomato_soil_2025.csv) and [preprocessing report](reports/preprocessing_2025/report.md) |
| Five publication graphs and their results CSVs | [Results and figure guide](results-final2/README.md) |
| R code that reproduces the graphs from the included CSVs | [R Markdown](results-final2/code/publication-figures.Rmd) |
| Python experiment code and tests | [Source](src/aoi_project/) and [tests](results-final2/tests/) |

## Prepare the dataset

The processed dataset is already included. To rebuild it, run these commands
from the repository root with Python 3.10 or newer.

```powershell
python -m pip install -e .
python preprocess_tomato.py
```

Preprocessing writes `data/processed/tomato_soil_2025.csv` and a methods report,
checksums, device mapping, and chronological split manifest under
`reports/preprocessing_2025/`. It retains and flags zero moisture readings
and preserves the original sampling times.

## Run the Python experiments

From the repository root, after installing the project dependencies:

```powershell
python run_agriculture.py --config configs/experiment.json --run-name tomato-2025-ack
python run_agriculture.py --config configs/publication_ack_burst.json --run-name tomato-2025-burst
python run_agriculture.py --config configs/publication_no_ack_bernoulli.json --run-name tomato-2025-no-ack
```

The runner writes each named run and its quality report under `results/runs/`.
To view or rebuild the five figures from the included results CSVs, use the
[results and figure guide](results-final2/README.md).

## Run the tests

From the repository root:

```powershell
python -m pip install -e . pytest
python -m pytest -q
```
