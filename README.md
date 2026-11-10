# Long-Tailed Conformal Prediction

DS4023 Machine Learning Group Project

## Overview

This project investigates conformal prediction for
long-tailed classification using Pl@ntNet-Trunc.

Task B prepares the data and model predictions. Task C implements frequency
and confusion grouping for conformal prediction.

## Dataset

Precomputed model probabilities and labels are used.

- 330 classes
- 17,906 calibration samples
- 33,000 test samples
- 230,267 training labels
- 12 classes without calibration samples

No model training is required.

## Setup

Create the project environment from the lockfile:

    uv sync --locked

Download and extract the dataset with Bash and `unzip`:

    bash scripts/download_data.sh

You can also extract the downloaded archive directly into `data/plantnet-trunc/`.
[Pl@ntNet-Trunc archive](https://drive.google.com/file/d/1a0SF6xbMDwxmAde2VgBoP4q_qYX2hslm/view)

## Task B

Run from the repository root:

    uv run --locked python -m src.data.data_loader
    uv run --locked python -m src.data.split
    uv run --locked python -m src.evaluation.metrics
    uv run --locked python -m scripts.plot_distribution
    uv run --locked python -m scripts.analyze_predictions

## Task C

Generate per-run and cross-seed metrics for both grouping methods:

    uv run --locked python -m scripts.analyze_grouped_coverage

## Prediction Analysis

| Metric | Accuracy |
|---|---:|
| Overall Top-1 | 59.99% |
| Rare-class Top-1 | 33.95% |
| Zero-calibration-class Top-1 | 6.00% |

## Documentation

See [Task B Documentation](docs/README_TASK_B.md)
for data preparation and [Task C Documentation](docs/README_TASK_C.md)
for grouping, calibration, and experiment output.

[Task C 中文说明](docs/README_TASK_C_ZH.md)

## Reference

Original research repository:
https://github.com/tiffanyding/long-tail-conformal
