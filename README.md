# Long-Tailed Conformal Prediction

DS4023 Machine Learning Group Project

## Overview

This project investigates conformal prediction for
long-tailed classification using Pl@ntNet-Trunc.

The current implementation covers Task B:
Dataset & Model Prediction.

## Dataset

Precomputed model probabilities and labels are used.

- 330 classes
- 17,906 calibration samples
- 33,000 test samples
- 230,267 training labels
- 12 classes without calibration samples

No model training is required.

## Setup

Install dependencies:

    python -m pip install numpy pandas matplotlib scikit-learn gdown

Download and extract the dataset:

    bash scripts/download_data.sh

## Task B

Run from the repository root:

    python -m src.data.data_loader
    python -m src.data.split
    python -m src.evaluation.metrics
    python -m scripts.plot_distribution
    python -m scripts.analyze_predictions

## Prediction Analysis

| Metric | Accuracy |
|---|---:|
| Overall Top-1 | 59.99% |
| Rare-class Top-1 | 33.95% |
| Zero-calibration-class Top-1 | 6.00% |

## Documentation

See [Task B Documentation](docs/README_TASK_B.md)
for detailed implementation and usage instructions.

## Reference

Original research repository:
https://github.com/tiffanyding/long-tail-conformal
