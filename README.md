# Long-Tailed Conformal Prediction

DS4023 Machine Learning Group Project

## Project Goal

Compare frequency-based and confusion-based grouping
for conformal prediction under limited calibration data.

The goal is to improve rare-class coverage while keeping
prediction sets small.

## Project Structure

- src/data/          Dataset loading and splitting (Member B)
- src/methods/       Baseline and proposed methods (Members A, C)
- src/evaluation/    Evaluation metrics (Members B, D)
- scripts/           Data download and analysis scripts
- results/figures/   Generated figures
- docs/              Documentation and reports
- data/              Local dataset (excluded from Git)

## Setup

Create a Python environment and install dependencies:

    python -m pip install numpy pandas matplotlib scikit-learn gdown

Download the precomputed dataset:

    bash scripts/download_data.sh

## Run Task B

Run from the repository root:

    python -m src.data.data_loader
    python -m src.data.split
    python -m src.evaluation.metrics
    python -m scripts.plot_distribution
    python -m scripts.analyze_predictions

## Dataset

Pl@ntNet-Trunc:
- 330 classes
- 17,906 calibration samples
- 33,000 test samples
- 230,267 training labels

The project uses precomputed model probabilities.
No model training is required.

## Documentation

See docs/README_TASK_B.md for Task B details.

## Original Research Code and Data

https://github.com/tiffanyding/long-tail-conformal

The original research code and dataset belong to their
respective authors.
