# Task B — Dataset & Model Prediction

## Dataset
Pl@ntNet-Trunc contains 330 classes.

- Training labels: 230267
- Calibration samples: 17906
- Test samples: 33000
- Classes absent from calibration: 12
- Training frequency range: 1–8021

The project uses precomputed probabilities and labels.
No model training is required.

## Environment
Python 3.14, NumPy, pandas, Matplotlib, scikit-learn.

## Data Source
https://github.com/tiffanyding/long-tail-conformal

Extract plantnet-trunc.zip into data/plantnet-trunc/.

## Modules

- src/data/data_loader.py: load and validate data
- src/data/split.py: reproducible budget splits
- src/evaluation/metrics.py: evaluation metrics
- scripts/plot_distribution.py: class distribution visualization

## Usage

Load data:

    from src.data.data_loader import load_data
    data = load_data()

Create a reproducible split:

    from src.data.split import get_split
    split = get_split(data, budget=0.3, seed=0)

Use the full budget for calibration:

    from src.data.split import get_full_calibration_split
    full = get_full_calibration_split(data, budget=0.3, seed=0)

Evaluate prediction sets:

    from src.evaluation.metrics import evaluate
    results = evaluate(
        prediction_sets,
        data["test_labels"],
        data["train_counts"]
    )

prediction_sets must be a boolean matrix of shape (N, 330).

## Experimental Protocol

Budgets: 10%, 30%, 100%.
Seeds: 0 through 9.

Standard split:
- 20% design set
- 80% calibration set

Full-budget calibration is available separately.

The independent test set must not be used for group learning
or threshold calibration.

Rare classes are the 110 classes with the lowest
training frequencies, using stable sorting for ties.

Overall coverage and average set size are weighted
using training class frequencies.

## Validation

Run:

    python -m src.data.data_loader
    python -m src.data.split
    python -m src.evaluation.metrics
    python -m scripts.plot_distribution

Expected output figure: class_distribution.png

## Model Prediction Analysis

Script: scripts/analyze_predictions.py

Dataset: Pl@ntNet-Trunc independent test set.

Results:
- Top-1 accuracy: 59.9909%
- Macro accuracy: 59.9909%
- Rare-class accuracy: 33.9455%
- Zero-calibration-class accuracy: 6.0000%
- Number of zero-calibration classes: 12

The test set contains 100 samples per class, so its
unweighted Top-1 accuracy equals macro accuracy.

Rare classes are the 110 least frequent training classes.

Zero-calibration classes have no samples in the
original calibration pool, but may have training samples.

The results show substantially lower Top-1 accuracy
for rare classes. This motivates evaluating prediction
set coverage, but does not establish any conformal
coverage guarantee.

Generated figures:
- class_distribution.png
- frequency_vs_accuracy.png

## Automatic Dataset Setup

From the project root, first install Python dependencies.

Then run:

    bash scripts/download_data.sh

This downloads the public Pl@ntNet-Trunc precomputed
predictions and labels from the original authors.

The extracted files are stored in data/plantnet-trunc/.

Dataset files are excluded from Git.

## Global Conformal Coverage Diagnostic

Script: scripts/analyze_coverage.py

Protocol:
- Dataset: Pl@ntNet-Trunc
- Method: Global split conformal prediction
- Target coverage: 90%
- Budget: 100%
- Random seed: 0
- Calibration samples: 14,325
- Score: 1 - predicted probability of the true class

Results:

| Metric | Value |
|---|---:|
| Global threshold | 0.787484 |
| Overall weighted coverage | 90.1039% |
| Macro coverage | 66.9697% |
| Common-class coverage | 89.5364% |
| Rare-class coverage | 40.4909% |
| Common–rare coverage gap | 49.0455 percentage points |
| Average prediction set size | 1.1538 |

Common and rare classes are defined as the highest-
and lowest-frequency 110 training classes, respectively.

This is a single-seed diagnostic using the standard
20/80 design/calibration split. It is not the final
multi-seed baseline experiment assigned to Member A.

Figure: results/figures/common_vs_rare_coverage.png

## Multi-seed Global Coverage Analysis

Script: scripts/analyze_coverage_multiseed.py

Experimental protocol:
- Budgets: 10%, 30%, 100%
- Random seeds: 0–9
- Target coverage: 90%
- Standard protocol: 20% design, 80% calibration
- Additional protocol: 100% of selected budget for calibration
- Total runs: 60
- Fixed independent test set

### Standard 80% Calibration Results

| Budget | Overall Coverage | Common Coverage | Rare Coverage | Coverage Gap | Avg Set Size |
|---|---:|---:|---:|---:|---:|
| 10% | 90.17% | 89.58% | 40.73% | 48.85 pp | 1.1586 |
| 30% | 90.03% | 89.42% | 40.35% | 49.07 pp | 1.1502 |
| 100% | 90.02% | 89.43% | 40.33% | 49.10 pp | 1.1505 |

All table values are means across 10 random seeds.

Rare-class coverage sample standard deviations:
- 10% budget: 1.42 percentage points
- 30% budget: 0.70 percentage points
- 100% budget: 0.17 percentage points

### Findings

Global conformal prediction achieves approximately 90%
frequency-weighted overall coverage, while rare-class
macro coverage remains around 40%.

Increasing calibration budget reduces variation across
random calibration splits, but does not substantially
eliminate the common-versus-rare coverage gap.

The experiments reuse the same independent test set.
The reported cross-seed variability reflects calibration
splitting, not uncertainty across independent test sets.

This analysis covers Global conformal prediction only.
Comparisons with frequency grouping, confusion grouping,
and other baselines belong to subsequent group experiments.

### Generated Outputs

- results/global_coverage_all_runs.csv
- results/global_coverage_summary.csv
- results/figures/rare_coverage_multiseed.png
- results/figures/common_coverage_multiseed.png
- results/figures/coverage_gap_multiseed.png
- results/figures/set_size_multiseed.png
