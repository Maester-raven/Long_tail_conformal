# Task A — Global and Classwise Baselines

Task A implements two split conformal baselines on the fixed Pl@ntNet-Trunc
model predictions. It uses Task B's data splits and evaluation metrics.

## Methods

The nonconformity score for each calibration example is

$$
s(x_i,y_i) = 1 - p(y_i \mid x_i).
$$

For a group with $n$ calibration scores and target coverage 90%, let

$$
k = \lceil 0.9(n+1) \rceil.
$$

The threshold is the $k$-th smallest score when $k \le n$, or
$+\infty$ otherwise. A candidate class is included when
$1-p(y\mid x) \le q_y$.

- **Global:** all classes share one threshold.
- **Classwise:** every class has its own threshold.

Both methods use the calibration and prediction functions in
`src/methods/grouped.py`. Global corresponds to a single group, while
Classwise corresponds to one group per class. This keeps the quantile
calculation consistent with Task C without changing its grouping methods.

## Small-sample behavior

When a class has at most 8 calibration examples, the classwise
threshold is infinite and that class appears in every prediction set.
With 9 examples, the threshold is the largest calibration score.

Thus the number of classes with at most 8 calibration examples is a
lower bound on the Classwise prediction set size. This helps explain
why Classwise achieves high rare-class coverage but produces long sets.

## Experimental protocol

- Budgets: 10%, 30%, 100%; random seeds: 0–9.
- Standard protocol: 20% design, 80% calibration.
- Both baselines use the same calibration split as Task C.
- Global is also evaluated using the full selected budget.
- Test labels are used for evaluation only.

Coverage and set size are computed using Task B's `evaluate`. Overall
coverage and average set size are weighted by training frequency.
Rare-class coverage is the mean coverage of the 110 rarest classes.

The nominal 90% split-conformal guarantee assumes exchangeability
between calibration and future samples. It does not imply 90%
coverage for every class under Global, nor automatically for a
reweighted test distribution.

## Results

Standard protocol, mean across 10 seeds:

| Method | Budget | Overall coverage | Rare coverage | Avg set size |
|---|---:|---:|---:|---:|
| Global | 10% | 90.17% | 40.73% | 1.16 |
| Classwise | 10% | 94.52% | 100.00% | 279.54 |
| Global | 30% | 90.03% | 40.35% | 1.15 |
| Classwise | 30% | 91.89% | 100.00% | 222.28 |
| Global | 100% | 90.02% | 40.33% | 1.15 |
| Classwise | 100% | 90.59% | 100.00% | 139.03 |

The Classwise set-size lower bounds averaged 278.6, 220.9, and
137.0 classes at the three budgets. All 110 rare classes have
infinite Classwise thresholds in every standard experiment.
Their 100% coverage reflects automatic inclusion, not perfect
classification accuracy.

## Run

From the project root, with the precomputed data in `data/plantnet-trunc/`:

```sh
uv sync --locked
uv run --locked python -m unittest discover -s tests -p test_baselines.py
uv run --locked python -m scripts.analyze_baselines
```

Outputs:

- `results/baselines_coverage_all_runs.csv`: 90 runs with method,
  protocol, budget, and seed.
- `results/baselines_coverage_summary.csv`: means and sample
  standard deviations over the ten seeds.

The `min_set_size` column counts classes forced into every prediction
set by infinite thresholds. `forced_rare_classes` counts how many of
those belong to the rare-class subset. These diagnostics can be used
in Task D's comparison with Task C.
