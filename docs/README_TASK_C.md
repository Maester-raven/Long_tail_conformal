# Task C — Grouped split conformal prediction

[中文说明](README_TASK_C_ZH.md)

Task C compares training-frequency and design-set confusion grouping on the
Pl@ntNet-Trunc predictions. Both methods share the same conformal calibration
and evaluation rules.

## Group construction and label budget

The standard protocol selects 10%, 30%, or 100% of the original calibration
pool with seeds 0–9. Each selected pool is split into 20% design and 80%
calibration using Task B's `get_split`. A class with fewer than three design
examples, including zero, goes into a shared fallback group in both paired
methods.

- **Confusion grouping.** Average the predicted probability vectors of design
  examples with true class $c$, set coordinate $c$ to zero, then divide by the
  sum of the remaining coordinates. Apply K-means to these vectors with five
  regular groups, the split seed, and `n_init=10`. A vector with no off-diagonal
  probability mass stays zero. With fewer than five distinct vectors, reduce
  the cluster count to the number of distinct vectors. A single distinct
  vector forms one regular group. An empty eligible set yields only the
  fallback group.
- **Paired frequency grouping.** Keep the same fallback classes and number of
  occupied regular groups. Stably sort the remaining classes by training
  frequency and cut the sequence into contiguous groups whose class counts
  differ by at most one. Ties follow class ID order.
- **Full-budget frequency grouping.** Apply the same stable sort and slicing
  to all classes with five groups, then use `get_full_calibration_split` to
  calibrate with the entire selected pool. This protocol uses training
  frequencies alone for grouping, so both group construction and calibration
  sample count differ from the paired frequency run.

## Thresholds and prediction sets

For each fixed group $g$, collect only the calibration scores of examples with
true labels in that group. If $n_g$ is the number of these examples, use

$$
s(x,y) = 1 - p(y\mid x),\qquad k_g = \lceil 0.9(n_g + 1) \rceil.
$$

When $k_g \le n_g$, $q_g$ is the $k_g$-th smallest score, counting from one.
Otherwise set $q_g = +\infty$. Thus at $n_g=0$ or $8$ the threshold is infinite,
and at $n_g=9$ it is the largest calibration score. The prediction set is

$$
C(x) = \{ y : 1 - p(y\mid x) \le q_{g(y)} \}.
$$

Each candidate uses the threshold of its own group. Classes in groups with
infinite thresholds belong to every prediction set, so their count is a
lower bound on the set size. The fallback group follows the same calibration
rule as regular groups. The conformal guarantee is group-level when calibration
and test examples are exchangeable within a group after group construction.
Individual species coverage is measured empirically.

## Run and validate

Follow the [project setup](../README.md) to obtain the precomputed predictions.
Run from the project root.

```sh
uv sync --locked
uv run --locked python -m unittest discover -s tests -p test_grouped.py
uv run --locked python -m scripts.analyze_grouped_coverage
```

The script uses test labels only in Task B's `evaluate`. Overall coverage and
average set size use training-frequency weights. Rare coverage is the
equal-class mean over the least frequent third of training classes. Macro
coverage gives every class equal weight. Common coverage and the common–rare
coverage gap follow the existing global analysis.

## Inputs for Task D

- `results/grouped_coverage_all_runs.csv` has 90 runs keyed by `method`,
  `protocol`, `budget`, and `seed`. Use `frequency` and `confusion` with
  `standard_80` for the paired comparison. The separate frequency protocol is
  `full_100`.
- `results/grouped_coverage_summary.csv` has nine rows, with cross-seed means
  and sample standard deviations by method, protocol, and budget. Its metric
  columns use the same `_mean` and `_std` suffixes as the global summary.
- `fallback_n_classes` counts classes assigned to the fallback group, and
  `fallback_class_fraction` divides that count by the number of dataset
  classes. `min_set_size` counts classes with infinite thresholds. All runs
  share the same test set, so the standard deviations describe variation over
  calibration pool splits.
