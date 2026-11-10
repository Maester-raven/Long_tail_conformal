from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data_loader import load_data
from src.data.split import BUDGETS, SEEDS, get_full_calibration_split, get_split
from src.evaluation.metrics import evaluate
from src.methods.grouped import (
    FALLBACK_GROUP,
    calibrate_group_thresholds,
    confusion_groups,
    frequency_groups,
    low_support_classes,
    predict_grouped,
)


OUTPUT_DIR = Path("results")
METRICS = (
    "overall_coverage",
    "macro_coverage",
    "common_coverage",
    "rare_coverage",
    "coverage_gap",
    "avg_set_size",
    "fallback_n_classes",
    "fallback_class_fraction",
    "min_set_size",
)


def evaluate_grouping(split, class_groups, method, protocol):
    thresholds = calibrate_group_thresholds(
        split["calib_probs"], split["calib_labels"], class_groups
    )
    prediction_sets = predict_grouped(split["test_probs"], class_groups, thresholds)
    counts = split["train_counts"]
    # Test labels are used only to score the completed prediction sets.
    result = evaluate(prediction_sets, split["test_labels"], counts)

    common = np.argsort(counts, kind="stable")[-(len(counts) // 3):]
    common_coverage = float(result["class_coverage"][common].mean())
    fallback_n_classes = int(np.count_nonzero(class_groups == FALLBACK_GROUP))
    return {
        "method": method,
        "protocol": protocol,
        "budget": split["budget"],
        "seed": split["seed"],
        "n_calibration": len(split["calib_labels"]),
        "overall_coverage": result["overall_coverage"],
        "macro_coverage": result["macro_coverage"],
        "common_coverage": common_coverage,
        "rare_coverage": result["rare_coverage"],
        "coverage_gap": common_coverage - result["rare_coverage"],
        "avg_set_size": result["avg_set_size"],
        "fallback_n_classes": fallback_n_classes,
        "fallback_class_fraction": fallback_n_classes / len(class_groups),
        # Classes with infinite thresholds appear in every prediction set.
        "min_set_size": int(np.count_nonzero(np.isposinf(thresholds[class_groups]))),
    }


def run_experiments(data, budget, seed):
    """Run the two paired comparisons and the full-budget frequency method."""
    # The paired methods share the split, fallback classes, and group count.
    split = get_split(data, budget=budget, seed=seed)
    fallback = low_support_classes(split["design_labels"], split["n_classes"])
    confusion = confusion_groups(
        split["design_probs"], split["design_labels"], fallback, seed
    )
    n_groups = len(np.unique(confusion[~fallback]))
    counts = data["train_counts"]
    frequency = frequency_groups(counts, fallback, n_groups)
    # Training counts define the groups, leaving the full budget for calibration.
    full = get_full_calibration_split(data, budget=budget, seed=seed)

    return [
        evaluate_grouping(split, frequency, "frequency", "standard_80"),
        evaluate_grouping(split, confusion, "confusion", "standard_80"),
        evaluate_grouping(full, frequency_groups(counts), "frequency", "full_100"),
    ]


def main():
    data = load_data()
    rows = [
        row
        for budget in BUDGETS
        for seed in SEEDS
        for row in run_experiments(data, budget, seed)
    ]
    runs = pd.DataFrame(rows)
    # Summarize the mean and sample standard deviation across seeds.
    summary = (
        runs.groupby(["method", "protocol", "budget"])[list(METRICS)]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUTPUT_DIR / "grouped_coverage_all_runs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "grouped_coverage_summary.csv", index=False)

    print(f"Completed {len(runs)} grouped experiments.")
    print(summary[[
        "method", "protocol", "budget", "overall_coverage_mean",
        "rare_coverage_mean", "avg_set_size_mean",
        "fallback_class_fraction_mean", "min_set_size_mean",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
