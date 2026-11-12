"""Run Global and Classwise baselines using Task B's data and metrics."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.data_loader import load_data
from src.data.split import BUDGETS, SEEDS, get_split, get_full_calibration_split
from src.evaluation.metrics import evaluate
from src.methods.baselines import fit_baseline, predict_baseline, min_set_size


OUTPUT_DIR = Path("results")
METRICS = (
    "overall_coverage",
    "macro_coverage",
    "common_coverage",
    "rare_coverage",
    "coverage_gap",
    "avg_set_size",
    "min_set_size",
    "zero_calibration_classes",
    "forced_rare_classes",
)


def evaluate_baseline(split, method, protocol):
    """Calibrate on the supplied split; use test labels only for evaluation."""
    groups, thresholds = fit_baseline(
        split["calib_probs"], split["calib_labels"], method
    )
    prediction_sets = predict_baseline(
        split["test_probs"], groups, thresholds
    )
    counts = split["train_counts"]
    results = evaluate(prediction_sets, split["test_labels"], counts)

    order = np.argsort(counts, kind="stable")
    n_rare = len(counts) // 3
    rare = order[:n_rare]
    common = order[-n_rare:]

    cal_counts = np.bincount(
        split["calib_labels"], minlength=len(counts)
    )
    forced = np.isposinf(thresholds[groups])
    common_coverage = float(results["class_coverage"][common].mean())

    return {
        "method": method,
        "protocol": protocol,
        "budget": split["budget"],
        "seed": split["seed"],
        "n_calibration": len(split["calib_labels"]),
        "overall_coverage": results["overall_coverage"],
        "macro_coverage": results["macro_coverage"],
        "common_coverage": common_coverage,
        "rare_coverage": results["rare_coverage"],
        "coverage_gap": common_coverage - results["rare_coverage"],
        "avg_set_size": results["avg_set_size"],
        "min_set_size": min_set_size(groups, thresholds),
        "zero_calibration_classes": int(np.count_nonzero(cal_counts == 0)),
        "forced_rare_classes": int(np.count_nonzero(forced[rare])),
    }


def run_experiments(data, budget, seed):
    """Use the standard split for both baselines; also run full-budget Global."""
    split = get_split(data, budget=budget, seed=seed)
    full = get_full_calibration_split(data, budget=budget, seed=seed)

    return [
        evaluate_baseline(split, "global", "standard_80"),
        evaluate_baseline(split, "classwise", "standard_80"),
        evaluate_baseline(full, "global", "full_100"),
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

    summary = (
        runs.groupby(["method", "protocol", "budget"])[list(METRICS)]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUTPUT_DIR / "baselines_coverage_all_runs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "baselines_coverage_summary.csv", index=False)

    print(f"Completed {len(runs)} baseline experiments.")
    print(summary[[
        "method", "protocol", "budget", "overall_coverage_mean",
        "rare_coverage_mean", "avg_set_size_mean", "min_set_size_mean",
    ]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
