from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.data_loader import load_data
from src.data.split import (
    BUDGETS,
    SEEDS,
    get_split,
    get_full_calibration_split,
)
from src.evaluation.metrics import evaluate


ALPHA = 0.1
OUTPUT_DIR = Path("results")
FIGURE_DIR = OUTPUT_DIR / "figures"


def global_threshold(probs, labels, alpha=ALPHA):
    """Finite-sample split-conformal threshold."""
    n = len(labels)

    if n == 0:
        return np.inf

    scores = 1.0 - probs[np.arange(n), labels]
    k = int(np.ceil((n + 1) * (1 - alpha)))

    if k > n:
        return np.inf

    return float(np.partition(scores, k - 1)[k - 1])


def run_experiment(data, budget, seed, protocol):
    """Run one experiment without using test labels for calibration."""

    if protocol == "standard_80":
        split = get_split(data, budget=budget, seed=seed)
    elif protocol == "full_100":
        split = get_full_calibration_split(
            data, budget=budget, seed=seed
        )
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    calib_probs = split["calib_probs"]
    calib_labels = split["calib_labels"]

    threshold = global_threshold(calib_probs, calib_labels)

    # The same test set is used only for final evaluation.
    prediction_sets = (1.0 - data["test_probs"]) <= threshold

    result = evaluate(
        prediction_sets,
        data["test_labels"],
        data["train_counts"],
    )

    # Fixed common and rare categories, based on TRAIN frequency.
    counts = data["train_counts"]
    n_classes = data["n_classes"]
    order = np.argsort(counts, kind="stable")
    n_group = n_classes // 3

    rare = order[:n_group]
    common = order[-n_group:]

    common_cov = float(result["class_coverage"][common].mean())
    rare_cov = float(result["class_coverage"][rare].mean())

    return {
        "protocol": protocol,
        "budget": budget,
        "seed": seed,
        "n_calibration": len(calib_labels),
        "threshold": threshold,
        "overall_coverage": result["overall_coverage"],
        "macro_coverage": result["macro_coverage"],
        "common_coverage": common_cov,
        "rare_coverage": rare_cov,
        "coverage_gap": common_cov - rare_cov,
        "avg_set_size": result["avg_set_size"],
    }


def plot_results(summary, metric, ylabel, filename):
    """Plot mean and sample standard deviation across seeds."""

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for protocol, label in [
        ("standard_80", "Standard: 80% calibration"),
        ("full_100", "Full-budget calibration"),
    ]:
        subset = summary[
            summary["protocol"] == protocol
        ].sort_values("budget")

        ax.errorbar(
            subset["budget"].to_numpy() * 100,
            subset[f"{metric}_mean"].to_numpy(),
            yerr=subset[f"{metric}_std"].to_numpy(),
            marker="o",
            capsize=4,
            linewidth=1.8,
            label=label,
        )

    ax.set_xticks([10, 30, 100])
    ax.set_xlabel("Calibration budget (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Global Conformal: {ylabel}")
    ax.grid(alpha=0.25)
    ax.legend()

    if "coverage" in metric:
        ax.axhline(0.9, linestyle="--", color="gray",
                   alpha=0.5, label="_nolegend_")
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=200)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    rows = []

    for protocol in ("standard_80", "full_100"):
        for budget in BUDGETS:
            for seed in SEEDS:
                row = run_experiment(
                    data, budget, seed, protocol
                )
                rows.append(row)

                print(
                    f"{protocol:12s} "
                    f"budget={budget:.0%} seed={seed} "
                    f"overall={row['overall_coverage']:.2%} "
                    f"rare={row['rare_coverage']:.2%}"
                )

    # Save every individual experiment.
    df = pd.DataFrame(rows)
    assert len(df) == 2 * len(BUDGETS) * len(SEEDS)
    assert not df.isna().any().any()

    df.to_csv(
        OUTPUT_DIR / "global_coverage_all_runs.csv",
        index=False,
    )

    # Aggregate by protocol and budget.
    metrics = [
        "threshold",
        "overall_coverage",
        "macro_coverage",
        "common_coverage",
        "rare_coverage",
        "coverage_gap",
        "avg_set_size",
    ]

    summary = (
        df.groupby(["protocol", "budget"], sort=True)[metrics]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]
    summary = summary.reset_index()

    summary.to_csv(
        OUTPUT_DIR / "global_coverage_summary.csv",
        index=False,
    )

    plot_results(
        summary,
        "rare_coverage",
        "Rare-class coverage",
        "rare_coverage_multiseed.png",
    )

    plot_results(
        summary,
        "common_coverage",
        "Common-class coverage",
        "common_coverage_multiseed.png",
    )

    plot_results(
        summary,
        "coverage_gap",
        "Common - rare coverage gap",
        "coverage_gap_multiseed.png",
    )

    plot_results(
        summary,
        "avg_set_size",
        "Average prediction set size",
        "set_size_multiseed.png",
    )

    print("\n=== MULTI-SEED SUMMARY ===")

    display_cols = [
        "protocol",
        "budget",
        "overall_coverage_mean",
        "common_coverage_mean",
        "rare_coverage_mean",
        "rare_coverage_std",
        "coverage_gap_mean",
        "avg_set_size_mean",
    ]

    print(
        summary[display_cols].to_string(
            index=False, float_format=lambda x: f"{x:.4f}"
        )
    )

    # Ensure the existing single-seed result is reproduced.
    previous = df[
        (df["protocol"] == "standard_80")
        & (df["budget"] == 1.0)
        & (df["seed"] == 0)
    ].iloc[0]

    assert np.isclose(
        previous["overall_coverage"], 0.901039,
        atol=1e-5
    )
    assert np.isclose(
        previous["rare_coverage"], 0.404909,
        atol=1e-5
    )

    print("\nPrevious seed-0 result reproduced.")
    print("All 60 experiments completed.")


if __name__ == "__main__":
    main()
