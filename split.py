import numpy as np
from data_loader import load_data

BUDGETS = (0.1, 0.3, 1.0)
SEEDS = tuple(range(10))


def get_split(data, budget=0.3, seed=42, design_ratio=0.2):
    """
    Split the original calibration pool into design and calibration sets.

    No test samples are used for splitting or calibration.
    The same budget and seed always produce identical indices.
    """

    if budget not in BUDGETS:
        raise ValueError(f"Budget must be one of {BUDGETS}")

    if not 0 < design_ratio < 1:
        raise ValueError("design_ratio must be between 0 and 1")

    n = len(data["cal_labels"])
    rng = np.random.default_rng(seed)

    # One reproducible permutation of the original calibration pool.
    indices = rng.permutation(n)

    budget_size = int(np.floor(n * budget + 0.5))
    selected = indices[:budget_size]

    design_size = int(np.floor(budget_size * design_ratio + 0.5))

    design_idx = selected[:design_size]
    calib_idx = selected[design_size:]

    assert len(np.intersect1d(design_idx, calib_idx)) == 0

    return {
        "budget": budget,
        "seed": seed,
        "design_indices": design_idx,
        "calib_indices": calib_idx,
        "design_probs": data["cal_probs"][design_idx],
        "design_labels": data["cal_labels"][design_idx],
        "calib_probs": data["cal_probs"][calib_idx],
        "calib_labels": data["cal_labels"][calib_idx],
        "test_probs": data["test_probs"],
        "test_labels": data["test_labels"],
        "train_counts": data["train_counts"],
        "n_classes": data["n_classes"],
    }


if __name__ == "__main__":
    data = load_data()

    for budget in BUDGETS:
        split = get_split(data, budget=budget, seed=0)

        print(f"\nBudget: {budget:.0%}")
        print("Design samples:", len(split["design_labels"]))
        print("Calibration samples:", len(split["calib_labels"]))
        print("Test samples:", len(split["test_labels"]))

        # Reproducibility check
        repeat = get_split(data, budget=budget, seed=0)
        assert np.array_equal(
            split["design_indices"],
            repeat["design_indices"]
        )
        assert np.array_equal(
            split["calib_indices"],
            repeat["calib_indices"]
        )

        # No overlap between design and calibration sets
        assert not np.intersect1d(
            split["design_indices"],
            split["calib_indices"]
        ).size

    # Different seeds should produce different samples
    a = get_split(data, budget=0.3, seed=0)
    b = get_split(data, budget=0.3, seed=1)
    assert not np.array_equal(a["design_indices"], b["design_indices"])

    print("\nAll split tests passed.")


def get_full_calibration_split(data, budget=0.3, seed=42):
    """
    Use the entire selected budget for calibration.

    Intended for methods that do not require a separate
    design set, such as global calibration.
    """
    split = get_split(data, budget=budget, seed=seed)

    full_indices = np.concatenate([
        split["design_indices"],
        split["calib_indices"],
    ])

    return {
        "budget": budget,
        "seed": seed,
        "calib_indices": full_indices,
        "calib_probs": data["cal_probs"][full_indices],
        "calib_labels": data["cal_labels"][full_indices],
        "test_probs": data["test_probs"],
        "test_labels": data["test_labels"],
        "train_counts": data["train_counts"],
        "n_classes": data["n_classes"],
    }
