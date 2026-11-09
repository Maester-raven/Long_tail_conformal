import numpy as np


def evaluate(prediction_sets, labels, train_counts):
    """
    Evaluate prediction sets under the estimated natural distribution.

    prediction_sets: boolean array (N, C)
    labels: integer array (N,)
    train_counts: number of training examples per class (C,)
    """
    prediction_sets = np.asarray(prediction_sets)
    labels = np.asarray(labels)
    train_counts = np.asarray(train_counts)

    if prediction_sets.ndim != 2:
        raise ValueError("prediction_sets must have shape (N, C)")

    n, n_classes = prediction_sets.shape

    if labels.shape != (n,):
        raise ValueError("labels shape mismatch")

    if train_counts.shape != (n_classes,):
        raise ValueError("train_counts shape mismatch")

    if prediction_sets.dtype != np.bool_:
        raise ValueError("prediction_sets must be boolean")

    if np.any((labels < 0) | (labels >= n_classes)):
        raise ValueError("Invalid class labels")

    if np.any(train_counts < 0) or train_counts.sum() <= 0:
        raise ValueError("Invalid train_counts")

    # Whether the true label is included in each prediction set.
    covered = prediction_sets[np.arange(n), labels]
    set_sizes = prediction_sets.sum(axis=1)

    class_counts = np.bincount(labels, minlength=n_classes)

    # Evaluation requires every class to appear in the test set.
    if np.any(class_counts == 0):
        raise ValueError("Test set must contain every class")

    class_coverage = (
        np.bincount(
            labels,
            weights=covered.astype(float),
            minlength=n_classes
        ) / class_counts
    )

    class_set_size = (
        np.bincount(
            labels,
            weights=set_sizes.astype(float),
            minlength=n_classes
        ) / class_counts
    )

    weights = train_counts / train_counts.sum()

    # Rare classes: bottom third by training frequency.
    # Stable ordering ensures deterministic tie handling.
    rare_count = n_classes // 3
    rare_classes = np.argsort(
        train_counts, kind="stable"
    )[:rare_count]

    return {
        "overall_coverage": float(
            np.dot(class_coverage, weights)
        ),
        "rare_coverage": float(
            class_coverage[rare_classes].mean()
        ),
        "avg_set_size": float(
            np.dot(class_set_size, weights)
        ),
        "macro_coverage": float(class_coverage.mean()),
        "rare_classes": rare_classes,
        "class_coverage": class_coverage,
        "class_set_size": class_set_size,
    }


if __name__ == "__main__":
    from data_loader import load_data

    data = load_data()

    # Sanity check: every class included in every prediction set.
    all_classes = np.ones_like(
        data["test_probs"], dtype=bool
    )

    result = evaluate(
        all_classes,
        data["test_labels"],
        data["train_counts"]
    )

    print("Overall coverage:", result["overall_coverage"])
    print("Rare coverage:", result["rare_coverage"])
    print("Average set size:", result["avg_set_size"])
    print("Macro coverage:", result["macro_coverage"])
    print("Number of rare classes:", len(result["rare_classes"]))

    assert np.isclose(result["overall_coverage"], 1.0)
    assert np.isclose(result["rare_coverage"], 1.0)
    assert np.isclose(result["avg_set_size"], data["n_classes"])
    assert np.isclose(result["macro_coverage"], 1.0)

    print("\nAll metric tests passed.")
