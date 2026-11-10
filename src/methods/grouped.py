import numpy as np
from sklearn.cluster import KMeans


ALPHA = 0.1
N_REGULAR_GROUPS = 5
FALLBACK_GROUP = N_REGULAR_GROUPS
MIN_DESIGN_SAMPLES = 3


def low_support_classes(design_labels, n_classes):
    """Return the mask of classes assigned to the shared fallback group."""
    # Count species absent from the design split as having zero examples.
    return np.bincount(design_labels, minlength=n_classes) < MIN_DESIGN_SAMPLES


def frequency_groups(train_counts, fallback=None, n_groups=N_REGULAR_GROUPS):
    """Split classes into equal-count groups ordered by training frequency."""
    if fallback is None:
        fallback = np.zeros(len(train_counts), dtype=bool)

    groups = np.full(len(train_counts), FALLBACK_GROUP, dtype=int)
    # Break frequency ties by class ID, then cut groups of nearly equal size.
    ordered = np.argsort(train_counts, kind="stable")
    eligible = ordered[~fallback[ordered]]
    if eligible.size:
        for group, classes in enumerate(np.array_split(eligible, n_groups)):
            groups[classes] = group
    return groups


def confusion_groups(design_probs, design_labels, fallback, seed):
    """Cluster normalized off-diagonal design probabilities by true class."""
    groups = np.full(design_probs.shape[1], FALLBACK_GROUP, dtype=int)
    eligible = np.flatnonzero(~fallback)
    if not eligible.size:
        return groups

    profiles = np.zeros((eligible.size, groups.size))
    for row, label in zip(profiles, eligible):
        row[:] = design_probs[design_labels == label].mean(axis=0, dtype=float)
        # Keep the probability assigned to other classes and normalize that mass.
        row[label] = 0.0
        mass = row.sum()
        if mass > 0:
            row /= mass

    # Cap the cluster count when the design split yields too few distinct profiles.
    n_groups = min(N_REGULAR_GROUPS, len(np.unique(profiles, axis=0)))
    if n_groups == 1:
        groups[eligible] = 0
    else:
        groups[eligible] = KMeans(
            n_clusters=n_groups, random_state=seed, n_init=10
        ).fit_predict(profiles)
    return groups


def calibrate_group_thresholds(calib_probs, calib_labels, class_groups):
    """Calibrate one split-conformal threshold per group on true-label scores."""
    # The true label determines both the calibration score and its group.
    calib_groups = class_groups[calib_labels]
    scores = 1.0 - calib_probs[np.arange(len(calib_labels)), calib_labels]
    thresholds = np.full(class_groups.max() + 1, np.inf)

    for group in np.unique(class_groups):
        group_scores = scores[calib_groups == group]
        n = len(group_scores)
        # The rank starts at one. If it exceeds n, the threshold stays infinite.
        k = int(np.ceil((n + 1) * (1.0 - ALPHA)))
        if k <= n:
            thresholds[group] = np.partition(group_scores, k - 1)[k - 1]
    return thresholds


def predict_grouped(test_probs, class_groups, thresholds):
    """Include each candidate whose score is within its own group's threshold."""
    # Broadcast one threshold per candidate class across all test examples.
    return (1.0 - test_probs) <= thresholds[class_groups]
