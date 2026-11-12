"""Global and classwise split conformal baselines (Task A).

Both baselines are special cases of the grouped calibration already
used by Task C. The model scores and splits come from Task B.
"""

import numpy as np

from src.methods.grouped import calibrate_group_thresholds, predict_grouped


def fit_baseline(calib_probs, calib_labels, method):
    """Return class-to-group assignments and calibrated thresholds."""
    n_classes = calib_probs.shape[1]

    if method == "global":
        groups = np.zeros(n_classes, dtype=int)
    elif method == "classwise":
        groups = np.arange(n_classes)
    else:
        raise ValueError(f"Unknown baseline: {method}")

    thresholds = calibrate_group_thresholds(
        calib_probs, calib_labels, groups
    )
    return groups, thresholds


def predict_baseline(test_probs, groups, thresholds):
    """Boolean prediction sets of shape (n_test, n_classes)."""
    return predict_grouped(test_probs, groups, thresholds)


def min_set_size(groups, thresholds):
    """Classes with infinite thresholds appear in every prediction set."""
    return int(np.count_nonzero(np.isposinf(thresholds[groups])))
