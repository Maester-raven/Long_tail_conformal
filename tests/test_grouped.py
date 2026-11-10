import unittest
from unittest.mock import patch

import numpy as np

from scripts import analyze_grouped_coverage
from src.data.split import get_full_calibration_split, get_split
from src.methods.grouped import (
    FALLBACK_GROUP,
    calibrate_group_thresholds,
    confusion_groups,
    frequency_groups,
    low_support_classes,
    predict_grouped,
)


class GroupedConformalTests(unittest.TestCase):
    def test_calibration_rank_at_small_sample_boundary(self):
        groups = np.array([0, 1, 2, 2, 3])
        labels = np.array([1] * 8 + [2, 3] * 4 + [2] + [4] * 19)
        scores = np.concatenate([
            np.arange(1, 9), np.arange(1, 10), np.arange(1, 20)
        ]) / 32
        probs = np.repeat((scores / 4)[:, None], len(groups), axis=1)
        probs[np.arange(len(labels)), labels] = 1.0 - scores

        thresholds = calibrate_group_thresholds(probs, labels, groups)

        np.testing.assert_array_equal(thresholds, [np.inf, np.inf, 9 / 32, 18 / 32])

    def test_prediction_uses_each_candidates_threshold_including_equality(self):
        groups = np.array([1, 0, 2])
        thresholds = np.array([0.25, 0.75, np.inf])
        probs = np.array([[0.25, 0.5, 0.25], [0.125, 0.75, 0.125]])

        prediction_sets = predict_grouped(probs, groups, thresholds)

        np.testing.assert_array_equal(prediction_sets, [[True, False, True], [False, True, True]])

    def test_frequency_splits_stably_into_nearly_equal_class_counts(self):
        train_counts = np.array([3, 3, 8, 1, 3, 2, 4])
        design_labels = np.repeat(np.arange(7), [3, 2, 3, 0, 3, 3, 3])
        fallback = low_support_classes(design_labels, len(train_counts))

        groups = frequency_groups(train_counts, fallback, n_groups=2)

        np.testing.assert_array_equal(groups, [0, FALLBACK_GROUP, 1, FALLBACK_GROUP, 0, 0, 1])

    def test_confusion_groups_normalize_off_diagonal_mass(self):
        labels = np.repeat(np.arange(5), [3, 3, 3, 3, 2])
        probs = np.array([
            [0.875, 0, 0, 0, 0, 0.125],
            [0, 0.25, 0, 0, 0, 0.75],
            [0, 0, 0.5, 0, 0.5, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ])[labels]
        fallback = low_support_classes(labels, probs.shape[1])

        groups = confusion_groups(probs, labels, fallback, seed=0)

        self.assertEqual(groups[0], groups[1])
        self.assertEqual(len(set(groups[:4])), 3)
        np.testing.assert_array_equal(groups[4:], [FALLBACK_GROUP, FALLBACK_GROUP])

    def test_all_classes_can_share_fallback_when_design_is_empty(self):
        probs = np.empty((0, 3))
        labels = np.empty(0, dtype=int)
        fallback = low_support_classes(labels, 3)

        groups = confusion_groups(probs, labels, fallback, seed=0)
        frequency = frequency_groups(np.array([2, 1, 3]), fallback, n_groups=0)
        thresholds = calibrate_group_thresholds(probs, labels, groups)

        np.testing.assert_array_equal(groups, [FALLBACK_GROUP] * 3)
        np.testing.assert_array_equal(groups, frequency)
        self.assertEqual(thresholds[FALLBACK_GROUP], np.inf)

    def test_experiments_use_design_and_calibration_for_their_own_roles(self):
        labels = np.tile(np.arange(6), 100)
        data = {
            "cal_probs": np.eye(6)[labels],
            "cal_labels": labels,
            "test_probs": np.eye(6),
            "test_labels": np.arange(6),
            "train_counts": np.arange(1, 7),
            "n_classes": 6,
        }
        split = get_split(data, budget=1.0, seed=0)
        full = get_full_calibration_split(data, budget=1.0, seed=0)

        with (
            patch.object(analyze_grouped_coverage, "confusion_groups", wraps=confusion_groups) as fit,
            patch.object(
                analyze_grouped_coverage, "calibrate_group_thresholds",
                wraps=calibrate_group_thresholds,
            ) as calibrate,
        ):
            rows = analyze_grouped_coverage.run_experiments(data, budget=1.0, seed=0)

        fit.assert_called_once()
        design_probs, design_labels, _, _ = fit.call_args.args
        np.testing.assert_array_equal(design_probs, split["design_probs"])
        np.testing.assert_array_equal(design_labels, split["design_labels"])

        expected = {len(split["calib_labels"]): split, len(full["calib_labels"]): full}
        seen = []
        for call in calibrate.call_args_list:
            calib_probs, calib_labels, groups = call.args
            selected = expected[len(calib_labels)]
            np.testing.assert_array_equal(calib_probs, selected["calib_probs"])
            np.testing.assert_array_equal(calib_labels, selected["calib_labels"])
            # One-hot design rows all yield the same zero profile.
            seen.append((len(calib_labels), len(np.unique(groups))))
        self.assertCountEqual(seen, [(480, 1), (480, 1), (600, 5)])
        self.assertCountEqual(
            [(row["method"], row["protocol"]) for row in rows],
            [("frequency", "standard_80"), ("confusion", "standard_80"),
             ("frequency", "full_100")],
        )


if __name__ == "__main__":
    unittest.main()
