import unittest

import numpy as np

from scripts import analyze_baselines
from scripts.analyze_coverage_multiseed import global_threshold
from src.methods.baselines import fit_baseline, predict_baseline, min_set_size
from src.data.split import get_split


class BaselineTests(unittest.TestCase):
    def test_global_matches_existing_global_threshold(self):
        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.ones(4), size=60)
        labels = rng.integers(0, 4, size=60)

        groups, thresholds = fit_baseline(probs, labels, "global")

        np.testing.assert_array_equal(groups, [0, 0, 0, 0])
        self.assertEqual(thresholds[0], global_threshold(probs, labels))

    def test_small_sample_thresholds(self):
        # At 90% target coverage, n=8 gives infinity; n=9 gives
        # the maximum score; n=19 gives the 18th smallest score.
        for n, expected in [(0, np.inf), (8, np.inf),
                            (9, 0.8), (19, 0.17)]:
            scores = np.arange(n) / 10 if n <= 9 else np.arange(n) / 100
            probs = (1.0 - scores)[:, None]
            labels = np.zeros(n, dtype=int)
            _, thresholds = fit_baseline(probs, labels, "global")
            self.assertTrue(np.isclose(thresholds[0], expected))

    def test_classwise_forced_inclusion(self):
        labels = np.array([0] * 9 + [1] * 3)
        probs = np.full((len(labels), 3), 0.05)
        probs[np.arange(len(labels)), labels] = 0.9

        groups, thresholds = fit_baseline(probs, labels, "classwise")
        self.assertTrue(np.isfinite(thresholds[0]))
        self.assertTrue(np.isinf(thresholds[1]))
        self.assertTrue(np.isinf(thresholds[2]))

        test_probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
        prediction_sets = predict_baseline(test_probs, groups, thresholds)
        self.assertEqual(min_set_size(groups, thresholds), 2)
        self.assertTrue(np.all(prediction_sets[:, 1:]))
        self.assertTrue(np.all(prediction_sets.sum(axis=1) >= 2))

    def test_run_uses_task_b_split(self):
        rng = np.random.default_rng(11)
        labels = np.tile(np.arange(3), 30)
        data = {
            "cal_probs": rng.dirichlet(np.ones(3), size=90),
            "cal_labels": labels,
            "test_probs": rng.dirichlet(np.ones(3), size=30),
            "test_labels": np.tile(np.arange(3), 10),
            "train_counts": np.array([1, 10, 100]),
            "n_classes": 3,
        }
        split = get_split(data, budget=0.3, seed=0)
        row = analyze_baselines.evaluate_baseline(
            split, "classwise", "standard_80"
        )

        self.assertEqual(row["n_calibration"], len(split["calib_labels"]))
        self.assertEqual(row["method"], "classwise")
        self.assertGreaterEqual(row["avg_set_size"], row["min_set_size"])


if __name__ == "__main__":
    unittest.main()
