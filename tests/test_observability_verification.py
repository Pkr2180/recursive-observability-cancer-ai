from __future__ import annotations

import unittest

import numpy as np

from src.observability.verification import (
    _benjamini_hochberg,
    _bootstrap_ci,
    _calibration,
    _ece,
    _normalized_reconstructability,
)


class ObservabilityVerificationTests(unittest.TestCase):
    def test_cluster_bootstrap_is_reproducible(self) -> None:
        values = np.arange(1, 11, dtype=float)
        first = _bootstrap_ci(
            [values],
            lambda sample: float(sample.mean()),
            replicates=300,
            rng=np.random.default_rng(7),
        )
        second = _bootstrap_ci(
            [values],
            lambda sample: float(sample.mean()),
            replicates=300,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(first, second)
        self.assertLess(first[0], values.mean())
        self.assertGreater(first[1], values.mean())

    def test_exact_reconstruction_scores_one(self) -> None:
        states = np.asarray([[1.0, -1.0], [0.5, 2.0]])
        self.assertEqual(_normalized_reconstructability(states, states), 1.0)

    def test_error_calibration_recovers_identity(self) -> None:
        error = np.linspace(0.1, 1.0, 20)
        profile = _calibration(error, error)
        self.assertAlmostEqual(profile["calibration_intercept"], 0.0, places=10)
        self.assertAlmostEqual(profile["calibration_slope"], 1.0, places=10)
        self.assertAlmostEqual(profile["pearson"], 1.0, places=10)

    def test_fdr_adjustment_is_monotone_by_rank(self) -> None:
        adjusted = _benjamini_hochberg([0.001, 0.02, 0.5])
        self.assertTrue(np.all(np.diff(adjusted) >= 0))
        self.assertTrue(np.all((adjusted >= 0) & (adjusted <= 1)))

    def test_ece_uses_fixed_probability_bins(self) -> None:
        y_true = np.array([0, 0, 1, 1], dtype=float)
        probability = np.array([0.1, 0.3, 0.7, 0.9], dtype=float)
        self.assertAlmostEqual(_ece(y_true, probability, bins=5), 0.2)


if __name__ == "__main__":
    unittest.main()
