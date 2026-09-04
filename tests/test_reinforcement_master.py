from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.validation.reinforcement_master import (
    _bootstrap_comparisons,
    _policy_forecast,
    _reward,
    _softmax_weights,
)


class ReinforcementMasterTests(unittest.TestCase):
    def test_softmax_policy_is_normalized_and_reward_ordered(self) -> None:
        q_values = np.asarray([[0.2, 0.8, 0.4], [0.9, 0.1, 0.2]])
        weights = _softmax_weights(q_values, temperature=0.1)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        self.assertEqual(int(np.argmax(weights[0])), 1)
        self.assertEqual(int(np.argmax(weights[1])), 0)

    def test_policy_forecast_applies_per_transition_weights(self) -> None:
        forecasts = np.asarray(
            [
                [[1.0, 0.0], [2.0, 0.0]],
                [[0.0, 1.0], [0.0, 2.0]],
                [[1.0, 1.0], [2.0, 2.0]],
            ]
        )
        weights = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
        expected = np.asarray([[1.0, 0.0], [1.0, 2.0]])
        np.testing.assert_allclose(_policy_forecast(forecasts, weights), expected)

    def test_reward_decreases_with_error(self) -> None:
        rewards = _reward(np.asarray([0.0, 0.5, 2.0]))
        self.assertTrue(np.all(np.diff(rewards) < 0))
        self.assertEqual(float(rewards[0]), 1.0)

    def test_cluster_bootstrap_is_reproducible(self) -> None:
        primary = np.asarray([0.8, 0.7, 0.9, 0.6, 0.75, 0.85])
        baseline = np.asarray([0.7, 0.65, 0.75, 0.55, 0.7, 0.8])
        groups = np.asarray(["a", "a", "b", "b", "c", "c"])
        first = _bootstrap_comparisons(primary, {"baseline": baseline}, groups, 100)
        second = _bootstrap_comparisons(primary, {"baseline": baseline}, groups, 100)
        pd.testing.assert_frame_equal(first, second)
        self.assertGreater(float(first.loc[0, "mean_reward_difference"]), 0)


if __name__ == "__main__":
    unittest.main()
