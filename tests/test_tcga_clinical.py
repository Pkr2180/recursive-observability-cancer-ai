from __future__ import annotations

import unittest

import pandas as pd

from src.preprocessing.tcga_clinical import _fixed_horizon
from src.validation.tcga_outcomes import _project_split


class TcgaClinicalTests(unittest.TestCase):
    def test_fixed_horizon_excludes_early_event_free_censoring(self) -> None:
        event = pd.Series([1, 1, 0, 0, pd.NA, 1])
        time = pd.Series([100, 900, 900, 100, 900, 730])
        observed = _fixed_horizon(event, time, 730)
        self.assertEqual(observed.iloc[0], 1)
        self.assertEqual(observed.iloc[1], 0)
        self.assertEqual(observed.iloc[2], 0)
        self.assertTrue(pd.isna(observed.iloc[3]))
        self.assertTrue(pd.isna(observed.iloc[4]))
        self.assertEqual(observed.iloc[5], 1)

    def test_project_split_is_deterministic_and_disjoint(self) -> None:
        projects = [f"TCGA-X{index:02d}" for index in range(33)]
        first = _project_split(projects)
        second = _project_split(list(reversed(projects)))
        self.assertEqual(first, second)
        sets = [set(first[name]) for name in ("fit", "calibration", "test")]
        self.assertFalse(sets[0] & sets[1])
        self.assertFalse(sets[0] & sets[2])
        self.assertFalse(sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(projects))


if __name__ == "__main__":
    unittest.main()
