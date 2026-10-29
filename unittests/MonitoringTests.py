#!/usr/bin/env python
"""
monitoring tests: the drift checks flag shifted data and accept similar data
"""

import unittest

import numpy as np

from unittests import production_snapshot  # noqa: F401  (adds the project root to sys.path)
from monitoring import check_drift


class MonitoringTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.X_train = rng.normal(0, 1, size=(400, 4))
        cls.y_train = rng.normal(100, 10, size=400)
        cls.rng = rng

    def test_01_no_drift_on_similar_data(self):
        X = self.rng.normal(0, 1, size=(80, 4))
        y = self.rng.normal(100, 10, size=80)
        report = check_drift(self.X_train, self.y_train, X, y)
        self.assertFalse(report["X_drift"])
        self.assertFalse(report["y_drift"])
        self.assertLess(report["outlier_rate"], 0.2)

    def test_02_drift_on_shifted_data(self):
        X = self.rng.normal(3, 1, size=(80, 4))
        y = self.rng.normal(160, 10, size=80)
        report = check_drift(self.X_train, self.y_train, X, y)
        self.assertTrue(report["X_drift"])
        self.assertTrue(report["y_drift"])
        self.assertGreater(report["outlier_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
