#!/usr/bin/env python
"""
logger tests: train and predict logs are created with the right contents (test logs only)
"""

import os
import unittest

from unittests import production_snapshot
from logger import PREDICT_HEADER, TRAIN_HEADER, log_path, read_log, update_predict_log, update_train_log


class LoggerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.before = production_snapshot()
        for kind in ("train", "predict"):
            if os.path.exists(log_path(kind, test=True)):
                os.remove(log_path(kind, test=True))

    def test_01_test_logs_are_separate(self):
        """test logs use their own files, never the production ones"""
        for kind in ("train", "predict"):
            self.assertTrue(os.path.basename(log_path(kind, test=True)).startswith("test-"))
            self.assertNotEqual(log_path(kind, test=True), log_path(kind, test=False))

    def test_02_train_log_created(self):
        """a train log is created with a header row"""
        path = update_train_log("all", ("2017-11-28", "2019-07-01"), 1234.5, "0.5s", 0.1, "unit test", test=True)
        self.assertTrue(os.path.exists(path))
        with open(path) as fh:
            self.assertEqual(fh.readline().strip(), ",".join(TRAIN_HEADER))

    def test_03_train_log_contents(self):
        """the latest train entry holds the logged values"""
        update_train_log("united_kingdom", ("2017-11-28", "2019-07-01"), 999.0, "0.2s", 0.1, "unit test", test=True)
        last = read_log("train", test=True)[-1]
        self.assertEqual(last["country"], "united_kingdom")
        self.assertEqual(last["date_range"], "2017-11-28:2019-07-01")
        self.assertEqual(float(last["eval_rmse"]), 999.0)
        self.assertEqual(last["model_version"], "0.1")

    def test_04_predict_log_created(self):
        """a predict log is created with a header row"""
        path = update_predict_log("all", "2018-01-05", 150000.0, "0.01s", 0.1, test=True)
        self.assertTrue(os.path.exists(path))
        with open(path) as fh:
            self.assertEqual(fh.readline().strip(), ",".join(PREDICT_HEADER))

    def test_05_predict_log_contents(self):
        """the latest predict entry holds the logged values and a unique id"""
        update_predict_log("france", "2019-02-01", 4321.0, "0.02s", 0.1, test=True)
        rows = read_log("predict", test=True)
        self.assertEqual(rows[-1]["country"], "france")
        self.assertEqual(rows[-1]["target_date"], "2019-02-01")
        self.assertEqual(float(rows[-1]["y_pred"]), 4321.0)
        self.assertEqual(len({r["unique_id"] for r in rows}), len(rows))

    def test_06_production_untouched(self):
        """logging in test mode never modifies production logs"""
        self.assertEqual(production_snapshot(), self.before)


if __name__ == "__main__":
    unittest.main()
