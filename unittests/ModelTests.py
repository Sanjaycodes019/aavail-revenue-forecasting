#!/usr/bin/env python
"""
model tests: training, loading and predicting (test mode only)
"""

import os
import unittest

import numpy as np

from unittests import production_snapshot
from model import TEST_COUNTRIES, model_load, model_path, model_predict, model_train


class ModelTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.before = production_snapshot()
        cls.selected = model_train(test=True)
        cls.all_data, cls.all_models = model_load(test=True)

    def test_01_train(self):
        """training saves one test model per test country"""
        self.assertEqual(sorted(self.selected), sorted(TEST_COUNTRIES))
        for country in TEST_COUNTRIES:
            self.assertTrue(os.path.exists(model_path(country, test=True)))

    def test_02_load(self):
        """loading returns a fitted model and feature table for every test country"""
        for country in TEST_COUNTRIES:
            self.assertIn(country, self.all_models)
            self.assertTrue(hasattr(self.all_models[country], "predict"))
            self.assertEqual(len(self.all_data[country]["X"]), len(self.all_data[country]["dates"]))

    def test_03_predict_all(self):
        """prediction for all countries combined"""
        result = model_predict("all", "2018", "01", "05", self.all_models, self.all_data, test=True)
        y_pred = result["y_pred"]
        self.assertEqual(len(y_pred), 1)
        self.assertTrue(np.isfinite(y_pred[0]))
        self.assertGreater(y_pred[0], 0)

    def test_04_predict_country(self):
        """prediction for a single country"""
        result = model_predict("united_kingdom", 2019, 3, 15, self.all_models, self.all_data, test=True)
        self.assertTrue(np.isfinite(result["y_pred"][0]))

    def test_05_invalid_inputs(self):
        """unknown countries and dates outside the data raise ValueError"""
        with self.assertRaises(ValueError):
            model_predict("atlantis", 2018, 1, 5, self.all_models, self.all_data, test=True)
        with self.assertRaises(ValueError):
            model_predict("all", 2030, 1, 5, self.all_models, self.all_data, test=True)
        with self.assertRaises(ValueError):
            model_predict("all", 2018, 2, 30, self.all_models, self.all_data, test=True)

    def test_06_production_untouched(self):
        """test-mode training and prediction never modify production models or logs"""
        self.assertEqual(production_snapshot(), self.before)


if __name__ == "__main__":
    unittest.main()
