#!/usr/bin/env python
"""
API tests: /train, /predict (single country and all), /logs - test mode only
"""

import unittest

from unittests import production_snapshot
from app import app


class ApiTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.before = production_snapshot()
        cls.client = app.test_client()

    def test_01_index(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("endpoints", resp.get_json())

    def test_02_train(self):
        """/train in test mode trains the test models"""
        resp = self.client.post("/train", json={"mode": "test"})
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["mode"], "test")
        self.assertIn("all", body["models"])

    def test_03_predict_country(self):
        """/predict returns a forecast for a specific country"""
        resp = self.client.post("/predict", json={"country": "united_kingdom", "year": "2018", "month": "01",
                                                  "day": "05", "mode": "test"})
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["country"], "united_kingdom")
        self.assertEqual(len(body["y_pred"]), 1)
        self.assertGreater(body["y_pred"][0], 0)

    def test_04_predict_all(self):
        """/predict returns a forecast for all countries combined (nested query format)"""
        resp = self.client.post("/predict", json={"query": {"country": "all", "year": 2019, "month": 6, "day": 1},
                                                  "mode": "test"})
        self.assertEqual(resp.status_code, 200)
        self.assertGreater(resp.get_json()["y_pred"][0], 0)

    def test_05_predict_bad_requests(self):
        """bad requests get a 400 with an error message"""
        cases = [
            {"country": "all", "year": "2018", "month": "01", "mode": "test"},                 # missing day
            {"country": "atlantis", "year": "2018", "month": "01", "day": "05", "mode": "test"},
            {"country": "all", "year": "2035", "month": "01", "day": "05", "mode": "test"},
        ]
        for body in cases:
            resp = self.client.post("/predict", json=body)
            self.assertEqual(resp.status_code, 400, body)
            self.assertIn("error", resp.get_json())
        self.assertEqual(self.client.post("/predict", data="not json").status_code, 400)

    def test_06_logs(self):
        """/logs serves existing log files and rejects missing or unsafe names"""
        self.assertEqual(self.client.get("/logs/test-predict.log").status_code, 200)
        self.assertEqual(self.client.get("/logs/test-train.log").status_code, 200)
        self.assertEqual(self.client.get("/logs/does-not-exist.log").status_code, 404)
        self.assertEqual(self.client.get("/logs/app.py").status_code, 400)
        ## path traversal never serves a file (rejected by routing or by the name check)
        self.assertIn(self.client.get("/logs/..%2Fapp.py").status_code, (400, 404))

    def test_07_production_untouched(self):
        """test-mode API calls never modify production models or logs"""
        self.assertEqual(production_snapshot(), self.before)


if __name__ == "__main__":
    unittest.main()
