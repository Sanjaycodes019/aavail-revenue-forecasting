#!/usr/bin/env python
"""
Flask API for the AAVAIL revenue forecasting models.

Endpoints:
    GET  /                  service info
    POST /train             retrain models            body: {"mode": "test"} trains test models only
    POST /predict           30-day revenue forecast   body: {"country": "all", "year": "2018", "month": "01", "day": "05"}
                            optional "mode": "test" uses the test models and test log
    GET  /logs/<filename>   download a log file

Usage:
    python app.py           # serves on 0.0.0.0:8080
"""

import os
import re

from flask import Flask, jsonify, request, send_from_directory

from logger import LOG_DIR
from model import MODEL_VERSION, model_load, model_predict, model_train

app = Flask(__name__)

## loaded models are cached per mode ('production' / 'test') and refreshed after training
_cache = {}


def get_models(test=False):
    key = "test" if test else "production"
    if key not in _cache:
        _cache[key] = model_load(test=test)
    return _cache[key]


def _is_test(payload):
    return str(payload.get("mode", "")).lower() == "test"


@app.route("/", methods=["GET"])
def index():
    return jsonify({"service": "AAVAIL revenue forecasting", "model_version": MODEL_VERSION,
                    "endpoints": ["POST /train", "POST /predict", "GET /logs/<filename>"]})


@app.route("/train", methods=["POST"])
def train():
    payload = request.get_json(silent=True) or {}
    test = _is_test(payload)
    selected = model_train(test=test)
    _cache.pop("test" if test else "production", None)
    return jsonify({"status": "trained", "mode": "test" if test else "production", "models": selected})


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400

    ## accept both a flat body and {"query": {...}}
    query = payload.get("query", payload)
    missing = [k for k in ("country", "year", "month", "day") if k not in query]
    if missing:
        return jsonify({"error": "missing field(s): {}".format(", ".join(missing))}), 400

    test = _is_test(payload)
    try:
        all_data, all_models = get_models(test)
    except FileNotFoundError as err:
        return jsonify({"error": str(err)}), 503

    try:
        result = model_predict(query["country"], query["year"], query["month"], query["day"],
                               all_models=all_models, all_data=all_data, test=test)
    except ValueError as err:
        return jsonify({"error": str(err)}), 400

    return jsonify({
        "country": str(query["country"]).lower(),
        "target_date": "{}-{}-{}".format(query["year"], str(query["month"]).zfill(2), str(query["day"]).zfill(2)),
        "y_pred": [float(v) for v in result["y_pred"]],
        "horizon_days": 30,
        "model_version": MODEL_VERSION,
    })


@app.route("/logs/<filename>", methods=["GET"])
def logs(filename):
    if not re.match(r"^[\w\-]+\.log$", filename):
        return jsonify({"error": "invalid log file name"}), 400
    if not os.path.exists(os.path.join(LOG_DIR, filename)):
        return jsonify({"error": "log file not found: {}".format(filename)}), 404
    return send_from_directory(LOG_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
