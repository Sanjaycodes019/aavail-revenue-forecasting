#!/usr/bin/env python
"""
Train and predict logging.

Production logs:  logs/train-YYYY-MM.log,  logs/predict-YYYY-MM.log
Test logs:        logs/test-train.log,     logs/test-predict.log

Each log is a csv file with a header row, one line per training run / prediction.
"""

import os
import csv
import time
import uuid
from datetime import date

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

TRAIN_HEADER = ["unique_id", "timestamp", "country", "date_range", "eval_rmse", "runtime",
                "model_version", "model_version_note"]
PREDICT_HEADER = ["unique_id", "timestamp", "country", "target_date", "y_pred", "runtime", "model_version"]


def log_path(kind, test=False):
    """kind is 'train' or 'predict'; test logs never share a file with production logs"""

    if kind not in ("train", "predict"):
        raise ValueError("unknown log kind: {}".format(kind))
    if test:
        return os.path.join(LOG_DIR, "test-{}.log".format(kind))
    today = date.today()
    return os.path.join(LOG_DIR, "{}-{}-{:02d}.log".format(kind, today.year, today.month))


def _append(path, header, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.writer(fh)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)
    return path


def update_train_log(country, date_range, eval_rmse, runtime, model_version, model_version_note, test=False):
    """append one line per trained model"""

    row = [uuid.uuid4().hex, time.strftime("%Y-%m-%d %H:%M:%S"), country,
           "{}:{}".format(*date_range), round(float(eval_rmse), 2), runtime, model_version, model_version_note]
    return _append(log_path("train", test), TRAIN_HEADER, row)


def update_predict_log(country, target_date, y_pred, runtime, model_version, test=False):
    """append one line per prediction"""

    row = [uuid.uuid4().hex, time.strftime("%Y-%m-%d %H:%M:%S"), country, target_date,
           round(float(y_pred), 2), runtime, model_version]
    return _append(log_path("predict", test), PREDICT_HEADER, row)


def read_log(kind, test=False):
    """return the rows of a log as a list of dicts (empty if the log does not exist)"""

    path = log_path(kind, test)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))
