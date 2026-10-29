#!/usr/bin/env python
"""
Performance and drift monitoring.

1. Input drift: compares the features of production days (data/cs-production) with the
   training features.
     - outlier rate: share of production days an IsolationForest (fit on training
       features) flags as outliers
     - Wasserstein distance between training and production features / targets, compared
       with a threshold from bootstrap samples of the training data itself
2. Performance: predicts every production day whose 30-day target is fully observed,
   compares with actual revenue (RMSE / MAE per month, model vs baseline).

Usage:
    python monitoring.py [country]      # default: all

Writes reports/monitoring-<country>.json and images/monitoring_<country>.png
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from ingestion import DATA_DIR, PRODUCTION_DIR, engineer_features, fetch_ts
from model import IMAGE_DIR, REPORT_DIR, baseline_predict, model_load, rmse

N_BOOTSTRAP = 200


def _scaled_distance(A, B):
    """mean Wasserstein distance over the columns of two (already scaled) feature matrices"""
    return float(np.mean([wasserstein_distance(A[:, j], B[:, j]) for j in range(A.shape[1])]))


def get_monitoring_tools(X, y, random_state=42):
    """
    fit the drift detectors on training data.
    returns a dict with the scaler, the outlier detector and bootstrap thresholds
    (95th percentile of distances between training resamples and the training data).
    """

    rng = np.random.default_rng(random_state)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    detector = IsolationForest(contamination=0.05, random_state=random_state).fit(Xs)

    n = min(100, len(X))
    x_dist, y_dist = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(len(X), size=n, replace=True)
        x_dist.append(_scaled_distance(Xs, Xs[idx]))
        y_dist.append(wasserstein_distance(y, y[idx]))

    return {"scaler": scaler, "detector": detector,
            "X_threshold": float(np.percentile(x_dist, 95)),
            "y_threshold": float(np.percentile(y_dist, 95))}


def check_drift(X_train, y_train, X_prod, y_prod):
    """compare production features/targets with training ones"""

    tools = get_monitoring_tools(X_train, y_train)
    Xs_train = tools["scaler"].transform(X_train)
    Xs_prod = tools["scaler"].transform(X_prod)

    outliers = tools["detector"].predict(Xs_prod) == -1
    x_dist = _scaled_distance(Xs_train, Xs_prod)
    y_dist = float(wasserstein_distance(y_train, y_prod))
    return {
        "n_production_days": int(len(X_prod)),
        "outlier_rate": float(outliers.mean()),
        "X_wasserstein": x_dist, "X_threshold": tools["X_threshold"], "X_drift": bool(x_dist > tools["X_threshold"]),
        "y_wasserstein": y_dist, "y_threshold": tools["y_threshold"], "y_drift": bool(y_dist > tools["y_threshold"]),
    }


def production_features(country, train_dir=DATA_DIR, prod_dir=PRODUCTION_DIR):
    """
    features and targets for production days. Production days are appended to the
    training series so the look-back features have their full history.
    """

    train_ts = fetch_ts(train_dir)[country]
    prod_all = fetch_ts(prod_dir)
    if country not in prod_all:
        raise ValueError("country '{}' is not among the top production countries".format(country))

    combined = pd.concat([train_ts, prod_all[country]], ignore_index=True)
    combined = combined.set_index("date").asfreq("D", fill_value=0).reset_index()
    combined["year_month"] = combined["date"].dt.strftime("%Y-%m")

    X, y, dates = engineer_features(combined, training=True)   # only days with a complete 30-day target
    is_prod = pd.to_datetime(dates) > train_ts["date"].max()
    return X[is_prod].reset_index(drop=True), y[is_prod], pd.to_datetime(dates[is_prod])


def monitor(country="all", save=True):
    all_data, all_models = model_load()
    if country not in all_models:
        raise ValueError("no production model for '{}'".format(country))

    X_train, y_train, _ = engineer_features(fetch_ts(DATA_DIR)[country])
    X_prod, y_prod, d_prod = production_features(country)

    report = {"country": country}
    report["drift"] = check_drift(X_train.values, y_train, X_prod.values, y_prod)

    y_model = all_models[country].predict(X_prod)
    y_base = baseline_predict(X_prod)
    report["performance"] = {
        "model_rmse": rmse(y_prod, y_model), "model_mae": float(mean_absolute_error(y_prod, y_model)),
        "baseline_rmse": rmse(y_prod, y_base), "baseline_mae": float(mean_absolute_error(y_prod, y_base)),
        "by_month": [
            {"month": m, "model_rmse": rmse(y_prod[mask], y_model[mask]), "baseline_rmse": rmse(y_prod[mask], y_base[mask])}
            for m in sorted(set(d_prod.strftime("%Y-%m")))
            for mask in [np.asarray(d_prod.strftime("%Y-%m") == m)]
        ],
    }

    if save:
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(os.path.join(REPORT_DIR, "monitoring-{}.json".format(country)), "w") as fh:
            json.dump(report, fh, indent=2)
        _plot(country, d_prod, y_prod, y_model, y_base, report)
    return report


def _plot(country, dates, y_true, y_model, y_base, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(IMAGE_DIR, exist_ok=True)
    perf = report["performance"]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(dates, y_true, color="black", lw=2.2, label="actual")
    ax.plot(dates, y_model, color="#4C72B0", lw=1.8, label="model (RMSE {:,.0f})".format(perf["model_rmse"]))
    ax.plot(dates, y_base, color="#999999", lw=1.6, ls="--", label="baseline (RMSE {:,.0f})".format(perf["baseline_rmse"]))
    d = report["drift"]
    ax.set_title("Production monitoring ({}): outlier rate {:.0%}, feature drift {}, target drift {}".format(
        country, d["outlier_rate"], "YES" if d["X_drift"] else "no", "YES" if d["y_drift"] else "no"))
    ax.set_ylabel("Revenue over next 30 days")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, "monitoring_{}.png".format(country)), dpi=120)
    plt.close(fig)


if __name__ == "__main__":

    country = sys.argv[1] if len(sys.argv) > 1 else "all"
    rep = monitor(country)
    d, p = rep["drift"], rep["performance"]
    print("country:               {}".format(country))
    print("production days:       {}".format(d["n_production_days"]))
    print("outlier rate:          {:.1%}".format(d["outlier_rate"]))
    print("feature drift:         {:.3f} (threshold {:.3f}) -> {}".format(d["X_wasserstein"], d["X_threshold"], d["X_drift"]))
    print("target drift:          {:,.0f} (threshold {:,.0f}) -> {}".format(d["y_wasserstein"], d["y_threshold"], d["y_drift"]))
    print("model RMSE / MAE:      {:,.0f} / {:,.0f}".format(p["model_rmse"], p["model_mae"]))
    print("baseline RMSE / MAE:   {:,.0f} / {:,.0f}".format(p["baseline_rmse"], p["baseline_mae"]))
    for row in p["by_month"]:
        print("  {}  model {:>10,.0f}   baseline {:>10,.0f}".format(row["month"], row["model_rmse"], row["baseline_rmse"]))
