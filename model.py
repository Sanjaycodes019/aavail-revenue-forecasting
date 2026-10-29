#!/usr/bin/env python
"""
Model training, comparison, loading and prediction.

- compare_models():  evaluate a baseline and several regressors on a time-ordered hold-out set
- model_train():     train one model per country (top 10 + 'all') and save it
- model_load():      load the saved models and the feature tables used for prediction
- model_predict():   forecast revenue for the 30 days starting at a given date

Usage:
    python model.py            # train production models
    python model.py compare    # compare models and write the comparison table/chart
"""

import os
import re
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ingestion import DATA_DIR, engineer_features, fetch_ts
from logger import update_predict_log, update_train_log

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

MODEL_VERSION = 0.1
MODEL_VERSION_NOTE = "best of linear regression / random forest / gradient boosting per country, target = revenue over next 30 days"
TEST_COUNTRIES = ["all", "united_kingdom"]
HOLDOUT_FRACTION = 0.25


def get_candidates(test=False):
    """the models that are compared; test=True uses small grids so unit tests run fast"""

    rf_grid = {"rf__n_estimators": [20]} if test else {"rf__n_estimators": [100, 200], "rf__max_depth": [None, 10]}
    gb_grid = {"gb__n_estimators": [20]} if test else {"gb__n_estimators": [100, 200], "gb__learning_rate": [0.05, 0.1]}
    cv = TimeSeriesSplit(n_splits=3)
    return {
        "linear_regression": Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]),
        "random_forest": GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(random_state=42))]),
            rf_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1),
        "gradient_boosting": GridSearchCV(
            Pipeline([("scaler", StandardScaler()), ("gb", GradientBoostingRegressor(random_state=42))]),
            gb_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1),
    }


def baseline_predict(X):
    """baseline: the next 30 days look like the previous 28 days, scaled to 30 days"""
    return X["previous_28"].values * 30 / 28


def time_split(X, y, dates, holdout=HOLDOUT_FRACTION):
    """oldest (1 - holdout) of the days for training, most recent days for testing"""
    cut = int(len(y) * (1 - holdout))
    return X.iloc[:cut], X.iloc[cut:], y[:cut], y[cut:], dates[:cut], dates[cut:]


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def model_path(country, test=False):
    prefix = "test" if test else "sl"
    version = re.sub(r"\.", "_", str(MODEL_VERSION))
    return os.path.join(MODEL_DIR, "{}-{}-{}.joblib".format(prefix, country, version))


######################################################################
# comparison
######################################################################

def compare_models(data_dir=DATA_DIR, countries=None, save=True):
    """
    evaluate the baseline and every candidate model on each country's most recent 25% of days.
    returns a DataFrame with one row per (country, model).
    """

    ts_data = fetch_ts(data_dir)
    countries = countries or list(ts_data)
    rows, holdout_preds = [], {}

    for country in countries:
        X, y, dates = engineer_features(ts_data[country])
        X_tr, X_te, y_tr, y_te, _, d_te = time_split(X, y, dates)
        preds = {"baseline": baseline_predict(X_te)}
        for name, model in get_candidates().items():
            model.fit(X_tr, y_tr)
            preds[name] = model.predict(X_te)
        ## the strategy used in production: the best candidate for this country
        best = min(get_candidates(), key=lambda m: rmse(y_te, preds[m]))
        preds["selected_per_country"] = preds[best]
        for name, p in preds.items():
            rows.append({"country": country, "model": name, "rmse": rmse(y_te, p), "mae": mean_absolute_error(y_te, p)})
        holdout_preds[country] = (d_te, y_te, preds)

    results = pd.DataFrame(rows)
    if save:
        os.makedirs(REPORT_DIR, exist_ok=True)
        results.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"), index=False)
        plot_comparison(results, holdout_preds)
    return results


def plot_comparison(results, holdout_preds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(IMAGE_DIR, exist_ok=True)
    order = ["baseline", "linear_regression", "random_forest", "gradient_boosting", "selected_per_country"]
    colors = {"baseline": "#999999", "linear_regression": "#4C72B0", "random_forest": "#DD8452",
              "gradient_boosting": "#55A868", "selected_per_country": "#8172B3"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 5.8), gridspec_kw={"width_ratios": [1.5, 1]})

    ## (left) forecasts vs actual for all countries combined
    d_te, y_te, preds = holdout_preds["all"]
    best_all = results[(results.country == "all") & results.model.isin(list(get_candidates()))].sort_values("rmse").iloc[0]
    ax1.plot(d_te, y_te, color="black", lw=2.2, label="actual")
    ax1.plot(d_te, preds["baseline"], color=colors["baseline"], lw=1.8, ls="--",
             label="baseline (RMSE {:,.0f})".format(rmse(y_te, preds["baseline"])))
    ax1.plot(d_te, preds[best_all.model], color=colors[best_all.model], lw=1.8,
             label="{} (RMSE {:,.0f})".format(best_all.model.replace("_", " "), best_all.rmse))
    ax1.set_title("All countries: 30-day revenue forecast vs baseline on the hold-out period")
    ax1.set_ylabel("Revenue over next 30 days")
    ax1.legend()

    ## (right) relative RMSE vs baseline, averaged over countries
    rel = results.pivot(index="country", columns="model", values="rmse")
    rel = rel.div(rel["baseline"], axis=0)[order].mean()
    bars = ax2.bar([m.replace("_", "\n") for m in order], rel.values, color=[colors[m] for m in order])
    for b, v in zip(bars, rel.values):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.01, "{:.2f}".format(v), ha="center", va="bottom")
    ax2.axhline(1, color="#999999", ls="--", lw=1)
    ax2.set_ylabel("RMSE relative to baseline (lower is better)")
    ax2.set_title("Average over {} series (top 10 countries + all)".format(len(holdout_preds)))

    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE_DIR, "model_vs_baseline.png"), dpi=120)
    plt.close(fig)


######################################################################
# train / load / predict
######################################################################

def model_train(data_dir=DATA_DIR, test=False):
    """
    train one model per country. Every candidate is evaluated on the most recent 25% of
    days; the best one is refit on all days and saved. test=True trains small models for
    a subset of countries and saves them as models/test-*.joblib (production models are
    untouched). Returns {country: selected model name}.
    """

    os.makedirs(MODEL_DIR, exist_ok=True)
    ts_data = fetch_ts(data_dir)
    if test:
        ts_data = {k: v for k, v in ts_data.items() if k in TEST_COUNTRIES}

    selected = {}
    for country, df in ts_data.items():
        t0 = time.time()
        X, y, dates = engineer_features(df)
        X_tr, X_te, y_tr, y_te, _, _ = time_split(X, y, dates)

        scores = {}
        for name, candidate in get_candidates(test).items():
            scores[name] = rmse(y_te, candidate.fit(X_tr, y_tr).predict(X_te))
        best = min(scores, key=scores.get)

        model = get_candidates(test)[best].fit(X, y)
        joblib.dump(model, model_path(country, test))
        selected[country] = best

        runtime = "{:.1f}s".format(time.time() - t0)
        date_range = (str(pd.Timestamp(dates[0]).date()), str(pd.Timestamp(dates[-1]).date()))
        update_train_log(country, date_range, scores[best], runtime, MODEL_VERSION,
                         "{} | selected: {}".format(MODEL_VERSION_NOTE, best), test=test)
    return selected


def model_load(data_dir=DATA_DIR, test=False):
    """
    returns (all_data, all_models):
      all_data[country]   = {"X": features, "y": targets, "dates": dates} for every day (incl. the last 30)
      all_models[country] = fitted model
    """

    prefix = "test" if test else "sl"
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError("no models found; run model_train() first")
    files = [f for f in os.listdir(MODEL_DIR) if f.startswith(prefix + "-") and f.endswith(".joblib")]
    if not files:
        raise FileNotFoundError("no '{}' models found in {}; run model_train() first".format(prefix, MODEL_DIR))

    all_models = {}
    for f in files:
        country = re.sub(r"^{}-|-\d+_\d+\.joblib$".format(prefix), "", f)
        all_models[country] = joblib.load(os.path.join(MODEL_DIR, f))

    ts_data = fetch_ts(data_dir)
    all_data = {}
    for country in all_models:
        X, y, dates = engineer_features(ts_data[country], training=False)
        all_data[country] = {"X": X, "y": y, "dates": pd.to_datetime(dates)}
    return all_data, all_models


def model_predict(country, year, month, day, all_models=None, all_data=None, test=False):
    """
    forecast revenue for the 30 days starting at year-month-day.
    returns {"y_pred": array([value]), "y_proba": None}
    """

    t0 = time.time()
    if all_models is None or all_data is None:
        all_data, all_models = model_load(test=test)

    country = str(country).strip().lower().replace(" ", "_")
    if country not in all_models:
        raise ValueError("no model for country '{}'; available: {}".format(country, sorted(all_models)))

    try:
        target = pd.Timestamp("{}-{}-{}".format(int(year), str(int(month)).zfill(2), str(int(day)).zfill(2)))
    except (ValueError, TypeError):
        raise ValueError("invalid date: {}-{}-{}".format(year, month, day))

    data = all_data[country]
    matches = np.where(data["dates"] == target)[0]
    if len(matches) == 0:
        raise ValueError("date {} is outside the available range {} to {}".format(
            target.date(), data["dates"].min().date(), data["dates"].max().date()))

    X_query = data["X"].iloc[[matches[0]]]
    y_pred = all_models[country].predict(X_query)

    runtime = "{:.3f}s".format(time.time() - t0)
    update_predict_log(country, str(target.date()), y_pred[0], runtime, MODEL_VERSION, test=test)
    return {"y_pred": y_pred, "y_proba": None}


if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        res = compare_models()
        summary = res.groupby("model")[["rmse", "mae"]].mean().sort_values("rmse")
        print(res.to_string(index=False))
        print("\nmean over countries:\n", summary.round(1))
    else:
        t0 = time.time()
        for country, name in model_train().items():
            print("{:<16} {}".format(country, name))
        print("trained production models in {:.0f}s".format(time.time() - t0))
        all_data, all_models = model_load()
        print("loaded models:", sorted(all_models))
        print("example prediction (all, 2018-01-05):", model_predict("all", 2018, 1, 5, all_models, all_data)["y_pred"])
