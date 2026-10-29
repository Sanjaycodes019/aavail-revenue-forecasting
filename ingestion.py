#!/usr/bin/env python
"""
Data ingestion for the AAVAIL revenue forecasting capstone.

- fetch_data():         read every JSON invoice file in a directory into one DataFrame
- convert_to_ts():      aggregate invoices into a daily time series
- fetch_ts():           build (and cache) the time series for the top 10 countries and 'all'
- engineer_features():  turn a daily time series into a supervised learning table

Usage:
    python ingestion.py [data_dir]
"""

import os
import re
import sys
import json
import time

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cs-train")
PRODUCTION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cs-production")
TOP_N_COUNTRIES = 10
FORECAST_DAYS = 30

## the source files do not use consistent column names
COLUMN_RENAMES = {"StreamID": "stream_id", "TimesViewed": "times_viewed", "total_price": "price"}
EXPECTED_COLUMNS = ["country", "customer_id", "day", "invoice", "month", "price", "stream_id", "times_viewed", "year"]


def country_key(name):
    """'United Kingdom' -> 'united_kingdom'"""
    return re.sub(r"\s+", "_", str(name).strip().lower())


def fetch_data(data_dir):
    """
    load every invoices-*.json file in data_dir into a single DataFrame,
    harmonize column names and add an invoice_date column
    """

    if not os.path.isdir(data_dir):
        raise FileNotFoundError("data directory does not exist: {}".format(data_dir))

    files = sorted(f for f in os.listdir(data_dir) if re.search(r"\.json$", f))
    if not files:
        raise FileNotFoundError("no json files found in {}".format(data_dir))

    frames = []
    for name in files:
        with open(os.path.join(data_dir, name)) as fh:
            df = pd.DataFrame(json.load(fh))
        df = df.rename(columns=COLUMN_RENAMES)
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError("{} is missing columns: {}".format(name, sorted(missing)))
        frames.append(df[EXPECTED_COLUMNS])

    df = pd.concat(frames, ignore_index=True)

    ## build a single date column
    df["invoice_date"] = pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2) + "-" + df["day"].astype(str).str.zfill(2))

    ## invoice ids sometimes carry letters (e.g. 'C' for cancellations); keep the numeric id
    df["invoice"] = df["invoice"].astype(str).str.replace(r"\D+", "", regex=True)

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["times_viewed"] = pd.to_numeric(df["times_viewed"], errors="coerce").fillna(0)

    ## a handful of negative prices are accounting adjustments, not purchases
    df = df[df["price"].notnull() & (df["price"] >= 0)]

    return df.sort_values("invoice_date").reset_index(drop=True)


def convert_to_ts(df, country=None):
    """
    aggregate invoices to one row per day; every day in the full date range of df
    is present (days without invoices get zeros), so all countries share one calendar
    columns: date, purchases, unique_invoices, unique_streams, total_views, year_month, revenue
    """

    days = pd.date_range(df["invoice_date"].min(), df["invoice_date"].max(), freq="D")

    if country is not None:
        df = df[df["country"].map(country_key) == country]
        if df.empty:
            raise ValueError("country not found: {}".format(country))

    grouped = df.groupby("invoice_date")
    ts = pd.DataFrame({
        "purchases": grouped.size(),
        "unique_invoices": grouped["invoice"].nunique(),
        "unique_streams": grouped["stream_id"].nunique(),
        "total_views": grouped["times_viewed"].sum(),
        "revenue": grouped["price"].sum(),
    }).reindex(days, fill_value=0)

    ts.index.name = "date"
    ts = ts.reset_index()
    ts["year_month"] = ts["date"].dt.strftime("%Y-%m")
    return ts[["date", "purchases", "unique_invoices", "unique_streams", "total_views", "year_month", "revenue"]]


def fetch_ts(data_dir, clean=False):
    """
    return {country: daily time series} for the top countries by revenue and 'all'.
    the series are cached as csv files in data_dir/ts-data; clean=True rebuilds them.
    """

    ts_dir = os.path.join(data_dir, "ts-data")

    if clean and os.path.isdir(ts_dir):
        for f in os.listdir(ts_dir):
            os.remove(os.path.join(ts_dir, f))

    if os.path.isdir(ts_dir) and any(f.endswith(".csv") for f in os.listdir(ts_dir)):
        out = {}
        for f in sorted(os.listdir(ts_dir)):
            if f.endswith(".csv"):
                key = re.sub(r"^ts-|\.csv$", "", f)
                out[key] = pd.read_csv(os.path.join(ts_dir, f), parse_dates=["date"])
        return out

    df = fetch_data(data_dir)
    revenue = df.groupby(df["country"].map(country_key))["price"].sum().sort_values(ascending=False)
    top = revenue.index[:TOP_N_COUNTRIES].tolist()

    out = {"all": convert_to_ts(df)}
    for c in top:
        out[c] = convert_to_ts(df, country=c)

    os.makedirs(ts_dir, exist_ok=True)
    for key, ts in out.items():
        ts.to_csv(os.path.join(ts_dir, "ts-{}.csv".format(key)), index=False)
    return out


def engineer_features(df, training=True):
    """
    build features from a daily time series. For each day:
      previous_7/14/28/70  revenue summed over the previous N days
      recent_invoices      mean daily unique invoices over the previous 30 days
      recent_views         mean daily views over the previous 30 days
    target: revenue summed over the next 30 days (including the day itself)

    training=True drops the last 30 days, whose target is not yet fully observed.
    returns X (DataFrame), y (array), dates (array of datetime64)
    """

    df = df.sort_values("date").reset_index(drop=True)
    rev = df.set_index("date")["revenue"]
    dates = df["date"].values

    feats = {}
    for n in (7, 14, 28, 70):
        ## shift(1): only use days strictly before the prediction date
        feats["previous_{}".format(n)] = rev.shift(1).rolling(n, min_periods=1).sum().fillna(0).values

    ## note: a 'same window last year' feature was tried and dropped. With only ~20 months
    ## of history it is zero for the whole first year and non-zero only from the Nov-2018
    ## peak onward, so models learned it as a spurious 'peak season' flag.

    inv = df.set_index("date")["unique_invoices"]
    views = df.set_index("date")["total_views"]
    feats["recent_invoices"] = inv.shift(1).rolling(30, min_periods=1).mean().fillna(0).values
    feats["recent_views"] = views.shift(1).rolling(30, min_periods=1).mean().fillna(0).values

    X = pd.DataFrame(feats)
    ## target: revenue for the next 30 days starting today
    y = rev[::-1].rolling(FORECAST_DAYS, min_periods=1).sum()[::-1].values

    if training:
        keep = np.arange(len(df)) < len(df) - FORECAST_DAYS
        X, y, dates = X[keep].reset_index(drop=True), y[keep], dates[keep]

    return X, y, dates


if __name__ == "__main__":

    data_dir = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    t0 = time.time()
    ts_all = fetch_ts(data_dir, clean=True)
    for key, ts in ts_all.items():
        print("{:<16} {} days  {} to {}  revenue {:,.0f}".format(
            key, len(ts), ts["date"].min().date(), ts["date"].max().date(), ts["revenue"].sum()))
    print("load time: {:.1f}s".format(time.time() - t0))
