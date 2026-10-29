# AAVAIL Revenue Forecasting – IBM AI Workflow Capstone

Forecasting the next 30 days of revenue for AAVAIL, overall and per country, with a model served through a Flask API, monitored for drift, covered by unit tests and packaged in Docker.

> Lines marked `TODO` are placeholders. Replace them with your own results as you build each part.

---

## 1. Business opportunity

AAVAIL management wants to predict revenue for the following month at any point in time, so managers can plan staffing and budgets. They want predictions both for the top ten countries by revenue and for all countries combined.

**Data science framing:** a supervised regression / time-series forecasting problem. The target is the revenue summed over the next 30 days.

**Testable hypotheses:**
- TODO: e.g. "A model using recent revenue, views and invoice counts beats a 30-day moving-average baseline."
- TODO: e.g. "Country-specific models outperform a single global model for the top countries."

---

## 2. Repository structure

```
├── data/
│   ├── cs-train/          # training JSON files
│   └── cs-production/     # production JSON files (used for monitoring)
├── ingestion.py           # loads JSON files and builds daily time series
├── EDA.ipynb              # exploratory data analysis with visualizations
├── model.py               # train / load / predict; compares several models
├── logger.py              # train and predict logging
├── monitoring.py          # performance and drift monitoring
├── app.py                 # Flask API
├── unittests/
│   ├── ApiTests.py
│   ├── ModelTests.py
│   └── LoggerTests.py
├── run-tests.py           # runs the full test suite
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 3. Setup

```bash
git clone https://github.com/Sanjaycodes019/aavail-revenue-forecasting.git
cd aavail-revenue-forecasting
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 4. Data ingestion

`ingestion.py` automates loading and preparing the data:

- Reads every JSON file in a data directory and makes column names consistent across files.
- Combines `year`, `month` and `day` into a single date.
- Adds up the data per day and per country into: `purchases`, `unique_invoices`, `unique_streams`, `total_views`, `revenue`.
- Writes one time-series CSV per country (top 10) plus `all`.

```bash
python ingestion.py
```

TODO: note anything you cleaned or fixed (e.g. inconsistent column names, negative prices).

---

## 5. Exploratory data analysis

See [`EDA.ipynb`](EDA.ipynb). Key findings:

- TODO: date range of the data and number of days.
- TODO: top countries by total revenue (chart).
- TODO: seasonal patterns in revenue / purchases / views (chart).
- TODO: how revenue relates to other features (chart).

TODO: add 1–2 key charts here as images, e.g. `![Revenue over time](images/revenue_over_time.png)`

---

## 6. Modeling

### Features
TODO: describe the features you engineered (e.g. revenue over the previous 7/14/28/70 days, recent invoices, views).

### Models compared

| Model | RMSE | MAE |
|---|---|---|
| Baseline (TODO: describe) | TODO | TODO |
| TODO: e.g. Random Forest | TODO | TODO |
| TODO: e.g. Gradient Boosting | TODO | TODO |
| TODO: e.g. Linear Regression | TODO | TODO |

TODO: which model you chose and why.

### Comparison with the baseline

TODO: add the chart comparing your model with the baseline, e.g. `![Model vs baseline](images/model_vs_baseline.png)`

---

## 7. API

Start the server:

```bash
python app.py
```

| Endpoint | Method | Description |
|---|---|---|
| `/train` | POST | Retrains the models (`{"mode": "test"}` trains test models only) |
| `/predict` | POST | Returns a revenue forecast |
| `/logs/<filename>` | GET | Downloads a log file |

Example: prediction for one country

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"country": "united_kingdom", "year": "2018", "month": "01", "day": "05"}'
```

Example: prediction for all countries combined

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"country": "all", "year": "2018", "month": "01", "day": "05"}'
```

TODO: check the request and response formats match your implementation.

---

## 8. Logging

`logger.py` writes one line per training run and per prediction:

- **Train log:** timestamp, date range of the training data, evaluation metric, runtime, model version.
- **Predict log:** unique ID, timestamp, country, target date, prediction, runtime, model version.

Logs are written to `logs/`. Test runs write to separate test log files so production logs are never touched.

---

## 9. Monitoring

`monitoring.py` checks whether incoming production data has drifted away from the training data:

- TODO: describe your method (e.g. outlier detection on input features, distance between training and production distributions).
- TODO: describe how prediction error is tracked over time against actual revenue.

```bash
python monitoring.py
```

---

## 10. Unit tests

Run the full suite with one command:

```bash
python run-tests.py
```

| Test file | Covers |
|---|---|
| `unittests/ApiTests.py` | `/train`, `/predict` (single country and `all`), `/logs` |
| `unittests/ModelTests.py` | training, loading and predicting |
| `unittests/LoggerTests.py` | train and predict log creation and contents |

**Test isolation:** all tests run in *test mode*, so models are saved as `test-*` files and logs as `test-*.log`. Running tests never overwrites production models or logs.

TODO: paste the output showing all tests passing.

---

## 11. Docker

Build and run:

```bash
docker build -t aavail-revenue-forecasting .
docker run -p 8080:8080 aavail-revenue-forecasting
```

The API is then available at `http://localhost:8080`.

---

## 12. Peer review checklist

| Review question | Where to find it |
|---|---|
| Unit tests for the API | `unittests/ApiTests.py` |
| Unit tests for the model | `unittests/ModelTests.py` |
| Unit tests for the logging | `unittests/LoggerTests.py` |
| All tests run with a single script and pass | `run-tests.py` ([section 10](#10-unit-tests)) |
| Mechanism to monitor performance | `monitoring.py` ([section 9](#9-monitoring)) |
| Read/write tests kept away from production | test mode ([section 10](#10-unit-tests)) |
| API works for one country and for all countries | [section 7](#7-api) |
| Data ingestion is a function or script | `ingestion.py` ([section 4](#4-data-ingestion)) |
| Multiple models compared | [section 6](#6-modeling) |
| EDA uses visualizations | `EDA.ipynb` ([section 5](#5-exploratory-data-analysis)) |
| Everything containerized in a working Docker image | `Dockerfile` ([section 11](#11-docker)) |
| Visualization comparing the model to the baseline | [section 6](#6-modeling) |
