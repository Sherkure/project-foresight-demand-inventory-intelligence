# Project FORESIGHT — FORESIGHT_FINAL

Merged project combining:
- **FORESIGHT_REAL** — genuine client data, cleaning pipeline, forecasting model,
  backtesting, forward forecast, and risk scoring (priority source for all
  data/modelling logic).
- **FORESIGHT_DASHBOARD** — Streamlit UI (`app/dashboard.py`), carried over
  **unmodified** for now. It is not yet wired to this backend's output schema —
  see "What remains" below.

Neither original project folder was modified. Everything here is newly
written or copied into `FORESIGHT_FINAL/`.

## Data

`data/raw/` contains the four real client extracts, copied unmodified from
FORESIGHT_REAL: `sales_daily.csv`, `sku_master.csv`, `calendar.csv`,
`inventory_snapshots.csv`. Nothing in this project is simulated or fabricated —
all downstream numbers are computed from these files.

## Pipeline (run in order)

```bash
python src/pipeline.py       # D1 — ingest + clean -> data/processed/analysis_ready.csv
python src/forecast.py       # D3 — weekly baseline, features, backtest, model, forward forecast
python src/risk_scoring.py   # D4 — stockout / overstock risk -> data/processed/risk_scoring.csv
```

Each script re-runs end-to-end from the raw data with no manual steps.

### `src/pipeline.py`
Cleans the four raw extracts (duplicate rows, missing values, invalid
negatives, category-label normalization) and joins them into one
analysis-ready dataset. Every cleaning decision is logged to
`data/processed/data_quality_log.txt`.

### `src/forecast.py`
- Aggregates cleaned daily sales to weekly SKU-level demand.
- Builds a seasonal-naive baseline (last week as this week's forecast).
- Engineers causal lag/rolling/calendar features — no future data ever
  enters a feature.
- Backtests with a rolling-origin split (train on the past, test on the
  most recent 6 weeks per SKU).
- Trains a gradient-boosted tree model (LightGBM if available, otherwise
  scikit-learn's `HistGradientBoostingRegressor` as a same-family fallback —
  this environment has no internet access to install LightGBM).
- Reports WAPE for baseline vs. model **honestly** — see
  `reports/backtest_results.json` for this run's actual numbers.
- Computes feature importance (native for LightGBM, permutation importance
  for the sklearn fallback) — `reports/feature_importance.csv` / `.png`.
- Produces a genuine 6-week-forward forecast per SKU —
  `data/processed/forecast_forward.csv`.

### `src/risk_scoring.py`
Preserved as-is from FORESIGHT_REAL: combines the forward forecast with the
latest real inventory snapshot per SKU to flag stockout risk, overstock risk,
and a recommended action (`Reorder now` / `Markdown / clear` / `Watch /
volatile` / `Healthy`) — transparent, rule-based, not a black box.

## Latest run's actual results

See `reports/backtest_results.json` for the current numbers (this file is
regenerated every run — nothing here is hard-coded in the README). As of the
run in this build:
- Baseline (seasonal-naive) WAPE and model WAPE are both reported, along with
  whether the model actually beat the baseline on the backtest.
- `data/processed/risk_scoring.csv` has one row per SKU with its risk flags.

## Dashboard integration (update)

`app/dashboard.py` is now wired to this backend's actual output schema
(`weekly_demand.csv`, `forecast_forward.csv`, `risk_scoring.csv`,
`backtest_results.json`, `feature_importance.csv`, `forecast_model.pkl`).
It has two tabs:

- **Demand Forecast** — Model Ready / Model WAPE / Baseline WAPE / WAPE
  Improvement (all read straight from `backtest_results.json`), a SKU
  selector, weekly sales history, a forecast-vs-actual chart (actual vs.
  the seasonal-naive baseline vs. the saved production model applied to
  historical weeks), next-week forecast, and feature importance.
- **Risk Scoring** — the existing stockout/overstock flags from
  `risk_scoring.py`, filterable by category and recommended action, with a
  rupee-value-at-stake figure computed transparently in the dashboard layer
  from real `sku_master.csv` unit costs/list prices combined with the real
  risk flags (shortfall x list price for stockout risk, surplus x unit cost
  for overstock risk) — nothing simulated.

Run: `streamlit run app/dashboard.py`

## What remains unfinished

- No deployed scoring service (FastAPI/D6) has been built yet in
  FORESIGHT_FINAL.
- No executive readout / EDA memo has been rewritten for the merged project
  (FORESIGHT_REAL's originals still exist in the untouched original folder).
- The rupee-value-at-stake figure is a dashboard-layer computation, not part
  of `src/risk_scoring.py` itself (which still preserves FORESIGHT_REAL's
  original binary stockout/overstock logic as instructed).
