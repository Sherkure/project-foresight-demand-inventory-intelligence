"""
FORESIGHT — Planning Dashboard (D5)

Wired to the FORESIGHT_FINAL backend (src/pipeline.py -> src/forecast.py ->
src/risk_scoring.py). Every number shown here is read directly from that
backend's genuine outputs:

  data/processed/weekly_demand.csv     - real weekly actuals per SKU
  data/processed/forecast_forward.csv  - genuine 6-week-forward ML forecast
  data/processed/risk_scoring.csv      - stockout / overstock flags
  reports/backtest_results.json        - honest held-out backtest metrics
  reports/feature_importance.csv       - model feature importance
  reports/forecast_model.pkl           - the trained production model

Nothing in this file fabricates, simulates, or hard-codes a forecast or a
metric. The only computation done in-app is: (1) re-applying the already
trained model to historical weeks purely so the "Forecast vs Actual" chart
has a fitted ML line to show (the authoritative WAPE numbers still come
straight from backtest_results.json, not from this chart), and (2) a
transparent rupee-value-at-stake calculation from real sku_master costs/
prices combined with the real risk_scoring.csv flags.

Run: streamlit run app/dashboard.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

# Must mirror src/forecast.py exactly - these are the causal features the
# saved model was trained on. Not a re-implementation of the model, just the
# feature recipe needed to hand rows to the already-trained model.
SEASON_LAG = 1
FEATURES = ["lag_1", "lag_2", "lag_4", "roll_mean_4", "roll_std_4", "week_of_year", "month"]

st.set_page_config(page_title="FORESIGHT — NorthBay Living", layout="wide")

st.title("📦 FORESIGHT — Demand & Inventory Planning")
st.caption("NorthBay Living · Reorder and markdown guidance, updated from the latest forecast run")


@st.cache_data
def load_core_data():
    weekly = pd.read_csv(DATA / "weekly_demand.csv", parse_dates=["date"])
    forecast_fwd = pd.read_csv(DATA / "forecast_forward.csv", parse_dates=["date"])
    risk_df = pd.read_csv(DATA / "risk_scoring.csv")
    sku_master = pd.read_csv(RAW / "sku_master.csv")
    with open(REPORTS / "backtest_results.json") as f:
        backtest_results = json.load(f)
    feat_imp = pd.read_csv(REPORTS / "feature_importance.csv", index_col=0)
    feat_imp.index.name = "feature"
    return weekly, forecast_fwd, risk_df, sku_master, backtest_results, feat_imp


@st.cache_resource
def load_model():
    return joblib.load(REPORTS / "forecast_model.pkl")


def engineer_features(sku_weekly: pd.DataFrame) -> pd.DataFrame:
    """Same causal feature engineering as src/forecast.py::add_baseline_and_features,
    applied here only so the saved model can be run for the display chart. The
    model is loaded, never retrained, in this file."""
    sku_weekly = sku_weekly.sort_values(["sku_id", "date"]).copy()
    sku_weekly["naive_pred"] = sku_weekly.groupby("sku_id")["weekly_units"].shift(SEASON_LAG)
    g = sku_weekly.groupby("sku_id")["weekly_units"]
    sku_weekly["lag_1"] = g.shift(1)
    sku_weekly["lag_2"] = g.shift(2)
    sku_weekly["lag_4"] = g.shift(4)
    sku_weekly["roll_mean_4"] = g.shift(1).rolling(4).mean()
    sku_weekly["roll_std_4"] = g.shift(1).rolling(4).std()
    sku_weekly["week_of_year"] = sku_weekly["date"].dt.isocalendar().week.astype(int)
    sku_weekly["month"] = sku_weekly["date"].dt.month
    return sku_weekly


# ---- Load backend outputs ----
data_load_error = None
try:
    weekly, forecast_fwd, risk, sku_master, backtest, feature_importance = load_core_data()
except FileNotFoundError as e:
    data_load_error = str(e)

if data_load_error:
    st.warning(
        "No data found yet. Run `python src/pipeline.py`, `src/forecast.py`, "
        f"and `src/risk_scoring.py` first, then reload. ({data_load_error})"
    )
    st.stop()

model = None
model_ready = False
try:
    model = load_model()
    model_ready = True
except Exception:
    model_ready = False

# ---- Derived (but not fabricated) fields ----
risk = risk.merge(
    sku_master[["sku_id", "category", "subcategory", "unit_cost", "list_price"]],
    on="sku_id", how="left",
)
# Rupee value at stake: shortfall x list price (lost sales exposure) for stockout
# risk, or surplus x unit cost (locked capital) for overstock risk. Every input
# (horizon_demand, available_stock, on_hand_units, unit_cost, list_price) comes
# straight from risk_scoring.csv / sku_master.csv - nothing simulated.
shortfall = (risk["horizon_demand"] - risk["available_stock"]).clip(lower=0)
excess = (risk["on_hand_units"] - risk["horizon_demand"]).clip(lower=0)
risk["rupee_value_at_stake"] = np.where(
    risk["stockout_risk"] == 1,
    shortfall * risk["list_price"],
    np.where(risk["overstock_risk"] == 1, excess * risk["unit_cost"], 0.0),
)
risk["quadrant"] = risk["recommended_action"]

# ---- Sidebar filters (apply to Risk Scoring tab) ----
st.sidebar.header("Risk Scoring filters")
categories = ["All"] + sorted(risk["category"].dropna().unique().tolist())
category_filter = st.sidebar.selectbox("Category", categories)
quadrant_filter = st.sidebar.multiselect(
    "Recommended action", options=sorted(risk["quadrant"].dropna().unique().tolist()),
    default=sorted(risk["quadrant"].dropna().unique().tolist()),
)
sku_search = st.sidebar.text_input("Search SKU ID")

filtered = risk.copy()
if category_filter != "All":
    filtered = filtered[filtered["category"] == category_filter]
filtered = filtered[filtered["quadrant"].isin(quadrant_filter)]
if sku_search:
    filtered = filtered[filtered["sku_id"].str.contains(sku_search, case=False)]

tab_forecast, tab_risk = st.tabs(["📈 Demand Forecast", "⚠️ Risk Scoring"])

# =====================================================================
# TAB 1 — DEMAND FORECAST
# =====================================================================
with tab_forecast:
    st.subheader("Model performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model Ready", "✅ Yes" if model_ready else "❌ No")
    c2.metric("Model WAPE", f"{backtest['model_wape'] * 100:.2f}%")
    c3.metric("Baseline WAPE", f"{backtest['baseline_wape'] * 100:.2f}%")
    wape_imp = backtest.get("wape_improvement_pct")
    c4.metric(
        "WAPE Improvement",
        f"{wape_imp:.2f}%" if wape_imp is not None else "n/a",
        delta="beats baseline" if backtest.get("model_beats_baseline") else "below baseline",
    )
    st.caption(
        f"Model: {backtest['model_name']} · backtest horizon {backtest['horizon_weeks']} weeks · "
        f"trained through {backtest['train_period_end']} · "
        f"{backtest['train_rows']} train rows / {backtest['test_rows']} held-out test rows"
    )

    st.divider()

    st.subheader("SKU demand explorer")
    sku_options = sorted(weekly["sku_id"].unique().tolist())
    chosen_sku = st.selectbox("SKU", sku_options)

    sku_hist = engineer_features(weekly[weekly["sku_id"] == chosen_sku])
    sku_future = forecast_fwd[forecast_fwd["sku_id"] == chosen_sku].sort_values("date")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Weekly Sales History**")
        st.line_chart(
            sku_hist.set_index("date")[["weekly_units"]].rename(columns={"weekly_units": "Actual units sold"})
        )

    with col_right:
        st.markdown("**Forecast vs Actual**")
        chart_df = sku_hist.set_index("date")[["weekly_units", "naive_pred"]].rename(
            columns={"weekly_units": "Actual", "naive_pred": "Seasonal-naive baseline"}
        )
        if model_ready:
            valid = sku_hist.dropna(subset=FEATURES).copy()
            if not valid.empty:
                valid["ml_forecast"] = np.clip(model.predict(valid[FEATURES]), 0, None)
                chart_df = chart_df.join(valid.set_index("date")["ml_forecast"].rename("ML forecast"))
        st.line_chart(chart_df)

    st.caption(
        "The ML forecast line is the saved production model "
        "(`reports/forecast_model.pkl`) applied to this SKU's historical weeks, "
        "shown alongside the seasonal-naive baseline for comparison. The "
        "Model WAPE / Baseline WAPE metrics above come from the proper "
        "held-out rolling-origin backtest in `reports/backtest_results.json`, "
        "not from this in-sample chart."
    )

    st.divider()

    st.subheader("Next Week Forecast")
    if not sku_future.empty:
        next_row = sku_future.iloc[0]
        st.metric(
            f"{chosen_sku} — week of {next_row['date'].date()}",
            f"{next_row['forecast']:.1f} units",
        )
        with st.expander("Full forward forecast (all forecasted weeks)"):
            st.dataframe(
                sku_future[["date", "forecast"]].rename(
                    columns={"date": "Week", "forecast": "Forecast (units)"}
                ),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.write("No forward forecast available for this SKU.")

    st.divider()

    st.subheader("Feature Importance")
    st.bar_chart(feature_importance["importance"])
    st.dataframe(
        feature_importance.reset_index().rename(columns={"feature": "Feature", "importance": "Importance"}),
        hide_index=True,
        use_container_width=True,
    )

# =====================================================================
# TAB 2 — RISK SCORING
# =====================================================================
with tab_risk:
    st.info(
        "Demand forecast and inventory position here are both computed from "
        "real client extracts (`data/raw/sales_daily.csv`, "
        "`data/raw/inventory_snapshots.csv`, `data/raw/sku_master.csv`) — see "
        "`README.md` and `data/processed/data_quality_log.txt` for the cleaning "
        "steps applied.",
        icon="ℹ️",
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("SKUs in view", len(filtered))
    col2.metric("Reorder Now", int((filtered["quadrant"] == "Reorder now").sum()))
    col3.metric("Markdown / Clear", int((filtered["quadrant"] == "Markdown / clear").sum()))
    col4.metric("Rupee value at stake", f"₹{filtered['rupee_value_at_stake'].sum():,.0f}")

    st.divider()

    st.subheader("Prioritised reorder / markdown list")
    if filtered.empty:
        st.write("No SKUs match the current filters.")
    else:
        show_cols = [
            "sku_id", "category", "recommended_action", "horizon_demand",
            "on_hand_units", "on_order_units", "available_stock",
            "stockout_risk", "overstock_risk", "rupee_value_at_stake",
        ]
        st.dataframe(
            filtered[show_cols].sort_values("rupee_value_at_stake", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

st.caption("FORESIGHT · Project engagement for NorthBay Living · Zidio Development")
