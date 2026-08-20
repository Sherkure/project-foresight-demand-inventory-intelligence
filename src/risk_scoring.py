"""
Project FORESIGHT — Risk Scoring (D4)

Combines the forward demand forecast (data/processed/forecast_forward.csv,
produced by src/forecast.py) with the real, current inventory position
(data/raw/inventory_snapshots.csv) to flag every SKU as stockout risk,
overstock risk, or healthy, with a transparent recommended action.

This logic is preserved as-is from FORESIGHT_REAL — it is not a black box:

  - Stockout risk: forecast demand over the horizon exceeds available stock
    (on-hand + on-order).
  - Overstock risk: on-hand stock is more than 2x the forecast demand over
    the horizon.
  - Recommended action is read directly off those two flags.

Run: python src/risk_scoring.py   (after src/pipeline.py and src/forecast.py)
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"


def recommend(row) -> str:
    if row["stockout_risk"] and not row["overstock_risk"]:
        return "Reorder now"
    if row["overstock_risk"] and not row["stockout_risk"]:
        return "Markdown / clear"
    if row["stockout_risk"] and row["overstock_risk"]:
        return "Watch / volatile"
    return "Healthy"


def main():
    forecast_forward = pd.read_csv(DATA_DIR / "forecast_forward.csv", parse_dates=["date"])
    inv = pd.read_csv(RAW_DIR / "inventory_snapshots.csv", parse_dates=["date"])
    latest_inv = inv.sort_values("date").groupby("sku_id").tail(1)

    horizon_demand = forecast_forward.groupby("sku_id")["forecast"].sum().reset_index()
    horizon_demand.columns = ["sku_id", "horizon_demand"]

    risk = horizon_demand.merge(latest_inv, on="sku_id", how="left")
    risk["available_stock"] = risk["on_hand_units"] + risk["on_order_units"]

    risk["stockout_risk"] = (risk["horizon_demand"] > risk["available_stock"]).astype(int)
    risk["overstock_risk"] = (risk["on_hand_units"] > 2 * risk["horizon_demand"].clip(lower=1)).astype(int)

    risk["recommended_action"] = risk.apply(recommend, axis=1)
    risk = risk[
        [
            "sku_id",
            "horizon_demand",
            "on_hand_units",
            "on_order_units",
            "available_stock",
            "stockout_risk",
            "overstock_risk",
            "recommended_action",
        ]
    ]

    risk.to_csv(DATA_DIR / "risk_scoring.csv", index=False)

    print("=== Risk scoring summary ===")
    print(risk["recommended_action"].value_counts())
    print(f"\nSaved -> data/processed/risk_scoring.csv ({risk.shape[0]} SKUs)")


if __name__ == "__main__":
    main()
