"""
Project FORESIGHT — Data Pipeline (D1)

Ingests the four real client extracts (sales_daily, sku_master, calendar,
inventory_snapshots) from data/raw/, cleans them, and produces one
analysis-ready dataset at data/processed/analysis_ready.csv.

No data is fabricated or simulated here: every row comes from the raw
extracts provided in the engagement brief. Cleaning steps only fix
duplicates / missing / invalid values found in that real data, and every
decision is logged.

Run:  python src/pipeline.py   (from the project root, or `python pipeline.py`
      from inside src/ — path handling below supports both)
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

log = []  # every cleaning decision -> data_quality_log.txt


def note(msg: str) -> None:
    print(msg)
    log.append(msg)


def load_raw():
    sales = pd.read_csv(RAW_DIR / "sales_daily.csv", parse_dates=["date"])
    sku_master = pd.read_csv(RAW_DIR / "sku_master.csv", parse_dates=["launch_date"])
    calendar = pd.read_csv(RAW_DIR / "calendar.csv", parse_dates=["date"])
    inventory = pd.read_csv(RAW_DIR / "inventory_snapshots.csv", parse_dates=["date"])
    note(
        f"Raw shapes -> sales:{sales.shape} sku_master:{sku_master.shape} "
        f"calendar:{calendar.shape} inventory:{inventory.shape}"
    )
    return sales, sku_master, calendar, inventory


def clean_sku_master(sku_master: pd.DataFrame) -> pd.DataFrame:
    dupes = sku_master.duplicated(subset="sku_id").sum()
    if dupes:
        note(f"sku_master: removed {dupes} duplicate sku_id rows")
        sku_master = sku_master.drop_duplicates(subset="sku_id", keep="first")

    before = sku_master["category"].unique().tolist()
    sku_master["category"] = sku_master["category"].astype(str).str.strip().str.title()
    after = sku_master["category"].unique().tolist()
    if before != after:
        note(f"sku_master: normalized category labels {before} -> {after}")

    n_missing_cost = sku_master["unit_cost"].isna().sum()
    if n_missing_cost:
        # fall back to 62% of list price (typical D2C home-goods COGS ratio) — documented assumption
        mask = sku_master["unit_cost"].isna()
        sku_master.loc[mask, "unit_cost"] = sku_master.loc[mask, "list_price"] * 0.62
        note(f"sku_master: {n_missing_cost} missing unit_cost values filled at 62% of list_price")

    return sku_master


def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    dupes = sales.duplicated().sum()
    if dupes:
        note(f"sales_daily: removed {dupes} exact duplicate rows")
        sales = sales.drop_duplicates()

    dupe_key = sales.duplicated(subset=["date", "sku_id"]).sum()
    if dupe_key:
        note(f"sales_daily: removed {dupe_key} duplicate (date, sku_id) rows, keeping first")
        sales = sales.drop_duplicates(subset=["date", "sku_id"], keep="first")

    missing_rev = sales["revenue"].isna().sum()
    if missing_rev:
        mask = sales["revenue"].isna()
        sales.loc[mask, "revenue"] = sales.loc[mask, "units_sold"] * sales.loc[mask, "unit_price"]
        note(f"sales_daily: {missing_rev} missing 'revenue' values reconstructed from units_sold * unit_price")

    missing_price = sales["unit_price"].isna().sum()
    if missing_price:
        mask = sales["unit_price"].isna() & (sales["units_sold"] > 0)
        sales.loc[mask, "unit_price"] = sales.loc[mask, "revenue"] / sales.loc[mask, "units_sold"]
        sales["unit_price"] = sales.groupby("sku_id")["unit_price"].transform(lambda s: s.fillna(s.median()))
        note(f"sales_daily: {missing_price} missing 'unit_price' values reconstructed / median-filled per SKU")

    neg = (sales["units_sold"] < 0).sum()
    if neg:
        note(f"sales_daily: {neg} negative units_sold rows clipped to 0")
        sales["units_sold"] = sales["units_sold"].clip(lower=0)

    neg_rev = (sales["revenue"] < 0).sum()
    if neg_rev:
        note(f"sales_daily: {neg_rev} negative revenue rows clipped to 0")
        sales["revenue"] = sales["revenue"].clip(lower=0)

    remaining_na = sales[["units_sold", "revenue", "unit_price"]].isna().any(axis=1).sum()
    if remaining_na:
        note(f"sales_daily: dropped {remaining_na} rows with unrecoverable missing values")
        sales = sales.dropna(subset=["units_sold", "revenue", "unit_price"])

    sales["units_sold"] = sales["units_sold"].astype(int)
    sales["promo_flag"] = sales["promo_flag"].fillna(0).astype(int)

    return sales


def clean_inventory(inventory: pd.DataFrame) -> pd.DataFrame:
    neg_stock = (inventory["on_hand_units"] < 0).sum()
    if neg_stock:
        inventory["on_hand_units"] = inventory["on_hand_units"].clip(lower=0)
        note(f"inventory_snapshots: {neg_stock} negative on_hand_units clipped to 0")

    neg_order = (inventory["on_order_units"] < 0).sum()
    if neg_order:
        inventory["on_order_units"] = inventory["on_order_units"].clip(lower=0)
        note(f"inventory_snapshots: {neg_order} negative on_order_units clipped to 0")

    dupes = inventory.duplicated(subset=["date", "sku_id"]).sum()
    if dupes:
        inventory = inventory.drop_duplicates(subset=["date", "sku_id"], keep="last")
        note(f"inventory_snapshots: removed {dupes} duplicate (date, sku_id) snapshot rows")

    return inventory


def build_analysis_ready(sales, sku_master, calendar, inventory) -> pd.DataFrame:
    df = sales.merge(sku_master, on="sku_id", how="left")
    df = df.merge(calendar, on="date", how="left", suffixes=("", "_cal"))

    inventory_sorted = inventory.sort_values("date")
    df_sorted = df.sort_values("date")
    merged = pd.merge_asof(
        df_sorted, inventory_sorted, on="date", by="sku_id", direction="backward"
    )

    if "is_holiday" in merged.columns:
        merged["is_holiday"] = merged["is_holiday"].fillna(0).astype(int)
    merged["promo_flag"] = merged["promo_flag"].fillna(0).astype(int)

    note(f"Final analysis-ready shape: {merged.shape}")
    return merged


def main():
    sales, sku_master, calendar, inventory = load_raw()
    sku_master = clean_sku_master(sku_master)
    sales = clean_sales(sales)
    inventory = clean_inventory(inventory)
    merged = build_analysis_ready(sales, sku_master, calendar, inventory)

    merged.to_csv(OUT_DIR / "analysis_ready.csv", index=False)
    with open(OUT_DIR / "data_quality_log.txt", "w") as f:
        f.write("\n".join(log))

    note(f"\nSaved analysis-ready dataset -> {OUT_DIR / 'analysis_ready.csv'}")
    note(f"Saved cleaning log -> {OUT_DIR / 'data_quality_log.txt'}")


if __name__ == "__main__":
    main()
