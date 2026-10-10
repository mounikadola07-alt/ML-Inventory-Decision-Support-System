"""
src/data_loader.py - Safe Dataset Ingestion, Validation, and Representative Benchmark Generation
Guarantees clean portability across local IDE and Google Colab environments.
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def load_walmart_sales(file_path: str = "data/Walmart_Sales.csv") -> pd.DataFrame:
    """
    Loads and validates Walmart weekly store sales dataset.
    """
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    elif os.path.exists("rawData.zip"):
        with zipfile.ZipFile("rawData.zip", "r") as z:
            with z.open("rawData/Walmart_sales/Walmart_Sales.csv") as f:
                df = pd.read_csv(f)
    elif os.path.exists("../rawData.zip"):
        with zipfile.ZipFile("../rawData.zip", "r") as z:
            with z.open("rawData/Walmart_sales/Walmart_Sales.csv") as f:
                df = pd.read_csv(f)
    else:
        raise FileNotFoundError(f"Walmart Sales dataset not found at {file_path} or in rawData.zip")

    # Parse dates explicitly
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", dayfirst=True)
    df = df.sort_values(by=["Store", "Date"]).reset_index(drop=True)
    return df


def load_online_retail(
    xlsx_path: str = "data/Online Retail.xlsx",
    cache_path: str = "data/Online_Retail_daily.csv"
) -> pd.DataFrame:
    """
    Loads and preprocesses UCI Online Retail transaction dataset into aggregated daily SKU demand.
    Caches aggregated daily series to CSV for sub-second subsequent loading.
    
    Data Cleaning Rules:
    - Eliminates duplicate transaction lines.
    - Filters out cancellations (InvoiceNo beginning with 'C') and non-positive quantities.
    - Excludes negative or zero unit prices.
    - Aggregates daily transaction volume and revenue per SKU (StockCode).
    """
    if os.path.exists(cache_path):
        df_daily = pd.read_csv(cache_path, dtype={"StockCode": str})
        df_daily["Date"] = pd.to_datetime(df_daily["Date"])
        df_daily = df_daily.sort_values(by=["StockCode", "Date"]).reset_index(drop=True)
        return df_daily
        
    if os.path.exists(xlsx_path):
        df_raw = pd.read_excel(xlsx_path)
    elif os.path.exists("rawData.zip"):
        with zipfile.ZipFile("rawData.zip", "r") as z:
            with z.open("rawData/Online_Retail/Online Retail.xlsx") as f:
                df_raw = pd.read_excel(f)
    elif os.path.exists("../rawData.zip"):
        with zipfile.ZipFile("../rawData.zip", "r") as z:
            with z.open("rawData/Online_Retail/Online Retail.xlsx") as f:
                df_raw = pd.read_excel(f)
    else:
        raise FileNotFoundError(f"Online Retail dataset not found at {xlsx_path} or in rawData.zip")

    # Clean transactions
    df_raw = df_raw.drop_duplicates().copy()
    valid_mask = (
        (~df_raw["InvoiceNo"].astype(str).str.startswith("C")) &
        (df_raw["Quantity"] > 0) &
        (df_raw["UnitPrice"] > 0)
    )
    df_valid = df_raw[valid_mask].copy()
    df_valid["InvoiceDate"] = pd.to_datetime(df_valid["InvoiceDate"])
    df_valid["Date"] = pd.to_datetime(df_valid["InvoiceDate"].dt.date)
    df_valid["Revenue"] = df_valid["Quantity"] * df_valid["UnitPrice"]
    df_valid["StockCode"] = df_valid["StockCode"].astype(str).str.strip()

    # Aggregate by SKU and Date
    df_daily = df_valid.groupby(["Date", "StockCode"]).agg(
        Daily_Quantity=("Quantity", "sum"),
        Daily_Revenue=("Revenue", "sum"),
        Avg_UnitPrice=("UnitPrice", "mean"),
        Transaction_Count=("InvoiceNo", "count")
    ).reset_index()

    df_daily = df_daily.sort_values(by=["StockCode", "Date"]).reset_index(drop=True)
    
    # Save cache for instant reload
    os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else ".", exist_ok=True)
    df_daily.to_csv(cache_path, index=False)
    return df_daily


def get_online_retail_continuous_panel(
    df_daily: pd.DataFrame,
    n_top_skus: int = 30,
    include_segments: bool = True
) -> pd.DataFrame:
    """
    Constructs a continuous daily SKU panel from real Online Retail data.
    Preserves calendar continuity by creating a complete Cartesian product [SKU, Date]
    and explicitly filling zero-sales days with 0.0.
    
    Selects representative SKUs across volume and frequency tiers:
    - High-volume / High-frequency
    - Medium-volume / Medium-frequency
    - Intermittent / Low-frequency
    """
    df_daily = df_daily.copy()
    df_daily["StockCode"] = df_daily["StockCode"].astype(str)
    
    # Compute SKU activity stats
    sku_stats = df_daily.groupby("StockCode").agg(
        total_qty=("Daily_Quantity", "sum"),
        active_days=("Date", "nunique")
    ).reset_index()

    if include_segments:
        # Segment by transaction frequency
        high_freq = sku_stats[sku_stats["active_days"] >= 200].sort_values("total_qty", ascending=False)
        med_freq = sku_stats[(sku_stats["active_days"] >= 50) & (sku_stats["active_days"] < 150)].sort_values("total_qty", ascending=False)
        low_freq = sku_stats[(sku_stats["active_days"] >= 15) & (sku_stats["active_days"] < 50)].sort_values("total_qty", ascending=False)
        
        per_seg = max(5, n_top_skus // 3)
        selected_skus = pd.concat([
            high_freq.head(per_seg),
            med_freq.head(per_seg),
            low_freq.head(per_seg)
        ])["StockCode"].unique()
    else:
        selected_skus = sku_stats.sort_values("total_qty", ascending=False).head(n_top_skus)["StockCode"].unique()

    # Filter to selected SKUs
    df_sub = df_daily[df_daily["StockCode"].isin(selected_skus)].copy()
    
    # Generate full Cartesian grid across date range
    min_date = df_sub["Date"].min()
    max_date = df_sub["Date"].max()
    full_dates = pd.date_range(min_date, max_date, freq="D")
    
    idx = pd.MultiIndex.from_product([selected_skus, full_dates], names=["StockCode", "Date"])
    panel = df_sub.set_index(["StockCode", "Date"]).reindex(idx).reset_index()
    
    # Explicit zero filling for demand and transaction counts
    panel["Daily_Quantity"] = panel["Daily_Quantity"].fillna(0.0)
    panel["Daily_Revenue"] = panel["Daily_Revenue"].fillna(0.0)
    panel["Transaction_Count"] = panel["Transaction_Count"].fillna(0)
    
    # Forward-fill unit prices within each SKU (strictly forward-fill only to eliminate backward leakage)
    panel["Avg_UnitPrice"] = panel.groupby("StockCode")["Avg_UnitPrice"].ffill().fillna(0.0)
    panel = panel.sort_values(by=["StockCode", "Date"]).reset_index(drop=True)
    
    # Attach segment label
    high_set = set(high_freq["StockCode"]) if include_segments else set()
    med_set = set(med_freq["StockCode"]) if include_segments else set()
    
    def assign_segment(sku):
        if sku in high_set:
            return "High_Volume"
        elif sku in med_set:
            return "Medium_Volume"
        return "Intermittent_Low"
        
    panel["Demand_Segment"] = panel["StockCode"].apply(assign_segment)
    return panel


def generate_benchmark_panel(
    dataset_type: str = "m5",
    n_entities: int = 20,
    n_periods: int = 180,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates deterministic, mathematically structured benchmark panels for verification
    and smoke tests when external gigabyte archives are absent.
    
    Profiles:
    - 'm5': Intermittent zero-inflated Poisson-Gamma grocery demand (~60% zeros).
    - 'online_retail': High-variance wholesale intermittent demand with long-tail batch spikes.
    """
    rng = np.random.RandomState(random_seed)
    records = []
    
    dates = pd.date_range("2022-01-01", periods=n_periods, freq="D")
    
    if dataset_type == "m5":
        for i in range(1, n_entities + 1):
            entity_id = f"FOODS_3_{i:03d}_CA_1"
            base_rate = rng.uniform(0.5, 3.5)
            price = round(rng.uniform(1.50, 8.99), 2)
            
            # Zero-inflation probability ~60%
            for d_idx, dt in enumerate(dates):
                # Day of week seasonality
                wday_factor = 1.3 if dt.dayofweek in [5, 6] else 0.9
                is_zero = rng.uniform(0, 1) < 0.60
                qty = 0.0 if is_zero else float(rng.poisson(lam=base_rate * wday_factor))
                records.append({
                    "id": entity_id,
                    "date": dt,
                    "Sales": qty,
                    "sell_price": price
                })
        df = pd.DataFrame(records)
        return df
        
    elif dataset_type == "online_retail":
        for i in range(1, n_entities + 1):
            sku = f"SKU_{i:04d}"
            base_price = round(rng.uniform(2.5, 45.0), 2)
            
            for dt in dates:
                # Intermittent order probability
                has_order = rng.uniform(0, 1) < 0.35
                if has_order:
                    # Lumpy lognormal wholesale quantity
                    qty = float(np.round(rng.lognormal(mean=2.2, sigma=1.2)))
                else:
                    qty = 0.0
                records.append({
                    "StockCode": sku,
                    "Date": dt,
                    "Daily_Quantity": qty,
                    "Avg_UnitPrice": base_price
                })
        df = pd.DataFrame(records)
        return df
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
