"""
tests/test_data_loader.py - Unit tests for data loading, preprocessing, and benchmark panel generation
"""

import os
import numpy as np
import pandas as pd
import pytest
from src.data_loader import (
    load_walmart_sales,
    load_online_retail,
    get_online_retail_continuous_panel,
    generate_benchmark_panel
)


def test_get_online_retail_continuous_panel_no_bfill_leakage():
    """
    Verifies that get_online_retail_continuous_panel strictly uses forward-fill only,
    preventing introductory transaction prices from leaking backward into pre-introduction dates.
    """
    # Create sample daily data where SKU_LATE is only active starting on day 3
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df_sample = pd.DataFrame([
        # SKU_EARLY is active on day 1 and day 4
        {"Date": dates[0], "StockCode": "SKU_EARLY", "Daily_Quantity": 10.0, "Daily_Revenue": 50.0, "Avg_UnitPrice": 5.0, "Transaction_Count": 1},
        {"Date": dates[3], "StockCode": "SKU_EARLY", "Daily_Quantity": 15.0, "Daily_Revenue": 90.0, "Avg_UnitPrice": 6.0, "Transaction_Count": 1},
        # SKU_LATE is introduced on day 3 at price 25.0, and has another transaction on day 5 at 30.0
        {"Date": dates[2], "StockCode": "SKU_LATE", "Daily_Quantity": 20.0, "Daily_Revenue": 500.0, "Avg_UnitPrice": 25.0, "Transaction_Count": 1},
        {"Date": dates[4], "StockCode": "SKU_LATE", "Daily_Quantity": 5.0, "Daily_Revenue": 150.0, "Avg_UnitPrice": 30.0, "Transaction_Count": 1},
    ])
    
    panel = get_online_retail_continuous_panel(df_sample, n_top_skus=2, include_segments=False)
    
    sku_late_panel = panel[panel["StockCode"] == "SKU_LATE"].sort_values("Date").reset_index(drop=True)
    assert len(sku_late_panel) == 5
    
    # Pre-introduction dates (Day 1 and Day 2): price must be 0.0, NEVER backward filled 25.0!
    assert sku_late_panel.loc[0, "Date"] == dates[0]
    assert sku_late_panel.loc[0, "Avg_UnitPrice"] == 0.0
    assert sku_late_panel.loc[0, "Daily_Quantity"] == 0.0

    assert sku_late_panel.loc[1, "Date"] == dates[1]
    assert sku_late_panel.loc[1, "Avg_UnitPrice"] == 0.0
    assert sku_late_panel.loc[1, "Daily_Quantity"] == 0.0

    # Introduction date (Day 3): price is 25.0
    assert sku_late_panel.loc[2, "Date"] == dates[2]
    assert sku_late_panel.loc[2, "Avg_UnitPrice"] == 25.0
    assert sku_late_panel.loc[2, "Daily_Quantity"] == 20.0

    # Inactive post-introduction date (Day 4): forward-filled with Day 3 price (25.0), zero demand
    assert sku_late_panel.loc[3, "Date"] == dates[3]
    assert sku_late_panel.loc[3, "Avg_UnitPrice"] == 25.0
    assert sku_late_panel.loc[3, "Daily_Quantity"] == 0.0

    # Subsequent transaction date (Day 5): updated price (30.0)
    assert sku_late_panel.loc[4, "Date"] == dates[4]
    assert sku_late_panel.loc[4, "Avg_UnitPrice"] == 30.0
    assert sku_late_panel.loc[4, "Daily_Quantity"] == 5.0


def test_get_online_retail_continuous_panel_cartesian_continuity():
    """
    Verifies that get_online_retail_continuous_panel produces a balanced Cartesian grid
    with explicit zero filling for demand and revenue.
    """
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df_sample = pd.DataFrame([
        {"Date": dates[0], "StockCode": "SKU_A", "Daily_Quantity": 5.0, "Daily_Revenue": 20.0, "Avg_UnitPrice": 4.0, "Transaction_Count": 1},
        {"Date": dates[9], "StockCode": "SKU_A", "Daily_Quantity": 8.0, "Daily_Revenue": 32.0, "Avg_UnitPrice": 4.0, "Transaction_Count": 1},
        {"Date": dates[1], "StockCode": "SKU_B", "Daily_Quantity": 2.0, "Daily_Revenue": 10.0, "Avg_UnitPrice": 5.0, "Transaction_Count": 1},
        {"Date": dates[8], "StockCode": "SKU_B", "Daily_Quantity": 6.0, "Daily_Revenue": 30.0, "Avg_UnitPrice": 5.0, "Transaction_Count": 1},
    ])
    
    panel = get_online_retail_continuous_panel(df_sample, n_top_skus=2, include_segments=False)
    # 2 SKUs x 10 days = 20 rows
    assert len(panel) == 20
    assert set(panel["StockCode"].unique()) == {"SKU_A", "SKU_B"}
    assert panel["Daily_Quantity"].isna().sum() == 0
    assert panel["Daily_Revenue"].isna().sum() == 0
    assert panel["Avg_UnitPrice"].isna().sum() == 0


def test_generate_benchmark_panel_m5():
    """
    Verifies properties of the synthetic intermittent M5 benchmark generator.
    """
    df_m5 = generate_benchmark_panel("m5", n_entities=10, n_periods=60, random_seed=42)
    assert len(df_m5) == 10 * 60
    assert set(df_m5.columns) == {"id", "date", "Sales", "sell_price"}
    assert df_m5["Sales"].min() >= 0.0
    assert df_m5["sell_price"].min() > 0.0
    # Verified intermittent demand: zero fraction should be ~60%
    zero_frac = (df_m5["Sales"] == 0.0).mean()
    assert 0.45 <= zero_frac <= 0.75, f"Unexpected zero fraction: {zero_frac:.2%}"


def test_generate_benchmark_panel_online_retail():
    """
    Verifies properties of the synthetic online retail benchmark generator.
    """
    df_retail = generate_benchmark_panel("online_retail", n_entities=8, n_periods=50, random_seed=42)
    assert len(df_retail) == 8 * 50
    assert set(df_retail.columns) == {"StockCode", "Date", "Daily_Quantity", "Avg_UnitPrice"}
    assert df_retail["Daily_Quantity"].min() >= 0.0
    assert df_retail["Avg_UnitPrice"].min() > 0.0


def test_generate_benchmark_panel_invalid_type():
    """
    Verifies that unknown dataset_type raises ValueError.
    """
    with pytest.raises(ValueError, match="Unknown dataset_type"):
        generate_benchmark_panel("unknown_regime")


def test_load_walmart_sales_structure():
    """
    Verifies Walmart dataset ingestion if file exists on disk.
    """
    if os.path.exists("data/Walmart_Sales.csv") or os.path.exists("rawData.zip"):
        df = load_walmart_sales()
        assert "Store" in df.columns
        assert "Date" in df.columns
        assert "Weekly_Sales" in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df["Date"])
        # Verify strict chronological sorting per store
        for store_id, grp in df.groupby("Store"):
            assert grp["Date"].is_monotonic_increasing


def test_load_online_retail_structure():
    """
    Verifies Online Retail ingestion and caching if data exists on disk.
    """
    if os.path.exists("data/Online_Retail_daily.csv") or os.path.exists("data/Online Retail.xlsx"):
        df_retail = load_online_retail()
        assert "Date" in df_retail.columns
        assert "StockCode" in df_retail.columns
        assert "Daily_Quantity" in df_retail.columns
        assert "Daily_Revenue" in df_retail.columns
        assert "Avg_UnitPrice" in df_retail.columns
        assert pd.api.types.is_datetime64_any_dtype(df_retail["Date"])
        assert len(df_retail) > 0
        assert (df_retail["Daily_Quantity"] > 0).all()
        assert (df_retail["Avg_UnitPrice"] > 0).all()

