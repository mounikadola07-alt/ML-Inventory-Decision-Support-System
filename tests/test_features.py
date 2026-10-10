"""
tests/test_features.py - Unit tests for feature engineering and panel construction
"""

import numpy as np
import pandas as pd
import pytest
from src.features import (
    add_cyclical_calendar_features,
    add_leak_free_lags,
    add_leak_free_rolling,
    clean_prices_forward_fill_only,
    build_continuous_daily_panel
)


def test_cyclical_calendar_features():
    df = pd.DataFrame({"Date": pd.date_range("2023-01-01", periods=10, freq="D")})
    res = add_cyclical_calendar_features(df, "Date")
    assert "sin_month" in res.columns
    assert "cos_month" in res.columns
    assert "sin_week" in res.columns
    assert "sin_wday" in res.columns
    assert (res["sin_month"] >= -1.0).all() and (res["sin_month"] <= 1.0).all()


def test_leak_free_lags_and_rolling_strict_shift():
    df = pd.DataFrame({
        "Store": [1, 1, 1, 1, 1],
        "Date": pd.date_range("2023-01-01", periods=5, freq="W"),
        "Sales": [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    
    # 1. Test Lags
    res = add_leak_free_lags(df, "Store", "Date", "Sales", lags=[1, 2])
    assert pd.isna(res.loc[0, "Sales_Lag1"])
    assert res.loc[1, "Sales_Lag1"] == 10.0
    assert res.loc[2, "Sales_Lag1"] == 20.0
    assert pd.isna(res.loc[1, "Sales_Lag2"])
    assert res.loc[2, "Sales_Lag2"] == 10.0

    # 2. Test Rolling shift(1) independence from current Sales
    res_roll = add_leak_free_rolling(df, "Store", "Date", "Sales", windows=[2])
    # At t=2 (Sales=30), rollmean2 of past 2 periods (10, 20) should be 15.0
    assert res_roll.loc[2, "Sales_RollMean2"] == 15.0
    
    # Modify Sales at t=2 to 999.0 and recompute - t=2 RollMean must STILL be 15.0!
    df_mod = df.copy()
    df_mod.loc[2, "Sales"] = 999.0
    res_mod = add_leak_free_rolling(df_mod, "Store", "Date", "Sales", windows=[2])
    assert res_mod.loc[2, "Sales_RollMean2"] == 15.0, "Leakage detected: current target leaked into RollMean"


def test_no_backward_price_leakage():
    # Store 1 has missing prices on day 1 and 2, price introduces on day 3
    df = pd.DataFrame({
        "Store": [1, 1, 1, 1],
        "Date": pd.date_range("2023-01-01", periods=4, freq="D"),
        "Price": [np.nan, np.nan, 9.99, np.nan]
    })
    cleaned = clean_prices_forward_fill_only(df, ["Store"], "Date", "Price")
    # Day 1 and Day 2 must NOT know about 9.99! Must be 0.0
    assert cleaned.loc[0, "Price"] == 0.0
    assert cleaned.loc[1, "Price"] == 0.0
    # Day 3 has 9.99, Day 4 forward fills 9.99
    assert cleaned.loc[2, "Price"] == 9.99
    assert cleaned.loc[3, "Price"] == 9.99


def test_continuous_daily_panel_creation():
    # SKU A sold on Jan 1 and Jan 4 only
    raw = pd.DataFrame({
        "SKU": ["A", "A"],
        "Date": ["2023-01-01", "2023-01-04"],
        "Quantity": [5, 10]
    })
    panel = build_continuous_daily_panel(raw, "SKU", "Date", "Quantity")
    # Must have 4 days: Jan 1, 2, 3, 4
    assert len(panel) == 4
    assert (panel["SKU"] == "A").all()
    quantities = panel["Quantity"].tolist()
    assert quantities == [5.0, 0.0, 0.0, 10.0], f"Zero demand days not correctly filled: {quantities}"


def test_demand_momentum_strict_shift():
    from src.features import add_demand_momentum
    df = pd.DataFrame({
        "Store": [1, 1, 1, 1],
        "Date": pd.date_range("2023-01-01", periods=4, freq="W"),
        "Sales": [10.0, 20.0, 30.0, 40.0]
    })
    res = add_demand_momentum(df, "Store", "Date", "Sales", base_window=2)
    assert "Sales_Momentum2" in res.columns
    # Row 0 has no past data -> fillna 1.0
    assert res.loc[0, "Sales_Momentum2"] == 1.0
    # Current period's Sales must not leak
    df_mod = df.copy()
    df_mod.loc[3, "Sales"] = 9999.0
    res_mod = add_demand_momentum(df_mod, "Store", "Date", "Sales", base_window=2)
    assert res_mod.loc[3, "Sales_Momentum2"] == res.loc[3, "Sales_Momentum2"]

