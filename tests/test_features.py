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


def test_rolling_cv_strict_shift():
    from src.features import add_rolling_coefficient_of_variation
    df = pd.DataFrame({
        "Store": [1, 1, 1, 1],
        "Date": pd.date_range("2023-01-01", periods=4, freq="W"),
        "Sales": [10.0, 20.0, 30.0, 40.0]
    })
    res = add_rolling_coefficient_of_variation(df, "Store", "Date", "Sales", windows=[2])
    assert "Sales_RollCV2" in res.columns
    # Mutating current Sales at t=3 must NOT alter t=3 feature value
    df_mod = df.copy()
    df_mod.loc[3, "Sales"] = 8888.0
    res_mod = add_rolling_coefficient_of_variation(df_mod, "Store", "Date", "Sales", windows=[2])
    assert res_mod.loc[3, "Sales_RollCV2"] == res.loc[3, "Sales_RollCV2"]


def test_demand_acceleration_and_jerk_strict_shift():
    from src.features import add_demand_acceleration_and_jerk
    df = pd.DataFrame({
        "Store": [1, 1, 1, 1, 1],
        "Date": pd.date_range("2023-01-01", periods=5, freq="W"),
        "Sales": [10.0, 20.0, 35.0, 55.0, 80.0]
    })
    res = add_demand_acceleration_and_jerk(df, "Store", "Date", "Sales")
    assert "Sales_Velocity" in res.columns
    assert "Sales_Acceleration" in res.columns
    assert "Sales_Jerk" in res.columns
    # Row 4 (Sales=80):
    # L1=55, L2=35, L3=20, L4=10
    # Velocity = 55 - 35 = 20
    # Accel = 55 - 2*35 + 20 = 5
    # Jerk = 55 - 3*35 + 3*20 - 10 = 55 - 105 + 60 - 10 = 0
    assert res.loc[4, "Sales_Velocity"] == 20.0
    assert res.loc[4, "Sales_Acceleration"] == 5.0
    assert res.loc[4, "Sales_Jerk"] == 0.0
    # Mutating current period row 4 Sales must NOT change these features
    df_mod = df.copy()
    df_mod.loc[4, "Sales"] = 9999.0
    res_mod = add_demand_acceleration_and_jerk(df_mod, "Store", "Date", "Sales")
    assert res_mod.loc[4, "Sales_Velocity"] == 20.0
    assert res_mod.loc[4, "Sales_Acceleration"] == 5.0
    assert res_mod.loc[4, "Sales_Jerk"] == 0.0


def test_multi_window_ewma_ratios_strict_shift():
    from src.features import add_multi_window_ewma_ratios
    df = pd.DataFrame({
        "Store": [1, 1, 1, 1],
        "Date": pd.date_range("2023-01-01", periods=4, freq="W"),
        "Sales": [10.0, 20.0, 30.0, 40.0]
    })
    res = add_multi_window_ewma_ratios(df, "Store", "Date", "Sales", spans=[3, 7])
    assert "Sales_EWMA3" in res.columns
    assert "Sales_EWMA7" in res.columns
    assert "Sales_EWMARatio_3_7" in res.columns
    # Strict shift: row 3 cannot know about row 3's Sales
    df_mod = df.copy()
    df_mod.loc[3, "Sales"] = 7777.0
    res_mod = add_multi_window_ewma_ratios(df_mod, "Store", "Date", "Sales", spans=[3, 7])
    assert res_mod.loc[3, "Sales_EWMARatio_3_7"] == res.loc[3, "Sales_EWMARatio_3_7"]


def test_leak_free_expanding_stats_strict_shift():
    from src.features import add_leak_free_expanding_stats
    df = pd.DataFrame({
        "Store": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "Date": pd.date_range("2023-01-01", periods=10, freq="W"),
        "Sales": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    })
    res = add_leak_free_expanding_stats(df, "Store", "Date", "Sales", min_periods=3)
    assert "Store_Hist_Expanding_Mean" in res.columns
    assert "Store_Hist_Expanding_Std" in res.columns
    # Row 9 (Sales=100): Expanding mean must only be computed over rows 0..8 (values 10..90)
    expected_mean = np.mean([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0])
    assert pytest.approx(res.loc[9, "Store_Hist_Expanding_Mean"]) == expected_mean
    # Altering row 9 Sales must NOT affect row 9 expanding stats
    df_mod = df.copy()
    df_mod.loc[9, "Sales"] = 99999.0
    res_mod = add_leak_free_expanding_stats(df_mod, "Store", "Date", "Sales", min_periods=3)
    assert res_mod.loc[9, "Store_Hist_Expanding_Mean"] == res.loc[9, "Store_Hist_Expanding_Mean"]


