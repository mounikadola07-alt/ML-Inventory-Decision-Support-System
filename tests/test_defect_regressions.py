"""
tests/test_defect_regressions.py - Regression tests preventing recurrence of audited defects
"""

import numpy as np
import pandas as pd
import pytest
from src.inventory import calculate_eoq, calculate_safety_stock
from src.features import clean_prices_forward_fill_only, build_continuous_daily_panel
from src.metrics import calc_bias_percentage


def test_regression_crit01_eoq_dimensional_scaling():
    """
    CRIT-01: In prior notebook cell 97, H was set to 0.25 * d_hat.
    This caused d_hat to cancel out, resulting in a ridiculous $1,442 batch size for a $1M/wk store.
    With proper carrying cost rate i=0.20 ($/$/yr), EOQ must scale with sqrt(annual_demand).
    """
    weekly_demand = 1_000_000.0  # $1M/week
    annual_demand = weekly_demand * 52.0
    setup_cost = 5000.0
    holding_rate = 0.20  # 20% annual
    
    # Correct calculation:
    correct_eoq = calculate_eoq(annual_demand, setup_cost, holding_rate)
    assert correct_eoq > 1_000_000.0, f"EOQ {correct_eoq} is unrealistically small for $52M annual sales"
    
    # Buggy calculation replication check:
    buggy_H = 0.25 * weekly_demand
    buggy_eoq = np.sqrt((2 * annual_demand * setup_cost) / buggy_H)
    # The buggy EOQ yields approximately 1442.22
    assert pytest.approx(buggy_eoq, rel=1e-2) == 1442.22
    # Verify our implemented function NEVER produces the buggy cancellation
    assert correct_eoq != buggy_eoq


def test_regression_crit02_continuous_panel_zero_inflation():
    """
    CRIT-02: Prior preprocessing omitted inactive dates, causing lags to span
    arbitrary calendar intervals rather than 1 day.
    Continuous daily panel must explicitly preserve calendar continuity.
    """
    tx_data = pd.DataFrame({
        "StockCode": ["SKU1", "SKU1"],
        "InvoiceDate": ["2023-01-01", "2023-01-05"],
        "Quantity": [10.0, 20.0]
    })
    panel = build_continuous_daily_panel(tx_data, "StockCode", "InvoiceDate", "Quantity")
    assert len(panel) == 5
    # The intervening days (Jan 2, Jan 3, Jan 4) must be strictly 0
    assert panel.loc[panel["InvoiceDate"] == "2023-01-02", "Quantity"].values[0] == 0.0
    assert panel.loc[panel["InvoiceDate"] == "2023-01-03", "Quantity"].values[0] == 0.0
    assert panel.loc[panel["InvoiceDate"] == "2023-01-04", "Quantity"].values[0] == 0.0


def test_regression_crit03_no_bfill_price_leakage():
    """
    CRIT-03: bfill() leaked future launch prices backwards into pre-release periods.
    """
    df = pd.DataFrame({
        "item_id": ["ITEM1", "ITEM1", "ITEM1"],
        "date": pd.date_range("2021-01-01", periods=3, freq="D"),
        "sell_price": [np.nan, np.nan, 4.99]
    })
    res = clean_prices_forward_fill_only(df, ["item_id"], "date", "sell_price")
    # Must not leak 4.99 backwards!
    assert res.loc[0, "sell_price"] == 0.0
    assert res.loc[1, "sell_price"] == 0.0
    assert res.loc[2, "sell_price"] == 4.99


def test_regression_high01_safety_stock_uses_sigma_error():
    """
    HIGH-01: Using raw demand variance sigma_d over-buffers safety stock.
    Using residual forecast standard error sigma_e produces efficient buffers.
    """
    sigma_demand = 100.0  # Total demand variance including predictable seasonals
    sigma_error = 25.0    # Forecast residual variance after machine learning model
    lead_time = 4.0
    
    ss_demand_std = calculate_safety_stock(sigma_demand, lead_time, service_level=0.95)
    ss_error_std = calculate_safety_stock(sigma_error, lead_time, service_level=0.95)
    
    # Safety stock sized from residual error should be significantly lower (here 4x lower)
    assert ss_error_std < ss_demand_std
    assert pytest.approx(ss_demand_std / ss_error_std, rel=1e-3) == 4.0


def test_regression_high04_log1p_negative_bias_verification():
    """
    HIGH-04: Log1p transformation induces negative bias on skewed demand due to Jensen's Inequality.
    """
    rng = np.random.RandomState(42)
    # Lognormal skewed demand
    true_demand = np.round(rng.lognormal(mean=2.0, sigma=1.0, size=500))
    log_demand = np.log1p(true_demand)
    
    # Perfect model in log space predicts mean of log_demand
    pred_log = np.full_like(log_demand, np.mean(log_demand))
    pred_raw = np.expm1(pred_log)
    
    bias_pct = calc_bias_percentage(true_demand, pred_raw)
    # By Jensen's inequality: E[log(1+Y)] < log(1 + E[Y]), expm1 will underestimate the mean
    assert bias_pct < -10.0, f"Expected significant negative bias, got {bias_pct:.2f}%"


def test_regression_phase_d_bias_undefined_no_silent_zero():
    """
    Phase D Requirement: Never silently substitute 0.0 for an undefined metric.
    Prior implementation returned 0.0 when sum(actual) == 0, falsely claiming 0% bias!
    """
    zeros_actual = np.zeros(20)
    nonzero_preds = np.ones(20) * 5.0
    bias_res = calc_bias_percentage(zeros_actual, nonzero_preds)
    assert np.isnan(bias_res), "Undefined bias must yield NaN, never silent 0.0"


def test_regression_recent_average_baseline_forecaster():
    """
    Phase B Requirement: Must compare against a simple recent-average baseline.
    """
    from src.models import RecentAverageModel
    df = pd.DataFrame({
        "Weekly_Sales_RollMean4": [100.0, 150.0, 200.0]
    })
    model = RecentAverageModel("Weekly_Sales_RollMean4")
    model.fit(df)
    preds = model.predict(df)
    assert np.array_equal(preds, np.array([100.0, 150.0, 200.0]))


def test_regression_online_retail_real_panel_segmentation():
    """
    Phase B Dataset 2: SKU segmentation across High, Medium, and Intermittent demand.
    """
    from src.data_loader import load_online_retail, get_online_retail_continuous_panel
    df_daily = load_online_retail()
    panel = get_online_retail_continuous_panel(df_daily, n_top_skus=15, include_segments=True)
    assert "Demand_Segment" in panel.columns
    assert set(panel["Demand_Segment"].unique()) == {"High_Volume", "Medium_Volume", "Intermittent_Low"}
    # Continuity assertion: every SKU must have the exact same number of days
    days_per_sku = panel.groupby("StockCode")["Date"].count()
    assert days_per_sku.nunique() == 1, "Calendar panel must be continuous and identical in length across all SKUs"


def test_regression_data_loader_price_leakage_eliminated():
    """
    CRIT-03 Extension: Verify that src/data_loader.py get_online_retail_continuous_panel
    does NOT leak future introductory unit prices backwards into pre-introduction dates.
    Prior bug at line 155 used .ffill().bfill().fillna(0.0), which leaked introductory prices backwards.
    """
    from src.data_loader import get_online_retail_continuous_panel
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    df_tx = pd.DataFrame([
        # Only sold on Day 3 at price 49.99
        {"Date": dates[2], "StockCode": "NEW_SKU", "Daily_Quantity": 5.0, "Daily_Revenue": 249.95, "Avg_UnitPrice": 49.99, "Transaction_Count": 1},
        # Another SKU sold on Day 1 and Day 4
        {"Date": dates[0], "StockCode": "OLD_SKU", "Daily_Quantity": 10.0, "Daily_Revenue": 100.0, "Avg_UnitPrice": 10.0, "Transaction_Count": 1},
        {"Date": dates[3], "StockCode": "OLD_SKU", "Daily_Quantity": 12.0, "Daily_Revenue": 120.0, "Avg_UnitPrice": 10.0, "Transaction_Count": 1},
    ])

    panel = get_online_retail_continuous_panel(df_tx, n_top_skus=2, include_segments=False)
    new_sku_rows = panel[panel["StockCode"] == "NEW_SKU"].sort_values("Date").reset_index(drop=True)

    # Day 1 and Day 2 must have Avg_UnitPrice == 0.0 (NO backward leak of 49.99!)
    assert new_sku_rows.loc[0, "Avg_UnitPrice"] == 0.0, "Day 1 price leaked from future!"
    assert new_sku_rows.loc[1, "Avg_UnitPrice"] == 0.0, "Day 2 price leaked from future!"
    # Day 3 must have introductory price
    assert new_sku_rows.loc[2, "Avg_UnitPrice"] == 49.99
    # Day 4 forward-fills Day 3 price
    assert new_sku_rows.loc[3, "Avg_UnitPrice"] == 49.99


