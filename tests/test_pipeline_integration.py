"""
tests/test_pipeline_integration.py - End-to-end integration test of the decision support pipeline
"""

import numpy as np
import pandas as pd
import pytest
from src.data_loader import load_walmart_sales, generate_benchmark_panel
from src.features import (
    add_cyclical_calendar_features,
    add_leak_free_lags,
    add_leak_free_rolling
)
from src.validation import split_locked_final_holdout, ChronologicalTimeSeriesSplit, run_walk_forward_cv
from src.models import (
    Lag1NaiveModel,
    SeasonalNaiveModel,
    create_xgboost_regressor,
    create_lightgbm_regressor,
    CrostonSBAModel
)
from src.metrics import evaluate_all_metrics
from src.inventory import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_eoq,
    InventorySimulation
)


def test_walmart_end_to_end_pipeline():
    # 1. Load real Walmart dataset
    df = load_walmart_sales("data/Walmart_Sales.csv")
    assert len(df) == 6435
    assert df["Store"].nunique() == 45

    # 2. Add features
    df = add_cyclical_calendar_features(df, "Date")
    df = add_leak_free_lags(df, "Store", "Date", "Weekly_Sales", lags=[1, 2, 4])
    df = add_leak_free_rolling(df, "Store", "Date", "Weekly_Sales", windows=[4, 12])
    
    # Drop warm-up rows
    df_clean = df.dropna().reset_index(drop=True)
    assert len(df_clean) > 5500

    # 3. Locked Holdout Split (last 26 weeks)
    dev_df, holdout_df = split_locked_final_holdout(df_clean, "Date", holdout_periods=26)
    assert dev_df["Date"].max() < holdout_df["Date"].min()

    # 4. Fit Baseline and XGBoost
    feature_cols = [
        "Store", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "sin_month", "cos_month", "sin_week", "cos_week",
        "Weekly_Sales_Lag1", "Weekly_Sales_Lag2", "Weekly_Sales_Lag4",
        "Weekly_Sales_RollMean4", "Weekly_Sales_RollMean12"
    ]
    
    X_dev, y_dev = dev_df[feature_cols], dev_df["Weekly_Sales"]
    X_holdout, y_holdout = holdout_df[feature_cols], holdout_df["Weekly_Sales"]
    
    # Naive Baseline
    naive = Lag1NaiveModel("Weekly_Sales_Lag1")
    naive_preds = naive.predict(X_holdout)
    m_naive = evaluate_all_metrics(y_holdout, naive_preds, "Lag1_Naive")
    
    # XGBoost
    xgb = create_xgboost_regressor(n_estimators=80, max_depth=6, learning_rate=0.1)
    xgb.fit(X_dev, y_dev)
    xgb_preds = xgb.predict(X_holdout)
    m_xgb = evaluate_all_metrics(y_holdout, xgb_preds, "XGBoost")
    
    # Assert ML outperforms Naive in MAE and RMSE
    assert m_xgb["MAE"] < m_naive["MAE"]
    assert m_xgb["RMSE"] < m_naive["RMSE"]
    assert abs(m_xgb["Bias_Percentage"]) < 2.0  # Safe bias bound

    # 5. Inventory Derivation & Simulation on Store 1
    s1_mask = holdout_df["Store"] == 1
    s1_actual = y_holdout[s1_mask].to_numpy()
    s1_preds = xgb_preds[s1_mask]
    
    sigma_e = float(np.sqrt(np.mean((s1_actual - s1_preds) ** 2)))
    d_hat = float(np.mean(s1_preds))
    
    # Lead time L=2 weeks
    ss = calculate_safety_stock(sigma_e, lead_time=2, service_level=0.95)
    rop = calculate_reorder_point(d_hat * 2, ss)
    annual_demand = d_hat * 52
    eoq = calculate_eoq(annual_demand=annual_demand, order_cost=2500.0, holding_cost_annual=0.20)
    
    assert ss > 0
    assert rop > ss
    assert eoq > 100000.0  # Sanity check: batch size for $1M/wk store must be >$100k, not $1,442!

    # 6. Chronological Inventory Simulation
    sim = InventorySimulation(lead_time=2, unit_cost=1.0, holding_rate_annual=0.20, order_setup_cost=2500.0, days_per_year=52)
    sim_res = sim.simulate(s1_actual, reorder_point=rop, order_quantity=eoq)
    assert sim_res["Fill_Rate_Pct"] > 90.0
    assert sim_res["Total_Cost"] > 0


def test_benchmark_panel_croston_and_tweedie():
    # Test intermittent grocery panel
    df_m5 = generate_benchmark_panel("m5", n_entities=5, n_periods=100)
    assert (df_m5["Sales"] == 0).mean() > 0.40  # Verified zero inflation
    
    # Fit Croston SBA
    croston = CrostonSBAModel()
    croston.fit(df_m5[["sell_price"]], df_m5["Sales"])
    preds = croston.predict(df_m5[["sell_price"]])
    assert len(preds) == len(df_m5)
    assert preds[0] > 0
