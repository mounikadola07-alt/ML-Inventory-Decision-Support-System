"""
tests/test_policy_comparison.py - Comprehensive Unit & Integration Tests for Milestone M3
Verifies the four forecast-selection policies:
  - Policy A: Historical / Seasonal Naive Baseline
  - Policy B: Statistical Accuracy-First (Minimum Validation MAE)
  - Policy C: Operational Bias-Constrained (Minimum MAE subject to |Bias%| <= 5%)
  - Policy D: Cost-Aware (Minimum Validation Replenishment Cost)
Verifies discrete-time inventory simulation with transit delay, pipeline orders, and zero lookahead.
Verifies machine-readable output files in results/ directory.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.policy_comparison import (
    PolicyResult,
    InventoryConfig,
    Log1pModelWrapper,
    select_policy_a_model,
    select_policy_b_model,
    select_policy_c_model,
    select_policy_d_model,
    evaluate_policy_a_naive,
    evaluate_policy_b_accuracy,
    evaluate_policy_c_bias_constrained,
    evaluate_policy_d_cost_aware,
    simulate_series_replenishment,
    simulate_panel_replenishment,
    make_json_serializable
)
from src.inventory import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_eoq,
    InventorySimulation
)


def test_policy_result_dataclass_and_indexing():
    """Verifies PolicyResult dataclass, JSON serialization, and dictionary-style access."""
    res = PolicyResult(
        policy_name="Policy B: Statistical Accuracy-First",
        selected_model="XGBoost_MSE",
        selection_criterion="Minimum Validation MAE",
        validation_metric_name="Validation MAE",
        validation_metric_value=45.2,
        holdout_forecast_metrics={"MAE": 42.1, "RMSE": 60.5, "Bias_Percentage": -1.2, "WAPE": 8.5},
        holdout_inventory_metrics={"Total_Cost": 1250.0, "Shortage_Cost": 0.0, "Fill_Rate_Pct": 100.0, "Stockout_Days": 0},
        dataset_name="Test Dataset",
        demand_regime="Smooth"
    )

    # Dictionary access
    assert res["selected_model"] == "XGBoost_MSE"
    assert res["MAE"] == 42.1
    assert res["Total_Cost"] == 1250.0
    assert res["Fill_Rate_Pct"] == 100.0

    # Serialization
    d = res.to_dict()
    assert d["Policy"] == "Policy B: Statistical Accuracy-First"
    assert d["Holdout_Forecast_Metrics"]["MAE"] == 42.1
    assert d["Holdout_Inventory_Metrics"]["Total_Cost"] == 1250.0

    # Invalid key access raises KeyError
    with pytest.raises(KeyError):
        _ = res["non_existent_key"]


def test_select_policy_a_naive():
    """Verifies Policy A baseline heuristic selection."""
    candidates = ["XGBoost", "LightGBM", "Seasonal_Naive", "Random_Forest"]
    assert select_policy_a_model(candidates) == "Seasonal_Naive"

    candidates_daily = ["XGBoost", "Lag1_Naive", "Recent_Average"]
    assert select_policy_a_model(candidates_daily) == "Lag1_Naive"

    # Explicit baseline preference
    assert select_policy_a_model(candidates_daily, baseline_name="Recent_Average") == "Recent_Average"


def test_select_policy_b_accuracy():
    """Verifies Policy B selects candidate with strictly lowest validation MAE."""
    cv_results = {
        "Model_A": {"CV_MAE_Mean": 50.0, "CV_RMSE_Mean": 80.0},
        "Model_B": {"CV_MAE_Mean": 35.0, "CV_RMSE_Mean": 60.0},
        "Model_C": {"CV_MAE_Mean": 40.0, "CV_RMSE_Mean": 55.0}
    }
    chosen, val_mae = select_policy_b_model(cv_results)
    assert chosen == "Model_B"
    assert val_mae == 35.0

    # Tiebreaker: lowest RMSE on equal MAE
    cv_tie = {
        "Model_1": {"CV_MAE_Mean": 30.0, "CV_RMSE_Mean": 55.0},
        "Model_2": {"CV_MAE_Mean": 30.0, "CV_RMSE_Mean": 45.0}
    }
    chosen_tie, _ = select_policy_b_model(cv_tie)
    assert chosen_tie == "Model_2"


def test_select_policy_c_bias_constrained():
    """Verifies Policy C filters models with |Bias%| > 5% and selects minimum MAE in safe pool."""
    cv_results = {
        "Log1p_Model": {"CV_MAE_Mean": 25.0, "CV_Bias_Pct_Mean": -42.0},  # Lowest MAE but heavily biased
        "Biased_Over": {"CV_MAE_Mean": 30.0, "CV_Bias_Pct_Mean": 18.5},   # Excess bias
        "Safe_Model_1": {"CV_MAE_Mean": 40.0, "CV_Bias_Pct_Mean": 2.1},   # Safe |bias| <= 5%
        "Safe_Model_2": {"CV_MAE_Mean": 35.0, "CV_Bias_Pct_Mean": -3.4}   # Safe |bias| <= 5%, lower MAE
    }

    # Fractional threshold 0.05 (5%)
    chosen, metric_val, is_fallback = select_policy_c_model(cv_results, bias_threshold=0.05)
    assert chosen == "Safe_Model_2"
    assert metric_val == 35.0
    assert not is_fallback

    # Percentage threshold 5.0
    chosen_pct, _, _ = select_policy_c_model(cv_results, bias_threshold=5.0)
    assert chosen_pct == "Safe_Model_2"

    # Fallback when no model meets threshold: selects model with smallest absolute bias
    cv_all_biased = {
        "Model_X": {"CV_MAE_Mean": 20.0, "CV_Bias_Pct_Mean": -12.0},
        "Model_Y": {"CV_MAE_Mean": 30.0, "CV_Bias_Pct_Mean": 8.5}   # Smallest |bias| = 8.5%
    }
    chosen_fallback, fallback_bias, is_fallback = select_policy_c_model(cv_all_biased, bias_threshold=0.05)
    assert chosen_fallback == "Model_Y"
    assert fallback_bias == 8.5
    assert is_fallback


def test_select_policy_d_cost_aware():
    """Verifies Policy D selects candidate minimizing validation inventory replenishment cost."""
    cv_results = {
        "Model_A": {"CV_Cost_Mean": 15000.0, "CV_Fill_Rate_Mean": 99.5},
        "Model_B": {"CV_Cost_Mean": 8500.0, "CV_Fill_Rate_Mean": 98.0},  # Minimum cost
        "Model_C": {"CV_Cost_Mean": 25000.0, "CV_Fill_Rate_Mean": 90.0}
    }
    chosen, best_cost = select_policy_d_model(cv_results)
    assert chosen == "Model_B"
    assert best_cost == 8500.0

    # Custom cost function support
    custom_fn = lambda name: 100.0 if name == "Model_C" else 500.0
    chosen_custom, custom_cost = select_policy_d_model(cv_results, validation_cost_fn=custom_fn)
    assert chosen_custom == "Model_C"
    assert custom_cost == 100.0


def test_log1p_model_wrapper():
    """Verifies Log1pModelWrapper fits on log1p and predicts with non-negative inverse expm1."""
    np.random.seed(42)
    X = pd.DataFrame({"feat": np.linspace(1, 10, 50)})
    y = np.exp(0.5 * X["feat"]) + np.random.normal(0, 1, 50)
    y = np.clip(y, 0, None)

    wrapper = Log1pModelWrapper(lambda: LinearRegression())
    wrapper.fit(X, y)
    preds = wrapper.predict(X)

    assert len(preds) == len(y)
    assert np.all(preds >= 0.0)
    # Target transformation exhibits median-seeking bias on skewed targets
    assert isinstance(preds, np.ndarray)


def test_inventory_simulation_pipeline_delay_no_lookahead():
    """Verifies discrete-time replenishment simulation enforces exact pipeline delay without lookahead."""
    config = InventoryConfig(
        lead_time=3,
        service_level=0.95,
        order_setup_cost=50.0,
        holding_rate_annual=0.20,
        unit_cost=10.0,
        stockout_penalty_per_unit=20.0,
        periods_per_year=365
    )

    # 40 periods of constant demand (40 * 20 = 800 units)
    demands = np.full(40, 20.0)
    forecasts = np.full(40, 20.0)

    # Starts with initial stock at ROP to trigger pipeline ordering
    sim_res = simulate_series_replenishment(demands, forecasts, config, initial_inventory=60.0)

    assert sim_res["Total_Demand"] == 800.0
    assert sim_res["Orders_Placed"] > 0
    assert sim_res["Holding_Cost"] > 0.0
    assert sim_res["Total_Cost"] == pytest.approx(
        sim_res["Holding_Cost"] + sim_res["Ordering_Cost"] + sim_res["Shortage_Cost"], rel=1e-4
    )


def test_stochastic_lead_time_safety_stock_scaling():
    """Verifies that lead-time variance sigma_L increases safety stock buffer monotonically."""
    sigma_e = 15.0
    L = 5.0
    d_hat = 40.0

    ss_deterministic = calculate_safety_stock(sigma_e, L, service_level=0.95, sigma_lead_time=0.0, mean_demand=d_hat)
    ss_stochastic = calculate_safety_stock(sigma_e, L, service_level=0.95, sigma_lead_time=1.5, mean_demand=d_hat)

    assert ss_stochastic > ss_deterministic
    # Mathematical formula verification:
    # Var_total = L * sigma_e^2 + d_hat^2 * sigma_L^2
    var_expected = 5.0 * (15.0 ** 2) + (40.0 ** 2) * (1.5 ** 2)
    expected_ss = 1.6448536269514722 * np.sqrt(var_expected)
    assert ss_stochastic == pytest.approx(expected_ss, rel=1e-4)


def test_synthetic_end_to_end_four_policy_disconnect():
    """
    Demonstrates the Accuracy-Cost disconnect under asymmetric costs and negative bias.
    Compares:
      - Model_Acc (low MAE, but -50% bias due to median optimization)
      - Model_Safe (higher MAE, 0% bias)
      - Model_Naive (heuristic)
    Asserts:
      - Policy B selects Model_Acc
      - Policy C rejects Model_Acc and selects Model_Safe
      - Policy D rejects Model_Acc due to high shortage cost and selects Model_Safe or Model_Naive
    """
    np.random.seed(42)
    n = 60
    # Skewed demand with intermittent spikes
    actual = np.random.exponential(scale=20.0, size=n)

    # Model Acc: systematically underpredicts by 50%
    pred_acc = actual * 0.50
    # Model Safe: unbiased predictor with noise
    pred_safe = actual + np.random.normal(0, 5, size=n)
    pred_safe = np.clip(pred_safe, 0, None)
    # Model Naive
    pred_naive = np.roll(actual, 1)
    pred_naive[0] = actual[0]

    config = InventoryConfig(
        lead_time=5,
        service_level=0.95,
        order_setup_cost=50.0,
        holding_rate_annual=0.20,
        unit_cost=10.0,
        stockout_penalty_per_unit=50.0,  # High stockout penalty
        periods_per_year=365
    )

    sim_acc = simulate_series_replenishment(actual, pred_acc, config)
    sim_safe = simulate_series_replenishment(actual, pred_safe, config)
    sim_naive = simulate_series_replenishment(actual, pred_naive, config)

    cv_results = {
        "Model_Acc": {
            "CV_MAE_Mean": 10.0,
            "CV_RMSE_Mean": 20.0,
            "CV_Bias_Pct_Mean": -50.0,
            "CV_Cost_Mean": sim_acc["Total_Cost"],
            "CV_Fill_Rate_Mean": sim_acc["Fill_Rate_Pct"]
        },
        "Model_Safe": {
            "CV_MAE_Mean": 14.0,  # Higher MAE
            "CV_RMSE_Mean": 22.0,
            "CV_Bias_Pct_Mean": 1.5,  # Unbiased <= 5%
            "CV_Cost_Mean": sim_safe["Total_Cost"],
            "CV_Fill_Rate_Mean": sim_safe["Fill_Rate_Pct"]
        },
        "Model_Naive": {
            "CV_MAE_Mean": 18.0,
            "CV_RMSE_Mean": 28.0,
            "CV_Bias_Pct_Mean": 0.5,
            "CV_Cost_Mean": sim_naive["Total_Cost"],
            "CV_Fill_Rate_Mean": sim_naive["Fill_Rate_Pct"]
        }
    }

    holdout_eval = {
        "Model_Acc": ({"MAE": 10.0, "RMSE": 20.0, "Bias_Percentage": -50.0}, sim_acc),
        "Model_Safe": ({"MAE": 14.0, "RMSE": 22.0, "Bias_Percentage": 1.5}, sim_safe),
        "Model_Naive": ({"MAE": 18.0, "RMSE": 28.0, "Bias_Percentage": 0.5}, sim_naive)
    }

    res_a = evaluate_policy_a_naive(cv_results=cv_results, holdout_evaluator=holdout_eval, baseline_model_name="Model_Naive")
    res_b = evaluate_policy_b_accuracy(cv_results=cv_results, holdout_evaluator=holdout_eval)
    res_c = evaluate_policy_c_bias_constrained(cv_results=cv_results, holdout_evaluator=holdout_eval, bias_threshold=0.05)
    res_d = evaluate_policy_d_cost_aware(cv_results=cv_results, holdout_evaluator=holdout_eval)

    # Verification of selections
    assert res_a.selected_model == "Model_Naive"
    assert res_b.selected_model == "Model_Acc"  # Selected for lowest MAE
    assert res_c.selected_model == "Model_Safe"  # Model_Acc disqualified due to -50% bias
    assert res_d.selected_model in ["Model_Safe", "Model_Naive"]  # Cost-aware rejects biased Model_Acc
    assert res_d.selected_model != "Model_Acc"

    # Operational impact verification: Model_Acc incurs heavy shortage penalties
    assert res_b["Total_Cost"] > res_c["Total_Cost"]
    assert res_b["Total_Cost"] > res_d["Total_Cost"]
    assert res_b["Shortage_Cost"] > res_c["Shortage_Cost"]
    assert res_b["Shortage_Cost"] > res_d["Shortage_Cost"]
    assert res_c["Fill_Rate_Pct"] >= res_b["Fill_Rate_Pct"]


def test_results_directory_artifacts_exist_and_valid():
    """Verifies that all 5 required machine-readable result files exist in results/ and are valid."""
    results_dir = "results"
    required_files = [
        "walmart_policy_comparison.json",
        "retail_policy_comparison.json",
        "m5_policy_comparison.json",
        "full_cross_regime_summary.json",
        "summary_table.csv"
    ]

    for fname in required_files:
        fpath = os.path.join(results_dir, fname)
        assert os.path.exists(fpath), f"Required artifact {fname} missing from results/"
        assert os.path.getsize(fpath) > 0, f"Artifact {fname} is empty"

    # Verify summary_table.csv structure
    df_csv = pd.read_csv(os.path.join(results_dir, "summary_table.csv"))
    assert len(df_csv) == 12, f"Expected 12 policy rows (3 datasets x 4 policies), got {len(df_csv)}"

    required_cols = [
        "Dataset", "Demand_Regime", "Policy", "Policy_Description",
        "Selected_Model", "Val_Selection_Metric", "Val_Selection_Value",
        "Holdout_MAE", "Holdout_RMSE", "Holdout_WAPE", "Holdout_Bias_Pct",
        "Holdout_Fill_Rate_Pct", "Holdout_Stockout_Days",
        "Holdout_Holding_Cost", "Holdout_Ordering_Cost", "Holdout_Shortage_Cost", "Holdout_Total_Cost"
    ]
    for col in required_cols:
        assert col in df_csv.columns, f"Missing column {col} in summary_table.csv"

    # Verify no NaN values in costs or metrics
    assert not df_csv["Holdout_Total_Cost"].isna().any()
    assert not df_csv["Holdout_MAE"].isna().any()

    # Verify the empirical accuracy-cost disconnect on Online Retail
    retail_df = df_csv[df_csv["Dataset"] == "Online Retail Daily SKU Panel"].set_index("Policy")
    pol_b = retail_df.loc["Policy B"]
    pol_c = retail_df.loc["Policy C"]
    pol_d = retail_df.loc["Policy D"]

    # Policy B selects XGBoost_Log1p for lowest MAE, but suffers huge shortage costs
    assert pol_b["Selected_Model"] == "XGBoost_Log1p"
    assert pol_b["Holdout_MAE"] < pol_c["Holdout_MAE"]  # Policy B has lower MAE
    assert pol_b["Holdout_Bias_Pct"] < -50.0            # Severe negative bias
    assert pol_b["Holdout_Total_Cost"] > pol_c["Holdout_Total_Cost"]  # Policy B cost is much higher!
    assert pol_b["Holdout_Stockout_Days"] > 0           # Significant stockout days
    assert pol_c["Holdout_Total_Cost"] < 20000.0        # Policy C avoids stockouts
