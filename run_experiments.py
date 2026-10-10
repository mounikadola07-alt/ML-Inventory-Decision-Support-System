"""
run_experiments.py - Reproducible End-to-End Walk-Forward Cross-Validation and Holdout Evaluation
Computes all metrics directly from model predictions and ground truth.
Includes both regression point forecasting and classification (ROC AUC, PR AUC, Accuracy) for stockout risk / demand surges.
Outputs machine-readable experiment_results.json.
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from src.data_loader import (
    load_walmart_sales,
    load_online_retail,
    get_online_retail_continuous_panel,
    generate_benchmark_panel
)
from src.features import (
    add_cyclical_calendar_features,
    add_leak_free_lags,
    add_leak_free_rolling,
    add_demand_momentum
)
from src.validation import (
    ChronologicalTimeSeriesSplit,
    split_locked_final_holdout,
    run_walk_forward_cv
)
from src.models import (
    Lag1NaiveModel,
    SeasonalNaiveModel,
    RecentAverageModel,
    CrostonSBAModel,
    create_random_forest,
    create_xgboost_regressor,
    create_lightgbm_regressor,
    create_random_forest_classifier,
    create_xgboost_classifier,
    create_lightgbm_classifier
)
from src.metrics import (
    evaluate_all_metrics,
    evaluate_classification_metrics,
    calc_mae,
    calc_rmse,
    calc_r2,
    calc_wape,
    calc_bias_percentage,
    calc_roc_auc,
    calc_accuracy,
    calc_f1
)
from src.inventory import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_eoq,
    InventorySimulation,
    run_inventory_sensitivity_analysis
)


def run_walmart_experiment() -> Dict[str, Any]:
    print(">>> [1/3] Running Walmart Weekly Store Sales Experiments...")
    df = load_walmart_sales("data/Walmart_Sales.csv")
    
    # Feature engineering with volatility and momentum
    df = add_cyclical_calendar_features(df, "Date")
    df = add_leak_free_lags(df, "Store", "Date", "Weekly_Sales", lags=[1, 2, 4, 52])
    df = add_leak_free_rolling(df, "Store", "Date", "Weekly_Sales", windows=[4, 12])
    df = add_demand_momentum(df, "Store", "Date", "Weekly_Sales", base_window=4)
    
    # Drop warm-up periods for 52-week seasonal lag
    df_clean = df.dropna().reset_index(drop=True)
    
    # 26 weeks locked holdout
    dev_df, holdout_df = split_locked_final_holdout(df_clean, "Date", holdout_periods=26)
    
    features = [
        "Store", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "sin_month", "cos_month", "sin_week", "cos_week",
        "Weekly_Sales_Lag1", "Weekly_Sales_Lag2", "Weekly_Sales_Lag4", "Weekly_Sales_Lag52",
        "Weekly_Sales_RollMean4", "Weekly_Sales_RollMean12",
        "Weekly_Sales_RollStd4", "Weekly_Sales_Momentum4"
    ]
    
    # 3-Fold Walk-Forward CV on dev set
    cv = ChronologicalTimeSeriesSplit(n_splits=3, horizon_periods=12, min_train_periods=24)
    cv_models = {
        "Lag1_Naive": lambda: Lag1NaiveModel("Weekly_Sales_Lag1"),
        "Seasonal_Naive": lambda: SeasonalNaiveModel("Weekly_Sales_Lag52"),
        "Recent_Average": lambda: RecentAverageModel("Weekly_Sales_RollMean4"),
        "Random_Forest": lambda: create_random_forest(n_estimators=120, max_depth=12),
        "XGBoost_MSE": lambda: create_xgboost_regressor(n_estimators=150, max_depth=4, learning_rate=0.03, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0),
        "LightGBM_MSE": lambda: create_lightgbm_regressor(n_estimators=150, num_leaves=20, learning_rate=0.03, subsample=0.85, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=2.0)
    }
    
    cv_results = {}
    for name, factory in cv_models.items():
        print(f"    Running 3-fold CV for Walmart: {name}")
        res = run_walk_forward_cv(dev_df, features, "Weekly_Sales", "Date", factory, cv, name)
        cv_results[name] = res
        
    # Locked Holdout Evaluation
    X_dev, y_dev = dev_df[features], dev_df["Weekly_Sales"]
    X_holdout, y_holdout = holdout_df[features], holdout_df["Weekly_Sales"]
    
    holdout_results = []
    trained_models = {}
    for name, factory in cv_models.items():
        m = factory()
        m.fit(X_dev, y_dev)
        preds = m.predict(X_holdout)
        metrics = evaluate_all_metrics(y_holdout, preds, name)
        holdout_results.append(metrics)
        trained_models[name] = (m, preds)

    # Store-Level Performance Breakdown & Error Distributions on Holdout
    store_results = []
    for store_id in sorted(holdout_df["Store"].unique()):
        mask = holdout_df["Store"] == store_id
        y_st = y_holdout[mask].to_numpy()
        pred_rf = trained_models["Random_Forest"][1][mask]
        pred_xgb = trained_models["XGBoost_MSE"][1][mask]
        pred_naive = trained_models["Lag1_Naive"][1][mask]
        
        store_results.append({
            "Store": int(store_id),
            "Mean_Sales": float(np.mean(y_st)),
            "RF_MAE": calc_mae(y_st, pred_rf),
            "RF_RMSE": calc_rmse(y_st, pred_rf),
            "RF_WAPE": calc_wape(y_st, pred_rf),
            "RF_Bias_Pct": calc_bias_percentage(y_st, pred_rf),
            "RF_R2": calc_r2(y_st, pred_rf),
            "XGB_MAE": calc_mae(y_st, pred_xgb),
            "Naive_MAE": calc_mae(y_st, pred_naive)
        })
    df_store = pd.DataFrame(store_results)
    store_summary = {
        "Total_Stores": int(len(df_store)),
        "RF_Store_MAE_Mean": float(df_store["RF_MAE"].mean()),
        "RF_Store_MAE_Median": float(df_store["RF_MAE"].median()),
        "RF_Store_MAE_Min": float(df_store["RF_MAE"].min()),
        "RF_Store_MAE_Max": float(df_store["RF_MAE"].max()),
        "RF_Store_R2_Mean": float(df_store["RF_R2"].mean()),
        "RF_Store_R2_Median": float(df_store["RF_R2"].median())
    }
        
    # Demand Surge / Stockout Risk Classification (75th Percentile Surge)
    print("    Running Walmart Demand Surge Classification (ROC AUC / Accuracy)...")
    store_p75 = dev_df.groupby("Store")["Weekly_Sales"].quantile(0.75).to_dict()
    y_dev_surge = (dev_df["Weekly_Sales"] > dev_df["Store"].map(store_p75)).astype(int)
    y_holdout_surge = (holdout_df["Weekly_Sales"] > holdout_df["Store"].map(store_p75)).astype(int)
    
    clf_models = {
        "Random_Forest_Clf": create_random_forest_classifier(n_estimators=100, max_depth=8, class_weight="balanced"),
        "LightGBM_Clf": create_lightgbm_classifier(n_estimators=100, num_leaves=15, max_depth=4, learning_rate=0.03, class_weight="balanced"),
        "XGBoost_Clf": create_xgboost_classifier(n_estimators=100, max_depth=4, learning_rate=0.03, scale_pos_weight=float((len(y_dev_surge)-sum(y_dev_surge))/sum(y_dev_surge)))
    }
    
    classification_results = []
    for c_name, clf in clf_models.items():
        clf.fit(X_dev, y_dev_surge)
        probs = clf.predict_proba(X_holdout)[:, 1]
        c_metrics = evaluate_classification_metrics(y_holdout_surge, probs, threshold=0.5, model_name=c_name)
        classification_results.append(c_metrics)
        
    # Inventory Replenishment & Simulation for Store 1
    s1_mask = holdout_df["Store"] == 1
    s1_actual = y_holdout[s1_mask].to_numpy()
    
    best_preds_s1 = trained_models["Random_Forest"][1][s1_mask]
    naive_preds_s1 = trained_models["Lag1_Naive"][1][s1_mask]
    
    # ML policy uses residual std sigma_e
    sigma_e_ml = float(np.sqrt(np.mean((s1_actual - best_preds_s1) ** 2)))
    d_hat_ml = float(np.mean(best_preds_s1))
    ss_ml = calculate_safety_stock(sigma_e_ml, lead_time=2, service_level=0.95)
    rop_ml = calculate_reorder_point(d_hat_ml * 2, ss_ml)
    eoq_ml = calculate_eoq(d_hat_ml * 52, order_cost=2500.0, holding_cost_annual=0.20)
    
    # Naive policy
    sigma_e_naive = float(np.sqrt(np.mean((s1_actual - naive_preds_s1) ** 2)))
    d_hat_naive = float(np.mean(naive_preds_s1))
    ss_naive = calculate_safety_stock(sigma_e_naive, lead_time=2, service_level=0.95)
    rop_naive = calculate_reorder_point(d_hat_naive * 2, ss_naive)
    eoq_naive = calculate_eoq(d_hat_naive * 52, order_cost=2500.0, holding_cost_annual=0.20)
    
    # Raw demand variance policy (HIGH-01 illustration)
    sigma_d_raw = float(np.std(s1_actual))
    ss_raw = calculate_safety_stock(sigma_d_raw, lead_time=2, service_level=0.95)
    rop_raw = calculate_reorder_point(d_hat_ml * 2, ss_raw)
    
    # Simulation (days_per_year=52 weeks, unit_cost=1.0 for monetary revenue inventory)
    sim = InventorySimulation(lead_time=2, unit_cost=1.0, holding_rate_annual=0.20, order_setup_cost=2500.0, days_per_year=52)
    sim_ml = sim.simulate(s1_actual, reorder_point=rop_ml, order_quantity=eoq_ml)
    sim_naive = sim.simulate(s1_actual, reorder_point=rop_naive, order_quantity=eoq_naive)
    sim_raw = sim.simulate(s1_actual, reorder_point=rop_raw, order_quantity=eoq_ml)
    
    return {
        "Dataset": "Walmart Weekly Store Sales",
        "Target": "Weekly_Sales ($ monetary store-level sales)",
        "Evaluation_Horizon": "26 weeks (locked chronological holdout)",
        "Observations_Holdout": len(y_holdout),
        "CV_Results": cv_results,
        "Holdout_Metrics": holdout_results,
        "Classification_Surge_Metrics": classification_results,
        "Store_Level_Summary": store_summary,
        "Store_Level_Per_Store": store_results[:5],
        "Inventory_Comparison_Store1": {
            "ML_Policy": {
                "Forecast_d_hat": round(d_hat_ml, 2),
                "Sigma_e": round(sigma_e_ml, 2),
                "Safety_Stock": round(ss_ml, 2),
                "ROP": round(rop_ml, 2),
                "EOQ": round(eoq_ml, 2),
                "Fill_Rate": f"{sim_ml['Fill_Rate_Pct']:.2f}%",
                "Total_Cost": round(sim_ml['Total_Cost'], 2),
                "Holding_Cost": round(sim_ml['Holding_Cost'], 2)
            },
            "Naive_Policy": {
                "Forecast_d_hat": round(d_hat_naive, 2),
                "Sigma_e": round(sigma_e_naive, 2),
                "Safety_Stock": round(ss_naive, 2),
                "ROP": round(rop_naive, 2),
                "EOQ": round(eoq_naive, 2),
                "Fill_Rate": f"{sim_naive['Fill_Rate_Pct']:.2f}%",
                "Total_Cost": round(sim_naive['Total_Cost'], 2)
            },
            "Raw_Demand_Std_Policy": {
                "Sigma_d_raw": round(sigma_d_raw, 2),
                "Safety_Stock": round(ss_raw, 2),
                "ROP": round(rop_raw, 2),
                "Total_Cost": round(sim_raw['Total_Cost'], 2),
                "Holding_Cost": round(sim_raw['Holding_Cost'], 2)
            }
        }
    }


def run_online_retail_experiment() -> Dict[str, Any]:
    print(">>> [2/3] Running Online Retail Panel Experiments (Real UCI Dataset)...")
    df_daily = load_online_retail("data/Online Retail.xlsx", cache_path="data/Online_Retail_daily.csv")
    df = get_online_retail_continuous_panel(df_daily, n_top_skus=30, include_segments=True)
    
    # Feature engineering
    df = add_cyclical_calendar_features(df, "Date")
    df = add_leak_free_lags(df, "StockCode", "Date", "Daily_Quantity", lags=[1, 2, 7])
    df = add_leak_free_rolling(df, "StockCode", "Date", "Daily_Quantity", windows=[7, 14])
    df = add_demand_momentum(df, "StockCode", "Date", "Daily_Quantity", base_window=7)
    
    df_clean = df.dropna().reset_index(drop=True)
    dev_df, holdout_df = split_locked_final_holdout(df_clean, "Date", holdout_periods=28)
    
    features = [
        "Avg_UnitPrice", "sin_wday", "cos_wday", "sin_month", "cos_month",
        "Daily_Quantity_Lag1", "Daily_Quantity_Lag2", "Daily_Quantity_Lag7",
        "Daily_Quantity_RollMean7", "Daily_Quantity_RollMean14",
        "Daily_Quantity_RollStd7", "Daily_Quantity_Momentum7"
    ]
    
    # 3-Fold Walk-Forward Cross Validation on development set
    cv = ChronologicalTimeSeriesSplit(n_splits=3, horizon_periods=14, min_train_periods=60)
    cv_models = {
        "Lag1_Naive": lambda: Lag1NaiveModel("Daily_Quantity_Lag1"),
        "Recent_Average": lambda: RecentAverageModel("Daily_Quantity_RollMean7"),
        "Random_Forest": lambda: create_random_forest(n_estimators=120, max_depth=8),
        "LightGBM_MSE": lambda: create_lightgbm_regressor(objective="regression", n_estimators=120, num_leaves=15, min_child_samples=20, learning_rate=0.03, reg_alpha=1.0, reg_lambda=5.0),
        "LightGBM_Tweedie": lambda: create_lightgbm_regressor(objective="tweedie", tweedie_variance_power=1.2, n_estimators=100, num_leaves=15, learning_rate=0.03)
    }
    
    cv_results = {}
    for name, factory in cv_models.items():
        print(f"    Running 3-fold CV for Online Retail: {name}")
        res = run_walk_forward_cv(dev_df, features, "Daily_Quantity", "Date", factory, cv, name)
        cv_results[name] = res
    
    # Locked Holdout Evaluation
    X_dev, y_dev = dev_df[features], dev_df["Daily_Quantity"]
    X_holdout, y_holdout = holdout_df[features], holdout_df["Daily_Quantity"]
    
    holdout_results = []
    trained_models = {}
    for name, factory in cv_models.items():
        m = factory()
        m.fit(X_dev, y_dev)
        preds = m.predict(X_holdout)
        metrics = evaluate_all_metrics(y_holdout, preds, name)
        holdout_results.append(metrics)
        trained_models[name] = preds
        
    # Tuned XGBoost MSE (anti-starvation operational regressor)
    xgb_mse = create_xgboost_regressor(n_estimators=120, max_depth=4, learning_rate=0.03, reg_alpha=1.0, reg_lambda=5.0)
    xgb_mse.fit(X_dev, y_dev)
    xgb_mse_preds = xgb_mse.predict(X_holdout)
    m_xgb_mse = evaluate_all_metrics(y_holdout, xgb_mse_preds, "XGBoost_MSE")
    holdout_results.append(m_xgb_mse)
    trained_models["XGBoost_MSE"] = xgb_mse_preds

    # Log1p XGBoost (demonstrating Jensen's inequality bias)
    xgb_log1p = create_xgboost_regressor(n_estimators=100, max_depth=6)
    xgb_log1p.fit(X_dev, np.log1p(y_dev))
    log_preds = xgb_log1p.predict(X_holdout)
    raw_preds = np.expm1(log_preds)
    m_log1p = evaluate_all_metrics(y_holdout, raw_preds, "XGBoost_Log1p")
    holdout_results.append(m_log1p)
    trained_models["XGBoost_Log1p"] = raw_preds
    
    # Demand Surge / Stockout Risk Classification (75th Percentile SKU Surge)
    print("    Running Online Retail Demand Surge Classification (ROC AUC / Accuracy)...")
    sku_p75 = dev_df.groupby("StockCode")["Daily_Quantity"].quantile(0.75).to_dict()
    y_dev_surge = (dev_df["Daily_Quantity"] > dev_df["StockCode"].map(sku_p75)).astype(int)
    y_holdout_surge = (holdout_df["Daily_Quantity"] > holdout_df["StockCode"].map(sku_p75)).astype(int)
    
    clf_models = {
        "Random_Forest_Clf": create_random_forest_classifier(n_estimators=100, max_depth=6, class_weight="balanced"),
        "LightGBM_Clf": create_lightgbm_classifier(n_estimators=100, num_leaves=15, max_depth=4, learning_rate=0.03, class_weight="balanced"),
        "XGBoost_Clf": create_xgboost_classifier(n_estimators=100, max_depth=3, learning_rate=0.03, scale_pos_weight=float((len(y_dev_surge)-sum(y_dev_surge))/sum(y_dev_surge)))
    }
    
    classification_results = []
    for c_name, clf in clf_models.items():
        clf.fit(X_dev, y_dev_surge)
        probs = clf.predict_proba(X_holdout)[:, 1]
        c_metrics = evaluate_classification_metrics(y_holdout_surge, probs, threshold=0.5, model_name=c_name)
        classification_results.append(c_metrics)

    # Segment-Level Evaluation across High_Volume, Medium_Volume, Intermittent_Low
    segment_results = {}
    for seg in holdout_df["Demand_Segment"].unique():
        seg_mask = holdout_df["Demand_Segment"] == seg
        y_seg = y_holdout[seg_mask].to_numpy()
        seg_results = []
        for model_name, p in trained_models.items():
            p_seg = p[seg_mask]
            m_s = evaluate_all_metrics(y_seg, p_seg, model_name)
            seg_results.append({
                "Model": model_name,
                "MAE": m_s["MAE"],
                "RMSE": m_s["RMSE"],
                "WAPE": m_s["WAPE"],
                "Bias_Percentage": m_s["Bias_Percentage"]
            })
        segment_results[seg] = seg_results
    
    return {
        "Dataset": "Online Retail Daily Demand (Real UCI Dataset)",
        "Target": "Daily_Quantity (continuous daily SKU panel with explicit zero demand)",
        "Total_SKUs": int(df["StockCode"].nunique()),
        "Zero_Demand_Pct": round(float((df["Daily_Quantity"] == 0).mean() * 100), 2),
        "Evaluation_Horizon": "28 days (locked chronological holdout)",
        "Observations_Holdout": len(y_holdout),
        "CV_Results": cv_results,
        "Holdout_Metrics": holdout_results,
        "Classification_Surge_Metrics": classification_results,
        "Segment_Holdout_Metrics": segment_results
    }


def run_m5_experiment() -> Dict[str, Any]:
    print(">>> [3/3] Running M5 Intermittent Grocery Panel Experiments...")
    df = generate_benchmark_panel("m5", n_entities=20, n_periods=180, random_seed=42)
    zero_ratio = float((df["Sales"] == 0).mean() * 100.0)
    
    # Feature engineering
    df = add_cyclical_calendar_features(df, "date")
    df = add_leak_free_lags(df, "id", "date", "Sales", lags=[1, 2, 7])
    df = add_leak_free_rolling(df, "id", "date", "Sales", windows=[7, 28])
    
    df_clean = df.dropna().reset_index(drop=True)
    dev_df, holdout_df = split_locked_final_holdout(df_clean, "date", holdout_periods=28)
    
    features = [
        "sell_price", "sin_wday", "cos_wday", "sin_month", "cos_month",
        "Sales_Lag1", "Sales_Lag2", "Sales_Lag7",
        "Sales_RollMean7", "Sales_RollMean28"
    ]
    
    X_dev, y_dev = dev_df[features], dev_df["Sales"]
    X_holdout, y_holdout = holdout_df[features], holdout_df["Sales"]
    
    models = {
        "Lag1_Naive": Lag1NaiveModel("Sales_Lag1"),
        "Recent_Average": RecentAverageModel("Sales_RollMean7"),
        "Croston_SBA": CrostonSBAModel(alpha=0.1),
        "Random_Forest": create_random_forest(n_estimators=100, max_depth=8),
        "XGBoost_MSE": create_xgboost_regressor(objective="reg:squarederror", n_estimators=100),
        "XGBoost_Tweedie_p1.5": create_xgboost_regressor(objective="reg:tweedie", tweedie_variance_power=1.5, n_estimators=100)
    }
    
    holdout_results = []
    for name, m in models.items():
        m.fit(X_dev, y_dev)
        preds = m.predict(X_holdout)
        metrics = evaluate_all_metrics(y_holdout, preds, name)
        holdout_results.append(metrics)
        
    # Intermittent Occurrence Classification: P(Sales > 0)
    print("    Running M5 Intermittent Occurrence Classification (ROC AUC / Accuracy)...")
    y_dev_occ = (y_dev > 0).astype(int)
    y_holdout_occ = (y_holdout > 0).astype(int)
    
    clf_models = {
        "Random_Forest_Clf": create_random_forest_classifier(n_estimators=100, max_depth=6, class_weight="balanced"),
        "LightGBM_Clf": create_lightgbm_classifier(n_estimators=100, num_leaves=15, max_depth=4, learning_rate=0.03, class_weight="balanced"),
        "XGBoost_Clf": create_xgboost_classifier(n_estimators=100, max_depth=4, learning_rate=0.03, scale_pos_weight=float((len(y_dev_occ)-sum(y_dev_occ))/sum(y_dev_occ)))
    }
    
    classification_results = []
    for c_name, clf in clf_models.items():
        clf.fit(X_dev, y_dev_occ)
        probs = clf.predict_proba(X_holdout)[:, 1]
        c_metrics = evaluate_classification_metrics(y_holdout_occ, probs, threshold=0.5, model_name=c_name)
        classification_results.append(c_metrics)
        
    return {
        "Dataset": "M5 Intermittent Grocery Demand",
        "Target": "Sales (unit daily count)",
        "Zero_Demand_Percentage": round(zero_ratio, 2),
        "Evaluation_Horizon": "28 days (locked chronological holdout)",
        "Observations_Holdout": len(y_holdout),
        "Holdout_Metrics": holdout_results,
        "Classification_Occurrence_Metrics": classification_results
    }


def main():
    print("================================================================================")
    print("STARTING SCIENTIFIC EXPERIMENT EXECUTION PIPELINE")
    print("================================================================================")
    
    walmart_res = run_walmart_experiment()
    online_res = run_online_retail_experiment()
    m5_res = run_m5_experiment()
    
    all_experiments = {
        "Walmart_Sales": walmart_res,
        "Online_Retail": online_res,
        "M5_Grocery": m5_res
    }
    
    out_file = "experiment_results.json"
    with open(out_file, "w") as f:
        json.dump(all_experiments, f, indent=2)
        
    print(f"\nSaved verified experiment results to {out_file} successfully!")
    print("================================================================================")


if __name__ == "__main__":
    main()
