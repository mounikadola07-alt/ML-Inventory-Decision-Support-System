"""
src/policy_comparison.py - Four-Policy Inventory Decision Support Engine & Cross-Regime Study
Implements and evaluates the 4 formal forecast selection policies:
  - Policy A: Historical / Seasonal Naive Baseline (Benchmark heuristic)
  - Policy B: Statistical Accuracy-First (Lowest Validation MAE)
  - Policy C: Bias-Constrained (Lowest Validation MAE subject to |Bias%| <= 5.0%)
  - Policy D: Cost-Aware (Lowest Estimated Inventory Replenishment Cost on Validation Folds)

Evaluates on identical locked chronological holdouts across three empirical demand regimes:
  1. Walmart Weekly Store Sales (Smooth: ADI=1.00, CV^2=0.29)
  2. Online Retail Daily SKU Panel (Lumpy/Erratic: 61.5% zeros, wholesale spikes)
  3. M5 Grocery Series (Intermittent: ADI=3.57, CV^2=0.25, 69.8% zeros)

Outputs machine-readable artifacts into results/ directory.
"""

import json
import os
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Tuple, Optional, Union, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

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
    split_locked_final_holdout
)
from src.models import (
    Lag1NaiveModel,
    SeasonalNaiveModel,
    RecentAverageModel,
    CrostonSBAModel,
    create_random_forest,
    create_xgboost_regressor,
    create_lightgbm_regressor
)
from src.metrics import (
    evaluate_all_metrics,
    calc_mae,
    calc_rmse,
    calc_wape,
    calc_r2,
    calc_bias_percentage,
    calc_bias_absolute
)
from src.inventory import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_eoq,
    InventorySimulation,
    get_z_score
)


# ==============================================================================
# Model Wrapper for Target Transformations (e.g. Log1p)
# ==============================================================================

class Log1pModelWrapper(BaseEstimator, RegressorMixin):
    """
    Wraps an estimator to train on log1p(y) and predict with expm1(pred).
    Demonstrates Jensen's inequality and asymmetric loss vulnerability under symmetric metrics.
    """

    def __init__(self, base_estimator_factory: Callable[[], Any]):
        self.base_estimator_factory = base_estimator_factory
        self.model_ = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        y_arr = np.asarray(y, dtype=np.float64)
        self.model_ = self.base_estimator_factory()
        self.model_.fit(X, np.log1p(np.clip(y_arr, 0.0, None)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("Model is not fitted yet.")
        preds = self.model_.predict(X)
        return np.clip(np.expm1(preds), 0.0, None)


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class InventoryConfig:
    """Configuration parameters for continuous-review (s, Q) replenishment simulation."""
    lead_time: int = 7
    service_level: float = 0.95
    order_setup_cost: float = 50.0
    holding_rate_annual: float = 0.20
    unit_cost: float = 10.0
    stockout_penalty_per_unit: float = 20.0
    periods_per_year: int = 365
    sigma_lead_time: float = 0.0


@dataclass
class PolicyResult:
    """Standardized result envelope for each forecast selection policy."""
    policy_name: str
    selected_model: str
    selection_criterion: str
    validation_metric_name: str
    validation_metric_value: float
    holdout_forecast_metrics: Dict[str, float]
    holdout_inventory_metrics: Dict[str, Any]
    dataset_name: str = ""
    demand_regime: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes PolicyResult to JSON-compliant dictionary."""
        inv_clean = {}
        for k, v in self.holdout_inventory_metrics.items():
            if isinstance(v, (np.floating, float)):
                inv_clean[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                inv_clean[k] = int(v)
            elif isinstance(v, (str, bool)):
                inv_clean[k] = v
            # Exclude large raw trajectory arrays from JSON summary
            elif not isinstance(v, (np.ndarray, list)):
                inv_clean[k] = v

        return {
            "Policy": self.policy_name,
            "Selected_Model": self.selected_model,
            "Selection_Criterion": self.selection_criterion,
            "Validation_Metric_Name": self.validation_metric_name,
            "Validation_Metric_Value": float(self.validation_metric_value),
            "Dataset": self.dataset_name,
            "Demand_Regime": self.demand_regime,
            "Holdout_Forecast_Metrics": {
                k: (float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v)
                for k, v in self.holdout_forecast_metrics.items()
            },
            "Holdout_Inventory_Metrics": inv_clean,
            "Extra": self.extra
        }

    def __getitem__(self, item: str) -> Any:
        """Enables dictionary-style indexing for backwards and test compatibility."""
        if hasattr(self, item):
            return getattr(self, item)
        if item in self.holdout_forecast_metrics:
            return self.holdout_forecast_metrics[item]
        if item in self.holdout_inventory_metrics:
            return self.holdout_inventory_metrics[item]
        aliases = {
            "Total_Cost": ("holdout_inventory_metrics", "Total_Cost"),
            "Holding_Cost": ("holdout_inventory_metrics", "Holding_Cost"),
            "Ordering_Cost": ("holdout_inventory_metrics", "Ordering_Cost"),
            "Shortage_Cost": ("holdout_inventory_metrics", "Shortage_Cost"),
            "Fill_Rate_Pct": ("holdout_inventory_metrics", "Fill_Rate_Pct"),
            "Stockout_Days": ("holdout_inventory_metrics", "Stockout_Days"),
            "Safety_Stock": ("holdout_inventory_metrics", "Safety_Stock"),
            "ROP": ("holdout_inventory_metrics", "ROP"),
            "EOQ": ("holdout_inventory_metrics", "EOQ"),
            "MAE": ("holdout_forecast_metrics", "MAE"),
            "RMSE": ("holdout_forecast_metrics", "RMSE"),
            "WAPE": ("holdout_forecast_metrics", "WAPE"),
            "Bias_Percentage": ("holdout_forecast_metrics", "Bias_Percentage"),
            "R2": ("holdout_forecast_metrics", "R2"),
        }
        if item in aliases:
            dict_name, key_name = aliases[item]
            subdict = getattr(self, dict_name)
            return subdict.get(key_name)
        raise KeyError(f"Key '{item}' not found in PolicyResult.")


# ==============================================================================
# Simulation Helpers
# ==============================================================================

def simulate_series_replenishment(
    actual_demands: np.ndarray,
    forecast_series: np.ndarray,
    config: InventoryConfig,
    sigma_error_override: Optional[float] = None,
    mean_forecast_override: Optional[float] = None,
    initial_inventory: Optional[float] = None
) -> Dict[str, Any]:
    """
    Executes a discrete-time continuous review (s, Q) replenishment simulation
    for a single demand series given actual demands and model forecasts.
    """
    actual_demands = np.asarray(actual_demands, dtype=np.float64)
    forecast_series = np.asarray(forecast_series, dtype=np.float64)

    # 1. Error standard error sigma_e and mean forecast d_hat
    if sigma_error_override is not None:
        sigma_e = float(sigma_error_override)
    else:
        sigma_e = float(np.sqrt(np.mean((actual_demands - forecast_series) ** 2)))

    if mean_forecast_override is not None:
        d_hat = float(max(0.001, mean_forecast_override))
    else:
        d_hat = float(max(0.001, np.mean(forecast_series)))

    # 2. Safety stock (residual-variance based with optional lead time jitter)
    ss = calculate_safety_stock(
        sigma_error=sigma_e,
        lead_time=config.lead_time,
        service_level=config.service_level,
        sigma_lead_time=config.sigma_lead_time,
        mean_demand=d_hat
    )

    # 3. Reorder Point (ROP)
    rop = calculate_reorder_point(
        mean_lead_time_demand=d_hat * config.lead_time,
        safety_stock=ss
    )

    # 4. Dimensionally consistent EOQ
    annual_demand = d_hat * config.periods_per_year
    holding_cost_annual = config.holding_rate_annual * config.unit_cost
    eoq = calculate_eoq(
        annual_demand=annual_demand,
        order_cost=config.order_setup_cost,
        holding_cost_annual=holding_cost_annual
    )

    # 5. Event simulation
    sim = InventorySimulation(
        lead_time=config.lead_time,
        unit_cost=config.unit_cost,
        holding_rate_annual=config.holding_rate_annual,
        order_setup_cost=config.order_setup_cost,
        stockout_penalty_per_unit=config.stockout_penalty_per_unit,
        days_per_year=config.periods_per_year
    )
    res = sim.simulate(actual_demands, reorder_point=rop, order_quantity=eoq, initial_inventory=initial_inventory)
    # Remove history arrays for clean JSON serialization
    res.pop("History_On_Hand", None)
    res.pop("History_Orders", None)

    # Augment with inventory control parameters
    res["Sigma_e"] = sigma_e
    res["Forecast_d_hat"] = d_hat
    res["Safety_Stock"] = ss
    res["ROP"] = rop
    res["EOQ"] = eoq
    return res


def make_json_serializable(val: Any) -> Any:
    """Recursively converts numpy structures to native Python types for JSON dumping."""
    if isinstance(val, (np.ndarray, list)):
        return [make_json_serializable(x) for x in val]
    elif isinstance(val, (np.floating, float)):
        return None if np.isnan(val) else float(val)
    elif isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, (np.bool_, bool)):
        return bool(val)
    elif isinstance(val, dict):
        return {str(k): make_json_serializable(v) for k, v in val.items()}
    return val


def simulate_panel_replenishment(
    df: pd.DataFrame,
    entity_col: str,
    target_col: str,
    pred_values: np.ndarray,
    config: InventoryConfig
) -> Dict[str, Any]:
    """
    Executes discrete-time inventory simulation across an entire multi-entity panel,
    aggregating operational costs and fill rates across all series.
    """
    df_eval = df.copy()
    df_eval["_pred"] = pred_values

    total_demand = 0.0
    total_served = 0.0
    total_lost_sales = 0.0
    total_stockout_days = 0
    total_orders_placed = 0
    total_holding_cost = 0.0
    total_ordering_cost = 0.0
    total_shortage_cost = 0.0
    total_cost = 0.0

    entity_results = {}
    entities = df_eval[entity_col].unique()

    for ent in entities:
        ent_mask = df_eval[entity_col] == ent
        y_ent = df_eval.loc[ent_mask, target_col].to_numpy()
        p_ent = df_eval.loc[ent_mask, "_pred"].to_numpy()

        res_ent = simulate_series_replenishment(y_ent, p_ent, config)
        entity_results[str(ent)] = res_ent

        total_demand += res_ent["Total_Demand"]
        total_served += res_ent["Total_Served"]
        total_lost_sales += res_ent["Total_Lost_Sales"]
        total_stockout_days += res_ent["Stockout_Days"]
        total_orders_placed += res_ent["Orders_Placed"]
        total_holding_cost += res_ent["Holding_Cost"]
        total_ordering_cost += res_ent["Ordering_Cost"]
        total_shortage_cost += res_ent["Shortage_Cost"]
        total_cost += res_ent["Total_Cost"]

    panel_fill_rate = (total_served / total_demand * 100.0) if total_demand > 0 else 100.0
    total_periods = len(df_eval)

    return {
        "Total_Entities": len(entities),
        "Total_Periods": total_periods,
        "Total_Demand": total_demand,
        "Total_Served": total_served,
        "Total_Lost_Sales": total_lost_sales,
        "Fill_Rate_Pct": float(panel_fill_rate),
        "Stockout_Days": int(total_stockout_days),
        "Orders_Placed": int(total_orders_placed),
        "Holding_Cost": float(total_holding_cost),
        "Ordering_Cost": float(total_ordering_cost),
        "Shortage_Cost": float(total_shortage_cost),
        "Total_Cost": float(total_cost),
        "Per_Entity_Sample": {str(k): entity_results[str(k)] for k in list(entities)[:3]}
    }


# ==============================================================================
# Model Selection Functions
# ==============================================================================

def select_policy_a_model(
    candidate_names: List[str],
    baseline_name: Optional[str] = None
) -> str:
    """
    Policy A: Historical / Seasonal Naive Baseline Selection.
    Selects the fixed benchmark heuristic.
    """
    if baseline_name and baseline_name in candidate_names:
        return baseline_name
    for preferred in ["Seasonal_Naive", "Lag1_Naive", "Recent_Average", "Croston_SBA"]:
        if preferred in candidate_names:
            return preferred
    return candidate_names[0]


def select_policy_b_model(
    cv_results: Dict[str, Any],
    candidate_names: Optional[List[str]] = None
) -> Tuple[str, float]:
    """
    Policy B: Statistical Accuracy-First Selection (Minimum Validation MAE).
    argmin_{M in M} MAE_val(M). Tiebreaker: lowest RMSE_val.
    """
    pool = candidate_names if candidate_names else list(cv_results.keys())
    best_model = None
    best_mae = float("inf")
    best_rmse = float("inf")

    for name in pool:
        metrics = cv_results[name]
        mae = float(metrics.get("CV_MAE_Mean", metrics.get("MAE", float("inf"))))
        rmse = float(metrics.get("CV_RMSE_Mean", metrics.get("RMSE", float("inf"))))

        if (mae < best_mae) or (np.isclose(mae, best_mae) and rmse < best_rmse):
            best_mae = mae
            best_rmse = rmse
            best_model = name

    if best_model is None:
        raise ValueError("Empty candidate pool in Policy B selection.")
    return best_model, best_mae


def select_policy_c_model(
    cv_results: Dict[str, Any],
    bias_threshold: float = 0.05,
    candidate_names: Optional[List[str]] = None
) -> Tuple[str, float, bool]:
    """
    Policy C: Operational Bias-Constrained Selection.
    Filters candidate models with |Bias%_val| <= bias_threshold (default 5.0%).
    Selects argmin_{M in M_safe} MAE_val(M).
    Fallback: argmin_{M in M} |Bias%_val(M)| if M_safe is empty.
    Returns: (selected_model, metric_value, is_fallback).
    """
    pool = candidate_names if candidate_names else list(cv_results.keys())
    # Handle fractional (0.05) vs percentage (5.0%)
    thr_pct = (bias_threshold * 100.0) if bias_threshold <= 1.0 else bias_threshold

    safe_candidates = []
    all_biases = {}

    for name in pool:
        metrics = cv_results[name]
        bias_pct = metrics.get("CV_Bias_Pct_Mean", metrics.get("Bias_Percentage", None))
        if bias_pct is None or np.isnan(bias_pct):
            bias_val = float("inf")
        else:
            bias_val = abs(float(bias_pct))

        all_biases[name] = bias_val
        mae = float(metrics.get("CV_MAE_Mean", metrics.get("MAE", float("inf"))))
        rmse = float(metrics.get("CV_RMSE_Mean", metrics.get("RMSE", float("inf"))))

        if bias_val <= thr_pct:
            safe_candidates.append((name, mae, rmse))

    if safe_candidates:
        # Sort by MAE then RMSE
        safe_candidates.sort(key=lambda item: (item[1], item[2]))
        selected_model = safe_candidates[0][0]
        selected_mae = safe_candidates[0][1]
        return selected_model, selected_mae, False
    else:
        # Fallback: candidate with minimum absolute bias
        best_model = min(all_biases.keys(), key=lambda k: all_biases[k])
        return best_model, all_biases[best_model], True


def select_policy_d_model(
    cv_results: Dict[str, Any],
    validation_cost_fn: Optional[Callable[[str], float]] = None,
    candidate_names: Optional[List[str]] = None
) -> Tuple[str, float]:
    """
    Policy D: Cost-Aware Selection.
    Selects model minimizing total estimated inventory replenishment cost on validation folds.
    argmin_{M in M} C_total^val(M). Tiebreaker: higher validation fill rate.
    """
    pool = candidate_names if candidate_names else list(cv_results.keys())
    best_model = None
    best_cost = float("inf")
    best_fill = -1.0

    for name in pool:
        if validation_cost_fn is not None:
            cost = float(validation_cost_fn(name))
            fill = 100.0
        else:
            metrics = cv_results[name]
            cost = float(metrics.get("CV_Cost_Mean", metrics.get("Total_Cost", float("inf"))))
            fill = float(metrics.get("CV_Fill_Rate_Mean", metrics.get("Fill_Rate_Pct", 0.0)))

        if (cost < best_cost) or (np.isclose(cost, best_cost) and fill > best_fill):
            best_cost = cost
            best_fill = fill
            best_model = name

    if best_model is None:
        raise ValueError("Empty candidate pool in Policy D selection.")
    return best_model, best_cost


# ==============================================================================
# Standard Policy Evaluation Functions (PROJECT.md Interface Contracts)
# ==============================================================================

def evaluate_policy_a_naive(
    candidate_models: Optional[Union[Dict[str, Any], List[str], str]] = None,
    cv_results: Optional[Dict[str, Any]] = None,
    holdout_evaluator: Optional[Union[Callable[[str], Tuple[Dict[str, float], Dict[str, Any]]], Dict[str, Any]]] = None,
    baseline_model_name: Optional[str] = None,
    dataset_name: str = "",
    demand_regime: str = "",
    **kwargs
) -> PolicyResult:
    """Evaluates Policy A (Historical / Seasonal Naive Baseline) on locked holdout."""
    if isinstance(candidate_models, str):
        chosen = candidate_models
    elif baseline_model_name:
        chosen = baseline_model_name
    elif isinstance(candidate_models, dict):
        chosen = select_policy_a_model(list(candidate_models.keys()), baseline_name=baseline_model_name)
    elif isinstance(candidate_models, list):
        chosen = select_policy_a_model(candidate_models, baseline_name=baseline_model_name)
    elif cv_results:
        chosen = select_policy_a_model(list(cv_results.keys()), baseline_name=baseline_model_name)
    else:
        chosen = "Baseline_Naive"

    val_metric_val = 0.0
    if cv_results and chosen in cv_results:
        val_metric_val = float(cv_results[chosen].get("CV_MAE_Mean", cv_results[chosen].get("MAE", 0.0)))

    holdout_f, holdout_i = _extract_holdout_metrics(chosen, holdout_evaluator, kwargs)

    return PolicyResult(
        policy_name="Policy A: Naive Baseline",
        selected_model=chosen,
        selection_criterion="Fixed Heuristic Benchmark",
        validation_metric_name="Zero-Parameter / Historical Heuristic",
        validation_metric_value=val_metric_val,
        holdout_forecast_metrics=holdout_f,
        holdout_inventory_metrics=holdout_i,
        dataset_name=dataset_name,
        demand_regime=demand_regime
    )


def evaluate_policy_b_accuracy(
    candidate_models: Optional[Union[Dict[str, Any], List[str]]] = None,
    cv_results: Optional[Dict[str, Any]] = None,
    holdout_evaluator: Optional[Union[Callable[[str], Tuple[Dict[str, float], Dict[str, Any]]], Dict[str, Any]]] = None,
    dataset_name: str = "",
    demand_regime: str = "",
    **kwargs
) -> PolicyResult:
    """Evaluates Policy B (Statistical Accuracy-First, lowest validation MAE) on locked holdout."""
    if not cv_results:
        raise ValueError("cv_results dictionary required for Policy B accuracy selection.")

    cand_names = list(candidate_models.keys()) if isinstance(candidate_models, dict) else (
        candidate_models if isinstance(candidate_models, list) else list(cv_results.keys())
    )
    chosen, best_mae = select_policy_b_model(cv_results, candidate_names=cand_names)
    holdout_f, holdout_i = _extract_holdout_metrics(chosen, holdout_evaluator, kwargs)

    return PolicyResult(
        policy_name="Policy B: Statistical Accuracy-First",
        selected_model=chosen,
        selection_criterion="Minimum Validation MAE (argmin MAE_val)",
        validation_metric_name="Validation MAE",
        validation_metric_value=best_mae,
        holdout_forecast_metrics=holdout_f,
        holdout_inventory_metrics=holdout_i,
        dataset_name=dataset_name,
        demand_regime=demand_regime
    )


def evaluate_policy_c_bias_constrained(
    candidate_models: Optional[Union[Dict[str, Any], List[str]]] = None,
    cv_results: Optional[Dict[str, Any]] = None,
    holdout_evaluator: Optional[Union[Callable[[str], Tuple[Dict[str, float], Dict[str, Any]]], Dict[str, Any]]] = None,
    bias_threshold: float = 0.05,
    dataset_name: str = "",
    demand_regime: str = "",
    **kwargs
) -> PolicyResult:
    """Evaluates Policy C (Operational Bias-Constrained, |Bias%| <= 5%) on locked holdout."""
    if not cv_results:
        raise ValueError("cv_results dictionary required for Policy C bias-constrained selection.")

    cand_names = list(candidate_models.keys()) if isinstance(candidate_models, dict) else (
        candidate_models if isinstance(candidate_models, list) else list(cv_results.keys())
    )
    chosen, metric_val, is_fallback = select_policy_c_model(
        cv_results,
        bias_threshold=bias_threshold,
        candidate_names=cand_names
    )
    holdout_f, holdout_i = _extract_holdout_metrics(chosen, holdout_evaluator, kwargs)

    thr_display = f"{bias_threshold * 100:.1f}%" if bias_threshold <= 1.0 else f"{bias_threshold:.1f}%"
    crit_desc = (
        f"Lowest MAE subject to |Bias%| <= {thr_display}"
        if not is_fallback else
        f"Fallback: Lowest |Bias%| (No model met |Bias%| <= {thr_display})"
    )

    return PolicyResult(
        policy_name="Policy C: Bias-Constrained",
        selected_model=chosen,
        selection_criterion=crit_desc,
        validation_metric_name="Validation MAE (Safe Pool)" if not is_fallback else "Validation |Bias%|",
        validation_metric_value=metric_val,
        holdout_forecast_metrics=holdout_f,
        holdout_inventory_metrics=holdout_i,
        dataset_name=dataset_name,
        demand_regime=demand_regime,
        extra={"Is_Fallback": is_fallback, "Bias_Threshold": thr_display}
    )


def evaluate_policy_d_cost_aware(
    candidate_models: Optional[Union[Dict[str, Any], List[str]]] = None,
    cv_results: Optional[Dict[str, Any]] = None,
    holdout_evaluator: Optional[Union[Callable[[str], Tuple[Dict[str, float], Dict[str, Any]]], Dict[str, Any]]] = None,
    validation_cost_fn: Optional[Callable[[str], float]] = None,
    dataset_name: str = "",
    demand_regime: str = "",
    **kwargs
) -> PolicyResult:
    """Evaluates Policy D (Cost-Aware, lowest validation inventory cost) on locked holdout."""
    if not cv_results and validation_cost_fn is None:
        raise ValueError("cv_results or validation_cost_fn required for Policy D cost-aware selection.")

    cand_names = list(candidate_models.keys()) if isinstance(candidate_models, dict) else (
        candidate_models if isinstance(candidate_models, list) else (list(cv_results.keys()) if cv_results else [])
    )
    chosen, best_cost = select_policy_d_model(
        cv_results or {},
        validation_cost_fn=validation_cost_fn,
        candidate_names=cand_names
    )
    holdout_f, holdout_i = _extract_holdout_metrics(chosen, holdout_evaluator, kwargs)

    return PolicyResult(
        policy_name="Policy D: Cost-Aware",
        selected_model=chosen,
        selection_criterion="Minimum Simulated Replenishment Cost on Walk-Forward Validation Folds",
        validation_metric_name="Validation Inventory Total Cost",
        validation_metric_value=best_cost,
        holdout_forecast_metrics=holdout_f,
        holdout_inventory_metrics=holdout_i,
        dataset_name=dataset_name,
        demand_regime=demand_regime
    )


def _extract_holdout_metrics(
    model_name: str,
    holdout_evaluator: Optional[Union[Callable[[str], Tuple[Dict[str, float], Dict[str, Any]]], Dict[str, Any]]],
    kwargs: Dict[str, Any]
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Helper to cleanly extract forecast error and inventory simulation metrics for a selected model."""
    if callable(holdout_evaluator):
        return holdout_evaluator(model_name)
    elif isinstance(holdout_evaluator, dict) and model_name in holdout_evaluator:
        val = holdout_evaluator[model_name]
        if isinstance(val, tuple) and len(val) == 2:
            return val[0], val[1]
        elif isinstance(val, dict):
            return val.get("Holdout_Forecast_Metrics", {}), val.get("Holdout_Inventory_Metrics", {})
    # Fallback from kwargs
    if "holdout_forecast_metrics" in kwargs and "holdout_inventory_metrics" in kwargs:
        return kwargs["holdout_forecast_metrics"], kwargs["holdout_inventory_metrics"]
    return {}, {}


# ==============================================================================
# Cross-Regime Empirical Study Runners
# ==============================================================================

def run_walmart_policy_study(output_dir: str = "results") -> Dict[str, Any]:
    """
    Executes walk-forward CV, inventory cost tracking, and 4-policy locked holdout evaluation
    for the strictly smooth Walmart Weekly Store Sales regime.
    """
    print(">>> [Policy Study 1/3] Running Walmart Weekly Store Sales (Smooth Regime)...")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    df_w = load_walmart_sales("data/Walmart_Sales.csv")
    df_w = add_cyclical_calendar_features(df_w, "Date")
    df_w = add_leak_free_lags(df_w, "Store", "Date", "Weekly_Sales", lags=[1, 2, 4, 52])
    df_w = add_leak_free_rolling(df_w, "Store", "Date", "Weekly_Sales", windows=[4, 12])
    df_w = add_demand_momentum(df_w, "Store", "Date", "Weekly_Sales", base_window=4)
    df_clean = df_w.dropna().reset_index(drop=True)

    dev_df, holdout_df = split_locked_final_holdout(df_clean, "Date", holdout_periods=26)

    features = [
        "Store", "Holiday_Flag", "Temperature", "Fuel_Price", "CPI", "Unemployment",
        "sin_month", "cos_month", "sin_week", "cos_week",
        "Weekly_Sales_Lag1", "Weekly_Sales_Lag2", "Weekly_Sales_Lag4", "Weekly_Sales_Lag52",
        "Weekly_Sales_RollMean4", "Weekly_Sales_RollMean12",
        "Weekly_Sales_RollStd4", "Weekly_Sales_Momentum4"
    ]

    cv_splitter = ChronologicalTimeSeriesSplit(n_splits=3, horizon_periods=12, min_train_periods=24)

    config = InventoryConfig(
        lead_time=2,
        service_level=0.95,
        order_setup_cost=2500.0,
        holding_rate_annual=0.20,
        unit_cost=1.0,
        stockout_penalty_per_unit=0.20,
        periods_per_year=52
    )

    models_factory = {
        "Seasonal_Naive": lambda: SeasonalNaiveModel("Weekly_Sales_Lag52"),
        "Lag1_Naive": lambda: Lag1NaiveModel("Weekly_Sales_Lag1"),
        "Recent_Average": lambda: RecentAverageModel("Weekly_Sales_RollMean4"),
        "Random_Forest": lambda: create_random_forest(n_estimators=100, max_depth=12),
        "XGBoost_MSE": lambda: create_xgboost_regressor(n_estimators=100, max_depth=4, learning_rate=0.03, reg_alpha=0.5, reg_lambda=2.0),
        "LightGBM_MSE": lambda: create_lightgbm_regressor(n_estimators=100, num_leaves=20, learning_rate=0.03, reg_alpha=0.5, reg_lambda=2.0)
    }

    cv_results = {}
    for name, factory in models_factory.items():
        print(f"    Walmart Walk-Forward CV: {name}...")
        fold_metrics = []
        fold_costs = []
        fold_fills = []

        for fold_i, (tr_idx, val_idx) in enumerate(cv_splitter.split(dev_df, "Date"), 1):
            tr, val = dev_df.loc[tr_idx], dev_df.loc[val_idx]
            m = factory()
            m.fit(tr[features], tr["Weekly_Sales"])
            preds = m.predict(val[features])
            y_val = val["Weekly_Sales"].to_numpy()

            m_dict = evaluate_all_metrics(y_val, preds, f"{name}_Fold_{fold_i}")
            fold_metrics.append(m_dict)

            # Store 1 validation simulation
            s1_mask = val["Store"] == 1
            y_s1 = val.loc[s1_mask, "Weekly_Sales"].to_numpy()
            p_s1 = preds[s1_mask]
            sim_res = simulate_series_replenishment(y_s1, p_s1, config)
            fold_costs.append(sim_res["Total_Cost"])
            fold_fills.append(sim_res["Fill_Rate_Pct"])

        cv_results[name] = {
            "Model": name,
            "CV_MAE_Mean": float(np.mean([f["MAE"] for f in fold_metrics])),
            "CV_RMSE_Mean": float(np.mean([f["RMSE"] for f in fold_metrics])),
            "CV_Bias_Pct_Mean": float(np.mean([f["Bias_Percentage"] for f in fold_metrics])),
            "CV_WAPE_Mean": float(np.mean([f["WAPE"] for f in fold_metrics])),
            "CV_Cost_Mean": float(np.mean(fold_costs)),
            "CV_Fill_Rate_Mean": float(np.mean(fold_fills)),
            "Fold_Details": fold_metrics
        }

    # Locked holdout fitting & simulation
    X_dev, y_dev = dev_df[features], dev_df["Weekly_Sales"]
    X_hold, y_hold = holdout_df[features], holdout_df["Weekly_Sales"]

    holdout_eval = {}
    for name, factory in models_factory.items():
        m = factory()
        m.fit(X_dev, y_dev)
        preds = m.predict(X_hold)
        f_metrics = evaluate_all_metrics(y_hold.to_numpy(), preds, name)

        # Simulation on Store 1
        s1_mask = holdout_df["Store"] == 1
        y_s1 = y_hold[s1_mask].to_numpy()
        p_s1 = preds[s1_mask]
        i_metrics = simulate_series_replenishment(y_s1, p_s1, config)

        holdout_eval[name] = (f_metrics, i_metrics)

    # Evaluate 4 Policies
    res_a = evaluate_policy_a_naive(
        cv_results=cv_results,
        holdout_evaluator=holdout_eval,
        baseline_model_name="Seasonal_Naive",
        dataset_name="Walmart Weekly Store Sales",
        demand_regime="Smooth (ADI=1.00, CV^2=0.29)"
    )
    res_b = evaluate_policy_b_accuracy(
        cv_results=cv_results,
        holdout_evaluator=holdout_eval,
        dataset_name="Walmart Weekly Store Sales",
        demand_regime="Smooth (ADI=1.00, CV^2=0.29)"
    )
    res_c = evaluate_policy_c_bias_constrained(
        cv_results=cv_results,
        holdout_evaluator=holdout_eval,
        bias_threshold=0.05,
        dataset_name="Walmart Weekly Store Sales",
        demand_regime="Smooth (ADI=1.00, CV^2=0.29)"
    )
    res_d = evaluate_policy_d_cost_aware(
        cv_results=cv_results,
        holdout_evaluator=holdout_eval,
        dataset_name="Walmart Weekly Store Sales",
        demand_regime="Smooth (ADI=1.00, CV^2=0.29)"
    )

    out = {
        "Dataset": "Walmart Weekly Store Sales",
        "Demand_Regime": "Smooth (ADI=1.00, CV^2=0.29)",
        "Holdout_Horizon": "26 weeks locked holdout",
        "Benchmark_Entity": "Store 1 ($ monetary inventory, L=2 weeks, alpha=95%, S=$2,500, i=0.20)",
        "CV_Candidate_Results": cv_results,
        "Policy_A_Naive": res_a.to_dict(),
        "Policy_B_Accuracy": res_b.to_dict(),
        "Policy_C_Bias_Constrained": res_c.to_dict(),
        "Policy_D_Cost_Aware": res_d.to_dict()
    }

    json_path = os.path.join(output_dir, "walmart_policy_comparison.json")
    with open(json_path, "w") as f:
        json.dump(make_json_serializable(out), f, indent=2)

    return out


def run_retail_policy_study(output_dir: str = "results") -> Dict[str, Any]:
    """
    Executes walk-forward CV, inventory cost tracking, and 4-policy locked holdout evaluation
    for the lumpy/erratic Online Retail continuous daily SKU panel.
    """
    print(">>> [Policy Study 2/3] Running Online Retail SKU Panel (Lumpy/Erratic Regime)...")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    df_daily = load_online_retail("data/Online Retail.xlsx", cache_path="data/Online_Retail_daily.csv")
    df = get_online_retail_continuous_panel(df_daily, n_top_skus=30, include_segments=True)
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

    cv_splitter = ChronologicalTimeSeriesSplit(n_splits=3, horizon_periods=14, min_train_periods=60)

    config = InventoryConfig(
        lead_time=7,
        service_level=0.95,
        order_setup_cost=50.0,
        holding_rate_annual=0.20,
        unit_cost=10.0,
        stockout_penalty_per_unit=25.0,
        periods_per_year=365
    )

    models_factory = {
        "Lag1_Naive": lambda: Lag1NaiveModel("Daily_Quantity_Lag1"),
        "Recent_Average": lambda: RecentAverageModel("Daily_Quantity_RollMean7"),
        "Random_Forest": lambda: create_random_forest(n_estimators=100, max_depth=8),
        "LightGBM_MSE": lambda: create_lightgbm_regressor(objective="regression", n_estimators=80, num_leaves=12, max_depth=3, learning_rate=0.02, reg_alpha=2.0, reg_lambda=5.0),
        "LightGBM_Tweedie": lambda: create_lightgbm_regressor(objective="tweedie", tweedie_variance_power=1.2, n_estimators=80, num_leaves=15, learning_rate=0.03),
        "XGBoost_MSE": lambda: create_xgboost_regressor(n_estimators=80, max_depth=3, learning_rate=0.03, reg_alpha=5.0, reg_lambda=10.0),
        "XGBoost_Log1p": lambda: Log1pModelWrapper(lambda: create_xgboost_regressor(n_estimators=80, max_depth=5, learning_rate=0.03))
    }

    cv_results = {}
    for name, factory in models_factory.items():
        print(f"    Online Retail Walk-Forward CV: {name}...")
        fold_metrics = []
        fold_costs = []
        fold_fills = []

        for fold_i, (tr_idx, val_idx) in enumerate(cv_splitter.split(dev_df, "Date"), 1):
            tr, val = dev_df.loc[tr_idx], dev_df.loc[val_idx]
            m = factory()
            m.fit(tr[features], tr["Daily_Quantity"])
            preds = np.clip(m.predict(val[features]), 0.0, None)
            y_val = val["Daily_Quantity"].to_numpy()

            m_dict = evaluate_all_metrics(y_val, preds, f"{name}_Fold_{fold_i}")
            fold_metrics.append(m_dict)

            # Fast panel simulation across all 30 SKUs on fold
            sim_panel = simulate_panel_replenishment(val, "StockCode", "Daily_Quantity", preds, config)
            fold_costs.append(sim_panel["Total_Cost"])
            fold_fills.append(sim_panel["Fill_Rate_Pct"])

        cv_results[name] = {
            "Model": name,
            "CV_MAE_Mean": float(np.mean([f["MAE"] for f in fold_metrics])),
            "CV_RMSE_Mean": float(np.mean([f["RMSE"] for f in fold_metrics])),
            "CV_Bias_Pct_Mean": float(np.mean([f["Bias_Percentage"] for f in fold_metrics])),
            "CV_WAPE_Mean": float(np.mean([f["WAPE"] for f in fold_metrics])),
            "CV_Cost_Mean": float(np.mean(fold_costs)),
            "CV_Fill_Rate_Mean": float(np.mean(fold_fills)),
            "Fold_Details": fold_metrics
        }

    # Locked holdout fitting & simulation
    X_dev, y_dev = dev_df[features], dev_df["Daily_Quantity"]
    X_hold, y_hold = holdout_df[features], holdout_df["Daily_Quantity"]

    holdout_eval = {}
    for name, factory in models_factory.items():
        m = factory()
        m.fit(X_dev, y_dev)
        preds = np.clip(m.predict(X_hold), 0.0, None)
        f_metrics = evaluate_all_metrics(y_hold.to_numpy(), preds, name)

        # Full panel simulation across all 30 SKUs on holdout
        i_metrics = simulate_panel_replenishment(holdout_df, "StockCode", "Daily_Quantity", preds, config)
        holdout_eval[name] = (f_metrics, i_metrics)

    res_a = evaluate_policy_a_naive(
        cv_results=cv_results,
        holdout_evaluator=holdout_eval,
        baseline_model_name="Lag1_Naive",
        dataset_name="Online Retail Daily SKU Panel",
        demand_regime="Lumpy/Erratic (61.5% zeros, wholesale spikes)"
    )
    res_b = evaluate_policy_b_accuracy(
        cv_results=cv_results,
        holdout_evaluator=holdout_eval,
        dataset_name="Online Retail Daily SKU Panel",
        demand_regime="Lumpy/Erratic (61.5% zeros, wholesale spikes)"
    )
    res_c = evaluate_policy_c_bias_constrained(
        cv_results=cv_results,
        holdout_evaluator=holdout_eval,
        bias_threshold=0.05,
        dataset_name="Online Retail Daily SKU Panel",
        demand_regime="Lumpy/Erratic (61.5% zeros, wholesale spikes)"
    )
    res_d = evaluate_policy_d_cost_aware(
        cv_results=cv_results,
        holdout_evaluator=holdout_eval,
        dataset_name="Online Retail Daily SKU Panel",
        demand_regime="Lumpy/Erratic (61.5% zeros, wholesale spikes)"
    )

    out = {
        "Dataset": "Online Retail Daily SKU Panel",
        "Demand_Regime": "Lumpy/Erratic (61.5% zeros, wholesale spikes)",
        "Holdout_Horizon": "28 days locked holdout (30 SKUs continuous panel)",
        "Benchmark_Parameters": "L=7 days, alpha=95%, S=$50, unit_cost=$10, p_shortage=$25",
        "CV_Candidate_Results": cv_results,
        "Policy_A_Naive": res_a.to_dict(),
        "Policy_B_Accuracy": res_b.to_dict(),
        "Policy_C_Bias_Constrained": res_c.to_dict(),
        "Policy_D_Cost_Aware": res_d.to_dict()
    }

    json_path = os.path.join(output_dir, "retail_policy_comparison.json")
    with open(json_path, "w") as f:
        json.dump(make_json_serializable(out), f, indent=2)

    return out


def run_m5_policy_study(output_dir: str = "results") -> Dict[str, Any]:
    """
    Executes walk-forward CV, inventory cost tracking, and 4-policy locked holdout evaluation
    for the zero-inflated intermittent M5 Grocery Panel.
    """
    print(">>> [Policy Study 3/3] Running M5 Grocery Panel (Intermittent Regime)...")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    df_m5 = generate_benchmark_panel("m5", n_entities=20, n_periods=180, random_seed=42)
    df_m5 = add_cyclical_calendar_features(df_m5, "date")
    df_m5 = add_leak_free_lags(df_m5, "id", "date", "Sales", lags=[1, 2, 7])
    df_m5 = add_leak_free_rolling(df_m5, "id", "date", "Sales", windows=[7, 28])
    df_clean = df_m5.dropna().reset_index(drop=True)

    dev_df, holdout_df = split_locked_final_holdout(df_clean, "date", holdout_periods=28)

    features = [
        "sell_price", "sin_wday", "cos_wday", "sin_month", "cos_month",
        "Sales_Lag1", "Sales_Lag2", "Sales_Lag7",
        "Sales_RollMean7", "Sales_RollMean28"
    ]

    cv_splitter = ChronologicalTimeSeriesSplit(n_splits=3, horizon_periods=14, min_train_periods=60)

    config = InventoryConfig(
        lead_time=7,
        service_level=0.95,
        order_setup_cost=20.0,
        holding_rate_annual=0.20,
        unit_cost=2.0,
        stockout_penalty_per_unit=5.0,
        periods_per_year=365
    )

    models_factory = {
        "Croston_SBA": lambda: CrostonSBAModel(alpha=0.1),
        "Lag1_Naive": lambda: Lag1NaiveModel("Sales_Lag1"),
        "Recent_Average": lambda: RecentAverageModel("Sales_RollMean7"),
        "Random_Forest": lambda: create_random_forest(n_estimators=80, max_depth=6),
        "XGBoost_MSE": lambda: create_xgboost_regressor(objective="reg:squarederror", n_estimators=60, max_depth=3, learning_rate=0.03, reg_alpha=2.0, reg_lambda=5.0),
        "XGBoost_Tweedie": lambda: create_xgboost_regressor(objective="reg:tweedie", tweedie_variance_power=1.2, n_estimators=60, max_depth=3, learning_rate=0.03, reg_alpha=2.0, reg_lambda=5.0)
    }

    cv_results = {}
    for name, factory in models_factory.items():
        print(f"    M5 Walk-Forward CV: {name}...")
        fold_metrics = []
        fold_costs = []
        fold_fills = []

        for fold_i, (tr_idx, val_idx) in enumerate(cv_splitter.split(dev_df, "date"), 1):
            tr, val = dev_df.loc[tr_idx], dev_df.loc[val_idx]
            m = factory()
            m.fit(tr[features], tr["Sales"])
            preds = np.clip(m.predict(val[features]), 0.0, None)
            y_val = val["Sales"].to_numpy()

            m_dict = evaluate_all_metrics(y_val, preds, f"{name}_Fold_{fold_i}")
            fold_metrics.append(m_dict)

            # Multi-entity panel simulation across 20 grocery items on fold
            sim_panel = simulate_panel_replenishment(val, "id", "Sales", preds, config)
            fold_costs.append(sim_panel["Total_Cost"])
            fold_fills.append(sim_panel["Fill_Rate_Pct"])

        cv_results[name] = {
            "Model": name,
            "CV_MAE_Mean": float(np.mean([f["MAE"] for f in fold_metrics])),
            "CV_RMSE_Mean": float(np.mean([f["RMSE"] for f in fold_metrics])),
            "CV_Bias_Pct_Mean": float(np.mean([f["Bias_Percentage"] for f in fold_metrics])),
            "CV_WAPE_Mean": float(np.mean([f["WAPE"] for f in fold_metrics])),
            "CV_Cost_Mean": float(np.mean(fold_costs)),
            "CV_Fill_Rate_Mean": float(np.mean(fold_fills)),
            "Fold_Details": fold_metrics
        }

    # Locked holdout evaluation
    X_dev, y_dev = dev_df[features], dev_df["Sales"]
    X_hold, y_hold = holdout_df[features], holdout_df["Sales"]

    holdout_eval = {}
    for name, factory in models_factory.items():
        m = factory()
        m.fit(X_dev, y_dev)
        preds = np.clip(m.predict(X_hold), 0.0, None)
        f_metrics = evaluate_all_metrics(y_hold.to_numpy(), preds, name)

        # Multi-entity panel simulation across all 20 grocery items on holdout
        i_metrics = simulate_panel_replenishment(holdout_df, "id", "Sales", preds, config)
        holdout_eval[name] = (f_metrics, i_metrics)

    res_a = evaluate_policy_a_naive(
        cv_results=cv_results,
        holdout_evaluator=holdout_eval,
        baseline_model_name="Croston_SBA",
        dataset_name="M5 Intermittent Grocery Panel",
        demand_regime="Intermittent (ADI=3.57, CV^2=0.25, 69.8% zeros)"
    )
    res_b = evaluate_policy_b_accuracy(
        cv_results=cv_results,
        holdout_evaluator=holdout_eval,
        dataset_name="M5 Intermittent Grocery Panel",
        demand_regime="Intermittent (ADI=3.57, CV^2=0.25, 69.8% zeros)"
    )
    res_c = evaluate_policy_c_bias_constrained(
        cv_results=cv_results,
        holdout_evaluator=holdout_eval,
        bias_threshold=0.05,
        dataset_name="M5 Intermittent Grocery Panel",
        demand_regime="Intermittent (ADI=3.57, CV^2=0.25, 69.8% zeros)"
    )
    res_d = evaluate_policy_d_cost_aware(
        cv_results=cv_results,
        holdout_evaluator=holdout_eval,
        dataset_name="M5 Intermittent Grocery Panel",
        demand_regime="Intermittent (ADI=3.57, CV^2=0.25, 69.8% zeros)"
    )

    out = {
        "Dataset": "M5 Intermittent Grocery Panel",
        "Demand_Regime": "Intermittent (ADI=3.57, CV^2=0.25, 69.8% zeros)",
        "Holdout_Horizon": "28 days locked holdout (20 grocery items panel)",
        "Benchmark_Parameters": "L=7 days, alpha=95%, S=$20, unit_cost=$2, p_shortage=$5",
        "CV_Candidate_Results": cv_results,
        "Policy_A_Naive": res_a.to_dict(),
        "Policy_B_Accuracy": res_b.to_dict(),
        "Policy_C_Bias_Constrained": res_c.to_dict(),
        "Policy_D_Cost_Aware": res_d.to_dict()
    }

    json_path = os.path.join(output_dir, "m5_policy_comparison.json")
    with open(json_path, "w") as f:
        json.dump(make_json_serializable(out), f, indent=2)

    return out


def run_full_cross_regime_study(output_dir: str = "results") -> Dict[str, Any]:
    """
    Executes the complete cross-regime empirical evaluation across Walmart, Online Retail, and M5,
    saving machine-readable JSON artifacts and summary_table.csv to output_dir.
    """
    print("================================================================================")
    print("STARTING COMPLETE FOUR-POLICY CROSS-REGIME INVENTORY DECISION STUDY")
    print("================================================================================")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    walmart_res = run_walmart_policy_study(output_dir=output_dir)
    retail_res = run_retail_policy_study(output_dir=output_dir)
    m5_res = run_m5_policy_study(output_dir=output_dir)

    full_summary = {
        "Study_Title": "Cross-Regime Empirical Investigation of Forecast Selection Policies vs Downstream Inventory Cost",
        "Regimes_Evaluated": [
            "Walmart Weekly Store Sales (Smooth: ADI=1.00, CV^2=0.29)",
            "Online Retail Daily SKU Panel (Lumpy/Erratic: 61.5% zeros, wholesale spikes)",
            "M5 Grocery Series (Intermittent: ADI=3.57, CV^2=0.25, 69.8% zeros)"
        ],
        "Policies_Evaluated": [
            "Policy A: Historical / Seasonal Naive Baseline",
            "Policy B: Statistical Accuracy-First (Lowest Validation MAE)",
            "Policy C: Operational Bias-Constrained (Lowest MAE with |Bias%| <= 5%)",
            "Policy D: Cost-Aware (Lowest Estimated Inventory Cost on Validation Folds)"
        ],
        "Walmart_Results": walmart_res,
        "Online_Retail_Results": retail_res,
        "M5_Grocery_Results": m5_res
    }

    full_json_path = os.path.join(output_dir, "full_cross_regime_summary.json")
    with open(full_json_path, "w") as f:
        json.dump(make_json_serializable(full_summary), f, indent=2)

    # Build CSV Summary Table
    csv_rows = []
    regimes_data = [
        ("Walmart Weekly Store Sales", "Smooth (ADI=1.00, CV^2=0.29)", walmart_res),
        ("Online Retail Daily SKU Panel", "Lumpy/Erratic (61.5% zeros, wholesale spikes)", retail_res),
        ("M5 Intermittent Grocery Panel", "Intermittent (ADI=3.57, CV^2=0.25, 69.8% zeros)", m5_res)
    ]

    policy_keys = [
        ("Policy A", "Policy_A_Naive", "Naive Baseline"),
        ("Policy B", "Policy_B_Accuracy", "Accuracy-First (Min MAE)"),
        ("Policy C", "Policy_C_Bias_Constrained", "Bias-Constrained (|Bias%|<=5%)"),
        ("Policy D", "Policy_D_Cost_Aware", "Cost-Aware (Min Cost)")
    ]

    for ds_label, regime_label, d_res in regimes_data:
        for pol_code, pol_key, pol_desc in policy_keys:
            p_data = d_res.get(pol_key, d_res.get(pol_key.replace("_", " "), {}))
            f_m = p_data.get("Holdout_Forecast_Metrics", {})
            i_m = p_data.get("Holdout_Inventory_Metrics", {})

            csv_rows.append({
                "Dataset": ds_label,
                "Demand_Regime": regime_label,
                "Policy": pol_code,
                "Policy_Description": pol_desc,
                "Selected_Model": p_data.get("Selected_Model", "N/A"),
                "Val_Selection_Metric": p_data.get("Validation_Metric_Name", "N/A"),
                "Val_Selection_Value": round(float(p_data.get("Validation_Metric_Value", 0.0)), 2),
                "Holdout_MAE": round(float(f_m.get("MAE", 0.0)), 2),
                "Holdout_RMSE": round(float(f_m.get("RMSE", 0.0)), 2),
                "Holdout_WAPE": round(float(f_m.get("WAPE", 0.0)), 2),
                "Holdout_Bias_Pct": round(float(f_m.get("Bias_Percentage", 0.0)), 2),
                "Holdout_Fill_Rate_Pct": round(float(i_m.get("Fill_Rate_Pct", 100.0)), 2),
                "Holdout_Stockout_Days": int(i_m.get("Stockout_Days", 0)),
                "Holdout_Holding_Cost": round(float(i_m.get("Holding_Cost", 0.0)), 2),
                "Holdout_Ordering_Cost": round(float(i_m.get("Ordering_Cost", 0.0)), 2),
                "Holdout_Shortage_Cost": round(float(i_m.get("Shortage_Cost", 0.0)), 2),
                "Holdout_Total_Cost": round(float(i_m.get("Total_Cost", 0.0)), 2)
            })

    df_csv = pd.DataFrame(csv_rows)
    csv_path = os.path.join(output_dir, "summary_table.csv")
    df_csv.to_csv(csv_path, index=False)

    print(f"\nSaved cross-regime artifacts successfully to {output_dir}/:")
    print(f"  - {os.path.join(output_dir, 'walmart_policy_comparison.json')}")
    print(f"  - {os.path.join(output_dir, 'retail_policy_comparison.json')}")
    print(f"  - {os.path.join(output_dir, 'm5_policy_comparison.json')}")
    print(f"  - {os.path.join(output_dir, 'full_cross_regime_summary.json')}")
    print(f"  - {os.path.join(output_dir, 'summary_table.csv')}")
    print("================================================================================")

    return full_summary


if __name__ == "__main__":
    run_full_cross_regime_study()
