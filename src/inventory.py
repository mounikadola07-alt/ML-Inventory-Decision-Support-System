"""
src/inventory.py - Scientifically Defensible Quantitative Inventory Control & Simulation
Implements stochastic continuous-review (s, Q) policies, dimensional validation,
and event-driven chronological replenishment simulation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import norm


def get_z_score(service_level: float = 0.95) -> float:
    """Computes exact standard normal inverse CDF quantile for target cycle service level."""
    if not (0.5 < service_level < 1.0):
        raise ValueError(f"Service level must be between 0.5 and 1.0 (got {service_level})")
    return float(norm.ppf(service_level))


def calculate_safety_stock(
    sigma_error: float,
    lead_time: float,
    service_level: float = 0.95,
    sigma_lead_time: float = 0.0,
    mean_demand: float = 0.0
) -> float:
    """
    Computes Safety Stock (SS) based on forecast error standard error (sigma_e)
    and optional lead-time uncertainty (sigma_L).
    Formula: SS = Z_alpha * sqrt(L * sigma_e^2 + d_hat^2 * sigma_L^2)
    
    CRITICAL: Sized from residual forecast error variance (sigma_e), NOT total demand variance.
    Predictable seasonality and trend are absorbed by the point forecast.
    """
    if sigma_error < 0 or lead_time <= 0:
        return 0.0
    z = get_z_score(service_level)
    variance_total = lead_time * (sigma_error ** 2) + (mean_demand ** 2) * (sigma_lead_time ** 2)
    return float(z * np.sqrt(max(0.0, variance_total)))


def calculate_reorder_point(
    mean_lead_time_demand: float,
    safety_stock: float
) -> float:
    """
    Computes continuous-review Reorder Point (ROP).
    Formula: ROP = d_hat * L + SS
    """
    return float(max(0.0, mean_lead_time_demand + safety_stock))


def calculate_eoq(
    annual_demand: float,
    order_cost: float,
    holding_cost_annual: float
) -> float:
    """
    Economic Order Quantity (EOQ).
    Formula: EOQ = sqrt( (2 * D * S) / H )
    
    Dimensional Rules:
    - Physical units: D in units/yr, S in $/order, H in $/unit/yr (where H = i * unit_cost).
    - Monetary value ($): D in $/yr, S in $/order, H = i (holding rate, $/$/yr, e.g. 0.20).
    """
    if annual_demand <= 0 or order_cost <= 0 or holding_cost_annual <= 0:
        return 0.0
    return float(np.sqrt((2.0 * annual_demand * order_cost) / holding_cost_annual))


class InventorySimulation:
    """
    Event-driven chronological inventory replenishment simulation.
    Strictly forbids lookahead bias; orders placed at day t arrive at day t + lead_time.
    """

    def __init__(
        self,
        lead_time: int = 7,
        unit_cost: float = 10.0,
        holding_rate_annual: float = 0.20,
        order_setup_cost: float = 50.0,
        stockout_penalty_per_unit: float = 20.0,
        days_per_year: int = 365
    ):
        self.lead_time = int(lead_time)
        self.unit_cost = float(unit_cost)
        self.holding_cost_per_day = (holding_rate_annual * unit_cost) / days_per_year
        self.order_cost = float(order_setup_cost)
        self.stockout_cost = float(stockout_penalty_per_unit)
        self.days_per_year = days_per_year

    def simulate(
        self,
        actual_demands: np.ndarray,
        reorder_point: float,
        order_quantity: float,
        initial_inventory: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes step-by-step chronological inventory simulation.
        
        At each period t:
        1. Receive orders placed at t - lead_time.
        2. Realize demand actual_demands[t].
        3. Serve demand from on-hand stock; unmet demand becomes lost sales.
        4. Check inventory position = (on_hand + on_order).
        5. If inventory position <= ROP, place replenishment order of size Q.
        """
        actual_demands = np.asarray(actual_demands, dtype=np.float64)
        n_periods = len(actual_demands)
        
        if initial_inventory is None:
            # Standard initial condition: ROP + Q
            on_hand = float(reorder_point + order_quantity)
        else:
            on_hand = float(initial_inventory)
            
        on_order = 0.0
        # Pipeline: list of tuples (arrival_time, quantity)
        pipeline: List[Tuple[int, float]] = []
        
        history_on_hand = np.zeros(n_periods)
        history_orders = np.zeros(n_periods)
        history_stockouts = np.zeros(n_periods)
        history_served = np.zeros(n_periods)
        
        total_orders_placed = 0
        total_demand = 0.0
        total_served = 0.0
        total_lost_sales = 0.0
        stockout_days = 0
        
        for t in range(n_periods):
            # 1. Receive deliveries
            arrivals = [qty for arr_t, qty in pipeline if arr_t == t]
            for qty in arrivals:
                on_hand += qty
                on_order -= qty
            pipeline = [item for item in pipeline if item[0] != t]
            
            # 2. Demand realization
            d_t = max(0.0, actual_demands[t])
            total_demand += d_t
            
            if on_hand >= d_t:
                served = d_t
                lost = 0.0
                on_hand -= d_t
            else:
                served = on_hand
                lost = d_t - on_hand
                on_hand = 0.0
                stockout_days += 1
                
            total_served += served
            total_lost_sales += lost
            
            history_on_hand[t] = on_hand
            history_stockouts[t] = lost
            history_served[t] = served
            
            # 3. Replenishment decision (Continuous Review: check inventory position)
            inv_position = on_hand + on_order
            if inv_position <= reorder_point:
                # Continuous-review (s, Q): order sufficient integer multiples of Q to bring IP > ROP
                q_single = float(order_quantity)
                if q_single <= 0:
                    q_order = 0.0
                else:
                    deficit = max(0.0, reorder_point - inv_position)
                    n_batches = max(1, int(np.ceil((deficit + 1e-6) / q_single)))
                    q_order = float(n_batches * q_single)

                if q_order > 0:
                    pipeline.append((t + self.lead_time, q_order))
                    on_order += q_order
                    history_orders[t] = q_order
                    total_orders_placed += 1
                
        # Cost Accounting
        holding_cost = float(np.sum(history_on_hand) * self.holding_cost_per_day)
        ordering_cost = float(total_orders_placed * self.order_cost)
        shortage_cost = float(total_lost_sales * self.stockout_cost)
        total_cost = holding_cost + ordering_cost + shortage_cost
        
        fill_rate = float((total_served / total_demand * 100.0) if total_demand > 0 else 100.0)
        stockout_rate = float((stockout_days / n_periods * 100.0) if n_periods > 0 else 0.0)
        
        return {
            "Periods": n_periods,
            "Total_Demand": total_demand,
            "Total_Served": total_served,
            "Total_Lost_Sales": total_lost_sales,
            "Fill_Rate_Pct": fill_rate,
            "Stockout_Days": stockout_days,
            "Stockout_Rate_Pct": stockout_rate,
            "Orders_Placed": total_orders_placed,
            "Average_On_Hand": float(np.mean(history_on_hand)),
            "Holding_Cost": holding_cost,
            "Ordering_Cost": ordering_cost,
            "Shortage_Cost": shortage_cost,
            "Total_Cost": total_cost,
            "History_On_Hand": history_on_hand,
            "History_Orders": history_orders
        }


def run_inventory_sensitivity_analysis(
    demands: np.ndarray,
    forecast_sigma: float,
    mean_forecast: float,
    lead_times: List[int] = [3, 7, 14],
    service_levels: List[float] = [0.90, 0.95, 0.99],
    holding_rate: float = 0.20,
    order_cost: float = 50.0,
    unit_cost: float = 10.0
) -> pd.DataFrame:
    """
    Evaluates policy sensitivity across varying lead times and service levels.
    """
    records = []
    annual_demand = mean_forecast * 365.0
    h_annual = holding_rate * unit_cost
    eoq = calculate_eoq(annual_demand, order_cost, h_annual)
    
    for L in lead_times:
        for alpha in service_levels:
            ss = calculate_safety_stock(forecast_sigma, L, alpha)
            rop = calculate_reorder_point(mean_forecast * L, ss)
            sim = InventorySimulation(lead_time=L, unit_cost=unit_cost, holding_rate_annual=holding_rate, order_setup_cost=order_cost)
            res = sim.simulate(demands, reorder_point=rop, order_quantity=eoq)
            records.append({
                "Lead_Time": L,
                "Service_Level_Target": f"{int(alpha*100)}%",
                "Safety_Stock": round(ss, 2),
                "ROP": round(rop, 2),
                "EOQ": round(eoq, 2),
                "Fill_Rate": f"{res['Fill_Rate_Pct']:.2f}%",
                "Stockout_Days": res['Stockout_Days'],
                "Avg_Inventory": round(res['Average_On_Hand'], 2),
                "Total_Cost": round(res['Total_Cost'], 2)
            })
    return pd.DataFrame(records)
