"""
tests/test_inventory.py - Unit tests for inventory optimization and simulation
"""

import numpy as np
import pytest
from src.inventory import (
    get_z_score,
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_eoq,
    InventorySimulation,
    run_inventory_sensitivity_analysis
)


def test_z_score():
    assert pytest.approx(get_z_score(0.95), rel=1e-3) == 1.645
    assert pytest.approx(get_z_score(0.99), rel=1e-3) == 2.326
    with pytest.raises(ValueError):
        get_z_score(1.5)


def test_safety_stock_and_rop():
    sigma_e = 10.0
    L = 4.0
    # SS = 1.644853 * 10 * 2 = 32.897
    ss = calculate_safety_stock(sigma_e, L, 0.95)
    assert pytest.approx(ss, rel=1e-3) == 32.897
    
    mean_lead_time_demand = 100.0
    rop = calculate_reorder_point(mean_lead_time_demand, ss)
    assert pytest.approx(rop, rel=1e-3) == 132.897


def test_eoq_dimensional_consistency():
    # D = 10,000 units/yr, S = $50, H = $2/unit/yr -> EOQ = sqrt(2 * 10000 * 50 / 2) = sqrt(500,000) = 707.1
    eoq = calculate_eoq(annual_demand=10000, order_cost=50, holding_cost_annual=2.0)
    assert pytest.approx(eoq, rel=1e-3) == 707.1068
    
    # Monetary inventory: D = $52,000,000/yr, S = $2,500, i = 0.20 -> EOQ = sqrt(2 * 52M * 2500 / 0.20) = $1,140,175
    eoq_dollar = calculate_eoq(annual_demand=52000000, order_cost=2500, holding_cost_annual=0.20)
    assert pytest.approx(eoq_dollar, rel=1e-3) == 1140175.4


def test_simulation_event_dynamics():
    # 30 days of constant demand 10 units/day
    demands = np.full(30, 10.0)
    lead_time = 3
    # ROP = 10*3 + 10 = 40, Q = 50
    sim = InventorySimulation(lead_time=lead_time, unit_cost=5.0, order_setup_cost=20.0)
    res = sim.simulate(demands, reorder_point=40.0, order_quantity=50.0, initial_inventory=60.0)
    
    assert res["Total_Demand"] == 300.0
    assert res["Fill_Rate_Pct"] == 100.0
    assert res["Stockout_Days"] == 0
    assert res["Total_Lost_Sales"] == 0.0
    assert res["Total_Cost"] == pytest.approx(res["Holding_Cost"] + res["Ordering_Cost"] + res["Shortage_Cost"])
    assert res["Orders_Placed"] > 0


def test_sensitivity_monotonicity():
    demands = np.random.RandomState(42).poisson(lam=15, size=90)
    df_sens = run_inventory_sensitivity_analysis(demands, forecast_sigma=4.0, mean_forecast=15.0)
    assert len(df_sens) == 9 # 3 lead times * 3 service levels
    # Within lead time 7, higher service level should give higher safety stock
    l7 = df_sens[df_sens["Lead_Time"] == 7]
    assert l7.iloc[0]["Safety_Stock"] < l7.iloc[1]["Safety_Stock"] < l7.iloc[2]["Safety_Stock"]


def test_safety_stock_stochastic_lead_time():
    sigma_e = 10.0
    L = 4.0
    mean_d = 50.0
    sigma_L = 1.0  # Lead time standard deviation of 1 day
    
    ss_deterministic = calculate_safety_stock(sigma_e, L, 0.95)
    ss_stochastic = calculate_safety_stock(sigma_e, L, 0.95, sigma_lead_time=sigma_L, mean_demand=mean_d)
    
    # Stochastic lead time increases total variance: L*sigma_e^2 + d^2*sigma_L^2 = 4*100 + 2500*1 = 2900
    # Deterministic variance: 4*100 = 400
    assert ss_stochastic > ss_deterministic
    # sqrt(2900) * 1.644853 = 53.85 * 1.644853 = 88.58
    assert pytest.approx(ss_stochastic, rel=1e-2) == 88.58


def test_simulation_multi_batch_ordering():
    # If a large demand spike creates a deficit of 150 below ROP, with Q=50 it should order 3 batches (150 units)
    demands = np.array([200.0, 0.0, 0.0, 0.0, 0.0])
    sim = InventorySimulation(lead_time=2, unit_cost=5.0, order_setup_cost=20.0)
    # Initial inventory 100, demand 200 -> on_hand drops to 0, ROP = 50, Q = 50
    # Deficit = 50 - 0 = 50 -> orders batches
    res = sim.simulate(demands, reorder_point=50.0, order_quantity=50.0, initial_inventory=100.0)
    assert res["Orders_Placed"] >= 1
    assert res["Total_Served"] == 100.0
    assert res["Total_Lost_Sales"] == 100.0

