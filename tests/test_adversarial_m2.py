"""
tests/test_adversarial_m2.py - Adversarial Stress Test Suite for Milestone M2
Authored by challenger_1 (Empirical Challenger) to stress-test metrics,
data loaders, and chronological validation boundaries.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Callable

from src.metrics import (
    calc_mae,
    calc_rmse,
    calc_r2,
    calc_mape,
    calc_wape,
    calc_bias_percentage,
    calc_bias_absolute,
    calc_underforecast_rate,
    calc_pinball_loss,
    calc_prediction_interval_coverage,
    evaluate_all_metrics
)
from src.data_loader import (
    get_online_retail_continuous_panel,
    generate_benchmark_panel
)
from src.validation import (
    ChronologicalTimeSeriesSplit,
    split_locked_final_holdout,
    run_walk_forward_cv
)


# ==============================================================================
# SECTION 1: ADVERSARIAL STRESS TESTS FOR src/metrics.py
# ==============================================================================

def test_metrics_all_zeros_actuals_and_forecasts():
    """
    Stress test with all-zero actuals and all-zero forecasts.
    Verifies that zero division does not raise unhandled ZeroDivisionError
    and percentage metrics correctly resolve to NaN.
    """
    n = 100
    y_true = np.zeros(n)
    y_pred = np.zeros(n)

    res = evaluate_all_metrics(y_true, y_pred, "AllZeroModel")
    assert res["MAE"] == 0.0
    assert res["RMSE"] == 0.0
    assert res["R2"] == 0.0
    assert np.isnan(res["MAPE_Filtered"])
    assert np.isnan(res["WAPE"])
    assert np.isnan(res["Bias_Percentage"])
    assert res["Bias_Absolute"] == 0.0
    assert res["Underforecast_Rate"] == 0.0
    assert res["Pinball_Loss_Median"] == 0.0


def test_metrics_all_zero_forecast_on_positive_actuals():
    """
    Stress test with positive actual demand and complete zero forecasts (total failure).
    """
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.zeros(4)

    assert calc_mae(y_true, y_pred) == 25.0
    assert calc_wape(y_true, y_pred) == 100.0
    assert calc_bias_percentage(y_true, y_pred) == -100.0
    assert calc_bias_absolute(y_true, y_pred) == -25.0
    assert calc_underforecast_rate(y_true, y_pred) == 1.0


def test_metrics_huge_array_million_elements_performance():
    """
    Stress test with 1,000,000 elements to verify execution speed (< 1.0s)
    and numeric stability under large sums.
    """
    import time
    rng = np.random.RandomState(42)
    n = 1_000_000
    y_true = rng.exponential(scale=15.0, size=n)
    y_pred = y_true + rng.normal(loc=0.0, scale=2.0, size=n)

    t0 = time.time()
    res = evaluate_all_metrics(y_true, y_pred, "HugeArrayModel")
    elapsed = time.time() - t0

    assert elapsed < 1.0, f"Evaluating 10^6 elements took too long: {elapsed:.3f}s"
    assert res["Observations"] == n
    assert np.isfinite(res["MAE"])
    assert np.isfinite(res["RMSE"])
    assert np.isfinite(res["WAPE"])
    assert np.isfinite(res["Bias_Percentage"])


def test_metrics_floating_point_precision_near_zero():
    """
    Stress test near-zero floating point edge cases.
    Identifies behavior when actual sum is near floating-point epsilon (e.g. 1e-12).
    """
    # 1e-12 is below default atol=1e-8 in np.isclose, so it safely returns NaN
    y_true = np.array([1e-12, 1e-12])
    y_pred = np.array([1.0, 1.0])

    assert np.isnan(calc_wape(y_true, y_pred))
    assert np.isnan(calc_bias_percentage(y_true, y_pred))


def test_metrics_nan_and_inf_handling():
    """
    Stress test handling of NaN and Inf.
    Verifies that scikit-learn based metrics raise ValueError while
    custom metrics evaluate gracefully or propagate NaN/Inf.
    """
    y_nan = np.array([1.0, np.nan, 3.0])
    y_clean = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Input contains NaN"):
        calc_mae(y_clean, y_nan)
    with pytest.raises(ValueError, match="Input contains NaN"):
        calc_rmse(y_clean, y_nan)
    with pytest.raises(ValueError, match="Input contains NaN"):
        calc_r2(y_clean, y_nan)

    # Custom formulas propagate NaN
    assert np.isnan(calc_wape(y_nan, y_clean))
    assert np.isnan(calc_bias_percentage(y_nan, y_clean))


def test_metrics_adversarial_negative_demand():
    """
    Stress test negative demand values.
    While physical retail demand is non-negative, financial or transformed demand
    may include negatives. When sum(y_true) == 0 due to offsetting signs,
    WAPE and Bias% must return NaN rather than division error.
    """
    y_true_zero_sum = np.array([25.0, -25.0])
    y_pred = np.array([20.0, -20.0])

    assert np.isnan(calc_wape(y_true_zero_sum, y_pred))
    assert np.isnan(calc_bias_percentage(y_true_zero_sum, y_pred))


def test_metrics_shape_broadcasting_vulnerability():
    """
    ADVERSARIAL CHALLENGE FINDING:
    Demonstrates that passing a 2D column vector (N, 1) alongside a 1D vector (N,)
    passes the length assertion `assert len(y_true) == len(y_pred)`, but triggers
    silent outer broadcasting (N, N) in `y_pred - y_true`, artificially inflating WAPE and Bias%.
    """
    y_true_1d = np.array([10.0, 20.0, 30.0])
    y_pred_2d = np.array([[12.0], [22.0], [32.0]])  # Shape (3, 1)

    # Lengths match
    assert len(y_true_1d) == len(y_pred_2d)

    # In 1D, true WAPE is 10.0% and true Bias% is +10.0%
    y_pred_1d = y_pred_2d.ravel()
    expected_wape = calc_wape(y_true_1d, y_pred_1d)
    expected_bias = calc_bias_percentage(y_true_1d, y_pred_1d)

    assert pytest.approx(expected_wape, rel=1e-5) == 10.0
    assert pytest.approx(expected_bias, rel=1e-5) == 10.0

    # Under 2D broadcasting:
    actual_bias_2d = calc_bias_percentage(y_true_1d, y_pred_2d)
    actual_wape_2d = calc_wape(y_true_1d, y_pred_2d)
    # Hardened metric functions flatten inputs via .ravel(), neutralizing broadcasting
    assert pytest.approx(actual_bias_2d, rel=1e-5) == expected_bias, (
        f"Broadcasting flaw not neutralized: got {actual_bias_2d} vs expected {expected_bias}"
    )
    assert pytest.approx(actual_wape_2d, rel=1e-5) == expected_wape, (
        f"Broadcasting flaw not neutralized in WAPE: got {actual_wape_2d} vs expected {expected_wape}"
    )


# ==============================================================================
# SECTION 2: ADVERSARIAL STRESS TESTS FOR src/data_loader.py (PRICE LEAK FIX)
# ==============================================================================

def test_data_loader_delayed_sku_introduction_no_backward_leak():
    """
    Adversarially challenge price leak prevention:
    Simulate multi-SKU panel where SKUs are introduced on staggered dates
    (Day 0, Day 5, Day 10, Day 20).
    Permute/shuffle the transaction table out of chronological order.
    Verify that across all pre-introduction dates, Avg_UnitPrice is strictly 0.0.
    """
    rng = np.random.RandomState(999)
    dates = pd.date_range("2022-01-01", periods=30, freq="D")
    records = []

    for sku_idx in range(4):
        sku = f"TEST_SKU_{sku_idx}"
        start_day = sku_idx * 5
        intro_price = 15.0 + sku_idx * 10.0

        for d in range(start_day, 30):
            if rng.uniform(0, 1) < 0.7:  # Intermittent transactions
                records.append({
                    "StockCode": sku,
                    "Date": dates[d],
                    "Daily_Quantity": float(rng.randint(1, 50)),
                    "Daily_Revenue": float(rng.randint(1, 50) * intro_price),
                    "Avg_UnitPrice": intro_price + (d - start_day) * 1.5,
                    "Transaction_Count": 1
                })

    # Permute rows randomly to challenge sort assumptions
    df_tx = pd.DataFrame(records).sample(frac=1.0, random_state=42).reset_index(drop=True)
    panel = get_online_retail_continuous_panel(df_tx, n_top_skus=4, include_segments=False)

    for sku in panel["StockCode"].unique():
        sku_panel = panel[panel["StockCode"] == sku].sort_values("Date").reset_index(drop=True)
        tx_sku = df_tx[df_tx["StockCode"] == sku].sort_values("Date").reset_index(drop=True)
        first_tx_date = tx_sku["Date"].min()

        # Pre-introduction window: all dates prior to first transaction MUST be 0.0
        pre_intro = sku_panel[sku_panel["Date"] < first_tx_date]
        if len(pre_intro) > 0:
            pre_intro_prices = pre_intro["Avg_UnitPrice"].values
            assert np.all(pre_intro_prices == 0.0), (
                f"Backward price leak detected for {sku} before {first_tx_date}! "
                f"Observed prices: {pre_intro_prices}"
            )

        # First active day must match the introductory price
        post_intro = sku_panel[sku_panel["Date"] >= first_tx_date]
        assert len(post_intro) > 0
        assert post_intro.iloc[0]["Avg_UnitPrice"] > 0.0


def test_data_loader_exact_historical_price_invariance():
    """
    Verify that for every SKU and every calendar date t,
    Avg_UnitPrice in the continuous panel equals the price of the most recent
    transaction on or prior to t, and never a transaction at t+1.
    """
    dates = pd.date_range("2022-03-01", periods=20, freq="D")
    records = []
    sku = "SKU_STRICT_HISTORY"

    tx_days = [3, 7, 12, 18]
    prices = [10.0, 12.5, 14.0, 9.99]

    for d, p in zip(tx_days, prices):
        records.append({
            "StockCode": sku,
            "Date": dates[d],
            "Daily_Quantity": 5.0,
            "Daily_Revenue": 5.0 * p,
            "Avg_UnitPrice": p,
            "Transaction_Count": 1
        })

    # Shuffle records
    df_tx = pd.DataFrame(records).sample(frac=1.0, random_state=77).reset_index(drop=True)
    panel = get_online_retail_continuous_panel(df_tx, n_top_skus=1, include_segments=False)
    sku_panel = panel[panel["StockCode"] == sku].sort_values("Date").reset_index(drop=True)
    tx_sku = df_tx[df_tx["StockCode"] == sku].sort_values("Date").reset_index(drop=True)

    for _, row in sku_panel.iterrows():
        dt = row["Date"]
        row_price = row["Avg_UnitPrice"]
        prior_tx = tx_sku[tx_sku["Date"] <= dt]
        expected_price = prior_tx.iloc[-1]["Avg_UnitPrice"] if len(prior_tx) > 0 else 0.0

        assert np.isclose(row_price, expected_price), (
            f"Price lookahead or discrepancy on {dt}: "
            f"expected {expected_price}, got {row_price}"
        )


# ==============================================================================
# SECTION 3: ADVERSARIAL STRESS TESTS FOR src/validation.py
# ==============================================================================

def test_validation_shuffled_data_strictly_leak_free():
    """
    Stress test ChronologicalTimeSeriesSplit with completely shuffled / out-of-order rows.
    Verifies that train and validation indices are strictly temporally disjoint.
    """
    rng = np.random.RandomState(42)
    n_periods = 60
    dates = pd.date_range("2021-01-01", periods=n_periods, freq="D")
    df = pd.DataFrame({
        "Date": np.repeat(dates, 3),
        "Store": np.tile([1, 2, 3], n_periods),
        "Weekly_Sales": rng.uniform(10, 100, size=n_periods * 3)
    })
    # Thoroughly shuffle DataFrame rows
    df_shuffled = df.sample(frac=1.0, random_state=88).reset_index(drop=True)

    cv = ChronologicalTimeSeriesSplit(n_splits=3, horizon_periods=10, min_train_periods=20)

    for fold_num, (tr_idx, val_idx) in enumerate(cv.split(df_shuffled, "Date"), start=1):
        tr_df = df_shuffled.loc[tr_idx]
        val_df = df_shuffled.loc[val_idx]

        # No index overlap
        assert len(set(tr_idx).intersection(set(val_idx))) == 0

        # Strict temporal inequality: max train < min val
        max_tr = tr_df["Date"].max()
        min_val = val_df["Date"].min()
        assert max_tr < min_val, f"Leakage detected in fold {fold_num}: {max_tr} >= {min_val}"


def test_validation_holdout_boundary_disjoint():
    """
    Stress test split_locked_final_holdout:
    Verifies that dev_df and holdout_df have zero temporal overlap.
    """
    dates = pd.date_range("2020-01-01", periods=100, freq="W")
    df = pd.DataFrame({"Date": dates, "Target": np.arange(100)})

    dev_df, holdout_df = split_locked_final_holdout(df, "Date", holdout_periods=26)

    assert dev_df["Date"].max() < holdout_df["Date"].min()
    assert len(holdout_df["Date"].unique()) == 26
    assert len(dev_df) + len(holdout_df) == len(df)


def test_validation_insufficient_periods_raises():
    """
    Boundary stress test:
    Verifies that ChronologicalTimeSeriesSplit rejects datasets with fewer
    periods than min_train_periods + n_splits * horizon_periods.
    """
    dates = pd.date_range("2021-01-01", periods=30, freq="D")
    df = pd.DataFrame({"Date": dates, "y": np.arange(30)})

    cv = ChronologicalTimeSeriesSplit(n_splits=3, horizon_periods=10, min_train_periods=10)
    # required = 10 + 3*10 = 40 > 30
    with pytest.raises(ValueError, match="Insufficient time periods"):
        list(cv.split(df, "Date"))
