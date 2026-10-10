"""
tests/test_metrics.py - Unit tests for metric module
"""

import numpy as np
import pytest
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


def test_perfect_predictions():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    assert calc_mae(y, y) == 0.0
    assert calc_rmse(y, y) == 0.0
    assert calc_r2(y, y) == 1.0
    assert calc_bias_percentage(y, y) == 0.0
    assert calc_bias_absolute(y, y) == 0.0
    assert calc_wape(y, y) == 0.0
    assert calc_underforecast_rate(y, y) == 0.0


def test_positive_and_negative_bias():
    y_true = np.array([100.0, 100.0, 100.0, 100.0])
    # Over-forecasting by 10%
    y_over = np.array([110.0, 110.0, 110.0, 110.0])
    assert pytest.approx(calc_bias_percentage(y_true, y_over), rel=1e-5) == 10.0
    assert pytest.approx(calc_bias_absolute(y_true, y_over), rel=1e-5) == 10.0
    
    # Under-forecasting by 20%
    y_under = np.array([80.0, 80.0, 80.0, 80.0])
    assert pytest.approx(calc_bias_percentage(y_true, y_under), rel=1e-5) == -20.0
    assert pytest.approx(calc_bias_absolute(y_true, y_under), rel=1e-5) == -20.0
    assert calc_underforecast_rate(y_true, y_under) == 1.0


def test_mape_and_wape_zero_handling():
    y_true = np.array([0.0, 0.0, 10.0, 20.0])
    y_pred = np.array([1.0, 2.0, 12.0, 18.0])
    
    # Filter policy ignores first two zeros
    # |10-12|/10 = 0.20, |20-18|/20 = 0.10 -> mean = 0.15 -> 15%
    mape_filt = calc_mape(y_true, y_pred, zero_policy="filter")
    assert pytest.approx(mape_filt, rel=1e-5) == 15.0
    
    # WAPE: sum(|diff|) = 1+2+2+2 = 7; sum(true) = 30; 7/30 = 23.333%
    wape = calc_wape(y_true, y_pred)
    assert pytest.approx(wape, rel=1e-5) == (7.0 / 30.0) * 100.0


def test_evaluate_all_metrics_dictionary():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 19.0, 29.0])
    res = evaluate_all_metrics(y_true, y_pred, "TestModel")
    assert res["Model"] == "TestModel"
    assert res["Observations"] == 3
    assert "MAE" in res
    assert "RMSE" in res
    assert "WAPE" in res
    assert "Bias_Percentage" in res


def test_length_mismatch_raises():
    with pytest.raises(AssertionError):
        evaluate_all_metrics(np.array([1, 2]), np.array([1, 2, 3]))


def test_bias_zero_denominator_returns_nan():
    # If all true demands are 0, percentage bias is undefined and MUST return np.nan
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([5.0, 10.0, 15.0])
    bias_pct = calc_bias_percentage(y_true, y_pred)
    assert np.isnan(bias_pct), "Zero actual total demand must return NaN, not 0.0!"


def test_pinball_loss_and_interval_coverage():
    from src.metrics import calc_pinball_loss, calc_prediction_interval_coverage
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert calc_pinball_loss(y_true, y_pred, quantile=0.5) == 0.0
    
    # Asymmetric pinball loss test
    y_under = np.array([5.0, 15.0, 25.0]) # under-forecast by 5
    # for q=0.8: err = +5 -> loss = 0.8 * 5 = 4.0
    assert pytest.approx(calc_pinball_loss(y_true, y_under, quantile=0.8), rel=1e-5) == 4.0
    
    # Coverage
    lower = np.array([8.0, 15.0, 25.0])
    upper = np.array([12.0, 25.0, 28.0]) # 3rd point (30) out of bounds
    cov = calc_prediction_interval_coverage(y_true, lower, upper)
    assert pytest.approx(cov, rel=1e-5) == (2.0 / 3.0) * 100.0


def test_wape_zero_denominator_returns_nan():
    """WAPE must return np.nan when sum(y_true) == 0, never silent 0.0 or division error."""
    y_true = np.zeros(10)
    y_pred = np.ones(10) * 5.0
    wape = calc_wape(y_true, y_pred)
    assert np.isnan(wape), "WAPE on zero actual total demand must return NaN!"


def test_metric_length_mismatch_assertions():
    """All metric functions must enforce length equality and raise AssertionError on mismatch."""
    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(AssertionError, match="Length mismatch"):
        calc_mae(a, b)
    with pytest.raises(AssertionError, match="Length mismatch"):
        calc_rmse(a, b)
    with pytest.raises(AssertionError, match="Length mismatch"):
        calc_r2(a, b)
    with pytest.raises(AssertionError, match="Length mismatch"):
        calc_wape(a, b)
    with pytest.raises(AssertionError, match="Length mismatch"):
        calc_bias_percentage(a, b)
    with pytest.raises(AssertionError, match="Length mismatch"):
        calc_bias_absolute(a, b)
    with pytest.raises(AssertionError, match="Length mismatch"):
        calc_underforecast_rate(a, b)
    with pytest.raises(AssertionError, match="Length mismatch"):
        calc_pinball_loss(a, b)


def test_calc_r2_edge_cases():
    """R2 should return 0.0 when Var(y_true) == 0 or len(y_true) < 2, and negative when worse than mean."""
    # Zero variance (constant actual)
    y_const = np.array([5.0, 5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 6.0, 4.0, 5.0])
    assert calc_r2(y_const, y_pred) == 0.0
    
    # Single sample
    assert calc_r2(np.array([10.0]), np.array([10.0])) == 0.0
    
    # Negative R2 (worse than mean predictor)
    y_true = np.array([1.0, 2.0, 3.0])
    y_terrible = np.array([100.0, 200.0, 300.0])
    assert calc_r2(y_true, y_terrible) < 0.0


def test_pinball_loss_quantile_bounds():
    """Quantile must be strictly between 0 and 1 exclusive."""
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="Quantile must be between 0 and 1"):
        calc_pinball_loss(y, y, quantile=0.0)
    with pytest.raises(ValueError, match="Quantile must be between 0 and 1"):
        calc_pinball_loss(y, y, quantile=1.0)
    with pytest.raises(ValueError, match="Quantile must be between 0 and 1"):
        calc_pinball_loss(y, y, quantile=-0.1)


def test_empty_arrays_handling():
    """Verify empty input handling across metric functions."""
    empty = np.array([], dtype=float)
    assert np.isnan(calc_wape(empty, empty))
    assert np.isnan(calc_bias_percentage(empty, empty))
    assert np.isnan(calc_mape(empty, empty))
    assert np.isnan(calc_bias_absolute(empty, empty))
    assert np.isnan(calc_underforecast_rate(empty, empty))
    assert np.isnan(calc_pinball_loss(empty, empty))
    assert calc_r2(empty, empty) == 0.0
    with pytest.raises(ValueError):
        calc_mae(empty, empty)
    with pytest.raises(ValueError):
        calc_rmse(empty, empty)


def test_signed_operational_bias_strict_conventions():
    """
    Directional Operational Convention:
    Positive Bias% means over-forecasting (y_pred > y_true), elevating inventory holding costs.
    Negative Bias% means under-forecasting (y_pred < y_true), inducing stockout and lost sales risk.
    """
    y_true = np.array([50.0, 50.0])
    y_over = np.array([60.0, 60.0]) # sum pred = 120, sum true = 100 -> +20%
    y_under = np.array([40.0, 40.0]) # sum pred = 80, sum true = 100 -> -20%

    assert calc_bias_percentage(y_true, y_over) == 20.0
    assert calc_bias_absolute(y_true, y_over) == 10.0
    assert calc_underforecast_rate(y_true, y_over) == 0.0

    assert calc_bias_percentage(y_true, y_under) == -20.0
    assert calc_bias_absolute(y_true, y_under) == -10.0
    assert calc_underforecast_rate(y_true, y_under) == 1.0


def test_classification_metrics_and_roc_auc():
    from src.metrics import calc_roc_auc, calc_accuracy, calc_f1, evaluate_classification_metrics
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    
    auc_val = calc_roc_auc(y_true, y_prob)
    assert auc_val == 1.0
    assert calc_accuracy(y_true, (y_prob >= 0.5).astype(int)) == 1.0
    assert calc_f1(y_true, (y_prob >= 0.5).astype(int)) == 1.0
    
    res = evaluate_classification_metrics(y_true, y_prob, threshold=0.5, model_name="PerfectClf")
    assert res["ROC_AUC"] == 1.0
    assert res["Accuracy"] == 1.0
    assert res["F1_Score"] == 1.0
    assert res["Brier_Score"] < 0.05
    
    # Degenerate single-class edge case returns NaN for ROC AUC
    y_single = np.array([0, 0, 0])
    y_single_prob = np.array([0.1, 0.2, 0.3])
    assert np.isnan(calc_roc_auc(y_single, y_single_prob))



