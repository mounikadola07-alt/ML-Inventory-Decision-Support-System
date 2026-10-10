"""
src/metrics.py - Production and Research Quality Forecasting Evaluation Metrics
Provides mathematically sound, leakage-free, and edge-case-hardened metrics.
Hardened against 2D column vector broadcasting vulnerabilities via .ravel().
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)


def calc_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    return float(mean_absolute_error(y_true, y_pred))


def calc_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calc_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R^2). Returns 0.0 or negative if worse than mean."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    if len(y_true) < 2 or np.all(y_true == y_true[0]):
        return 0.0
    return float(r2_score(y_true, y_pred))


def calc_mape(y_true: np.ndarray, y_pred: np.ndarray, zero_policy: str = "filter") -> float:
    """
    Mean Absolute Percentage Error (MAPE).
    
    Zero Denominator Policies:
    - 'filter': Calculates MAPE exclusively on rows where y_true > 0 (transparently drops zeros).
    - 'epsilon': Adds a small numerical epsilon (1e-6) to denominator.
    
    Returns percentage value (e.g. 5.2 for 5.2%).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    if len(y_true) == 0:
        return np.nan
    
    if zero_policy == "filter":
        valid = y_true > 0
        if not np.any(valid):
            return np.nan
        return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])) * 100.0)
    elif zero_policy == "epsilon":
        denom = np.where(y_true == 0, 1e-6, y_true)
        return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    else:
        raise ValueError(f"Unknown zero_policy: {zero_policy}. Choose 'filter' or 'epsilon'.")


def calc_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error (WAPE).
    Formula: WAPE = 100 * sum(|y_true - y_pred|) / sum(y_true)
    Stable under intermittent zero-demand distributions.
    Returns np.nan when sum(y_true) == 0.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    if len(y_true) == 0:
        return np.nan
    sum_true = float(np.sum(y_true))
    if sum_true == 0.0 or np.isclose(sum_true, 0.0):
        return np.nan
    return float((np.sum(np.abs(y_true - y_pred)) / sum_true) * 100.0)


def calc_bias_percentage(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Aggregate Operational Percentage Bias.
    Formula: Bias% = 100 * sum(y_pred - y_true) / sum(y_true)
    
    Convention:
    - Positive (> 0): Aggregate over-forecasting (holding cost risk).
    - Negative (< 0): Aggregate under-forecasting (stockout / starvation risk).
    - Returns np.nan when sum(y_true) == 0 (undefined bias; never silently returns 0.0).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    if len(y_true) == 0:
        return np.nan
    sum_true = float(np.sum(y_true))
    if sum_true == 0.0 or np.isclose(sum_true, 0.0):
        return np.nan
    return float((np.sum(y_pred - y_true) / sum_true) * 100.0)


def calc_bias_absolute(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed forecast error (y_pred - y_true)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(y_pred - y_true))


def calc_underforecast_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of periods where forecast is strictly less than actual demand (stockout pressure)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(y_pred < y_true))


def calc_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5) -> float:
    """
    Quantile / Pinball loss: L_q(y, y_hat) = mean(max(q*(y - y_hat), (q-1)*(y - y_hat))).
    Evaluates asymmetric cost between under-forecasting and over-forecasting.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"
    if len(y_true) == 0:
        return np.nan
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"Quantile must be between 0 and 1 exclusive (got {quantile})")
    err = y_true - y_pred
    return float(np.mean(np.maximum(quantile * err, (quantile - 1.0) * err)))


def calc_prediction_interval_coverage(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray
) -> float:
    """
    Empirical Coverage Percentage of prediction interval [lower_bound, upper_bound].
    Returns coverage percentage between 0.0% and 100.0%.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    lower_bound = np.asarray(lower_bound, dtype=np.float64).ravel()
    upper_bound = np.asarray(upper_bound, dtype=np.float64).ravel()
    assert len(y_true) == len(lower_bound) == len(upper_bound), (
        f"Length mismatch: {len(y_true)} vs {len(lower_bound)} vs {len(upper_bound)}"
    )
    if len(y_true) == 0:
        return np.nan
    in_bounds = (y_true >= lower_bound) & (y_true <= upper_bound)
    return float(np.mean(in_bounds) * 100.0)


def evaluate_all_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    model_name: str = "Candidate Model"
) -> Dict[str, Any]:
    """
    Computes standard suite of operational and statistical metrics.
    Guarantees no fabricated metrics; all computed directly from true and pred vectors.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    
    assert len(y_t) == len(y_p), f"Length mismatch: {len(y_t)} vs {len(y_p)}"
    
    return {
        "Model": model_name,
        "Observations": int(len(y_t)),
        "MAE": calc_mae(y_t, y_p),
        "RMSE": calc_rmse(y_t, y_p),
        "R2": calc_r2(y_t, y_p),
        "MAPE_Filtered": calc_mape(y_t, y_p, zero_policy="filter"),
        "WAPE": calc_wape(y_t, y_p),
        "Bias_Percentage": calc_bias_percentage(y_t, y_p),
        "Bias_Absolute": calc_bias_absolute(y_t, y_p),
        "Underforecast_Rate": calc_underforecast_rate(y_t, y_p),
        "Pinball_Loss_Median": calc_pinball_loss(y_t, y_p, quantile=0.5)
    }


def calc_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area Under the ROC Curve. Returns np.nan if only one class is present."""
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def calc_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Classification Accuracy."""
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int32).ravel()
    return float(accuracy_score(y_true, y_pred))


def calc_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 Score (Binary)."""
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int32).ravel()
    return float(f1_score(y_true, y_pred, zero_division=0.0))


def calc_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Area Under Precision-Recall Curve (Average Precision)."""
    y_true = np.asarray(y_true, dtype=np.int32).ravel()
    y_prob = np.asarray(y_prob, dtype=np.float64).ravel()
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_prob))


def evaluate_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_prob: Union[pd.Series, np.ndarray],
    threshold: float = 0.5,
    model_name: str = "Candidate Classifier"
) -> Dict[str, Any]:
    """
    Computes classification metrics for inventory stockout risk and demand surge prediction.
    """
    y_t = np.asarray(y_true, dtype=np.int32).ravel()
    y_p = np.asarray(y_prob, dtype=np.float64).ravel()
    assert len(y_t) == len(y_p), f"Length mismatch: {len(y_t)} vs {len(y_p)}"
    
    y_pred_bin = (y_p >= threshold).astype(np.int32)
    
    roc_auc = calc_roc_auc(y_t, y_p)
    pr_auc = calc_pr_auc(y_t, y_p)
    acc = calc_accuracy(y_t, y_pred_bin)
    f1 = calc_f1(y_t, y_pred_bin)
    precision = float(precision_score(y_t, y_pred_bin, zero_division=0.0))
    recall = float(recall_score(y_t, y_pred_bin, zero_division=0.0))
    brier_score = float(np.mean((y_p - y_t) ** 2))
    
    return {
        "Model": model_name,
        "Observations": int(len(y_t)),
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "Accuracy": acc,
        "F1_Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Brier_Score": brier_score,
        "Threshold": threshold
    }

