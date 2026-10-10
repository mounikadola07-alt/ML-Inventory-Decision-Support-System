"""
tests/test_validation.py - Unit tests for chronological walk-forward validation
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from src.validation import (
    ChronologicalTimeSeriesSplit,
    split_locked_final_holdout,
    run_walk_forward_cv
)


def test_chronological_splits_strictly_expanding_and_disjoint():
    dates = pd.date_range("2021-01-01", periods=60, freq="W")
    df = pd.DataFrame({
        "Date": np.repeat(dates, 2),
        "Store": np.tile([1, 2], 60),
        "Target": np.random.randn(120),
        "Feature": np.random.randn(120)
    })
    
    cv = ChronologicalTimeSeriesSplit(n_splits=3, horizon_periods=6, min_train_periods=20)
    
    prev_train_len = 0
    for fold_i, (tr_idx, val_idx) in enumerate(cv.split(df, "Date")):
        tr_dates = df.loc[tr_idx, "Date"]
        val_dates = df.loc[val_idx, "Date"]
        
        # Strict temporal order
        assert tr_dates.max() < val_dates.min(), f"Fold {fold_i} has temporal overlap"
        # Expanding train set
        assert len(tr_idx) > prev_train_len
        prev_train_len = len(tr_idx)
        # Validation horizon is exactly 6 periods * 2 stores = 12 rows
        assert len(val_idx) == 12


def test_split_locked_final_holdout():
    dates = pd.date_range("2020-01-01", periods=50, freq="W")
    df = pd.DataFrame({"Date": dates, "Sales": range(50)})
    
    dev_df, holdout_df = split_locked_final_holdout(df, "Date", holdout_periods=10)
    assert len(holdout_df) == 10
    assert len(dev_df) == 40
    assert dev_df["Date"].max() < holdout_df["Date"].min()


def test_run_walk_forward_cv_execution():
    dates = pd.date_range("2021-01-01", periods=40, freq="W")
    df = pd.DataFrame({
        "Date": dates,
        "Feat1": np.arange(40, dtype=float),
        "Target": np.arange(40, dtype=float) * 2.0 + 1.0
    })
    cv = ChronologicalTimeSeriesSplit(n_splits=2, horizon_periods=5, min_train_periods=20)
    summary = run_walk_forward_cv(
        df=df,
        feature_cols=["Feat1"],
        target_col="Target",
        time_col="Date",
        model_factory=lambda: Ridge(),
        cv_splitter=cv,
        model_name="RidgeTest"
    )
    assert summary["Model"] == "RidgeTest"
    assert "CV_MAE_Mean" in summary
    assert "CV_RMSE_Mean" in summary
    assert len(summary["Fold_Details"]) == 2
