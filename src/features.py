"""
src/features.py - Leakage-Free Feature Engineering & Panel Construction
Enforces strictly backward-looking lag, rolling window, cyclical harmonics,
and continuous SKU-date panel re-indexing with zero future contamination.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def add_cyclical_calendar_features(
    df: pd.DataFrame,
    date_col: str = "Date"
) -> pd.DataFrame:
    """
    Computes smooth cyclical trigonometric harmonics for annual, seasonal, and weekly cycles.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    
    month = dates.dt.month
    week = dates.dt.isocalendar().week.astype(int)
    wday = dates.dt.dayofweek
    
    df["sin_month"] = np.sin(2.0 * np.pi * month / 12.0)
    df["cos_month"] = np.cos(2.0 * np.pi * month / 12.0)
    df["sin_week"] = np.sin(2.0 * np.pi * week / 52.0)
    df["cos_week"] = np.cos(2.0 * np.pi * week / 52.0)
    df["sin_wday"] = np.sin(2.0 * np.pi * wday / 7.0)
    df["cos_wday"] = np.cos(2.0 * np.pi * wday / 7.0)
    
    return df


def add_leak_free_lags(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    target_col: str,
    lags: List[int] = [1, 2, 4, 8]
) -> pd.DataFrame:
    """
    Generates autoregressive lag features strictly shifting past values.
    Requires data to be sorted chronologically within each group.
    """
    df = df.sort_values(by=[group_col, time_col]).copy()
    for lag in lags:
        assert lag >= 1, "Lag must be >= 1 to prevent same-period lookahead"
        df[f"{target_col}_Lag{lag}"] = df.groupby(group_col)[target_col].shift(lag)
    return df


def add_leak_free_rolling(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    target_col: str,
    windows: List[int] = [4, 12]
) -> pd.DataFrame:
    """
    Generates rolling mean and std features.
    Enforces shift(1) prior to rolling aggregation so the current period is never included.
    """
    df = df.sort_values(by=[group_col, time_col]).copy()
    for w in windows:
        # shift(1) ensures strictly historical observations
        shifted = df.groupby(group_col)[target_col].shift(1)
        df[f"{target_col}_RollMean{w}"] = df.groupby(group_col)[target_col].transform(
            lambda s: s.shift(1).rolling(w, min_periods=1).mean()
        )
        df[f"{target_col}_RollStd{w}"] = df.groupby(group_col)[target_col].transform(
            lambda s: s.shift(1).rolling(w, min_periods=2).std().fillna(0.0)
        )
    return df


def add_demand_momentum(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    target_col: str,
    base_window: int = 4
) -> pd.DataFrame:
    """
    Computes leak-free demand momentum ratio: Lag1 / (RollMean{base_window} + 1.0).
    Captures acceleration or deceleration in recent demand velocity.
    """
    df = df.sort_values(by=[group_col, time_col]).copy()
    roll_col = f"{target_col}_RollMean{base_window}"
    lag_col = f"{target_col}_Lag1"
    if roll_col not in df.columns:
        df[roll_col] = df.groupby(group_col)[target_col].transform(
            lambda s: s.shift(1).rolling(base_window, min_periods=1).mean()
        )
    if lag_col not in df.columns:
        df[lag_col] = df.groupby(group_col)[target_col].shift(1)
    df[f"{target_col}_Momentum{base_window}"] = (df[lag_col] / (df[roll_col].fillna(0.0) + 1.0)).fillna(1.0)
    return df



def clean_prices_forward_fill_only(
    df: pd.DataFrame,
    group_cols: List[str],
    time_col: str,
    price_col: str
) -> pd.DataFrame:
    """
    Cleans price series using forward-fill only.
    Strictly forbids bfill() to eliminate backward leakage of future prices.
    Pre-release periods without prior price observations are filled with 0.0 or group min.
    """
    df = df.sort_values(by=group_cols + [time_col]).copy()
    df[price_col] = df.groupby(group_cols)[price_col].ffill().fillna(0.0)
    return df


def build_continuous_daily_panel(
    df_raw: pd.DataFrame,
    sku_col: str,
    date_col: str,
    qty_col: str,
    min_date: Optional[pd.Timestamp] = None,
    max_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Builds a continuous daily SKU panel including explicit zero-demand calendar dates.
    Resolves the fatal data representation mismatch where inactive days were omitted.
    """
    df = df_raw.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.floor('D')
    
    # Aggregate transaction records to daily total
    daily_agg = df.groupby([sku_col, date_col])[qty_col].sum().reset_index()
    
    start_dt = min_date if min_date is not None else daily_agg[date_col].min()
    end_dt = max_date if max_date is not None else daily_agg[date_col].max()
    
    all_dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
    all_skus = daily_agg[sku_col].unique()
    
    # Construct complete Cartesian grid
    full_index = pd.MultiIndex.from_product([all_skus, all_dates], names=[sku_col, date_col])
    panel = daily_agg.set_index([sku_col, date_col]).reindex(full_index, fill_value=0.0).reset_index()
    
    return panel
