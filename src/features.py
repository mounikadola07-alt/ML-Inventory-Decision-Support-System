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


def add_rolling_coefficient_of_variation(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    target_col: str,
    windows: List[int] = [7, 14, 28],
    epsilon: float = 1.0
) -> pd.DataFrame:
    """
    Computes leak-free rolling coefficient of variation (CV = RollStd / (RollMean + eps)).
    Enforces shift(1) so current and future observations are never observed.
    """
    df = df.sort_values(by=[group_col, time_col]).copy()
    for w in windows:
        mean_col = f"{target_col}_RollMean{w}"
        std_col = f"{target_col}_RollStd{w}"
        if mean_col not in df.columns:
            df[mean_col] = df.groupby(group_col)[target_col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean()
            )
        if std_col not in df.columns:
            df[std_col] = df.groupby(group_col)[target_col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=2).std().fillna(0.0)
            )
        df[f"{target_col}_RollCV{w}"] = (df[std_col] / (df[mean_col] + epsilon)).fillna(0.0)
    return df


def add_demand_acceleration_and_jerk(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    target_col: str
) -> pd.DataFrame:
    """
    Computes discrete finite difference dynamics: velocity, acceleration, and jerk.
    All differences use strictly historical lags (Lag1, Lag2, Lag3, Lag4) to eliminate lookahead bias.
    Velocity = Lag1 - Lag2
    Acceleration = Lag1 - 2*Lag2 + Lag3
    Jerk = Lag1 - 3*Lag2 + 3*Lag3 - Lag4
    """
    df = df.sort_values(by=[group_col, time_col]).copy()
    for lag in [1, 2, 3, 4]:
        lag_col = f"{target_col}_Lag{lag}"
        if lag_col not in df.columns:
            df[lag_col] = df.groupby(group_col)[target_col].shift(lag)

    l1 = df[f"{target_col}_Lag1"]
    l2 = df[f"{target_col}_Lag2"]
    l3 = df[f"{target_col}_Lag3"]
    l4 = df[f"{target_col}_Lag4"]

    df[f"{target_col}_Velocity"] = (l1 - l2).fillna(0.0)
    df[f"{target_col}_Acceleration"] = (l1 - 2.0 * l2 + l3).fillna(0.0)
    df[f"{target_col}_Jerk"] = (l1 - 3.0 * l2 + 3.0 * l3 - l4).fillna(0.0)
    return df


def add_multi_window_ewma_ratios(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    target_col: str,
    spans: List[int] = [3, 7, 14],
    epsilon: float = 1.0
) -> pd.DataFrame:
    """
    Computes leak-free multi-window exponentially weighted moving averages (EWMA) and fast/slow ratios.
    Enforces shift(1) before ewma calculation.
    """
    df = df.sort_values(by=[group_col, time_col]).copy()
    for span in spans:
        ewma_col = f"{target_col}_EWMA{span}"
        df[ewma_col] = df.groupby(group_col)[target_col].transform(
            lambda s: s.shift(1).ewm(span=span, min_periods=1).mean()
        )

    if 3 in spans and 7 in spans:
        df[f"{target_col}_EWMARatio_3_7"] = (
            df[f"{target_col}_EWMA3"] / (df[f"{target_col}_EWMA7"] + epsilon)
        ).fillna(1.0)
    if 7 in spans and 14 in spans:
        df[f"{target_col}_EWMARatio_7_14"] = (
            df[f"{target_col}_EWMA7"] / (df[f"{target_col}_EWMA14"] + epsilon)
        ).fillna(1.0)
    return df


def add_calendar_interaction_features(
    df: pd.DataFrame,
    date_col: str = "Date",
    demand_mean_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Computes non-leaking calendar interaction features: weekend indicator,
    month-end indicator, and calendar-demand harmonic interactions.
    """
    df = df.copy()
    dates = pd.to_datetime(df[date_col])
    wday = dates.dt.dayofweek
    day = dates.dt.day

    df["is_weekend"] = (wday >= 5).astype(float)
    df["day_of_month"] = day.astype(float)
    df["is_month_end"] = (day >= 25).astype(float)

    if demand_mean_col is not None and demand_mean_col in df.columns:
        df[f"weekend_x_{demand_mean_col}"] = df["is_weekend"] * df[demand_mean_col]
        if "sin_month" in df.columns:
            df[f"month_x_{demand_mean_col}"] = df["sin_month"] * df[demand_mean_col]
    return df


def add_leak_free_expanding_stats(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    target_col: str,
    min_periods: int = 7,
    epsilon: float = 1.0
) -> pd.DataFrame:
    """
    Computes strictly backward-looking cumulative expanding statistics per group entity.
    Uses shift(1).expanding() to guarantee zero current or future information leakage.
    """
    df = df.sort_values(by=[group_col, time_col]).copy()
    exp_mean_col = f"{group_col}_Hist_Expanding_Mean"
    exp_std_col = f"{group_col}_Hist_Expanding_Std"
    exp_max_col = f"{group_col}_Hist_Expanding_Max"

    df[exp_mean_col] = df.groupby(group_col)[target_col].transform(
        lambda s: s.shift(1).expanding(min_periods=min_periods).mean()
    ).fillna(0.0)
    df[exp_std_col] = df.groupby(group_col)[target_col].transform(
        lambda s: s.shift(1).expanding(min_periods=min_periods).std()
    ).fillna(0.0)
    df[exp_max_col] = df.groupby(group_col)[target_col].transform(
        lambda s: s.shift(1).expanding(min_periods=min_periods).max()
    ).fillna(0.0)

    lag1_col = f"{target_col}_Lag1"
    if lag1_col in df.columns:
        df[f"Ratio_{lag1_col}_ExpMean"] = (
            df[lag1_col] / (df[exp_mean_col] + epsilon)
        ).fillna(1.0)

    roll_col = f"{target_col}_RollMean7"
    if roll_col in df.columns:
        df[f"Ratio_{roll_col}_ExpMean"] = (
            df[roll_col] / (df[exp_mean_col] + epsilon)
        ).fillna(1.0)
    return df

