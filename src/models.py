"""
src/models.py - Standardized Estimators and Forecasters
Provides classical time-series baselines (Naive, Seasonal Naive, Croston SBA)
and production tree ensembles (Random Forest, XGBoost, LightGBM).
"""

import numpy as np
import pandas as pd
from typing import Optional, Union
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier


class Lag1NaiveModel(BaseEstimator, RegressorMixin):
    """Predicts directly from the Lag1 feature column."""

    def __init__(self, lag_col_name: str = "Weekly_Sales_Lag1"):
        self.lag_col_name = lag_col_name

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.lag_col_name in X.columns:
            return X[self.lag_col_name].fillna(0.0).to_numpy()
        elif isinstance(X, np.ndarray):
            return X[:, 0]
        else:
            raise ValueError(f"Column {self.lag_col_name} not found in features.")


class SeasonalNaiveModel(BaseEstimator, RegressorMixin):
    """Predicts from the seasonal lag feature column (e.g. Lag52)."""

    def __init__(self, seasonal_col_name: str = "Weekly_Sales_Lag52"):
        self.seasonal_col_name = seasonal_col_name

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.seasonal_col_name in X.columns:
            return X[self.seasonal_col_name].fillna(0.0).to_numpy()
        elif isinstance(X, np.ndarray):
            return X[:, 0]
        else:
            raise ValueError(f"Column {self.seasonal_col_name} not found in features.")


class RecentAverageModel(BaseEstimator, RegressorMixin):
    """Predicts from a rolling recent-average feature column (e.g. RollMean4 or RollMean7)."""

    def __init__(self, rolling_col_name: str = "Weekly_Sales_RollMean4"):
        self.rolling_col_name = rolling_col_name

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(X, pd.DataFrame) and self.rolling_col_name in X.columns:
            return X[self.rolling_col_name].fillna(0.0).to_numpy()
        elif isinstance(X, np.ndarray):
            return X[:, 0]
        else:
            raise ValueError(f"Column {self.rolling_col_name} not found in features.")


class CrostonSBAModel(BaseEstimator, RegressorMixin):
    """
    Syntetos-Boylan Approximation (SBA) for intermittent demand forecasting.
    Deflates Croston's forecast by (1 - alpha/2) to eliminate positive bias on intermittent zeros.
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.demand_level_ = 1.0
        self.interval_level_ = 1.0
        self.forecast_ = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y_vals = np.asarray(y, dtype=float)
        p = 1.0  # inter-arrival interval
        z = y_vals[y_vals > 0][0] if np.any(y_vals > 0) else 1.0  # non-zero demand level
        q = 1.0  # periods since last positive demand
        
        for val in y_vals:
            if val > 0:
                z = self.alpha * val + (1.0 - self.alpha) * z
                p = self.alpha * q + (1.0 - self.alpha) * p
                q = 1.0
            else:
                q += 1.0
                
        self.demand_level_ = z
        self.interval_level_ = max(1.0, p)
        # SBA factor
        sba_factor = 1.0 - (self.alpha / 2.0)
        self.forecast_ = float(sba_factor * (self.demand_level_ / self.interval_level_))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.forecast_)


def create_random_forest(
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    random_state: int = 42
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1
    )


def create_xgboost_regressor(
    objective: str = "reg:squarederror",
    tweedie_variance_power: float = 1.5,
    learning_rate: float = 0.05,
    n_estimators: int = 150,
    max_depth: int = 6,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42
) -> XGBRegressor:
    kwargs = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "random_state": random_state,
        "n_jobs": -1
    }
    if objective == "reg:tweedie":
        kwargs["objective"] = "reg:tweedie"
        kwargs["tweedie_variance_power"] = tweedie_variance_power
    else:
        kwargs["objective"] = "reg:squarederror"
    return XGBRegressor(**kwargs)


def create_lightgbm_regressor(
    objective: str = "regression",
    tweedie_variance_power: float = 1.5,
    learning_rate: float = 0.05,
    n_estimators: int = 150,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    random_state: int = 42
) -> LGBMRegressor:
    kwargs = {
        "n_estimators": n_estimators,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "random_state": random_state,
        "n_jobs": -1,
        "verbose": -1
    }
    if objective == "tweedie":
        kwargs["objective"] = "tweedie"
        kwargs["tweedie_variance_power"] = tweedie_variance_power
    else:
        kwargs["objective"] = "regression"
    return LGBMRegressor(**kwargs)


def create_random_forest_classifier(
    n_estimators: int = 100,
    max_depth: Optional[int] = 8,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42
) -> RandomForestClassifier:
    """Creates a regularized Random Forest Classifier for stockout risk & surge prediction."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=3,
        class_weight=class_weight,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1
    )


def create_xgboost_classifier(
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.03,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    reg_alpha: float = 0.5,
    reg_lambda: float = 2.0,
    scale_pos_weight: float = 1.0,
    random_state: int = 42
) -> XGBClassifier:
    """Creates a regularized XGBoost Classifier for stockout risk & surge prediction."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )


def create_lightgbm_classifier(
    n_estimators: int = 100,
    num_leaves: int = 15,
    max_depth: int = 4,
    learning_rate: float = 0.03,
    min_child_samples: int = 20,
    subsample: float = 0.85,
    colsample_bytree: float = 0.85,
    reg_alpha: float = 0.5,
    reg_lambda: float = 2.0,
    class_weight: Optional[str] = "balanced",
    random_state: int = 42
) -> LGBMClassifier:
    """Creates a regularized LightGBM Classifier for stockout risk & surge prediction."""
    return LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )

