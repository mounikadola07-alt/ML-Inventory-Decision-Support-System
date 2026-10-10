"""
src/validation.py - Chronological Rolling-Origin / Expanding-Window Validation Engine
Enforces strict temporal ordering, locked holdout isolation, identical evaluation row sets,
and reproducible walk-forward cross-validation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Generator, Dict, Any, Callable
from src.metrics import evaluate_all_metrics


class ChronologicalTimeSeriesSplit:
    """
    Expanding-window (rolling-origin) chronological cross-validation splitter.
    Ensures that for every fold k: max(train_time) < min(val_time).
    """

    def __init__(
        self,
        n_splits: int = 3,
        horizon_periods: int = 12,
        min_train_periods: int = 24
    ):
        self.n_splits = n_splits
        self.horizon_periods = horizon_periods
        self.min_train_periods = min_train_periods

    def split(
        self,
        df: pd.DataFrame,
        time_col: str
    ) -> Generator[Tuple[pd.Index, pd.Index], None, None]:
        """
        Yields (train_indices, val_indices) across n_splits expanding chronological windows.
        """
        unique_times = np.sort(df[time_col].unique())
        total_periods = len(unique_times)
        
        required_periods = self.min_train_periods + self.n_splits * self.horizon_periods
        if total_periods < required_periods:
            raise ValueError(
                f"Insufficient time periods ({total_periods}) for {self.n_splits} splits "
                f"with min_train={self.min_train_periods} and horizon={self.horizon_periods}."
            )
            
        # Origin offsets
        step = self.horizon_periods
        # The last fold evaluates up to the last available time in df
        for i in range(self.n_splits):
            origin_idx = self.min_train_periods + i * step
            val_end_idx = min(origin_idx + self.horizon_periods, total_periods)
            
            origin_time = unique_times[origin_idx - 1]
            val_end_time = unique_times[val_end_idx - 1]
            
            train_mask = df[time_col] <= origin_time
            val_mask = (df[time_col] > origin_time) & (df[time_col] <= val_end_time)
            
            train_idx = df.index[train_mask]
            val_idx = df.index[val_mask]
            
            # Assert strict temporal separation
            max_tr = df.loc[train_idx, time_col].max()
            min_val = df.loc[val_idx, time_col].min()
            assert max_tr < min_val, f"Leakage detected: max train {max_tr} >= min val {min_val}"
            
            yield train_idx, val_idx


def split_locked_final_holdout(
    df: pd.DataFrame,
    time_col: str,
    holdout_periods: int = 26
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partitions dataset into:
    1. Development dataset (Train/Val for model exploration and hyperparameter tuning)
    2. Locked Final Holdout (untouched until final locked evaluation)
    """
    unique_times = np.sort(df[time_col].unique())
    cutoff_time = unique_times[-holdout_periods - 1]
    
    dev_df = df[df[time_col] <= cutoff_time].copy()
    holdout_df = df[df[time_col] > cutoff_time].copy()
    
    assert dev_df[time_col].max() < holdout_df[time_col].min(), "Holdout overlap detected"
    return dev_df, holdout_df


def run_walk_forward_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    time_col: str,
    model_factory: Callable[[], Any],
    cv_splitter: ChronologicalTimeSeriesSplit,
    model_name: str = "Candidate"
) -> Dict[str, Any]:
    """
    Evaluates a model architecture across walk-forward CV folds.
    Returns aggregated metrics and fold variance.
    """
    fold_metrics = []
    
    for fold_num, (tr_idx, val_idx) in enumerate(cv_splitter.split(df, time_col), start=1):
        X_tr = df.loc[tr_idx, feature_cols]
        y_tr = df.loc[tr_idx, target_col]
        X_val = df.loc[val_idx, feature_cols]
        y_val = df.loc[val_idx, target_col]
        
        # Fit model strictly on fold training data
        model = model_factory()
        model.fit(X_tr, y_tr)
        
        preds = model.predict(X_val)
        m = evaluate_all_metrics(y_val, preds, model_name=f"{model_name}_Fold_{fold_num}")
        m["Fold"] = fold_num
        fold_metrics.append(m)
        
    df_folds = pd.DataFrame(fold_metrics)
    summary = {
        "Model": model_name,
        "CV_MAE_Mean": float(df_folds["MAE"].mean()),
        "CV_MAE_Std": float(df_folds["MAE"].std()),
        "CV_RMSE_Mean": float(df_folds["RMSE"].mean()),
        "CV_RMSE_Std": float(df_folds["RMSE"].std()),
        "CV_Bias_Pct_Mean": float(df_folds["Bias_Percentage"].mean()),
        "CV_WAPE_Mean": float(df_folds["WAPE"].mean()),
        "Fold_Details": fold_metrics
    }
    return summary
