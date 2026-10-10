"""
tests/test_classification_models.py - Unit tests for classification models, threshold calibrations,
and expanding walk-forward cross validation.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models import (
    calibrate_threshold_youden_j,
    calibrate_threshold_f1,
    create_calibrated_classifier,
    create_soft_voting_ensemble,
    create_random_forest_classifier,
    create_lightgbm_classifier,
    create_xgboost_classifier
)
from src.validation import ChronologicalTimeSeriesSplit, run_walk_forward_classification_cv
from src.metrics import evaluate_classification_metrics


def test_youden_j_threshold_calibration():
    # Construct synthetic ground truth and probabilities
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    
    thresh = calibrate_threshold_youden_j(y_true, y_prob)
    # Optimum separation is between 0.4 and 0.6 (e.g. 0.6)
    assert 0.4 <= thresh <= 0.7
    # Boundary tests
    assert calibrate_threshold_youden_j(np.zeros(5), np.full(5, 0.5)) == 0.5


def test_f1_threshold_calibration():
    y_true = np.array([0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.5, 0.6, 0.8, 0.9])
    
    thresh = calibrate_threshold_f1(y_true, y_prob)
    assert 0.3 <= thresh <= 0.6


def test_calibrated_classifier_platt_scaling():
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    base = LogisticRegression()
    cal = create_calibrated_classifier(base, method="sigmoid", cv=2)
    cal.fit(X, y)
    probs = cal.predict_proba(X)[:, 1]
    
    assert len(probs) == 100
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_soft_voting_ensemble():
    X = np.random.randn(80, 4)
    y = (X[:, 0] > 0).astype(int)
    
    m1 = LogisticRegression()
    m2 = RandomForestClassifier(n_estimators=10, random_state=42)
    ens = create_soft_voting_ensemble([("lr", m1), ("rf", m2)], weights=[0.5, 0.5])
    ens.fit(X, y)
    probs = ens.predict_proba(X)[:, 1]
    
    assert len(probs) == 80
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_walk_forward_classification_cv():
    dates = pd.date_range("2021-01-01", periods=60, freq="W")
    df = pd.DataFrame({
        "Date": dates,
        "Feat1": np.random.randn(60),
        "Surge": (np.random.randn(60) > 0).astype(int)
    })
    
    cv = ChronologicalTimeSeriesSplit(n_splits=3, horizon_periods=10, min_train_periods=25)
    res = run_walk_forward_classification_cv(
        df=df,
        feature_cols=["Feat1"],
        target_col="Surge",
        time_col="Date",
        model_factory=lambda: LogisticRegression(),
        cv_splitter=cv,
        threshold=0.5,
        model_name="LogReg"
    )
    
    assert res["Model"] == "LogReg"
    assert "CV_ROC_AUC_Mean" in res
    assert "CV_Accuracy_Mean" in res
    assert len(res["Fold_Details"]) == 3
