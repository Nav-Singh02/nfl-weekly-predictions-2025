# src/models.py
from __future__ import annotations
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from xgboost import XGBClassifier


def fit_models(X: np.ndarray, y: np.ndarray):
    """
    Fit three base models and wrap each with **isotonic** calibration (cv=5):
      - RandomForest
      - XGBoost (binary:logistic)
      - LogisticRegression (with feature scaling)
    Returns a tuple of calibrated estimators.
    """
    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    rf_cal = CalibratedClassifierCV(rf, method="isotonic", cv=5)
    rf_cal.fit(X, y)

    # --- XGBoost ---
    xgb = XGBClassifier(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )
    xgb_cal = CalibratedClassifierCV(xgb, method="isotonic", cv=5)
    xgb_cal.fit(X, y)

    # --- Logistic Regression (with scaling) ---
    lr = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42),
    )
    lr_cal = CalibratedClassifierCV(lr, method="isotonic", cv=5)
    lr_cal.fit(X, y)

    return rf_cal, xgb_cal, lr_cal


def predict_proba(trio, X: np.ndarray):
    """
    Return class-1 probabilities from each calibrated model.
    """
    rf_cal, xgb_cal, lr_cal = trio
    p_rf = rf_cal.predict_proba(X)[:, 1]
    p_xgb = xgb_cal.predict_proba(X)[:, 1]
    p_lr = lr_cal.predict_proba(X)[:, 1]
    return p_rf, p_xgb, p_lr
