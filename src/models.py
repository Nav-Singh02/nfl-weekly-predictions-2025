# src/models.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Any, Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

@dataclass
class Trio:
    # Hold calibrated wrappers (type Any so we can store sklearn wrappers)
    rf: Any
    xgb: Any
    lr: Any

def fit_models(X, y) -> Trio:
    """
    Fit three base models, then wrap each with sigmoid calibration
    (Platt scaling) using 5-fold CV so predicted probabilities are
    better aligned with observed outcomes.
    """
    rf_base = RandomForestClassifier(n_estimators=400, random_state=42)
    xgb_base = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=42
    )
    lr_base = LogisticRegression(max_iter=1000)

    # Calibrate each model's probabilities
    rf  = CalibratedClassifierCV(rf_base,  method="sigmoid", cv=5)
    xgb = CalibratedClassifierCV(xgb_base, method="sigmoid", cv=5)
    lr  = CalibratedClassifierCV(lr_base,  method="sigmoid", cv=5)

    rf.fit(X, y)
    xgb.fit(X, y)
    lr.fit(X, y)

    return Trio(rf, xgb, lr)

def predict_proba(trio: Trio, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_rf  = trio.rf.predict_proba(X)[:, 1]
    p_xgb = trio.xgb.predict_proba(X)[:, 1]
    p_lr  = trio.lr.predict_proba(X)[:, 1]
    return p_rf, p_xgb, p_lr
