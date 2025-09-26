# src/ensemble.py
import numpy as np

def combine_probs(p_rf, p_xgb, p_lr, weights=None):
    """Simple mean to start (mirrors the original ensemble idea)."""
    if weights is None:
        return np.vstack([p_rf, p_xgb, p_lr]).mean(axis=0)
    w = np.asarray(weights) / np.sum(weights)
    return np.vstack([p_rf, p_xgb, p_lr]).T.dot(w)
