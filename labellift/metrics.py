"""Ranking metrics for the LabelLift backtest.

Fraud teams operate at a fixed review budget, so top-k% precision and recall are
the metrics that matter most. These helpers rank transactions by predicted score
and measure how much true fraud is captured in the riskiest k%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _top_k_indices(y_score: pd.Series, k_percent: float) -> np.ndarray:
    """Return the indices of the highest-scoring k% of rows (at least one row)."""
    scores = np.asarray(y_score, dtype=float)
    n = scores.shape[0]
    k = max(1, int(round(n * k_percent / 100.0)))
    return np.argsort(-scores, kind="stable")[:k]


def precision_at_k_percent(y_true: pd.Series, y_score: pd.Series, k_percent: float) -> float:
    """Fraction of the top-k% highest-scoring transactions that are truly fraudulent."""
    y_true_arr = np.asarray(y_true, dtype=float)
    top = _top_k_indices(y_score, k_percent)
    return float(y_true_arr[top].mean())


def recall_at_k_percent(y_true: pd.Series, y_score: pd.Series, k_percent: float) -> float:
    """Fraction of all true fraud captured within the top-k% highest-scoring transactions."""
    y_true_arr = np.asarray(y_true, dtype=float)
    total_positive = y_true_arr.sum()
    if total_positive <= 0:
        return 0.0
    top = _top_k_indices(y_score, k_percent)
    return float(y_true_arr[top].sum() / total_positive)
