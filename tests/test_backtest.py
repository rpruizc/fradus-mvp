"""Tests for the backtest engine (Agent 5) and its ranking-metric helpers."""

import numpy as np
import pandas as pd
import pytest

from labellift.backtest import run_backtest
from labellift.estimator import compute_pseudo_labels
from labellift.metrics import precision_at_k_percent, recall_at_k_percent
from labellift.propensities import estimate_propensities
from labellift.synthetic_data import generate_synthetic_transactions

METRIC_COLUMNS = [
    "roc_auc",
    "average_precision",
    "precision_at_1_percent",
    "precision_at_5_percent",
    "recall_at_1_percent",
    "recall_at_5_percent",
    "mean_predicted_fraud_rate",
    "true_fraud_rate",
]
BOUNDED_COLUMNS = [
    "precision_at_1_percent",
    "precision_at_5_percent",
    "recall_at_1_percent",
    "recall_at_5_percent",
]


@pytest.fixture(scope="module")
def backtest_metrics() -> pd.DataFrame:
    """Run the full pipeline on a small deterministic sample and return metrics."""
    df = generate_synthetic_transactions(n=80_000, seed=42)
    df = estimate_propensities(df)
    df = compute_pseudo_labels(df)
    return run_backtest(df)


def test_backtest_returns_two_rows(backtest_metrics: pd.DataFrame) -> None:
    assert len(backtest_metrics) == 2
    assert set(backtest_metrics["model"]) == {
        "raw_observed_label_model",
        "labellift_pseudo_label_model",
    }


def test_metric_columns_exist(backtest_metrics: pd.DataFrame) -> None:
    for column in METRIC_COLUMNS:
        assert column in backtest_metrics.columns


def test_metric_values_finite(backtest_metrics: pd.DataFrame) -> None:
    assert np.isfinite(backtest_metrics[METRIC_COLUMNS].to_numpy()).all()


def test_precision_recall_in_unit_interval(backtest_metrics: pd.DataFrame) -> None:
    for column in BOUNDED_COLUMNS:
        values = backtest_metrics[column]
        assert values.min() >= 0.0, (column, values.min())
        assert values.max() <= 1.0, (column, values.max())


# --- metric helper unit tests --------------------------------------------------
def test_precision_at_k_percent_perfect_ranking() -> None:
    # 100 rows, 10 positives ranked at the very top.
    y_true = pd.Series([1] * 10 + [0] * 90)
    y_score = pd.Series(list(range(100, 0, -1)))
    # Top 10% = 10 rows, all positive.
    assert precision_at_k_percent(y_true, y_score, 10.0) == 1.0
    assert recall_at_k_percent(y_true, y_score, 10.0) == 1.0


def test_recall_at_k_percent_partial() -> None:
    y_true = pd.Series([1] * 10 + [0] * 90)
    y_score = pd.Series(list(range(100, 0, -1)))
    # Top 5% = 5 rows, all positive -> recall 5/10.
    assert recall_at_k_percent(y_true, y_score, 5.0) == 0.5


def test_recall_with_no_positives_is_zero() -> None:
    y_true = pd.Series([0, 0, 0, 0])
    y_score = pd.Series([0.1, 0.9, 0.5, 0.2])
    assert recall_at_k_percent(y_true, y_score, 50.0) == 0.0
