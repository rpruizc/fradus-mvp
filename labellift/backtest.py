"""Backtest: does training on LabelLift pseudo-labels recover more true fraud?

Two models are trained on the same features and the same training rows:

- Model A (``raw_observed_label_model``) — logistic regression on the raw observed
  corrupted labels, the status quo for fraud teams.
- Model B (``labellift_pseudo_label_model``) — a gradient-boosted regressor on the
  continuous LabelLift pseudo-labels.

Both are evaluated against synthetic ground-truth ``true_fraud`` on a held-out test
split. Using ``true_fraud`` is only valid because this is simulated data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .config import BACKTEST_METRICS_CSV, CORRECTED_LABELS_CSV
from .metrics import precision_at_k_percent, recall_at_k_percent

FEATURE_COLUMNS = [
    "issuer_id",
    "mcc",
    "amount",
    "country_pair",
    "channel",
    "device_risk",
    "account_age_days",
    "prior_decline_count",
    "segment",
]
CATEGORICAL_FEATURES = ["issuer_id", "mcc", "country_pair", "channel", "segment"]
NUMERIC_FEATURES = ["amount", "device_risk", "account_age_days", "prior_decline_count"]

TRAIN_FRACTION = 0.70


def _build_logistic_pipeline() -> Pipeline:
    """One-hot + logistic-regression pipeline for the raw-label classifier (Model A)."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def _build_regressor_pipeline() -> Pipeline:
    """Ordinal-encoded gradient-boosted regressor for the pseudo-label model (Model B)."""
    n_categorical = len(CATEGORICAL_FEATURES)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = HistGradientBoostingRegressor(
        random_state=42,
        categorical_features=list(range(n_categorical)),
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def _compute_metrics(y_true: pd.Series, y_score: np.ndarray) -> dict:
    """Compute the full metric block comparing scores against synthetic ground truth."""
    y_score = pd.Series(np.asarray(y_score, dtype=float))
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "precision_at_1_percent": precision_at_k_percent(y_true, y_score, 1.0),
        "precision_at_5_percent": precision_at_k_percent(y_true, y_score, 5.0),
        "recall_at_1_percent": recall_at_k_percent(y_true, y_score, 1.0),
        "recall_at_5_percent": recall_at_k_percent(y_true, y_score, 5.0),
        "mean_predicted_fraud_rate": float(np.asarray(y_score, dtype=float).mean()),
        "true_fraud_rate": float(np.asarray(y_true, dtype=float).mean()),
    }


def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Compare raw-label training against LabelLift pseudo-label training.

    Returns a metrics dataframe with one row per model:
    - raw_observed_label_model
    - labellift_pseudo_label_model
    """
    ordered = df.sort_values("transaction_id").reset_index(drop=True)
    n_train = int(len(ordered) * TRAIN_FRACTION)
    train_df = ordered.iloc[:n_train]
    test_df = ordered.iloc[n_train:]

    y_test = test_df["true_fraud"].reset_index(drop=True)

    # Model A: raw observed corrupted labels, observed training rows only.
    observed_train = train_df[train_df["observed"] == 1]
    model_a = _build_logistic_pipeline()
    model_a.fit(
        observed_train[FEATURE_COLUMNS],
        observed_train["observed_label_corrupted"].astype(int),
    )
    score_a = model_a.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]

    # Model B: LabelLift pseudo-labels, all training rows.
    model_b = _build_regressor_pipeline()
    model_b.fit(train_df[FEATURE_COLUMNS], train_df["labellift_pseudo_label"])
    score_b = np.clip(model_b.predict(test_df[FEATURE_COLUMNS]), 0.0, 1.0)

    rows = [
        {"model": "raw_observed_label_model", **_compute_metrics(y_test, score_a)},
        {"model": "labellift_pseudo_label_model", **_compute_metrics(y_test, score_b)},
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Run the backtest on the corrected-label dataset and save the metrics."""
    df = pd.read_csv(CORRECTED_LABELS_CSV)
    metrics = run_backtest(df)
    BACKTEST_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(BACKTEST_METRICS_CSV, index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(metrics.to_string(index=False))
    print(f"\nWrote backtest metrics to {BACKTEST_METRICS_CSV}")


if __name__ == "__main__":
    main()
