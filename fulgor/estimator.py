"""Collapsed Fulgor pseudo-label estimator (MVP).

This is the MVP estimator, not the full production sequential triply robust
estimator. It combines two corrections:

1. Label-noise correction — undo the false-negative / false-positive miscoding of
   observed labels.
2. Inverse-observation weighting — upweight the residual of the rare observed
   labels by ``1 / q_hat_total`` to recover fraud signal that censorship removed.

The result is a continuous pseudo-label per transaction that an existing fraud
model can be trained on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CORRECTED_LABELS_CSV,
    LABEL_CORRUPTION_FALSE_NEGATIVE,
    LABEL_CORRUPTION_FALSE_POSITIVE,
    SYNTHETIC_TRANSACTIONS_CSV,
)

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

REQUIRED_PROPENSITY_COLUMNS = [
    "e_hat_authorization",
    "r_hat_reporting",
    "p_hat_maturity",
    "q_hat_total",
]


def _build_logistic_pipeline() -> Pipeline:
    """Build a one-hot + logistic-regression pipeline over the feature columns."""
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


def train_baseline_fraud_model(df: pd.DataFrame) -> tuple[object, pd.Series]:
    """Train a baseline fraud probability model using observed corrupted labels only.

    Return the fitted model and fraud probability predictions for all rows.
    """
    observed_mask = df["observed"] == 1
    pipeline = _build_logistic_pipeline()
    pipeline.fit(
        df.loc[observed_mask, FEATURE_COLUMNS],
        df.loc[observed_mask, "observed_label_corrupted"].astype(int),
    )
    scores = pipeline.predict_proba(df[FEATURE_COLUMNS])[:, 1]
    return pipeline, pd.Series(scores, index=df.index, name="baseline_fraud_score")


def apply_label_corruption_correction(
    y_observed: pd.Series,
    false_negative_rate: float = LABEL_CORRUPTION_FALSE_NEGATIVE,
    false_positive_rate: float = LABEL_CORRUPTION_FALSE_POSITIVE,
) -> pd.Series:
    """Apply binary label-noise correction:

        y_corr = (y - eps01) / (1 - eps10 - eps01)

    where ``eps01`` is the false-positive rate (a true 0 observed as 1) and
    ``eps10`` is the false-negative rate (a true 1 observed as 0). Corrected
    values are clipped to ``[0, 1]``. NaN inputs (unobserved rows) stay NaN.
    """
    eps01 = false_positive_rate
    eps10 = false_negative_rate
    corrected = (y_observed - eps01) / (1.0 - eps10 - eps01)
    return corrected.clip(lower=0.0, upper=1.0)


def compute_pseudo_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Fulgor pseudo-labels using the collapsed residual-weighted estimator.

    Required output columns:
    - baseline_fraud_score
    - observed_label_corrected
    - inverse_observation_weight
    - fulgor_pseudo_label
    - label_bias_delta

    Raises:
        ValueError: if the required propensity columns are not present in ``df``.
    """
    missing = [c for c in REQUIRED_PROPENSITY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "compute_pseudo_labels requires propensity columns "
            f"{REQUIRED_PROPENSITY_COLUMNS}; missing: {missing}. "
            "Run estimate_propensities() first."
        )

    out = df.copy()
    _, baseline_score = train_baseline_fraud_model(df)

    observed_label_corrected = apply_label_corruption_correction(df["observed_label_corrupted"])
    inverse_observation_weight = 1.0 / df["q_hat_total"]

    observed = df["observed"].to_numpy(dtype=float)
    baseline = baseline_score.to_numpy()
    # Unobserved rows carry NaN corrected labels; they are masked out by ``observed``.
    corrected_filled = observed_label_corrected.fillna(0.0).to_numpy()

    pseudo = baseline + observed * inverse_observation_weight.to_numpy() * (
        corrected_filled - baseline
    )
    pseudo = np.clip(pseudo, 0.0, 1.0)

    out["baseline_fraud_score"] = baseline
    out["observed_label_corrected"] = observed_label_corrected
    out["inverse_observation_weight"] = inverse_observation_weight
    out["fulgor_pseudo_label"] = pseudo
    out["label_bias_delta"] = out["fulgor_pseudo_label"] - out["baseline_fraud_score"]
    return out


def main() -> None:
    """Load synthetic data, estimate propensities, compute pseudo-labels, and save them."""
    from .propensities import estimate_propensities

    df = pd.read_csv(SYNTHETIC_TRANSACTIONS_CSV)
    df = estimate_propensities(df)
    df = compute_pseudo_labels(df)
    CORRECTED_LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CORRECTED_LABELS_CSV, index=False)
    print(f"Wrote corrected labels for {len(df):,} rows to {CORRECTED_LABELS_CSV}")


if __name__ == "__main__":
    main()
