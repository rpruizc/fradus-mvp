"""Observation-propensity models for LabelLift.

A transaction only becomes an observable fraud label if it survives three
sequential censorship gates: authorization, reporting, and maturity. This module
estimates the probability of clearing each gate with plain scikit-learn logistic
regression, then combines them into a single total observation propensity.

    q_hat_total = e_hat_authorization * r_hat_reporting * p_hat_maturity
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import PROPENSITY_CEILING, PROPENSITY_FLOOR

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


def _fit_predict_proba(
    X_train: pd.DataFrame, y_train: pd.Series, X_all: pd.DataFrame
) -> np.ndarray:
    """Fit a logistic pipeline on a subset and predict positive-class probability for all rows."""
    pipeline = _build_logistic_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline.predict_proba(X_all)[:, 1]


def estimate_propensities(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate authorization, reporting, maturity, and total observation propensities.

    Returns a copy of df with:
    - e_hat_authorization
    - r_hat_reporting
    - p_hat_maturity
    - q_hat_total

    Each propensity model is fit on its relevant subset of rows but produces a
    predicted probability for *every* row. Individual propensities are clipped to
    ``[PROPENSITY_FLOOR, PROPENSITY_CEILING]`` and the total is their product.
    """
    out = df.copy()
    features = df[FEATURE_COLUMNS]

    # Authorization model: trained on all rows.
    e_hat = _fit_predict_proba(features, df["authorized"], features)

    # Reporting model: trained only on authorized rows.
    auth_mask = df["authorized"] == 1
    r_hat = _fit_predict_proba(features[auth_mask], df.loc[auth_mask, "reported"], features)

    # Maturity model: trained only on authorized & reported rows.
    reported_mask = auth_mask & (df["reported"] == 1)
    p_hat = _fit_predict_proba(
        features[reported_mask], df.loc[reported_mask, "matured_by_training"], features
    )

    e_hat = np.clip(e_hat, PROPENSITY_FLOOR, PROPENSITY_CEILING)
    r_hat = np.clip(r_hat, PROPENSITY_FLOOR, PROPENSITY_CEILING)
    p_hat = np.clip(p_hat, PROPENSITY_FLOOR, PROPENSITY_CEILING)

    out["e_hat_authorization"] = e_hat
    out["r_hat_reporting"] = r_hat
    out["p_hat_maturity"] = p_hat
    out["q_hat_total"] = e_hat * r_hat * p_hat
    return out


def main() -> None:
    """Estimate propensities on the synthetic dataset and print a short summary."""
    from .config import SYNTHETIC_TRANSACTIONS_CSV

    df = pd.read_csv(SYNTHETIC_TRANSACTIONS_CSV)
    out = estimate_propensities(df)
    columns = ["e_hat_authorization", "r_hat_reporting", "p_hat_maturity", "q_hat_total"]
    print(out[columns].describe())


if __name__ == "__main__":
    main()
