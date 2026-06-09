"""Deterministic synthetic payment-label observation pipeline.

This module simulates the journey a transaction takes before it can ever become a
fraud-training label:

    true_fraud  ->  authorized  ->  reported  ->  matured_by_training  ->  observed

Each stage censors the data. The point of the LabelLift MVP is that fraud models
only ever see the final ``observed`` slice, which is a biased sample of true fraud.

All randomness flows from a single seeded NumPy generator so the dataset is exactly
reproducible. No real financial, cardholder, issuer, or merchant data is used — every
value here is synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    LABEL_CORRUPTION_FALSE_NEGATIVE,
    LABEL_CORRUPTION_FALSE_POSITIVE,
    N_TRANSACTIONS,
    RANDOM_SEED,
    SYNTHETIC_TRANSACTIONS_CSV,
    TRAINING_WINDOW_DAYS,
)

# --- Fixed synthetic design ----------------------------------------------------
N_ISSUERS = 50
N_MERCHANTS = 2_000

MCC_VALUES = [
    "digital_goods",
    "travel",
    "marketplace",
    "electronics",
    "grocery",
    "gaming",
    "subscription",
    "luxury",
]
COUNTRY_PAIRS = ["domestic", "cross_border", "high_risk_cross_border"]
CHANNELS = ["card_present", "card_not_present", "wallet"]
HIGH_RISK_MCCS = {"gaming", "luxury", "digital_goods", "travel"}

# Categorical sampling weights (must each sum to 1.0).
MCC_PROBS = [0.16, 0.10, 0.14, 0.16, 0.18, 0.08, 0.12, 0.06]
COUNTRY_PAIR_PROBS = [0.70, 0.22, 0.08]
CHANNEL_PROBS = [0.45, 0.45, 0.10]

# --- True-fraud logistic coefficients ------------------------------------------
FRAUD_INTERCEPT = -6.45
FRAUD_COEF_AMOUNT = 0.55
FRAUD_COEF_CROSS_BORDER = 0.70
FRAUD_COEF_HIGH_RISK_CB = 1.50
FRAUD_COEF_CNP = 0.85
FRAUD_COEF_WALLET = 0.20
FRAUD_COEF_DEVICE = 0.90
FRAUD_COEF_AGE = -0.50  # young accounts (low age z-score) are riskier
FRAUD_COEF_DECLINE = 0.45
FRAUD_COEF_HIGH_RISK_MCC = 0.50

# --- Authorization logistic coefficients ---------------------------------------
AUTH_INTERCEPT = 2.30
AUTH_COEF_FRAUD_PROB = 3.50  # subtracted: higher fraud prob -> more declines
AUTH_COEF_HIGH_RISK_CB = 0.80
AUTH_COEF_DEVICE = 0.40
AUTH_COEF_CNP = 0.30
AUTH_COEF_DECLINE = 0.25

# --- Reporting logistic coefficients -------------------------------------------
# Reporting depends only on observable covariates (a function of the fraud-risk
# score), never on the latent ``true_fraud`` flag. This keeps the observation
# process missing-at-random given X, which is exactly the assumption LabelLift's
# inverse-propensity correction relies on. Risky transactions are reported more
# often, so true fraud is still reported more than legit traffic on average.
REPORT_INTERCEPT = -4.65
REPORT_COEF_RISK = 3.20  # higher fraud-risk score -> more likely to be reported
REPORT_COEF_AMOUNT = 0.35  # low-value transactions are less likely to be reported
ISSUER_REPORT_EFFECT_SD = 0.45
SEGMENT_REPORT_EFFECT_SD = 0.35

# --- Maturity logistic coefficients --------------------------------------------
MATURITY_INTERCEPT = 1.35
MATURITY_COEF_CROSS_BORDER = 0.90
MATURITY_COEF_HIGH_RISK_CB = 1.40
MATURITY_COEF_TRAVEL = 1.00
MATURITY_COEF_MARKETPLACE = 0.80
MATURITY_COEF_SLOW_ISSUER = 1.30
N_SLOW_ISSUERS = 10  # ISSUER_001 .. ISSUER_010 settle slowly

MAX_DELAY_DAYS = 120


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def _zscore(x: np.ndarray) -> np.ndarray:
    """Standardize an array to zero mean and unit variance."""
    return (x - x.mean()) / x.std()


def generate_synthetic_transactions(
    n: int = N_TRANSACTIONS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate the deterministic synthetic transaction dataset.

    Args:
        n: Number of transactions to generate.
        seed: Random seed controlling every stochastic draw.

    Returns:
        A dataframe with one row per synthetic transaction and every column
        required by the LabelLift pipeline.
    """
    rng = np.random.default_rng(seed)

    # --- Identifiers & raw features --------------------------------------------
    issuer_labels = np.array([f"ISSUER_{i + 1:03d}" for i in range(N_ISSUERS)])
    merchant_labels = np.array([f"MERCHANT_{i + 1:04d}" for i in range(N_MERCHANTS)])

    issuer_idx = rng.integers(0, N_ISSUERS, n)
    merchant_idx = rng.integers(0, N_MERCHANTS, n)
    mcc_idx = rng.choice(len(MCC_VALUES), size=n, p=MCC_PROBS)
    cp_idx = rng.choice(len(COUNTRY_PAIRS), size=n, p=COUNTRY_PAIR_PROBS)
    ch_idx = rng.choice(len(CHANNELS), size=n, p=CHANNEL_PROBS)

    issuer_id = issuer_labels[issuer_idx]
    merchant_id = merchant_labels[merchant_idx]
    mcc = np.array(MCC_VALUES)[mcc_idx]
    country_pair = np.array(COUNTRY_PAIRS)[cp_idx]
    channel = np.array(CHANNELS)[ch_idx]

    amount = np.round(np.clip(np.exp(rng.normal(3.4, 1.0, n)), 1.0, 50_000.0), 2)
    device_risk = np.round(rng.beta(2.0, 5.0, n), 4)
    account_age_days = np.clip(rng.exponential(500.0, n), 0, 3650).astype(int)
    prior_decline_count = rng.poisson(0.30, n).astype(int)

    segment = np.char.add(
        np.char.add(np.char.add(np.char.add(mcc, "__"), country_pair), "__"), channel
    )

    # --- Convenience flags -----------------------------------------------------
    cross_border = (country_pair == "cross_border").astype(float)
    high_risk_cb = (country_pair == "high_risk_cross_border").astype(float)
    cnp = (channel == "card_not_present").astype(float)
    wallet = (channel == "wallet").astype(float)
    high_risk_mcc = np.isin(mcc, list(HIGH_RISK_MCCS)).astype(float)
    travel = (mcc == "travel").astype(float)
    marketplace = (mcc == "marketplace").astype(float)
    slow_issuer = (issuer_idx < N_SLOW_ISSUERS).astype(float)

    amount_z = _zscore(np.log(amount))
    device_z = _zscore(device_risk)
    age_z = _zscore(account_age_days.astype(float))

    # --- 1. True fraud ---------------------------------------------------------
    logit_fraud = (
        FRAUD_INTERCEPT
        + FRAUD_COEF_AMOUNT * amount_z
        + FRAUD_COEF_CROSS_BORDER * cross_border
        + FRAUD_COEF_HIGH_RISK_CB * high_risk_cb
        + FRAUD_COEF_CNP * cnp
        + FRAUD_COEF_WALLET * wallet
        + FRAUD_COEF_DEVICE * device_z
        + FRAUD_COEF_AGE * age_z
        + FRAUD_COEF_DECLINE * prior_decline_count
        + FRAUD_COEF_HIGH_RISK_MCC * high_risk_mcc
    )
    true_fraud_probability = _sigmoid(logit_fraud)
    true_fraud = (rng.random(n) < true_fraud_probability).astype(int)

    # --- 2. Authorization (declined transactions are never labeled) ------------
    logit_auth = (
        AUTH_INTERCEPT
        - AUTH_COEF_FRAUD_PROB * true_fraud_probability
        - AUTH_COEF_HIGH_RISK_CB * high_risk_cb
        - AUTH_COEF_DEVICE * device_z
        - AUTH_COEF_CNP * cnp
        - AUTH_COEF_DECLINE * (prior_decline_count > 0).astype(float)
    )
    authorization_probability = _sigmoid(logit_auth)
    authorized = (rng.random(n) < authorization_probability).astype(int)

    # --- 3. Reporting (only meaningful when authorized) ------------------------
    issuer_report_effect = rng.normal(0.0, ISSUER_REPORT_EFFECT_SD, N_ISSUERS)[issuer_idx]
    seg_unique, seg_codes = np.unique(segment, return_inverse=True)
    segment_report_effect = rng.normal(0.0, SEGMENT_REPORT_EFFECT_SD, len(seg_unique))[seg_codes]

    fraud_risk_z = _zscore(logit_fraud)  # covariate-only risk signal
    logit_report = (
        REPORT_INTERCEPT
        + REPORT_COEF_RISK * fraud_risk_z
        + REPORT_COEF_AMOUNT * amount_z
        + issuer_report_effect
        + segment_report_effect
    )
    reporting_probability = np.where(authorized == 1, _sigmoid(logit_report), 0.0)
    reported = ((rng.random(n) < reporting_probability) & (authorized == 1)).astype(int)

    # --- 4. Delay & maturity (only meaningful when authorized and reported) ----
    logit_maturity = (
        MATURITY_INTERCEPT
        - MATURITY_COEF_CROSS_BORDER * cross_border
        - MATURITY_COEF_HIGH_RISK_CB * high_risk_cb
        - MATURITY_COEF_TRAVEL * travel
        - MATURITY_COEF_MARKETPLACE * marketplace
        - MATURITY_COEF_SLOW_ISSUER * slow_issuer
    )
    maturity_probability = _sigmoid(logit_maturity)

    applicable = (authorized == 1) & (reported == 1)
    matured_draw = rng.random(n) < maturity_probability
    fast = applicable & matured_draw
    slow = applicable & ~matured_draw

    label_delay_days = np.full(n, np.nan)
    label_delay_days[fast] = rng.integers(1, TRAINING_WINDOW_DAYS + 1, fast.sum())
    label_delay_days[slow] = rng.integers(
        TRAINING_WINDOW_DAYS + 1, MAX_DELAY_DAYS + 1, slow.sum()
    )
    matured_by_training = (applicable & (label_delay_days <= TRAINING_WINDOW_DAYS)).astype(int)

    # --- 5. Observation & label corruption -------------------------------------
    observed = ((authorized == 1) & (reported == 1) & (matured_by_training == 1)).astype(int)

    observed_label_raw = np.where(observed == 1, true_fraud.astype(float), np.nan)

    corrupted = true_fraud.astype(float).copy()
    flip = rng.random(n)
    fn_mask = (observed == 1) & (true_fraud == 1) & (flip < LABEL_CORRUPTION_FALSE_NEGATIVE)
    fp_mask = (observed == 1) & (true_fraud == 0) & (flip < LABEL_CORRUPTION_FALSE_POSITIVE)
    corrupted[fn_mask] = 0.0
    corrupted[fp_mask] = 1.0
    observed_label_corrupted = np.where(observed == 1, corrupted, np.nan)

    transaction_id = np.array([f"TXN_{i:08d}" for i in range(n)])

    df = pd.DataFrame(
        {
            "transaction_id": transaction_id,
            "issuer_id": issuer_id,
            "merchant_id": merchant_id,
            "mcc": mcc,
            "amount": amount,
            "country_pair": country_pair,
            "channel": channel,
            "device_risk": device_risk,
            "account_age_days": account_age_days,
            "prior_decline_count": prior_decline_count,
            "true_fraud_probability": np.round(true_fraud_probability, 6),
            "true_fraud": true_fraud,
            "authorization_probability": np.round(authorization_probability, 6),
            "authorized": authorized,
            "reporting_probability": np.round(reporting_probability, 6),
            "reported": reported,
            "label_delay_days": label_delay_days,
            "maturity_probability": np.round(maturity_probability, 6),
            "matured_by_training": matured_by_training,
            "observed_label_raw": observed_label_raw,
            "observed_label_corrupted": observed_label_corrupted,
            "observed": observed,
            "segment": segment,
        }
    )
    return df


def save_synthetic_transactions(df: pd.DataFrame, path=SYNTHETIC_TRANSACTIONS_CSV) -> None:
    """Write the synthetic dataset to ``path`` as CSV, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """Generate the full synthetic dataset and write it to the data directory."""
    df = generate_synthetic_transactions()
    save_synthetic_transactions(df)
    print(f"Wrote {len(df):,} synthetic transactions to {SYNTHETIC_TRANSACTIONS_CSV}")


if __name__ == "__main__":
    main()
