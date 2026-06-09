"""Tests for the synthetic payment-label data generator."""

from pathlib import Path

import pandas as pd
import pytest

from labellift.config import N_TRANSACTIONS, SYNTHETIC_TRANSACTIONS_CSV
from labellift.synthetic_data import generate_synthetic_transactions

REQUIRED_COLUMNS = [
    "transaction_id",
    "issuer_id",
    "merchant_id",
    "mcc",
    "amount",
    "country_pair",
    "channel",
    "device_risk",
    "account_age_days",
    "prior_decline_count",
    "true_fraud_probability",
    "true_fraud",
    "authorization_probability",
    "authorized",
    "reporting_probability",
    "reported",
    "label_delay_days",
    "maturity_probability",
    "matured_by_training",
    "observed_label_raw",
    "observed_label_corrupted",
    "observed",
    "segment",
]


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """A fast, deterministic 100k-row sample for distribution checks."""
    return generate_synthetic_transactions(n=100_000, seed=42)


def test_required_columns_exist(sample_df: pd.DataFrame) -> None:
    for column in REQUIRED_COLUMNS:
        assert column in sample_df.columns, f"missing column: {column}"


def test_segment_format(sample_df: pd.DataFrame) -> None:
    row = sample_df.iloc[0]
    assert row["segment"] == f"{row['mcc']}__{row['country_pair']}__{row['channel']}"


def test_true_fraud_rate_in_range(sample_df: pd.DataFrame) -> None:
    rate = sample_df["true_fraud"].mean()
    assert 0.008 <= rate <= 0.015, rate


def test_authorization_rate_in_range(sample_df: pd.DataFrame) -> None:
    rate = sample_df["authorized"].mean()
    assert 0.82 <= rate <= 0.92, rate


def test_fraud_authorization_lower_than_legit(sample_df: pd.DataFrame) -> None:
    fraud_auth = sample_df.loc[sample_df["true_fraud"] == 1, "authorized"].mean()
    legit_auth = sample_df.loc[sample_df["true_fraud"] == 0, "authorized"].mean()
    assert fraud_auth < legit_auth, (fraud_auth, legit_auth)


def test_observed_label_rate_below_true_fraud_rate(sample_df: pd.DataFrame) -> None:
    true_rate = sample_df["true_fraud"].mean()
    observed_fraud_rate = (
        (sample_df["observed"] == 1) & (sample_df["observed_label_corrupted"] == 1)
    ).mean()
    assert observed_fraud_rate < true_rate, (observed_fraud_rate, true_rate)


def test_no_observed_label_when_not_observed(sample_df: pd.DataFrame) -> None:
    unobserved = sample_df[sample_df["observed"] == 0]
    assert unobserved["observed_label_raw"].isna().all()
    assert unobserved["observed_label_corrupted"].isna().all()


@pytest.mark.skipif(
    not Path(SYNTHETIC_TRANSACTIONS_CSV).exists(),
    reason="full dataset not generated; run `python -m labellift.synthetic_data`",
)
def test_full_dataset_has_one_million_rows() -> None:
    df = pd.read_csv(SYNTHETIC_TRANSACTIONS_CSV)
    assert len(df) == N_TRANSACTIONS
    for column in REQUIRED_COLUMNS:
        assert column in df.columns, f"missing column: {column}"
