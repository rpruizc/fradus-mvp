"""Tests for the propensity models (Agent 3) and the LabelLift estimator (Agent 4)."""

import numpy as np
import pandas as pd
import pytest

from labellift.config import PROPENSITY_CEILING, PROPENSITY_FLOOR
from labellift.estimator import (
    apply_label_corruption_correction,
    compute_pseudo_labels,
)
from labellift.propensities import estimate_propensities
from labellift.synthetic_data import generate_synthetic_transactions

PROPENSITY_COLUMNS = [
    "e_hat_authorization",
    "r_hat_reporting",
    "p_hat_maturity",
    "q_hat_total",
]
ESTIMATOR_COLUMNS = [
    "baseline_fraud_score",
    "observed_label_corrected",
    "inverse_observation_weight",
    "labellift_pseudo_label",
    "label_bias_delta",
]


@pytest.fixture(scope="module")
def base_df() -> pd.DataFrame:
    """Deterministic 80k-row sample (fast enough to fit logistic models in tests)."""
    return generate_synthetic_transactions(n=80_000, seed=42)


@pytest.fixture(scope="module")
def propensity_df(base_df: pd.DataFrame) -> pd.DataFrame:
    """Sample with the four estimated propensity columns attached."""
    return estimate_propensities(base_df)


@pytest.fixture(scope="module")
def pseudo_df(propensity_df: pd.DataFrame) -> pd.DataFrame:
    """Sample with the LabelLift pseudo-label columns attached."""
    return compute_pseudo_labels(propensity_df)


# --- Agent 3: propensities -----------------------------------------------------
def test_propensity_columns_exist(propensity_df: pd.DataFrame) -> None:
    for column in PROPENSITY_COLUMNS:
        assert column in propensity_df.columns


def test_individual_propensities_within_bounds(propensity_df: pd.DataFrame) -> None:
    for column in ["e_hat_authorization", "r_hat_reporting", "p_hat_maturity"]:
        values = propensity_df[column]
        assert values.min() >= PROPENSITY_FLOOR - 1e-9, (column, values.min())
        assert values.max() <= PROPENSITY_CEILING + 1e-9, (column, values.max())


def test_total_propensity_positive_and_below_one(propensity_df: pd.DataFrame) -> None:
    q = propensity_df["q_hat_total"]
    assert q.min() > 0.0
    assert q.max() < 1.0


def test_total_propensity_is_product(propensity_df: pd.DataFrame) -> None:
    expected = (
        propensity_df["e_hat_authorization"]
        * propensity_df["r_hat_reporting"]
        * propensity_df["p_hat_maturity"]
    )
    np.testing.assert_allclose(propensity_df["q_hat_total"], expected, rtol=1e-9)


def test_no_nans_in_propensities(propensity_df: pd.DataFrame) -> None:
    assert not propensity_df[PROPENSITY_COLUMNS].isna().any().any()


def test_estimate_propensities_does_not_mutate_input(base_df: pd.DataFrame) -> None:
    before = set(base_df.columns)
    estimate_propensities(base_df)
    assert set(base_df.columns) == before


# --- Agent 4: estimator --------------------------------------------------------
def test_estimator_columns_exist(pseudo_df: pd.DataFrame) -> None:
    for column in ESTIMATOR_COLUMNS:
        assert column in pseudo_df.columns


def test_pseudo_labels_in_unit_interval(pseudo_df: pd.DataFrame) -> None:
    pseudo = pseudo_df["labellift_pseudo_label"]
    assert pseudo.min() >= 0.0
    assert pseudo.max() <= 1.0
    assert not pseudo.isna().any()


def test_inverse_weights_positive(pseudo_df: pd.DataFrame) -> None:
    assert (pseudo_df["inverse_observation_weight"] > 0).all()


def test_corrected_labels_in_unit_interval_where_observed(pseudo_df: pd.DataFrame) -> None:
    corrected = pseudo_df.loc[pseudo_df["observed"] == 1, "observed_label_corrected"]
    assert corrected.min() >= 0.0
    assert corrected.max() <= 1.0
    assert not corrected.isna().any()


def test_missing_propensities_raises(base_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        compute_pseudo_labels(base_df)


def test_label_corruption_correction_formula() -> None:
    y = pd.Series([0.0, 1.0])
    corrected = apply_label_corruption_correction(y, 0.06, 0.08)
    assert corrected.min() >= 0.0
    assert corrected.max() <= 1.0
