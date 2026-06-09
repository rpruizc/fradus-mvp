"""Fulgor MVP — investor-facing Streamlit dashboard.

Run with:

    streamlit run app.py

The dashboard tells one story: fraud models are trained on chargebacks, chargebacks
are a censored and corrupted view of true fraud, and Fulgor reconstructs corrected
pseudo-labels so existing fraud models can be trained on a de-biased target.

No authentication, no database, no real payment data — everything is synthetic.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st

from fulgor.config import (
    BACKTEST_METRICS_CSV,
    CORRECTED_LABELS_CSV,
    DEMO_BACKTEST_METRICS_CSV,
    DEMO_BLINDSPOT_ATLAS_CSV,
    DEMO_FUNNEL_JSON,
    DEMO_KPIS_JSON,
    DEMO_MANIFEST_JSON,
    DEMO_SCORE_SAMPLE_CSV,
    DEMO_TOP_BIAS_CSV,
    LABEL_CORRUPTION_FALSE_NEGATIVE,
    LABEL_CORRUPTION_FALSE_POSITIVE,
    SYNTHETIC_TRANSACTIONS_CSV,
)
from fulgor.plots import (
    backtest_bar,
    blindspot_scatter,
    label_problem_funnel,
    pseudo_label_histogram,
)

FN = LABEL_CORRUPTION_FALSE_NEGATIVE
FP = LABEL_CORRUPTION_FALSE_POSITIVE

# Only the columns the dashboard actually needs (keeps the 1M-row load light).
LOAD_COLUMNS = [
    "transaction_id",
    "issuer_id",
    "mcc",
    "amount",
    "country_pair",
    "channel",
    "true_fraud",
    "authorized",
    "reported",
    "matured_by_training",
    "observed",
    "observed_label_corrupted",
    "q_hat_total",
    "baseline_fraud_score",
    "fulgor_pseudo_label",
    "label_bias_delta",
]

DEMO_ARTIFACTS = [
    DEMO_MANIFEST_JSON,
    DEMO_KPIS_JSON,
    DEMO_FUNNEL_JSON,
    DEMO_BLINDSPOT_ATLAS_CSV,
    DEMO_SCORE_SAMPLE_CSV,
    DEMO_TOP_BIAS_CSV,
    DEMO_BACKTEST_METRICS_CSV,
]

st.set_page_config(page_title="Fulgor", page_icon="🛰️", layout="wide")


# --- Data loading & preparation (cached) ---------------------------------------
def _demo_artifacts_available() -> bool:
    """Return True when the compact hosted-demo data bundle is present."""
    return all(path.exists() for path in DEMO_ARTIFACTS)


def _demo_artifacts_mtime() -> float:
    """Cache key for the compact hosted-demo data bundle."""
    return max(path.stat().st_mtime for path in DEMO_ARTIFACTS)


def _should_use_demo_artifacts() -> bool:
    """Prefer compact artifacts unless a local full-data run is explicitly requested."""
    return _demo_artifacts_available() and os.environ.get("FULGOR_USE_FULL_DATA") != "1"


@st.cache_data(show_spinner=False)
def load_demo_kpis(mtime: float) -> dict:
    """Load precomputed KPI values for hosted self-service deployments."""
    return json.loads(DEMO_KPIS_JSON.read_text())


@st.cache_data(show_spinner=False)
def load_demo_funnel(mtime: float) -> dict[str, int]:
    """Load precomputed funnel counts for hosted self-service deployments."""
    return json.loads(DEMO_FUNNEL_JSON.read_text())


@st.cache_data(show_spinner=False)
def load_demo_atlas(mtime: float) -> pd.DataFrame:
    """Load precomputed issuer blindspot rows for hosted self-service deployments."""
    return pd.read_csv(DEMO_BLINDSPOT_ATLAS_CSV)


@st.cache_data(show_spinner=False)
def load_demo_scores(mtime: float) -> pd.DataFrame:
    """Load precomputed score samples for hosted self-service deployments."""
    return pd.read_csv(DEMO_SCORE_SAMPLE_CSV)


@st.cache_data(show_spinner=False)
def load_demo_top_bias(mtime: float) -> pd.DataFrame:
    """Load precomputed top upward corrections for hosted self-service deployments."""
    return pd.read_csv(DEMO_TOP_BIAS_CSV)


@st.cache_data(show_spinner=False)
def load_demo_backtest_metrics(mtime: float) -> pd.DataFrame:
    """Load precomputed backtest metrics for hosted self-service deployments."""
    return pd.read_csv(DEMO_BACKTEST_METRICS_CSV)


@st.cache_data(show_spinner=False)
def load_corrected_labels(mtime: float) -> pd.DataFrame:
    """Load the corrected-label dataset (cache keyed on file mtime)."""
    df = pd.read_csv(CORRECTED_LABELS_CSV, usecols=LOAD_COLUMNS)
    corrected_unclipped = (df["observed_label_corrupted"].fillna(0.0) - FP) / (1.0 - FN - FP)
    # Horvitz-Thompson numerator: de-noised, inverse-propensity-weighted positives.
    df["_ht_contribution"] = df["observed"] * corrected_unclipped / df["q_hat_total"]
    observed_fraud = (df["observed"] == 1) & (df["observed_label_corrupted"] == 1)
    df["_observed_fraud"] = observed_fraud.astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_backtest_metrics(mtime: float) -> pd.DataFrame:
    """Load the backtest metrics table (cache keyed on file mtime)."""
    return pd.read_csv(BACKTEST_METRICS_CSV)


@st.cache_data(show_spinner=False)
def compute_kpis(mtime: float) -> dict:
    """Compute the four hero KPIs from the corrected-label dataset."""
    df = load_corrected_labels(mtime)
    n = len(df)
    true_rate = float(df["true_fraud"].mean())
    observed_rate = float(df["_observed_fraud"].mean())
    corrected_rate = float(df["_ht_contribution"].sum() / n)
    undercount = corrected_rate / max(observed_rate, 1e-9)
    return {
        "n": n,
        "true_rate": true_rate,
        "observed_rate": observed_rate,
        "corrected_rate": corrected_rate,
        "undercount": undercount,
    }


@st.cache_data(show_spinner=False)
def compute_funnel(mtime: float) -> dict[str, int]:
    """Count how many true frauds survive each censorship stage."""
    df = load_corrected_labels(mtime)
    tf = df["true_fraud"] == 1
    authorized = tf & (df["authorized"] == 1)
    reported = authorized & (df["reported"] == 1)
    matured = reported & (df["matured_by_training"] == 1)
    return {
        "All transactions": int(len(df)),
        "True fraud": int(tf.sum()),
        "Authorized true fraud": int(authorized.sum()),
        "Reported true fraud": int(reported.sum()),
        "Matured true fraud": int(matured.sum()),
        "Observed correctly labeled fraud": int((tf & (df["_observed_fraud"] == 1)).sum()),
    }


@st.cache_data(show_spinner=False)
def compute_blindspot_atlas(mtime: float) -> pd.DataFrame:
    """Aggregate the censorship and fraud-rate story per issuer."""
    df = load_corrected_labels(mtime)
    grouped = df.groupby("issuer_id")
    atlas = grouped.agg(
        transaction_count=("true_fraud", "size"),
        true_fraud_rate=("true_fraud", "mean"),
        observed_fraud_rate=("_observed_fraud", "mean"),
        ht_sum=("_ht_contribution", "sum"),
        authorized_sum=("authorized", "sum"),
        reported_sum=("reported", "sum"),
        matured_sum=("matured_by_training", "sum"),
    )
    atlas["corrected_fraud_rate"] = (atlas["ht_sum"] / atlas["transaction_count"]).clip(lower=0.0)
    atlas["authorization_rate"] = atlas["authorized_sum"] / atlas["transaction_count"]
    atlas["reporting_rate"] = atlas["reported_sum"] / atlas["authorized_sum"]
    atlas["maturity_rate"] = atlas["matured_sum"] / atlas["reported_sum"].clip(lower=1)
    observed_floor = atlas["observed_fraud_rate"].clip(lower=0.0001)
    atlas["blindspot_multiplier"] = atlas["corrected_fraud_rate"] / observed_floor
    atlas = atlas.reset_index()[
        [
            "issuer_id",
            "transaction_count",
            "true_fraud_rate",
            "observed_fraud_rate",
            "corrected_fraud_rate",
            "authorization_rate",
            "reporting_rate",
            "maturity_rate",
            "blindspot_multiplier",
        ]
    ]
    return atlas.sort_values("blindspot_multiplier", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_top_bias(mtime: float) -> pd.DataFrame:
    """Return the 20 transactions with the largest Fulgor label-bias correction."""
    df = load_corrected_labels(mtime)
    columns = [
        "transaction_id",
        "issuer_id",
        "mcc",
        "amount",
        "country_pair",
        "channel",
        "baseline_fraud_score",
        "fulgor_pseudo_label",
        "label_bias_delta",
    ]
    return df.nlargest(20, "label_bias_delta")[columns].reset_index(drop=True)


@st.cache_data(show_spinner=False)
def sample_scores(mtime: float, n: int = 50_000) -> pd.DataFrame:
    """Down-sampled scores for a fast, readable histogram."""
    df = load_corrected_labels(mtime)
    sample = df.sample(min(n, len(df)), random_state=42)
    return sample[["baseline_fraud_score", "fulgor_pseudo_label"]].reset_index(drop=True)


# --- Pipeline runners (for the on-demand buttons) ------------------------------
def _run_generate() -> None:
    from fulgor import synthetic_data

    with st.spinner("Generating 1,000,000 synthetic transactions…"):
        synthetic_data.main()


def _run_correction() -> None:
    from fulgor import estimator

    with st.spinner("Estimating propensities and computing Fulgor pseudo-labels…"):
        estimator.main()


def _run_backtest() -> None:
    from fulgor import backtest

    with st.spinner("Running the raw-vs-Fulgor backtest…"):
        backtest.main()


def _gate_or_run() -> bool:
    """Render generation buttons for any missing artifact. Return True if all exist."""
    if not SYNTHETIC_TRANSACTIONS_CSV.exists():
        st.info("Step 1 of 3 — the synthetic transaction dataset has not been generated yet.")
        if st.button("Generate synthetic data", type="primary"):
            _run_generate()
            st.rerun()
        return False
    if not CORRECTED_LABELS_CSV.exists():
        st.info("Step 2 of 3 — corrected pseudo-labels have not been computed yet.")
        if st.button("Run Fulgor correction", type="primary"):
            _run_correction()
            st.rerun()
        return False
    if not BACKTEST_METRICS_CSV.exists():
        st.info("Step 3 of 3 — the backtest has not been run yet.")
        if st.button("Run backtest", type="primary"):
            _run_backtest()
            st.rerun()
        return False
    return True


# --- Section renderers ---------------------------------------------------------
def render_hero(kpis: dict) -> None:
    """Section 1 — Hero."""
    st.title("Fulgor")
    st.subheader("Causal label infrastructure for fraud AI")
    st.markdown(
        "Fraud models are trained on chargebacks. Chargebacks are not ground truth. "
        "Fulgor reconstructs corrected pseudo-labels from declined, unreported, "
        "delayed, and miscoded transactions."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True synthetic fraud rate", f"{kpis['true_rate']:.2%}")
    c2.metric("Observed corrupted fraud rate", f"{kpis['observed_rate']:.2%}")
    c3.metric("Fulgor corrected fraud rate", f"{kpis['corrected_rate']:.2%}")
    c4.metric("Observed fraud undercount multiplier", f"{kpis['undercount']:.2f}×")
    st.caption(
        "Synthetic MVP demo. No real payment data used. The corrected rate is an "
        "inverse-propensity reconstruction; the true rate is shown only because the "
        "data is simulated."
    )


def render_reviewer_brief(kpis: dict, metrics: pd.DataFrame) -> None:
    """Short self-service summary for reviewers who open the app without a walkthrough."""
    indexed = metrics.set_index("model")
    raw_ap = float(indexed.loc["raw_observed_label_model", "average_precision"])
    lift_ap = float(indexed.loc["fulgor_pseudo_label_model", "average_precision"])
    ap_lift = (lift_ap / raw_ap - 1.0) * 100.0
    hidden_share = 1.0 - (kpis["observed_rate"] / max(kpis["true_rate"], 1e-9))
    st.info(
        "Reviewer summary: Fulgor is an offline label-reconstruction layer, not a new "
        "authorization model. This deterministic demo shows that raw chargeback labels miss "
        f"{hidden_share:.0%} of synthetic fraud prevalence; the corrected labels recover the "
        f"missing signal and improve average precision by {ap_lift:.1f}% in the synthetic "
        "backtest. In a customer pilot, the same workflow runs on historical logs and is "
        "evaluated on a later matured holdout window."
    )


def render_label_problem(funnel_counts: dict[str, int]) -> None:
    """Section 2 — The label problem."""
    st.header("The label problem")
    st.plotly_chart(label_problem_funnel(funnel_counts), width="stretch")
    st.markdown("**The model only sees the final slice, not the full fraud process.**")


def render_blindspot_atlas(atlas: pd.DataFrame) -> None:
    """Section 3 — Blindspot Atlas."""
    st.header("Blindspot Atlas")
    st.markdown(
        "Each issuer under-observes fraud differently, depending on how aggressively it "
        "declines, how often fraud is reported, and how fast disputes mature. The "
        "**blindspot multiplier** is how much Fulgor lifts the observed fraud rate."
    )
    # Streamlit's NumberColumn `format` applies printf to the raw value without
    # scaling, so convert the fraction columns to percentage points for display.
    percent_columns = [
        "true_fraud_rate",
        "observed_fraud_rate",
        "corrected_fraud_rate",
        "authorization_rate",
        "reporting_rate",
        "maturity_rate",
    ]
    display = atlas.head(15).copy()
    display[percent_columns] = display[percent_columns] * 100.0
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "issuer_id": "Issuer",
            "transaction_count": st.column_config.NumberColumn("Transactions", format="%d"),
            "true_fraud_rate": st.column_config.NumberColumn("True fraud", format="%.2f%%"),
            "observed_fraud_rate": st.column_config.NumberColumn("Observed", format="%.2f%%"),
            "corrected_fraud_rate": st.column_config.NumberColumn("Corrected", format="%.2f%%"),
            "authorization_rate": st.column_config.NumberColumn("Auth rate", format="%.1f%%"),
            "reporting_rate": st.column_config.NumberColumn("Report rate", format="%.1f%%"),
            "maturity_rate": st.column_config.NumberColumn("Maturity rate", format="%.1f%%"),
            "blindspot_multiplier": st.column_config.NumberColumn("Blindspot ×", format="%.2f×"),
        },
    )
    st.caption("Rates shown as percentages. Top 15 issuers by blindspot multiplier.")
    st.plotly_chart(blindspot_scatter(atlas), width="stretch")


def render_pseudo_labels(scores: pd.DataFrame, top_bias: pd.DataFrame) -> None:
    """Section 4 — Corrected pseudo-labels."""
    st.header("Corrected pseudo-labels")
    st.plotly_chart(pseudo_label_histogram(scores), width="stretch")
    st.markdown("**Transactions Fulgor re-scores most aggressively upward**")
    st.dataframe(
        top_bias,
        width="stretch",
        hide_index=True,
        column_config={
            "transaction_id": "Transaction",
            "issuer_id": "Issuer",
            "mcc": "MCC",
            "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
            "country_pair": "Country pair",
            "channel": "Channel",
            "baseline_fraud_score": st.column_config.NumberColumn("Baseline", format="%.3f"),
            "fulgor_pseudo_label": st.column_config.NumberColumn("Fulgor", format="%.3f"),
            "label_bias_delta": st.column_config.NumberColumn("Bias Δ", format="%.3f"),
        },
    )


def render_backtest(metrics: pd.DataFrame) -> None:
    """Section 5 — Backtest."""
    st.header("Backtest")
    display = metrics.copy()
    display["model"] = display["model"].map(
        {
            "raw_observed_label_model": "Raw observed labels",
            "fulgor_pseudo_label_model": "Fulgor pseudo-labels",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)
    st.plotly_chart(backtest_bar(metrics), width="stretch")
    st.markdown(
        "This synthetic backtest uses `true_fraud` only because this is simulated data. "
        "In a real customer pilot, evaluation uses a later matured holdout window."
    )


def render_pipeline() -> None:
    """Section 6 — How it plugs into fraud teams."""
    st.header("How it plugs into fraud teams")
    st.markdown(
        "1. Ingest historical transaction and chargeback data.\n"
        "2. Estimate authorization, reporting, and maturity propensities.\n"
        "3. Correct corrupted labels.\n"
        "4. Generate Fulgor pseudo-labels.\n"
        "5. Train the customer's existing fraud model on corrected labels."
    )


def render_limitations() -> None:
    """Section 7 — MVP limitations."""
    st.header("MVP limitations")
    st.markdown(
        "- Synthetic data only.\n"
        "- Collapsed estimator only, not full sequential nested estimator.\n"
        "- No empirical Bayes issuer shrinkage yet.\n"
        "- No production data connectors.\n"
        "- No real customer backtest yet.\n"
        "- No claim of production fraud-detection lift yet."
    )


def render_dashboard(
    kpis: dict,
    funnel_counts: dict[str, int],
    atlas: pd.DataFrame,
    scores: pd.DataFrame,
    top_bias: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    """Render the full self-service dashboard from prepared dataframes."""
    render_hero(kpis)
    render_reviewer_brief(kpis, metrics)
    st.divider()
    render_label_problem(funnel_counts)
    st.divider()
    render_blindspot_atlas(atlas)
    st.divider()
    render_pseudo_labels(scores, top_bias)
    st.divider()
    render_backtest(metrics)
    st.divider()
    render_pipeline()
    st.divider()
    render_limitations()


def main() -> None:
    """Render the full Fulgor dashboard, gating on the presence of data artifacts."""
    if _should_use_demo_artifacts():
        demo_mtime = _demo_artifacts_mtime()
        with st.spinner("Loading self-service demo…"):
            kpis = load_demo_kpis(demo_mtime)
            funnel_counts = load_demo_funnel(demo_mtime)
            atlas = load_demo_atlas(demo_mtime)
            scores = load_demo_scores(demo_mtime)
            top_bias = load_demo_top_bias(demo_mtime)
            metrics = load_demo_backtest_metrics(demo_mtime)
        render_dashboard(kpis, funnel_counts, atlas, scores, top_bias, metrics)
        return

    if not _gate_or_run():
        return

    labels_mtime = CORRECTED_LABELS_CSV.stat().st_mtime
    metrics_mtime = BACKTEST_METRICS_CSV.stat().st_mtime

    with st.spinner("Loading synthetic results…"):
        kpis = compute_kpis(labels_mtime)
        funnel_counts = compute_funnel(labels_mtime)
        atlas = compute_blindspot_atlas(labels_mtime)
        scores = sample_scores(labels_mtime)
        top_bias = compute_top_bias(labels_mtime)
        metrics = load_backtest_metrics(metrics_mtime)

    render_dashboard(kpis, funnel_counts, atlas, scores, top_bias, metrics)


# Streamlit executes this module as "__main__"; the guard keeps `import app` safe.
if __name__ == "__main__":
    main()
