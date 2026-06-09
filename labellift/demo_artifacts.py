"""Build compact artifacts for the hosted self-service demo.

The full synthetic CSVs are intentionally git-ignored because they are large.
This module distills them into the small tables the Streamlit app needs at
runtime, so a hosted deployment can render immediately without a manual setup
step.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .config import (
    BACKTEST_METRICS_CSV,
    CORRECTED_LABELS_CSV,
    DEMO_BACKTEST_METRICS_CSV,
    DEMO_BLINDSPOT_ATLAS_CSV,
    DEMO_DATA_DIR,
    DEMO_FUNNEL_JSON,
    DEMO_KPIS_JSON,
    DEMO_MANIFEST_JSON,
    DEMO_SCORE_SAMPLE_CSV,
    DEMO_TOP_BIAS_CSV,
    LABEL_CORRUPTION_FALSE_NEGATIVE,
    LABEL_CORRUPTION_FALSE_POSITIVE,
)

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
    "labellift_pseudo_label",
    "label_bias_delta",
]


def _write_json(path: Path, payload: dict) -> None:
    """Write a compact JSON file with stable key ordering."""
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _prepare_labels(corrected_labels_path: Path) -> pd.DataFrame:
    """Load the corrected-label data and attach reusable summary columns."""
    df = pd.read_csv(corrected_labels_path, usecols=LOAD_COLUMNS)
    denominator = 1.0 - LABEL_CORRUPTION_FALSE_NEGATIVE - LABEL_CORRUPTION_FALSE_POSITIVE
    corrected_unclipped = (
        df["observed_label_corrupted"].fillna(0.0) - LABEL_CORRUPTION_FALSE_POSITIVE
    ) / denominator
    df["_ht_contribution"] = df["observed"] * corrected_unclipped / df["q_hat_total"]
    df["_observed_fraud"] = (
        (df["observed"] == 1) & (df["observed_label_corrupted"] == 1)
    ).astype(int)
    return df


def _compute_kpis(df: pd.DataFrame) -> dict:
    """Compute hero metrics for the hosted demo."""
    n = len(df)
    true_rate = float(df["true_fraud"].mean())
    observed_rate = float(df["_observed_fraud"].mean())
    corrected_rate = float(df["_ht_contribution"].sum() / n)
    return {
        "n": int(n),
        "true_rate": true_rate,
        "observed_rate": observed_rate,
        "corrected_rate": corrected_rate,
        "undercount": corrected_rate / max(observed_rate, 1e-9),
    }


def _compute_funnel(df: pd.DataFrame) -> dict[str, int]:
    """Count how many true frauds survive each censorship stage."""
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


def _compute_blindspot_atlas(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate issuer-level blind spots for the hosted dashboard."""
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
    return (
        atlas.reset_index()[
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
        .sort_values("blindspot_multiplier", ascending=False)
        .reset_index(drop=True)
    )


def _compute_top_bias(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """Return transactions with the largest upward label correction."""
    columns = [
        "transaction_id",
        "issuer_id",
        "mcc",
        "amount",
        "country_pair",
        "channel",
        "baseline_fraud_score",
        "labellift_pseudo_label",
        "label_bias_delta",
    ]
    return df.nlargest(n, "label_bias_delta")[columns].reset_index(drop=True)


def _sample_scores(df: pd.DataFrame, n: int = 50_000) -> pd.DataFrame:
    """Down-sample score distributions for fast hosted rendering."""
    sample = df.sample(min(n, len(df)), random_state=42)
    return sample[["baseline_fraud_score", "labellift_pseudo_label"]].reset_index(drop=True)


def build_demo_artifacts(
    corrected_labels_path: Path = CORRECTED_LABELS_CSV,
    backtest_metrics_path: Path = BACKTEST_METRICS_CSV,
    output_dir: Path = DEMO_DATA_DIR,
) -> None:
    """Build all compact hosted-demo artifacts from generated full CSV outputs."""
    if not corrected_labels_path.exists():
        raise FileNotFoundError(
            f"Missing {corrected_labels_path}. Run `python -m labellift.estimator` first."
        )
    if not backtest_metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {backtest_metrics_path}. Run `python -m labellift.backtest` first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_json = output_dir / DEMO_MANIFEST_JSON.name
    kpis_json = output_dir / DEMO_KPIS_JSON.name
    funnel_json = output_dir / DEMO_FUNNEL_JSON.name
    blindspot_atlas_csv = output_dir / DEMO_BLINDSPOT_ATLAS_CSV.name
    score_sample_csv = output_dir / DEMO_SCORE_SAMPLE_CSV.name
    top_bias_csv = output_dir / DEMO_TOP_BIAS_CSV.name
    backtest_metrics_csv = output_dir / DEMO_BACKTEST_METRICS_CSV.name

    df = _prepare_labels(corrected_labels_path)

    _write_json(kpis_json, _compute_kpis(df))
    _write_json(funnel_json, _compute_funnel(df))
    _compute_blindspot_atlas(df).to_csv(blindspot_atlas_csv, index=False)
    _compute_top_bias(df).to_csv(top_bias_csv, index=False)
    _sample_scores(df).to_csv(score_sample_csv, index=False)
    pd.read_csv(backtest_metrics_path).to_csv(backtest_metrics_csv, index=False)
    _write_json(
        manifest_json,
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "source": "deterministic synthetic data, seed 42",
            "source_rows": int(len(df)),
            "synthetic_only": True,
            "files": [
                kpis_json.name,
                funnel_json.name,
                blindspot_atlas_csv.name,
                score_sample_csv.name,
                top_bias_csv.name,
                backtest_metrics_csv.name,
            ],
        },
    )


def main() -> None:
    """CLI entrypoint for rebuilding hosted-demo artifacts."""
    build_demo_artifacts()
    print(f"Wrote compact hosted-demo artifacts to {DEMO_DATA_DIR}")


if __name__ == "__main__":
    main()
