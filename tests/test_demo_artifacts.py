"""Tests for compact hosted-demo artifact generation."""

import json

import pandas as pd

from labellift.demo_artifacts import build_demo_artifacts


def test_build_demo_artifacts_writes_compact_bundle(tmp_path) -> None:
    corrected_labels = pd.DataFrame(
        {
            "transaction_id": ["TXN_1", "TXN_2", "TXN_3", "TXN_4"],
            "issuer_id": ["ISSUER_A", "ISSUER_A", "ISSUER_B", "ISSUER_B"],
            "mcc": ["digital_goods", "grocery", "travel", "gaming"],
            "amount": [100.0, 25.0, 500.0, 75.0],
            "country_pair": ["domestic", "domestic", "cross_border", "domestic"],
            "channel": ["card_not_present", "card_present", "wallet", "card_not_present"],
            "true_fraud": [1, 0, 1, 0],
            "authorized": [1, 1, 0, 1],
            "reported": [1, 0, 0, 1],
            "matured_by_training": [1, 0, 0, 1],
            "observed": [1, 0, 0, 1],
            "observed_label_corrupted": [1.0, None, None, 0.0],
            "q_hat_total": [0.50, 0.40, 0.20, 0.60],
            "baseline_fraud_score": [0.30, 0.10, 0.70, 0.20],
            "labellift_pseudo_label": [0.95, 0.10, 0.70, 0.00],
            "label_bias_delta": [0.65, 0.00, 0.00, -0.20],
        }
    )
    metrics = pd.DataFrame(
        {
            "model": ["raw_observed_label_model", "labellift_pseudo_label_model"],
            "average_precision": [0.2, 0.3],
        }
    )

    corrected_path = tmp_path / "corrected_labels.csv"
    metrics_path = tmp_path / "backtest_metrics.csv"
    output_dir = tmp_path / "demo_data"
    corrected_labels.to_csv(corrected_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    build_demo_artifacts(corrected_path, metrics_path, output_dir)

    expected_files = {
        "manifest.json",
        "kpis.json",
        "funnel.json",
        "blindspot_atlas.csv",
        "score_sample.csv",
        "top_bias.csv",
        "backtest_metrics.csv",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}

    kpis = json.loads((output_dir / "kpis.json").read_text())
    assert kpis["n"] == 4

    atlas = pd.read_csv(output_dir / "blindspot_atlas.csv")
    assert set(atlas["issuer_id"]) == {"ISSUER_A", "ISSUER_B"}
