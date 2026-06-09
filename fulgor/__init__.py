"""Fulgor MVP package.

Causal label infrastructure for fraud AI. This package implements a synthetic
demonstration of the Fulgor label-reconstruction pipeline:

- ``synthetic_data``  — generate the payment-label observation pipeline.
- ``propensities``    — estimate authorization, reporting, and maturity propensities.
- ``estimator``       — collapsed residual-weighted pseudo-label estimator.
- ``backtest``        — compare raw-label vs. pseudo-label training.
- ``metrics``         — precision/recall helpers at top-k%.
- ``plots``           — Plotly figures for the Streamlit dashboard.

All synthetic data is reproducible from ``config.RANDOM_SEED``.
"""

__version__ = "0.1.0"
