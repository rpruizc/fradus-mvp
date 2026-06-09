# LabelLift MVP

Causal label infrastructure for fraud AI. **This MVP uses synthetic data only.**

## What this is

LabelLift is an offline label-reconstruction engine for fraud models. Fraud teams
train on chargebacks, but chargebacks are a censored and corrupted view of true
fraud: declined transactions are never labeled, approved fraud is often never
reported, reported fraud often matures after the training window, and observed
labels are miscoded. This MVP simulates that pipeline on 1,000,000 synthetic
transactions, estimates the authorization / reporting / maturity propensities,
reconstructs corrected pseudo-labels, and shows in a backtest that training on
the corrected labels recovers more true synthetic fraud than training on the raw
observed labels.

## What this is not

- Not a production fraud-detection system and not a replacement for a customer's
  authorization or decisioning stack.
- Not trained on any real financial, cardholder, issuer, or merchant data — every
  value is synthetic and generated from seed `42`.
- Not the full production estimator. This is the collapsed MVP estimator, not the
  sequential nested triply robust version, and it has no empirical Bayes issuer
  shrinkage yet.
- Not a claim of customer traction, revenue, or live deployment.

## How to run

```bash
# 1. Create an environment with Python 3.11 and install dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Build the synthetic dataset, corrected labels, and backtest metrics
python -m labellift.synthetic_data    # -> data/synthetic_transactions.csv
python -m labellift.estimator         # -> data/corrected_labels.csv
python -m labellift.backtest          # -> data/backtest_metrics.csv

# 3. Launch the dashboard
streamlit run app.py
```

The dashboard also regenerates any missing data file from a button, so you can
simply run `streamlit run app.py` on a clean checkout and click through
*Generate synthetic data → Run LabelLift correction → Run backtest*.

Run the test suite with `pytest` and lint with `ruff check .`.

## Demo flow

1. **Hero** — true vs. observed vs. LabelLift-corrected fraud rate, and the
   observed undercount multiplier.
2. **The label problem** — a funnel showing how few true frauds survive
   authorization, reporting, maturity, and miscoding to become a usable label.
3. **Blindspot Atlas** — per-issuer table and scatter of where fraud is most
   under-observed.
4. **Corrected pseudo-labels** — score distributions and the transactions
   LabelLift re-scores most aggressively upward.
5. **Backtest** — raw-label training vs. LabelLift-label training, scored against
   synthetic ground truth.
6. **How it plugs into fraud teams** — the five-step pipeline.
7. **MVP limitations** — what this demo deliberately does not do.

### Backtest result

On the synthetic backtest, the LabelLift pseudo-label model beats the raw observed
label model on all three headline metrics — average precision, precision@1%, and
recall@1% — as well as ROC-AUC. The lift is modest and synthetic; it demonstrates
the mechanism, not a production fraud-detection improvement. If you re-tune the
pipeline and LabelLift no longer wins, tune the synthetic data generator, not the
metrics.

## File structure

```text
labellift-mvp/
  app.py                      # Streamlit dashboard (the investor demo)
  README.md
  requirements.txt
  pyproject.toml              # ruff line length 100
  data/
    synthetic_transactions.csv
    corrected_labels.csv
    backtest_metrics.csv
  labellift/
    __init__.py
    config.py                 # seed 42, dataset + corruption + propensity constants
    synthetic_data.py         # the payment-label observation pipeline
    propensities.py           # authorization / reporting / maturity propensities
    estimator.py              # collapsed residual-weighted pseudo-label estimator
    backtest.py               # raw-label vs. pseudo-label training comparison
    metrics.py                # precision/recall at top-k%
    plots.py                  # Plotly figures for the dashboard
  tests/
    test_synthetic_data.py
    test_estimator.py
    test_backtest.py
  artifacts/
    techstars_one_pager.md
    demo_script.md
    application_answers.md
    design_partner_targets.csv
    outbound_messages.md
    QA_REPORT.md
    screenshots/
```

## Known limitations

- Synthetic data only; no real payment data and no production data connectors.
- Collapsed estimator only, not the full sequential nested estimator.
- No empirical Bayes issuer shrinkage yet, so per-issuer corrected rates in the
  Blindspot Atlas are noisier than the aggregate.
- No real customer backtest yet and no claim of production fraud-detection lift.
- Inverse-propensity weighting recovers the aggregate fraud rate well but is
  variance-sensitive where observation propensities are very small.
