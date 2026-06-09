# QA Report

Integration and end-to-end QA for the LabelLift MVP. Environment: Python 3.11.15,
fixed stack from `requirements.txt` (pandas 3.0, numpy 2.4, scikit-learn 1.9,
scipy, plotly, streamlit 1.58, pytest, ruff). All data is synthetic, seed `42`.

## Commands run

```bash
python -m labellift.synthetic_data    # -> data/synthetic_transactions.csv (1,000,000 rows, 176 MB)
python -m labellift.estimator         # -> data/corrected_labels.csv (304 MB)
python -m labellift.backtest          # -> data/backtest_metrics.csv (2 rows)
pytest                                # 27 passed
ruff check .                          # All checks passed
streamlit run app.py                  # dashboard renders, all 7 sections, no errors
```

## Pass/fail status

| Check | Status |
| --- | --- |
| Synthetic dataset exists (1,000,000 rows) | PASS |
| Corrected labels dataset exists | PASS |
| Backtest metrics dataset exists | PASS |
| `pytest` (27 tests across 3 files) | PASS |
| `ruff check .` | PASS |
| Dashboard loads via `streamlit run app.py` | PASS |
| All charts render (funnel, scatter, histogram, bar) | PASS |
| No Python tracebacks / Streamlit error boxes in UI | PASS |
| No console errors in browser | PASS |
| README instructions accurate | PASS |
| No file-path / import / missing-dependency errors | PASS |
| No real customer traction claims | PASS |
| No real payment data | PASS |
| No exposed secrets | PASS |
| No external / paid API calls | PASS |

### Synthetic data targets (verified)

| Target | Required | Observed |
| --- | --- | --- |
| True fraud rate | 0.8%–1.5% | 1.10% |
| Authorization rate | 82%–92% | 86.8% |
| Fraud auth rate < legit auth rate | yes | 73.8% < 87.0% |
| Reporting rate among authorized true fraud | 45%–75% | 47.9% |
| Maturity rate among reported true fraud | 45%–80% | 53.7% |
| Observed label rate < true fraud rate | yes | 0.55% < 1.10% |

### Backtest result (LabelLift vs. raw observed labels)

| Metric | Raw | LabelLift | LabelLift wins |
| --- | --- | --- | --- |
| Average precision | 0.0713 | 0.0740 | yes |
| Precision @ 1% | 0.1390 | 0.1447 | yes |
| Recall @ 1% | 0.1278 | 0.1330 | yes |
| ROC-AUC | 0.8054 | 0.8103 | yes |

LabelLift beats raw-label training on all three acceptance metrics (and ROC-AUC).
The lift is modest and synthetic; it demonstrates the mechanism, not a production
fraud-detection improvement.

## Bugs found

1. **Observation was outcome-dependent (MNAR).** The first synthetic generator made
   the reporting probability depend directly on the latent `true_fraud` flag. That
   violates the missing-at-random assumption LabelLift's inverse-propensity
   correction relies on, so the corrected fraud rate was unrecoverable (it overshot
   true prevalence by more than 10x).
2. **Label-noise correction did nothing in aggregate.** The clipped, per-row
   corrected label `(y - eps01)/(1 - eps10 - eps01)` collapses back to the raw
   binary label after clipping, so it did not remove the false-positive miscoding
   from any rate estimate.
3. **Blindspot Atlas table rates were 100x too small.** Streamlit's
   `NumberColumn(format=...)` applies printf to the raw fraction without scaling, so
   `0.868` rendered as `0.9%` instead of `86.8%`.
4. **Chart title overlapped the legend** in the backtest bar chart and the
   pseudo-label histogram.
5. **scikit-learn deprecation warning** from `LogisticRegression(n_jobs=-1)` (no-op
   since sklearn 1.8).

## Bugs fixed

1. Reporting now depends only on covariates (a function of the fraud-risk score),
   keeping observation missing-at-random. The Horvitz-Thompson reconstruction now
   recovers ~0.99% against a true 1.10% rate.
2. The dashboard's rate KPIs use the unclipped label correction in the
   inverse-propensity sum, which cancels false-positive and false-negative miscoding
   in aggregate. The clipped correction is retained only as the per-row training
   label (which must stay in `[0, 1]`).
3. The Blindspot Atlas converts fraction columns to percentage points before display.
4. Both chart legends were moved below the plot area.
5. Removed the deprecated `n_jobs` argument from every `LogisticRegression`.

## Known remaining issues

- Per-issuer corrected fraud rates in the Blindspot Atlas are noisier than the
  aggregate and can overshoot true prevalence, because there is no empirical Bayes
  issuer shrinkage yet (a deliberate MVP limitation).
- `data/corrected_labels.csv` is ~304 MB; it is regenerable and git-ignored. First
  dashboard load reads it once (~15 s) and then caches.
- The synthetic dataset is large, so the propensity and backtest tests run on
  deterministic sub-samples for speed; the full 1,000,000-row count is asserted
  against the generated CSV.

## Demo readiness

Ready. A new user can follow the README, run the three module commands (or click the
in-app generation buttons), and `streamlit run app.py` opens the full investor demo:
hero KPIs, the label-problem funnel, the Blindspot Atlas, corrected pseudo-labels,
the backtest, the integration pipeline, and the limitations. Screenshots are captured
in `artifacts/screenshots/`. No credentials, database, or network access required.
