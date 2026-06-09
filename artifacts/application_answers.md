# LabelLift Application Answers

Paste-ready. Each answer is under 120 words. No revenue, customer, or live-deployment
claims.

## Company one-liner

LabelLift is causal label infrastructure for fraud AI: we reconstruct corrected
pseudo-labels so banks and payment companies stop training fraud models on biased
chargeback labels.

## What are you building?

We build an offline historical backtest that turns a customer's raw chargeback labels
into corrected pseudo-labels for their existing fraud models. We estimate the
probability each historical transaction was authorized, reported, and matured in time,
then reweight and de-noise the observed labels. The output drops into the training
pipeline a customer already runs — no change to their production authorization or
decisioning stack. It is causal label infrastructure that sits upstream of model
training, not another fraud model.

## What problem are you solving?

Fraud models are trained on chargebacks, but chargebacks are not ground truth.
Declined transactions are never labeled, approved fraud is often never reported,
reported fraud frequently matures after the training window, and observed labels get
miscoded. Models therefore learn from a biased sample that undercounts true fraud, and
the bias concentrates in specific issuers and corridors. Teams experience this as
unexplained model decay and uneven performance, with no tooling to correct the
training target itself.

## Who is the customer?

Mid-size fintech card issuers and issuer processors that already run a fraud model and
hold historical authorization and chargeback data, but lack the causal-inference
tooling to correct their training labels. Buyers are heads of fraud, risk-data science
leads, and fraud-platform owners. We expand from issuers to PSPs, acquirers, and
networks, which all train fraud models on the same kind of biased labels.

## Why now?

Payments are shifting card-not-present and cross-border, which widens the gap between
true fraud and observed labels. Teams have invested in models and features but still
train them on years of accumulated chargeback bias. Inverse-propensity and causal
methods are now standard, and the inputs — authorization logs, dispute timing, reason
codes — already sit inside every issuer and processor. The correction was impractical
to productize before and is straightforward now.

## What have you built?

We have built a synthetic-data MVP that simulates the full payment-label pipeline on
one million transactions, estimates authorization, reporting, and maturity
propensities, reconstructs corrected pseudo-labels with a collapsed residual-weighted
estimator, and visualizes fraud blind spots per issuer. A backtest shows that training
on corrected pseudo-labels recovers more true synthetic fraud than training on raw
observed labels. No real payment data is used; everything is reproducible from a fixed
seed.

## What is your wedge?

A paid offline historical backtest. A customer hands us historical authorization and
chargeback data; we return corrected pseudo-labels and a measured lift on their own
matured holdout window. It is low-risk, touches nothing in production, and proves value
on the customer's data before any model change. The backtest is how we convert a fraud
team into a design partner and a design partner into a paid pilot.

## What is your business model?

A $50k paid historical backtest as the entry wedge, then $60k–$300k annual SaaS
contracts for fintech issuers and PSPs, scaling to $500k+ enterprise contracts for
large processors, acquirers, and networks. Pricing follows portfolio size and the
number of models and entities served. The backtest both generates revenue and
de-risks the pilot decision.

## Why can this become large?

Every issuer, processor, acquirer, PSP, and network trains fraud models on biased
labels, so corrected-label infrastructure is horizontal, not a point tool. It is
model-agnostic and sits beside whatever model a customer already runs, which makes it
expand naturally across portfolios and entities. As more historical backtests run, the
propensity and label-correction methodology compounds into a data and benchmarking
advantage that is hard to copy.

## What will you accomplish during Techstars?

Secure 3–5 design partners and run historical label audits in month one. Demonstrate
corrected-label lift on matured holdout windows in month two. Convert at least two
design partners into paid pilots in month three. Alongside, harden the estimator from
the collapsed MVP toward the full sequential version and stand up the first production
data connectors.

## Current status

We have built a synthetic-data MVP that simulates the payment-label pipeline, estimates
observation propensities, reconstructs corrected pseudo-labels, and visualizes fraud
blind spots. The next step is running paid historical backtests with design partners.

## Risks

The correction depends on observation being explainable from covariates; strongly
outcome-dependent censoring is harder and needs richer features. Inverse-propensity
weighting is variance-sensitive where observation probabilities are very small, which
production shrinkage must tame. Data access and procurement cycles at issuers are slow,
so the paid backtest exists partly to shorten them. Finally, lift must be proven on
real matured holdouts, not only synthetic data.
