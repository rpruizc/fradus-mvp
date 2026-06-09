# Fulgor

## One-liner

Fulgor fixes corrupted fraud-training labels before banks and payment companies
train their fraud models.

## Problem

Fraud models are trained on chargebacks, but chargebacks are not ground truth. The
training labels are systematically censored and corrupted before a model ever sees
them:

- Declined transactions are never labeled.
- Approved fraud may never be reported.
- Reported fraud may arrive too late for training.
- Observed labels can be miscoded.

The result is that models learn from a biased sample that undercounts true fraud,
and the bias is concentrated in specific issuers, segments, and corridors — exactly
where teams least want a blind spot.

## Solution

Fulgor is an offline label-reconstruction engine. It estimates the probability
that each historical transaction was authorized, reported, and matured in time,
then reweights and de-noises the observed labels into corrected pseudo-labels.
Customers train their existing fraud models on these corrected pseudo-labels. It is
not a replacement of production authorization systems — it sits upstream of model
training and improves the target the model learns from.

## Why now

Fraud losses and false declines are both rising as payments move card-not-present
and cross-border. Teams have invested heavily in models and features but still
train them on chargeback labels that are years of accumulated bias. Causal and
inverse-propensity methods are now standard tooling, and the data to run them
(authorization logs, dispute timing, chargeback reason codes) already sits in every
issuer and processor.

## Initial customer

Mid-size fintech card issuers and issuer processors that already run an in-house or
vendor fraud model and have historical authorization and chargeback data, but lack
the causal-inference tooling to correct their training labels. They feel the pain as
unexplained model decay and uneven performance across portfolios.

## MVP

A synthetic-data MVP that simulates the full payment-label pipeline on 1,000,000
transactions, estimates authorization / reporting / maturity propensities,
reconstructs corrected pseudo-labels with a collapsed residual-weighted estimator,
and visualizes fraud blind spots. A backtest shows that training on corrected
pseudo-labels recovers more true synthetic fraud than training on raw observed
labels. No real payment data is used.

## Business model

- A $50k paid historical backtest as the entry wedge.
- $60k–$300k annual SaaS contracts for fintech issuers and PSPs.
- $500k+ enterprise contracts for large processors, acquirers, or networks.

## Why this can be venture-scale

Every issuer, processor, acquirer, PSP, and network trains fraud models on biased
labels, so the corrected-label layer is horizontal infrastructure rather than a
point tool. It is model-agnostic and sits beside whatever model the customer already
runs, which makes it expand-friendly across portfolios and entities. As more
historical backtests run, the propensity and label-correction methodology compounds
into a defensible data and benchmarking moat.

## 90-day Techstars plan

- Month 1: secure 3–5 design partners and run historical label audits.
- Month 2: demonstrate corrected-label lift on matured holdout windows.
- Month 3: convert at least two design partners into paid pilots.
