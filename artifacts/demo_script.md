# Fulgor — 3-minute demo script

Target length: ~3 minutes. Run `streamlit run app.py` and scroll top to bottom.

## 0:00 — Open on the hero (beat 1 + 2)

> "Every fraud team I talk to treats chargebacks as ground truth. They train their
> models on them. But chargebacks are not ground truth — they're the *end* of a long
> pipeline that quietly throws away most of the real fraud."

Point at the four KPI cards.

> "In this synthetic portfolio the true fraud rate is about 1.1%. But the labels the
> model actually gets to see? About half a percent. The observed labels undercount
> real fraud by nearly two times — and that gap is exactly what Fulgor rebuilds."

## 0:35 — The label problem funnel (beat 2)

> "Here's why. Of every true fraud, some are declined and never labeled. Of what's
> approved, only some get reported. Of what's reported, only some mature inside the
> training window. And then a chunk is simply miscoded."

Trace the funnel down.

> "By the time you reach the bottom, the model is training on a small, biased slice —
> not the full fraud process. Chargebacks are censored four different ways."

## 1:10 — Fulgor corrects it (beat 3)

> "Fulgor is an offline engine. It estimates the probability each transaction was
> authorized, reported, and matured, then reweights and de-noises the observed labels
> into corrected pseudo-labels."

Point back at the hero's "Fulgor corrected fraud rate."

> "Notice the corrected rate lands right back near the true 1.1%. We reconstructed the
> fraud the raw labels had hidden — without ever seeing the ground truth."

## 1:40 — Blindspot Atlas (beat 4)

> "And the bias isn't uniform. This is the Blindspot Atlas. Each row is an issuer.
> The blindspot multiplier is how badly that issuer under-observes fraud."

Point at the top rows and the scatter.

> "Some issuers are off by five or six times. These are the portfolios where a fraud
> model is most confidently wrong. Fulgor shows you exactly where to look."

## 2:10 — Backtest (beat 5)

> "Does correcting the labels actually help? We split the data, train one model on the
> raw observed labels and one on Fulgor pseudo-labels, and score both against the
> synthetic ground truth."

Point at the bar chart.

> "Fulgor wins on average precision, on precision at the top 1%, and on recall at
> the top 1%. Same model, same features — just a better training target. In a real
> pilot we'd evaluate on a later matured holdout instead of synthetic truth."

## 2:40 — The wedge (beat 6)

> "We don't ask anyone to rip out their stack. The wedge is a paid historical
> backtest: give us your logs, we hand back corrected labels and the lift number on
> your own data. That's how we turn a fraud team into a design partner, and a design
> partner into a paid pilot."

> "Fulgor — causal label infrastructure for fraud AI."
