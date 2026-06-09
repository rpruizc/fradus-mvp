# Fulgor Paper Implementation Roadmap

This roadmap turns the paper, "Causal Label Recovery in Payment Networks: A
Sequential Triply Robust Estimator Under Authorization, Reporting, Delay, and
Label Corruption", into versioned product releases.

The goal is to move from the current synthetic, collapsed-estimator demo to a
customer-testable implementation of the paper's operational architecture. Each
sprint must end with a releasable app state that can be opened in a browser and
tested by a customer or reviewer without the developer narrating the flow.

## Agent Rules

- Treat this file as the sprint source of truth.
- Do not widen scope beyond the task being implemented.
- Every code change must be traceable to the paper source listed in the task.
- Every task must include tests before it is marked complete.
- Keep the current synthetic demo working unless the task explicitly changes it.
- Preserve self-service browser behavior: the modern web app must load from
  checked-in demo artifacts when no full local CSVs are present.
- Do not claim production lift from synthetic data. Any customer-facing text must
  distinguish synthetic demo results from pilot results on real historical data.
- Treat `application-ui-v4/` as a reference-only UI kit. Do not import runtime
  code from that directory. When a reference component is needed, copy and adapt
  the React example into `apps/web/src/` with product-specific props, tests, and
  no placeholder Tailwind UI demo data.

## Paper Coverage Matrix

These are the 12 implementation areas required for full paper coverage.

1. Real data ingestion and schema contract
   - Paper source: Sections 3.1-3.4, 10.3, 10.5.
   - Current state: synthetic-only generator and CSV outputs.

2. Temporal matured-holdout backtest
   - Paper source: Sections 10.5, 10.6, 11.7.
   - Current state: synthetic holdout scored against `true_fraud`.

3. Algorithm 1 cross-fitting
   - Paper source: Definition 31, Algorithm 1, Theorem 32.
   - Current state: nuisance models fit and predict on the same data.

4. STR pseudo-outcomes and final conditional pseudo-labels
   - Paper source: Definitions 15-16, Equation 22, Section 10.1.
   - Current state: clipped collapsed residual pseudo-label only.

5. Empirical Bayes issuer shrinkage
   - Paper source: Section 9, Definition 38, Equations 33-37.
   - Current state: clipping only; no issuer-level shrinkage.

6. Diagnostics, uncertainty, and validation
   - Paper source: Definition 33, Proposition 34, Sections 8, 11.7.
   - Current state: dashboard metrics only.

7. Conditional maturity windows and optimal training delay
   - Paper source: Section 3.6, Definition 43, Theorem 46.
   - Current state: fixed 30-day synthetic training window.

8. Full nested sequential STR estimator
   - Paper source: Definition 15, Definition 16, Sections 3.4-3.5, 10.3.
   - Current state: collapsed special case only.

9. Segment-specific label-corruption rates
   - Paper source: Definition 18, Sections 5.1-5.4, Section 11.7.
   - Current state: global constants for false negative and false positive rates.

10. Positivity violation handling
    - Paper source: Assumption 8, Section 11.3.
    - Current state: propensities are clipped but unreliable regions are not
      explicitly flagged.

11. Flexible nuisance-model registry and calibration
    - Paper source: Sections 10.2-10.3, 11.7-11.8.
    - Current state: fixed logistic regressions for propensities and baseline
      score.

12. Offline-online product packaging
    - Paper source: Section 10.5, Algorithm 1.
    - Current state: Streamlit demo plus separate Python module commands.

## Release v0.1 - Product Architecture Migration

Release goal: leave Streamlit behind and establish the production-shaped app
architecture before implementing the rest of the paper. The estimator remains a
Python package and offline pipeline, because the paper's Section 10.5 architecture
defines Fulgor as an offline label-reconstruction engine whose outputs feed
downstream fraud models. The browser product becomes a modern web UI backed by an
API, not a Streamlit script.

Architecture decision for this release:

- Core package: keep `fulgor/` as the Python estimator and artifact package.
- Pipeline: add a Python CLI/job layer that produces durable run artifacts.
- API: add a FastAPI service in `services/api/fulgor_api/`.
- Web UI: add a React + TypeScript + Vite app in `apps/web/`.
- UI reference: use `application-ui-v4/react/` as the design/component reference
  for application shells, stats, tables, forms, alerts, tabs, drawers, modals,
  and page examples. Copy/adapt selected components into `apps/web/src/`; do not
  import from `application-ui-v4/`.
- Hosting shape: the FastAPI service must be able to serve the built web UI and
  expose JSON endpoints, so the self-service demo can run as one deployable web
  process.
- Streamlit: keep `app.py` only until modern UI parity exists, then remove it in
  FLGR-000-l.

Customer test: run one local command, open a browser, and see the same
self-service synthetic demo currently available in Streamlit, rendered by the new
web UI and backed by API responses from compact demo artifacts.

### Tasks

- [x] FLGR-000-a - Add architecture decision record for replacing Streamlit.
  - Paper source: Section 10.5 offline-online architecture.
  - Add `docs/architecture.md` that states the package + pipeline + API + web UI
    boundary, the chosen stack (`fulgor/`, FastAPI, React + TypeScript +
    Vite, Tailwind CSS, Headless UI, Heroicons), and why the estimator stays
    offline.
  - Document `application-ui-v4/` as the reference-only component source and
    state that copied/adapted components must live under `apps/web/src/`.
  - The document must explicitly say Streamlit is not the product shell.
  - Test: no code test required; README must link to this document.

- [ ] FLGR-000-b - Create package boundary for product artifacts.
  - Paper source: Algorithm 1 outputs and Section 10.5 Phase 1.
  - Add `fulgor/artifacts.py` with typed path helpers for demo artifacts,
    run manifests, summaries, diagnostics, corrected labels, and reports.
  - Existing `fulgor/demo_artifacts.py` must use these helpers instead of
    hard-coded output paths.
  - Test: unit test verifies every artifact path is rooted in either
    `artifacts/demo_data/` or a caller-provided run directory.

- [ ] FLGR-000-c - Add pipeline run summary contract.
  - Paper source: Algorithm 1 ensures pseudo-labels, point estimate, confidence
    interval.
  - Add `fulgor/run_summary.py` with dataclasses or typed dictionaries for
    `DemoSummary`, `EstimatorSummary`, `BacktestSummary`, and `RunSummary`.
  - The current compact demo artifact reader must be able to produce a
    `DemoSummary`.
  - Test: demo artifacts load into `DemoSummary` with stable KPI and metric
    fields.

- [ ] FLGR-000-d - Add FastAPI backend scaffold.
  - Paper source: Section 10.5 Phase 1 exposes offline reconstruction outputs to
    Phase 2 consumers.
  - Add `services/api/fulgor_api/main.py` with a FastAPI app, `/health`, and
    `/api/version`.
  - `/health` must return `{ "status": "ok" }`.
  - Test: FastAPI test client verifies both endpoints.

- [ ] FLGR-000-e - Add demo artifact API endpoints.
  - Paper source: Section 10.1 pseudo-label outputs and Section 10.5 Phase 1
    outputs.
  - Add endpoints:
    - `GET /api/demo/summary`
    - `GET /api/demo/funnel`
    - `GET /api/demo/blindspot-atlas`
    - `GET /api/demo/scores`
    - `GET /api/demo/top-bias`
    - `GET /api/demo/backtest`
  - All endpoints must read from checked-in compact artifacts, not from
    `data/*.csv`.
  - Test: API tests pass with only `artifacts/demo_data/` present.

- [ ] FLGR-000-f - Add React + TypeScript + Vite web app scaffold.
  - Paper source: Section 10.5 requires a user-facing offline reconstruction
    product surface; framework choice is an implementation detail fixed by this
    task.
  - Create `apps/web/` with Vite, React, TypeScript, Tailwind CSS, Headless UI,
    Heroicons, and a test runner.
  - Configure Tailwind so classes used by adapted `application-ui-v4/react/`
    examples compile correctly, including dark-mode-safe classes where retained.
  - Create local folders `apps/web/src/components/ui/`,
    `apps/web/src/components/layout/`, `apps/web/src/components/charts/`, and
    `apps/web/src/features/demo/`.
  - The app must call `/api/health` or `/health` on startup and render a
    recoverable error state if the API is unavailable.
  - Test: frontend test renders the app shell and an API-unavailable state.

- [ ] FLGR-000-g - Rebuild the current demo dashboard in React.
  - Paper source: Section 10.1 pseudo-labels and Section 10.5 offline-online
    architecture.
  - Use `application-ui-v4/react/page-examples/home-screens/01-sidebar.jsx` or
    `application-ui-v4/react/application-shells/sidebar/03-sidebar-with-header.jsx`
    as the reference for the app shell, then copy/adapt the needed layout into
    `apps/web/src/components/layout/AppShell.tsx`.
  - Use `application-ui-v4/react/data-display/stats/05-with-shared-borders.jsx`
    as the reference for hero KPI cards, copied/adapted into a local KPI
    component.
  - Use `application-ui-v4/react/feedback/alerts/05-with-accent-border.jsx` as
    the reference for reviewer summary and warning panels.
  - Use `application-ui-v4/react/lists/tables/12-with-condensed-content.jsx` or
    `application-ui-v4/react/lists/tables/13-with-sortable-headings.jsx` as the
    reference for dense fraud analytics tables.
  - Implement React views for: hero KPIs, reviewer summary, label-problem
    funnel, Blindspot Atlas, pseudo-label distribution, top-bias table, backtest,
    pipeline explanation, and limitations.
  - The displayed numbers must come from API responses, not embedded constants.
  - Test: mocked API responses render the expected KPI text and backtest values.

- [ ] FLGR-000-h - Add chart components for the modern UI.
  - Paper source: Sections 10.1 and 11.7 require visual inspection of
    pseudo-labels, propensities, and diagnostics.
  - Chart components must visually fit the local components adapted from
    `application-ui-v4/react/`; do not use a conflicting chart theme.
  - Use a React chart library already added in `apps/web/package.json` and build
    reusable chart components for funnel, scatter, histogram, and grouped bars.
  - Test: component tests verify empty-data states and normal-data states.

- [ ] FLGR-000-i - Serve the built web UI from FastAPI.
  - Paper source: Section 10.5 operational architecture.
  - Add backend static-file serving for `apps/web/dist`.
  - Unknown non-API routes must return the web app entrypoint so browser refresh
    works.
  - Test: FastAPI test client returns HTML for `/` and JSON for `/api/demo/summary`.

- [ ] FLGR-000-j - Add one-command local product launch.
  - Paper source: Section 10.5 periodic offline reconstruction must not be
    required for online demo viewing.
  - Add a documented command, for example `uv run python -m
    services.api.fulgor_api.main` or a `Makefile` target, that starts the API
    and serves the built web UI.
  - The command must not require full `data/*.csv` files.
  - Test: smoke test starts the app process, calls `/health`, and exits cleanly.

- [ ] FLGR-000-k - Add deployment packaging for the modern app.
  - Paper source: Section 10.5 one offline-online product architecture.
  - Add deployment docs and config for a single web service that installs Python
    dependencies, builds `apps/web`, and starts FastAPI.
  - Do not require Streamlit Community Cloud.
  - Test: CI/build command verifies the frontend build output exists and FastAPI
    can import after build.

- [ ] FLGR-000-l - Remove Streamlit from the product path.
  - Paper source: Section 10.5; Streamlit is not part of the paper architecture.
  - After modern UI parity is verified, remove `app.py`, `.streamlit/`, and the
    `streamlit` dependency from `requirements.txt` and `pyproject.toml`.
  - Keep `artifacts/demo_data/` and the demo artifact builder.
  - Test: `rg -n "streamlit|Streamlit" .` returns matches only in migration
    history/docs explaining removal, not runtime code.

## Release v0.2 - Pilot Data Contract

Release goal: a customer can open the app, upload a historical transaction CSV,
and receive a deterministic validation report that maps their data to the paper's
observed-data structure. This release does not estimate STR yet on customer data.

Customer test: upload a sample CSV and confirm the app identifies required
columns, derived paper variables, missing values, and unusable rows.

### Tasks

- [ ] FLGR-001 - Add canonical paper schema constants.
  - Paper source: Section 3.1, Section 3.2.
  - Implement `fulgor/schema.py` with named column groups for `X`, `I`,
    `Delta`, `A`, `R`, `M`, `O`, `O_Y_tilde`, and optional stage histories
    `W1`, `W2`.
  - Include required raw input aliases: `transaction_id`, `transaction_time`,
    `issuer_id`, `authorized`, `reported`, `label_arrival_time`,
    `observed_label`.
  - Test: unit test asserts every paper variable has a canonical column name.

- [ ] FLGR-002 - Add a schema validation result object.
  - Paper source: Section 3.2 observed data `Z_t`.
  - Implement a pure-Python validator that returns `errors`, `warnings`,
    `derived_columns`, and `row_counts`.
  - The validator must not call the API layer or the web UI.
  - Test: missing `issuer_id` produces an error; missing optional `W1` produces
    no error.

- [ ] FLGR-003 - Derive paper gate variables from raw customer fields.
  - Paper source: Definitions 2-4 and Equation 2.
  - Implement `derive_observation_gates(df, training_time)` that computes:
    `available_maturity_days`, `authorized`, `reported`,
    `matured_by_training`, and `observed = authorized * reported * matured`.
  - `reported` must be meaningful only where `authorized == 1`.
  - Test: declined rows always have `observed == 0`.

- [ ] FLGR-004 - Add deterministic pilot fixture CSVs.
  - Paper source: Section 10.5 "historical transactions with sufficiently
    matured observation windows."
  - Add small fixtures under `tests/fixtures/` for valid data, missing required
    columns, invalid timestamps, and never-reported rows.
  - Test: fixtures are loadable and validation outcomes match expectations.

- [ ] FLGR-005 - Add a CSV ingestion module.
  - Paper source: Section 10.3 nuisance function estimation inputs.
  - Implement `fulgor/ingestion.py` with `load_pilot_csv(path)` and
    `validate_and_prepare_pilot_data(path, training_time)`.
  - Output must be a dataframe with the canonical schema from FLGR-001.
  - Test: valid fixture returns canonical columns and stable row counts.

- [ ] FLGR-006 - Add a browser upload validation screen.
  - Paper source: Section 10.5 Phase 1 offline reconstruction.
  - Add a React app section and API endpoint that accept a CSV upload and show
    the validation report.
  - Use `application-ui-v4/react/forms/form-layouts/03-two-column.jsx` as the
    reference for the upload/configuration form layout and
    `application-ui-v4/react/feedback/alerts/03-with-actions.jsx` or
    `05-with-accent-border.jsx` as the reference for validation errors and
    warnings. Copy/adapt these into local `apps/web/src/components/` files.
  - The existing synthetic demo must remain the default when no upload is
    provided.
  - Test: API upload test validates the fixture CSV; frontend test renders the
    validation report from a mocked API response.

- [ ] FLGR-007 - Add a `validate-data` CLI entrypoint.
  - Paper source: Section 10.5 Phase 1 offline reconstruction.
  - Add a module command that validates a CSV and writes a JSON validation report.
  - Test: CLI function writes JSON for the valid fixture.

- [ ] FLGR-008 - Document the customer CSV contract.
  - Paper source: Sections 3.1-3.4.
  - Add `docs/pilot-data-contract.md` with required fields, optional fields,
    accepted timestamp formats, and the exact derivation of `A`, `R`, `M`, `O`.
  - Test: no code test required; README must link to the document.

## Release v0.3 - Customer-Testable Matured-Holdout Backtest

Release goal: a customer can upload historical data and test a raw-label baseline
against a later matured holdout using the paper's offline evaluation framing.

Customer test: upload pilot data with dates, choose train and holdout windows, and
download a metrics report that clearly says which labels were used for evaluation.

### Tasks

- [ ] FLGR-009 - Add temporal window splitter.
  - Paper source: Section 10.6.1, training window `[T-W-Delta, T-Delta]`.
  - Implement `build_temporal_windows(df, train_end, train_width_days,
    maturity_delay_days, holdout_width_days)`.
  - Test: fixture rows fall into train, gap, holdout, or excluded windows
    deterministically.

- [ ] FLGR-010 - Add later-matured holdout label builder.
  - Paper source: Section 10.5 Phase 1 and Phase 2.
  - Implement holdout labels using rows whose labels have matured by the later
    evaluation time.
  - Do not use `true_fraud` for pilot evaluation.
  - Test: synthetic rows with `true_fraud` present still use matured observed
    labels when pilot mode is selected.

- [ ] FLGR-011 - Split synthetic backtest mode from pilot backtest mode.
  - Paper source: Section 10.1 conditional pseudo-labels and Section 10.5
    deployment architecture.
  - Keep `true_fraud` evaluation only for synthetic demo mode.
  - Add a mode flag or function split that makes pilot mode impossible to score
    against `true_fraud`.
  - Test: pilot mode raises if metric code tries to read `true_fraud`.

- [ ] FLGR-012 - Add pilot baseline model.
  - Paper source: Section 6 naive estimator comparison.
  - Implement raw observed-label training on training-window observed labels.
  - Use the same feature set as the corrected-label model for fair comparison.
  - Test: baseline trains on the valid fixture without reading unobserved labels.

- [ ] FLGR-013 - Add pilot evaluation metrics.
  - Paper source: Section 11.8 proper scoring rules and Section 10.5 pilot
    architecture.
  - Compute average precision, ROC-AUC when both classes exist, precision@1%,
    precision@5%, recall@1%, recall@5%, Brier score, and log loss against
    later-matured holdout observed labels.
  - Test: metrics return finite values or documented `None` when a metric is not
    identifiable because the holdout has one class.

- [ ] FLGR-014 - Add pilot backtest report artifact.
  - Paper source: Section 10.5 Phase 1 and Phase 2.
  - Write `artifacts/pilot_reports/<run_id>/metrics.csv`,
    `window_config.json`, and `label_counts.json`.
  - Test: report builder creates all expected files from fixtures.

- [ ] FLGR-015 - Add app controls for temporal backtest.
  - Paper source: Section 10.6.1.
  - In the upload flow, let the user set `train_end`, `train_width_days`,
    `maturity_delay_days`, and `holdout_width_days`.
  - Use local form controls adapted from `application-ui-v4/react/forms/input-groups/`,
    `select-menus/`, and `radio-groups/`; do not hand-roll visually unrelated
    controls.
  - Test: helper functions produce the same config shown in the app.

## Release v0.4 - Cross-Fitted Collapsed STR

Release goal: the app produces cross-fitted collapsed STR pseudo-outcomes,
corrected pseudo-labels, a fraud-rate estimate, variance, and confidence interval.

Customer test: run the pipeline on the sample data and download a row-level output
with `str_pseudo_outcome` and `fulgor_pseudo_label`.

### Tasks

- [ ] FLGR-016 - Add K-fold assignment utility.
  - Paper source: Definition 31.
  - Implement deterministic fold assignment with configurable `K` and seed.
  - Store `fold_id` in the output dataframe.
  - Test: every row is assigned exactly one fold and fold assignments are stable.

- [ ] FLGR-017 - Add cross-fitted propensity estimation.
  - Paper source: Definition 31 and Algorithm 1 line 2.
  - For each fold, fit `e_hat`, `r_hat`, `p_hat` on rows excluding that fold and
    predict for rows inside that fold.
  - Test: monkeypatch the estimator to prove fold rows are not used in their own
    nuisance fit.

- [ ] FLGR-018 - Add unclipped corruption-corrected outcome.
  - Paper source: Equation 21.
  - Implement `y_tilde_corr = (y_tilde - eps01) / (1 - eps10 - eps01)`.
  - Do not clip this column.
  - Test: observed label 0 with eps01 > 0 can produce a negative corrected value.

- [ ] FLGR-019 - Add collapsed STR pseudo-outcome.
  - Paper source: Equation 17 and Equation 22 collapsed special case.
  - Implement `str_pseudo_outcome = f_hat + O / q_hat_total *
    (y_tilde_corr - f_hat)`.
  - Keep `str_pseudo_outcome` unclipped.
  - Test: formula matches hand-calculated fixture values.

- [ ] FLGR-020 - Add plug-in fraud-rate estimate and confidence interval.
  - Paper source: Algorithm 1 lines 9-11 and Definition 33.
  - Compute `psi_hat = mean(str_pseudo_outcome)`, `sigma_hat_squared`,
    `standard_error`, and 95% CI.
  - Test: constant pseudo-outcomes produce zero variance and zero-width CI.

- [ ] FLGR-021 - Train final conditional pseudo-label model.
  - Paper source: Section 10.1, Equations 39-40.
  - Regress `str_pseudo_outcome` on transaction features `X` and write
    `fulgor_pseudo_label`.
  - The final pseudo-label may be clipped to `[0, 1]`; the pseudo-outcome must
    remain unclipped.
  - Test: output has both columns and only `fulgor_pseudo_label` is bounded.

- [ ] FLGR-022 - Add cross-fitted collapsed STR pipeline function.
  - Paper source: Algorithm 1.
  - Implement `run_collapsed_str_crossfit(df, k, corruption_rates, model_config)`.
  - Output row-level labels and run-level summary.
  - Test: function runs end-to-end on fixtures and synthetic subsample.

- [ ] FLGR-023 - Update hosted demo artifacts for STR outputs.
  - Paper source: Section 10.1.
  - Add compact summaries for pseudo-outcome distribution, `psi_hat`, CI, and
    final pseudo-label distribution.
  - Test: `fulgor.demo_artifacts` writes the new files.

## Release v0.5 - Empirical Bayes Shrinkage and Positivity

Release goal: issuer-level reporting and maturity propensities are stabilized,
and the app explicitly flags regions where the paper says causal recovery is not
identified.

Customer test: upload data with low-volume issuers and confirm the app shows
local, global, and EB-shrunk propensities plus positivity warnings.

### Tasks

- [ ] FLGR-024 - Add issuer-level propensity summary tables.
  - Paper source: Section 9.1.
  - Compute issuer counts, local mean authorization rate, reporting rate, maturity
    rate, and sampling variance proxies.
  - Test: low-volume and high-volume fixture issuers produce expected counts.

- [ ] FLGR-025 - Implement EB shrinkage weight.
  - Paper source: Definition 38, Equation 34.
  - Implement `lambda_i = sigma_B_squared / (sigma_B_squared +
    sigma_i_squared / n_i)`.
  - Clamp only for numerical validity, not as a replacement for the formula.
  - Test: large `n_i` yields lambda closer to 1 than small `n_i`.

- [ ] FLGR-026 - Apply EB shrinkage to reporting propensity.
  - Paper source: Equation 33.
  - Blend issuer-local and pooled global reporting propensities using `lambda_i`.
  - Test: small issuer estimate moves toward global estimate.

- [ ] FLGR-027 - Apply EB shrinkage to maturity propensity.
  - Paper source: Section 9.4, Equation 35.
  - Blend issuer-local and pooled global maturity propensities using delay-aware
    inputs where available.
  - Test: maturity shrinkage uses the maturity lambda, not the reporting lambda.

- [ ] FLGR-028 - Apply EB shrinkage to authorization propensity.
  - Paper source: Section 9.5, Equation 36.
  - Implement the same construction for authorization, while keeping it
    separately configurable because the paper says it is often more stable.
  - Test: disabling authorization shrinkage preserves unshrunk `e_hat`.

- [ ] FLGR-029 - Add shrinkage mode to STR pipeline.
  - Paper source: Equation 37 and Algorithm 1 line 3.
  - The cross-fitted pipeline must support `none`, `reporting_delay`, and `all`
    shrinkage modes.
  - Test: pipeline output includes both raw and EB propensity columns.

- [ ] FLGR-030 - Add positivity diagnostics.
  - Paper source: Assumption 8 and Section 11.3.
  - Compute min, p1, p5, median, and histogram bins for `e_hat`, `r_hat`,
    `p_hat`, and `q_hat_total`.
  - Test: fixture with zero reporting segment is flagged.

- [ ] FLGR-031 - Add identified-region flags.
  - Paper source: Section 11.3 structural positivity violation.
  - Add row-level `positivity_status` values: `identified`,
    `near_zero_propensity`, `structural_zero`, `model_extrapolation`.
  - Test: rows in an always-declined issuer/MCC segment are not marked
    `identified`.

- [ ] FLGR-032 - Surface shrinkage and positivity in the app.
  - Paper source: Sections 9.2 and 11.3.
  - Add an "Estimator Reliability" section with shrinkage effects and positivity
    warnings before backtest results.
  - Use local table, badge, and alert components adapted from
    `application-ui-v4/react/lists/tables/`,
    `application-ui-v4/react/elements/badges/`, and
    `application-ui-v4/react/feedback/alerts/`.
  - Test: API response schema and frontend component render fixture reliability
    summaries.

## Release v0.6 - Full Nested Sequential STR

Release goal: implement the paper's general sequential score with distinct
histories and nested regressions, while preserving the collapsed estimator as a
special case.

Customer test: run with only pre-authorization fields and see collapsed/full
equivalence; upload extra post-authorization fields and see full STR use them.

### Tasks

- [ ] FLGR-033 - Add stage history configuration.
  - Paper source: Definition 5.
  - Implement a config object with `H0`, `H1`, and `H2` feature lists.
  - Default `H0 = H1 = H2` for collapsed compatibility.
  - Test: config validation rejects `H2` features that are not present.

- [ ] FLGR-034 - Add post-authorization and post-reporting feature support.
  - Paper source: Section 3.5.
  - Extend schema validation to accept optional `W1` and `W2` feature columns.
  - Test: fixture with `three_ds_result` as `W1` and `evidence_quality` as `W2`
    validates.

- [ ] FLGR-035 - Implement `mu2` regression.
  - Paper source: Section 10.3 nested regressions step 1.
  - Regress `y_tilde_corr` on `H2` among fully observed rows.
  - Test: `mu2_hat` is predicted for all rows where `H2` is available.

- [ ] FLGR-036 - Implement `mu1` regression.
  - Paper source: Section 10.3 nested regressions step 2.
  - Regress `mu2_hat` on `H1` among approved-and-reported rows.
  - Test: `mu1_hat` is predicted for all rows where `H1` is available.

- [ ] FLGR-037 - Implement `mu0` regression.
  - Paper source: Section 10.3 nested regressions step 3.
  - Regress `mu1_hat` on `H0` among all transactions.
  - Test: `mu0_hat` is predicted for every input row.

- [ ] FLGR-038 - Implement general sequential score.
  - Paper source: Definition 15 and Equation 22.
  - Compute `phi_corr = mu0 + A/e*(mu1-mu0) + A*R/(e*r)*(mu2-mu1)
    + A*R*M/(e*r*p)*(y_tilde_corr-mu2)`.
  - Test: hand-calculated fixture matches exactly.

- [ ] FLGR-039 - Cross-fit all nested nuisance regressions.
  - Paper source: Definition 31 and Algorithm 1 line 2.
  - Fit `e`, `r`, `p`, `mu0`, `mu1`, `mu2` outside each held-out fold.
  - Test: fold-leakage monkeypatch covers all six nuisance models.

- [ ] FLGR-040 - Prove collapsed equivalence in tests.
  - Paper source: Section 4.2.
  - When `H0 = H1 = H2` and `mu0 = mu1 = mu2 = f_hat`, full STR output must
    match the collapsed score within numerical tolerance.
  - Test: deterministic fixture asserts equality.

- [ ] FLGR-041 - Add estimator mode selector.
  - Paper source: Sections 4.2-4.3.
  - Support `collapsed_str` and `sequential_str` modes in CLI and app.
  - Test: mode selector routes to the correct function.

## Release v0.7 - Corruption Rates, Sensitivity, and Validation Diagnostics

Release goal: support paper-defined class-conditional corruption rates, sensitivity
sweeps, and indirect validation diagnostics when ground truth is unavailable.

Customer test: change corruption assumptions in the app and see corrected fraud
rate, CI, and stability diagnostics update.

### Tasks

- [ ] FLGR-042 - Add corruption-rate table schema.
  - Paper source: Definition 18.
  - Define columns for segment keys, `eps10`, `eps01`, source, effective date,
    and confidence level.
  - Test: invalid rows where `eps10 + eps01 >= 1` are rejected.

- [ ] FLGR-043 - Implement segment-level corruption matching.
  - Paper source: Section 5.1.
  - Match each transaction to the most specific available corruption-rate row,
    falling back to global rates only when no segment row exists.
  - Test: issuer/MCC-specific rate overrides global rate.

- [ ] FLGR-044 - Replace global-only corruption correction.
  - Paper source: Equation 21.
  - Use row-level `eps10_hat` and `eps01_hat` in `y_tilde_corr`.
  - Test: two rows with identical labels but different segment rates produce
    different corrected outcomes.

- [ ] FLGR-045 - Add corruption sensitivity grid.
  - Paper source: Section 11.7 sensitivity sweep.
  - Vary `eps10` and `eps01` over user-configurable ranges while enforcing
    `eps10 + eps01 < 1`.
  - Test: output grid contains corrected fraud-rate estimates for every valid
    pair.

- [ ] FLGR-046 - Add ignorability sensitivity parameters.
  - Paper source: Section 11.2 and Appendix B.
  - Add configuration for `GammaA` and `GammaR` and compute the paper's displayed
    bias-bound summaries.
  - Test: `GammaA = GammaR = 1` returns zero additional bias bound.

- [ ] FLGR-047 - Add covariate balance diagnostics.
  - Paper source: Section 11.7.
  - Compute standardized mean differences before and after inverse-propensity
    weighting for numeric and one-hot categorical features.
  - Test: known balanced fixture returns near-zero weighted differences.

- [ ] FLGR-048 - Add propensity overlap plots and tables.
  - Paper source: Section 11.7.
  - Produce compact data for histograms of `e_hat`, `r_hat`, `p_hat`, and
    `q_hat_total`.
  - Test: histogram bin counts sum to row count.

- [ ] FLGR-049 - Add nuisance cross-validation diagnostics.
  - Paper source: Section 11.7.
  - Report held-out AUC/log loss/Brier for authorization, reporting, and maturity
    nuisance models where class variation exists.
  - Test: one-class folds return documented `None` metrics, not crashes.

- [ ] FLGR-050 - Add maturity-window stability diagnostic.
  - Paper source: Section 11.7.
  - Re-estimate `psi_hat` across multiple maturity windows and report drift.
  - Test: synthetic fixture with stable delays yields bounded variation.

## Release v0.8 - Conditional Delay and Optimal Training Delay

Release goal: maturity is modeled as `p0(x, i, Delta)` and the app recommends the
paper's optimal training delay for STR versus naive training.

Customer test: choose maturity-delay assumptions and see the estimated maturity
curve, `Delta*_STR`, and naive comparison.

### Tasks

- [ ] FLGR-051 - Make `Delta` first-class in schema and features.
  - Paper source: Section 3.1 and Definition 43.
  - Store `available_maturity_days` as canonical `Delta`.
  - Include `Delta` in maturity propensity features by default.
  - Test: maturity model receives `Delta` in its feature list.

- [ ] FLGR-052 - Estimate conditional maturity propensity.
  - Paper source: Section 3.6 and Section 10.3.
  - Fit `p_hat_maturity = P(tau <= Delta | A=1, R=1, H2)`.
  - Test: rows with longer `Delta` have nondecreasing average maturity in a
    monotone synthetic fixture.

- [ ] FLGR-053 - Estimate network maturity curve.
  - Paper source: Definition 43.
  - Compute `pbar(delta)` over a configured grid of delta days.
  - Test: `pbar(delta)` is nondecreasing after monotonic smoothing.

- [ ] FLGR-054 - Add exponential maturity fit.
  - Paper source: Section 10.6.4.
  - Fit `pbar(delta) = 1 - exp(-lambda * delta)` or document why the fit failed.
  - Test: generated exponential fixture recovers lambda within tolerance.

- [ ] FLGR-055 - Add Weibull maturity fit.
  - Paper source: Section 10.6.1.
  - Fit `pbar(delta) = pbar_infinity * (1 - exp(-(lambda * delta)^beta))`.
  - Test: generated Weibull fixture recovers monotone curve and finite
    parameters.

- [ ] FLGR-056 - Compute heterogeneity penalty inputs.
  - Paper source: Definition 45.
  - Compute `e_bar`, `r_bar`, `p_bar(Delta)`, `gamma`, and an estimated
    heterogeneity penalty `eta` from propensity variation.
  - Test: zero-variance propensities produce `eta` near 1.

- [ ] FLGR-057 - Compute STR optimal delay.
  - Paper source: Theorem 46, Equation 57.
  - Implement `Delta*_STR = max(0, 1/lambda * log(pi*(1-pi)*eta*lambda /
    (nu*n*e_bar*r_bar*gamma)))`.
  - Test: increasing `n` decreases `Delta*_STR` in a controlled fixture.

- [ ] FLGR-058 - Add naive delay comparison.
  - Paper source: Section 10.6.5 and Theorem 46 parts iii-iv.
  - Estimate or configure selection contrast `zeta` and show the naive waiting
    penalty separately from STR variance.
  - Test: when naive bias is zero, freshness gain is zero or documented.

- [ ] FLGR-059 - Add optimal-delay app section.
  - Paper source: Section 10.6.
  - Show maturity curve, STR delay recommendation, naive comparison, and required
    inputs/assumptions.
  - Use local stats and description-list components adapted from
    `application-ui-v4/react/data-display/stats/` and
    `application-ui-v4/react/data-display/description-lists/`.
  - Test: rendering helper accepts a precomputed delay summary.

## Release v0.9 - Flexible Nuisance Models and Calibration

Release goal: nuisance and downstream models are configurable, cross-validated,
and calibrated without changing estimator semantics.

Customer test: select logistic or gradient-boosted nuisance models and compare
diagnostics without changing the data upload.

### Tasks

- [ ] FLGR-060 - Add model registry.
  - Paper source: Section 10.2 and Section 10.3.
  - Implement named model builders for `logistic`, `hist_gradient_boosting`, and
    `random_forest` where dependencies already exist or are added explicitly.
  - Test: every registered model supports `fit` and probability or regression
    prediction as required.

- [ ] FLGR-061 - Add estimator model config object.
  - Paper source: Section 10.2 choice of function class `G`.
  - Allow separate model choices for `e`, `r`, `p`, `mu0`, `mu1`, `mu2`, and final
    pseudo-label regression.
  - Test: invalid model names fail validation before fitting.

- [ ] FLGR-062 - Add probability calibration option.
  - Paper source: Section 11.8 proper scoring rules.
  - Support calibrated nuisance probabilities using held-out folds only.
  - Test: calibration does not train on the held-out prediction fold.

- [ ] FLGR-063 - Add model-performance diagnostics.
  - Paper source: Section 11.7 cross-validation of nuisance models.
  - Persist per-fold metrics for each nuisance model and final downstream model.
  - Test: metrics artifact has one row per fold per nuisance model.

- [ ] FLGR-064 - Add model artifact serialization.
  - Paper source: Section 10.5 Phase 2.
  - Save fitted downstream pseudo-label model and model config for reproducible
    scoring.
  - Test: serialized model reloads and returns identical predictions on fixture.

- [ ] FLGR-065 - Add model registry UI controls.
  - Paper source: Section 10.2.
  - Expose model choices in an advanced app panel with safe defaults.
  - Use local tabs, select-menu, toggle, and drawer/modal components adapted from
    `application-ui-v4/react/navigation/tabs/`,
    `application-ui-v4/react/forms/select-menus/`,
    `application-ui-v4/react/forms/toggles/`, and
    `application-ui-v4/react/overlays/`.
  - Test: app helper maps UI config to validated model config.

## Release v1.0 - Offline-Online Product Package

Release goal: the implementation is usable as a product workflow: ingest data,
run STR, produce corrected labels, provide diagnostics, run a matured-holdout
backtest, and export a pilot report.

Customer test: upload a CSV, run the pipeline, inspect diagnostics, download
corrected labels and a pilot report, and use the corrected-label file in their own
modeling environment.

### Tasks

- [ ] FLGR-066 - Add run manifest format.
  - Paper source: Algorithm 1 inputs and outputs.
  - Define `run_manifest.json` with data hash, row counts, schema version,
    estimator mode, fold count, shrinkage mode, corruption config, model config,
    and generated artifact paths.
  - Test: manifest validates against a JSON schema or explicit validator.

- [ ] FLGR-067 - Add one-command offline reconstruction CLI.
  - Paper source: Section 10.5 Phase 1 and Algorithm 1.
  - Implement `python -m fulgor.run_pipeline --input ... --output ...`.
  - The command must validate data, run selected estimator, write corrected
    labels, summary, diagnostics, and manifest.
  - Test: CLI function runs on fixture and writes every required artifact.

- [ ] FLGR-068 - Add corrected-label export.
  - Paper source: Section 10.1, Equation 40.
  - Export a CSV with original transaction ID, fold ID, propensities,
    `str_pseudo_outcome`, `fulgor_pseudo_label`, positivity status, and
    diagnostics keys.
  - Test: export never includes synthetic-only `true_fraud` in pilot mode.

- [ ] FLGR-069 - Add pilot report generator.
  - Paper source: Sections 10.5 and 11.7.
  - Generate Markdown and JSON reports with: data validation, estimator config,
    fraud-rate estimate and CI, diagnostics, sensitivity sweep, backtest metrics,
    and limitations.
  - Test: report includes every required section for fixture runs.

- [ ] FLGR-070 - Add downloadable artifacts in browser app.
  - Paper source: Section 10.5 Phase 1 output and Phase 2 training data.
  - Add download buttons for corrected labels, diagnostics JSON, and pilot report.
  - Use local button and action-panel components adapted from
    `application-ui-v4/react/elements/buttons/` and
    `application-ui-v4/react/forms/action-panels/`.
  - Test: download payload builders return nonempty bytes with stable filenames.

- [ ] FLGR-071 - Add local run storage policy.
  - Paper source: Section 10.5 offline reconstruction.
  - Store customer-uploaded run artifacts under a configurable local `runs/`
    directory that is git-ignored.
  - Test: `.gitignore` excludes `runs/`; run outputs are not created under
    checked-in artifact directories unless explicitly using synthetic demo mode.

- [ ] FLGR-072 - Add hosted demo/pilot mode separation.
  - Paper source: Section 10.5 architecture.
  - Keep checked-in synthetic artifacts for public demo mode. Pilot uploads must
    run in session/local storage and must not modify checked-in demo artifacts.
  - Test: upload flow does not write to `artifacts/demo_data/`.

- [ ] FLGR-073 - Add run status and failure states.
  - Paper source: Section 10.5 operational architecture.
  - The app must show validation errors, estimation failures, one-class metric
    warnings, positivity failures, and successful run completion clearly.
  - Use local alert, empty-state, progress, and notification components adapted
    from `application-ui-v4/react/feedback/`,
    `application-ui-v4/react/navigation/progress-bars/`, and
    `application-ui-v4/react/overlays/notifications/`.
  - Test: fixture-induced failures render deterministic error summaries.

- [ ] FLGR-074 - Update README and customer instructions.
  - Paper source: Section 10.5.
  - Document local launch, hosted demo launch, pilot CSV upload, output files, and
    what customers should inspect after each run.
  - Test: no code test required; README must link to `docs/pilot-data-contract.md`.

## Release v1.1 - Paper Completeness Hardening

Release goal: close operational edge cases discussed in the paper after the main
offline-online product flow is usable.

Customer test: run edge-case fixtures for non-monotone labels, policy shifts,
and processor/PSP composite observation paths and inspect explicit warnings.

### Tasks

- [ ] FLGR-075 - Add non-monotone auxiliary-label support.
  - Paper source: Section 11.5.
  - Allow declined transactions with auxiliary labels to be marked as a separate
    observation channel, not silently folded into the monotone STR channel.
  - Test: auxiliary declined labels are excluded from monotone STR fitting but
    included in an auxiliary validation table.

- [ ] FLGR-076 - Add time-varying nuisance warning.
  - Paper source: Section 11.4.
  - Compute nuisance diagnostics by calendar period and flag abrupt changes in
    authorization, reporting, or maturity distributions.
  - Test: fixture with a policy shift is flagged.

- [ ] FLGR-077 - Add composite actor metadata.
  - Paper source: Section 11 discussion of merchant, PSP, processor, issuer
    composite propensities.
  - Extend schema to optionally include `merchant_id`, `psp_id`, `processor_id`,
    and `acquirer_id` and record which actors are observable.
  - Test: missing optional actor IDs do not block runs; present IDs appear in
    diagnostic grouping options.

- [ ] FLGR-078 - Add first-party/scam exclusion controls.
  - Paper source: Section 3.1 and Section 11.9.
  - Add config for excluding fraud types outside unauthorized third-party fraud
    when the input data contains `fraud_type`.
  - Test: first-party/scam rows are excluded or separately reported according to
    config.

- [ ] FLGR-079 - Add production readiness QA script.
  - Paper source: Section 10.5 and Section 11.7.
  - Add one script that runs unit tests, lint, synthetic demo artifact rebuild,
    fixture pilot pipeline, and report generation.
  - Test: script exits nonzero if any required artifact is missing.

- [ ] FLGR-080 - Freeze v1 paper-complete acceptance report.
  - Paper source: all sections mapped above.
  - Add `artifacts/paper_implementation_matrix.md` with every paper construct,
    implementation file, test file, and known limitation.
  - Test: no code test required; matrix must link every item in the coverage
    matrix to completed tasks.
