# LabelLift Product Architecture

Status: accepted for Release v0.1.

Paper source: Section 10.5 offline-online architecture.

## Decision

LabelLift will move from a Streamlit demo script to a production-shaped offline-online
product architecture:

- Core package: `labellift/` remains the Python estimator and artifact package.
- Pipeline: Python CLI and job entrypoints produce durable run artifacts.
- API: FastAPI in `services/api/labellift_api/` exposes health, version, and
  artifact-backed JSON endpoints.
- Web UI: React + TypeScript + Vite in `apps/web/` renders the browser product.
- Styling and UI primitives: Tailwind CSS, Headless UI, and Heroicons.
- Hosting shape: the FastAPI service serves JSON endpoints and the built web UI
  as one self-service demo process.

Streamlit is not the product shell. `app.py` can remain temporarily as the legacy
synthetic demo until the modern UI reaches parity, but new product surface area
belongs behind the FastAPI service and React web app.

## Boundary

`labellift/` owns estimator logic, synthetic demo generation, artifact contracts,
and offline reconstruction code. It must not depend on the web UI or API layer.

The pipeline layer owns scheduled or manually triggered offline runs. Its output
is durable files: manifests, summaries, diagnostics, corrected labels, and reports.
Those artifacts are the handoff from offline reconstruction to online product
surfaces.

The API layer owns transport concerns. It reads compact checked-in demo artifacts
or caller-provided run artifacts and returns JSON suitable for the web UI. It must
not recompute the estimator on page load.

The web UI owns browser interaction, navigation, charts, tables, empty states, and
customer-testable review flows. It calls the API and does not read local CSV files
directly.

## Why The Estimator Stays Offline

Section 10.5 describes LabelLift as an offline label-reconstruction engine whose
outputs feed downstream fraud modeling and review workflows. The estimator uses
historical transactions, delayed reports, matured holdout windows, nuisance-model
fits, diagnostics, and corrected label outputs. Those are batch reconstruction
concerns, not interactive request-path work.

Keeping estimation offline gives the product clearer operational boundaries:

- Browser sessions remain fast and reproducible because they load prepared
  artifacts.
- The API can serve compact demo outputs without requiring full `data/*.csv`
  files.
- Later pilot runs can separate customer data ingestion and reconstruction from
  artifact review.
- Downstream fraud model training receives versioned outputs instead of ad hoc
  UI state.

## UI Reference Policy

`application-ui-v4/` is a reference-only component source. It may guide app shell,
stats, tables, forms, alerts, tabs, drawers, modals, and page composition.

Runtime code must not import from `application-ui-v4/`. Any selected example must
be copied, adapted for LabelLift-specific props and data, tested locally, and
committed under `apps/web/src/`.

## Demo And Privacy Guardrails

The v0.1 self-service demo uses deterministic synthetic artifacts only. Demo-only
routes, reset commands, replay drivers, and prototype bridges must stay gated to
local, demo, or staging mode.

No production customer data, production certificates, real provider secrets, or
live customer endpoints belong in this architecture. Customer-facing copy must
continue to distinguish synthetic demo results from pilot results on real
historical data.

## Consequences

Future Release v0.1 tasks should build toward this boundary:

- Artifact helpers define stable paths before API endpoints depend on them.
- Run summary contracts define stable JSON before React views consume them.
- FastAPI becomes the single web process that can serve both API responses and
  the built React app.
- Streamlit remains a temporary migration aid only and is removed from the
  product path after modern UI parity is verified.
