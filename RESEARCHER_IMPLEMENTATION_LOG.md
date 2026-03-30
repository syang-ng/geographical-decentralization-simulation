# Researcher Implementation Log

This log is intentionally short and skim-first. It tracks the core engineering story: what changed, why it mattered, and where the truth boundary was kept.

## Core Story

- Exactness stayed the constraint.
  Speed work was only accepted when it preserved fixed-seed outputs and did not relax the research model into an approximation path.

- The simulation engine got faster in exact mode.
  The main gains came from caching, precomputation, redundant-work removal, lighter export plumbing, and avoiding repeated process startup.

- The explorer became a delivery layer around the same engine.
  Warm Python workers, async jobs, SSE updates, artifact manifests, and worker-side parsing improved responsiveness without changing canonical simulation math.

- The dashboard baseline moved toward the published researcher demo.
  The local website now uses the published datasets, public scenario names, and published-style metadata instead of older internal-only packaging.

- The MVP website is now frozen, not live.
  The local dashboard should be understood as a checked-in copy of the published static demo plus a small local compatibility layer. It does not depend on a live upstream feed.

## Exact-Preserving Engine Changes

- `simulation.py`
  Added deterministic seed handling, exact-result caching under `.simulation_cache`, lighter export plumbing, and optional `--full-history` / `--verbose` controls.

- `models.py`, `validator_agent.py`, `distribution.py`, `source_agent.py`
  Reduced repeated exact work by precomputing latency-related inputs, memoizing repeated threshold inputs, avoiding unnecessary agent stepping, and removing repeated threshold-path overhead.

- `simulation.py`, `explorer/server/simulation_worker.py`
  Added smaller derived exact artifacts such as `proposal_time_avg.json`, `attestation_sum.json`, and `top_regions_final.json` so the frontend can render faster without reparsing the heaviest raw traces.

- `tests/simulation_benchmark.py`
  Extended the benchmark/regression harness so exact runs can be compared across repo states with fixed seeds and file-hash checks.

## Explorer Runtime And UI Story

- `explorer/server/simulation-runtime.ts`, `explorer/server/simulation_worker.py`, `explorer/server/index.ts`
  Moved exact runs behind warm long-lived workers, async job execution, job deduplication, cancellation, stale-job dropping, and manifest/artifact endpoints.

- `explorer/src/pages/SimulationLabPage.tsx`, `explorer/src/workers/simulationArtifactWorker.ts`
  The browser now loads manifest-first, parses heavy artifacts off the main thread, and renders exact outputs more responsively.

- `explorer/server/catalog.ts`, `explorer/server/study-context.ts`, `explorer/server/index.ts`
  Tightened the research-assistant flow with smaller atomic tools, stricter provenance, and bounded simulation-copilot schemas so the UI can help users explore without fabricating data or mutating research truth.

## Dashboard Baseline Story

- `dashboard/simulations/`
  The local dashboard now uses the checked-in published researcher `data.json` payloads.

- `dashboard/assets/research-catalog.js`, `dashboard/index.html`, `dashboard/viewer.html`
  The launcher/viewer were aligned to the published public labels, intro copy, and published dataset packaging.

- `dashboard/README.md`, `dashboard/BASELINE.md`
  The MVP dashboard is now documented as a frozen local baseline of the published static website, not a runtime sync against a live upstream source.

- `preprocess_data.py`
  New locally generated preprocessed outputs can now emit published-style top-level metadata (`v`, `cost`, `delta`, `cutoff`, `gamma`, `description`) and published-style source labels by default.

## Validation Story

- Fixed-seed exactness remained the acceptance bar for engine work.

- The benchmark suite and hash comparisons were used to confirm that exact-preserving runtime changes did not alter canonical outputs.

- The published dashboard baseline was verified separately from the engine.
  The local dashboard dataset copies were checked against the published website payloads and matched across all 31 published datasets.

## Guard Rails That Matter

- No hidden approximation mode was introduced into the canonical exact path.

- The simulation copilot can suggest supported runs and views, but it cannot invent new metrics or fabricate chart values.

- The dashboard baseline is frozen locally so the MVP does not drift with an unseen upstream website change.

- Maintenance scripts for dashboard sync/verification still exist, but they are outside the normal MVP runtime path.

## Current Bottom Line

- Engine-side performance work: yes, materially improved.
- Research truth boundary: kept intact for the exact path.
- Dashboard parity: exact on published dataset payloads, close but not byte-for-byte identical on frontend source.
- MVP status: uses frozen published baseline data locally; no live website dependency at runtime.
