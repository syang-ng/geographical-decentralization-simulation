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

- `explorer/server/simulation_worker.py`, `explorer/server/simulation-runtime.ts`, `explorer/server/index.ts`
  Added prebuilt exact overview bundles, immutable gzip/Brotli delivery for artifacts and overview sidecars, and runtime-side manifest priming so the client can ask for render-ready exact views instead of rebuilding them from raw traces.

- `explorer/server/simulation-runtime.ts`
  Added isolated canonical prewarm in a dedicated background worker so common exact presets fill the shared cache without occupying the live user queue.

- `explorer/src/pages/SimulationLabPage.tsx`, `explorer/src/lib/simulation-api.ts`
  The lab now reads those prebuilt overview bundles directly and reuses parsed artifact blocks by artifact hash in browser session storage, which reduces repeated parse work on revisit.

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

## Contradictions Noted During Accuracy Review

- Copy and prompt framing had drifted to older paper structure in a few places.
  Section labels and experiment references no longer matched the current paper draft cleanly.

- The SE3 summary had become wrong in the reader layer and simulation guidance.
  Local copy said the transient decentralizing dip belonged to MSP with misaligned sources, while the current paper ties the brief dip to SSP under the real-Ethereum validator start with poorly connected relay placement.

- The gamma card crossed the truth boundary.
  It used a synthetic `% centralization index` chart and invented values rather than paper-backed directional language.

- The Simulation Lab mixed two different defaults.
  Website interactive defaults used 1,000 slots and 0.0001 ETH migration cost, but the copy presented them as if they were the paper reference runs, whose frozen public dataset family is centered on 10,000 slots and 0.002 ETH migration cost.

- The frozen dashboard overpromised experimentation.
  Its copy implied live parameter variation, even though the MVP dashboard only switches among checked-in published datasets and viewer controls.

- The dashboard catalog contained at least one metadata contradiction.
  The SE2 external `cost_0.0` entry pointed at the zero-cost dataset path but reported `metadata.cost = 0.002`.

- These were copy, prompt, and catalog contradictions.
  They did not require changing the canonical exact simulation math.

## Service Hardening And Summary-First Exact Exports

- `explorer/server/simulation-runtime.ts`
  Added bounded queue capacity, per-client active-job limits, and retention-based pruning for completed / failed / cancelled jobs so the long-lived website runtime does not keep unbounded in-memory job state forever.

- `explorer/server/index.ts`
  Added local request throttles and input bounds around `/api/explore`, `/api/simulation-copilot`, and `/api/simulations`. This is an engineering control layer only; it does not alter exact simulation semantics.

- `explorer/server/exploration-store.ts`
  Tightened the local persistence layer with indexed exact-query lookups, bounded item retention, configurable data-file placement, and atomic-on-write replacement. This is still a local-file store rather than a true shared production database, but it reduces avoidable blocking and lookup cost.

- `models.py`, `simulation.py`, `explorer/server/simulation_worker.py`
  Added a summary-first exact export path for the website worker. The explorer worker now asks the canonical simulator for exact summary artifacts without emitting the heaviest raw per-validator trace files by default, while the original CLI/research path still keeps its full raw export behavior unless explicitly changed.

- `dashboard/viewer.html`
  Removed forced no-cache fetches for static assets and throttled redraws to the browser animation frame so the frozen dashboard baseline feels less heavy without changing the published data it renders.

- Validation:
  `npm run build`, `npm run smoke`, `python -m py_compile simulation.py models.py explorer/server/simulation_worker.py`, and `python tests/simulation_benchmark.py --repeat 1 --strict` all passed after the earlier runtime-hardening batch.

- Validation for the exact overview delivery batch:
  `npm --prefix explorer run build` and `python -m py_compile explorer/server/simulation_worker.py` passed after adding isolated prewarm, prebuilt overview sidecars, and immutable compressed delivery.

- Truth-first note:
  The summary-first worker path is a delivery optimization around the same exact engine. Fixed-seed benchmark hashes remained unchanged, so this batch did not move the canonical research outputs on the checked scenarios.
