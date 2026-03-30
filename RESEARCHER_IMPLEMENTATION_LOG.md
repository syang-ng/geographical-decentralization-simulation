# Researcher Implementation Log

Date: 2026-03-28

This log records engineering changes made to improve exact-mode runtime and playground responsiveness without changing the simulation outputs for a fixed seed and configuration.

## Changes

- `explorer/server/index.ts`, `explorer/server/simulation-runtime.ts`, `explorer/server/simulation_worker.py`
  Added a warm exact-mode simulation service for the explorer. The Node API now keeps a long-lived Python worker process alive, queues simulation jobs asynchronously, deduplicates in-flight identical configs by hash, and exposes manifest/artifact endpoints instead of launching the CLI per browser request.

- `explorer/server/simulation-runtime.ts`
  The explorer backend now uses a small warm worker pool instead of a single worker process. Jobs are dispatched to idle exact workers concurrently, and each worker owns at most one exact simulation at a time. This improves multi-user playground responsiveness without changing any simulation result.

- `explorer/server/index.ts`, `explorer/src/pages/SimulationLabPage.tsx`
  Replaced status polling with SSE. The browser now listens to per-job exact status events instead of repeatedly polling the job endpoint. This reduces request volume and makes the UI feel more responsive under repeated LLM-driven experimentation.

- `explorer/server/index.ts`, `explorer/server/simulation-runtime.ts`, `explorer/src/pages/SimulationLabPage.tsx`
  Added explicit cancellation plus queued-job stale dropping. Superseded queued jobs from the same browser client are cancelled before they consume CPU, and queued jobs also age out after a timeout. Active exact runs can be cancelled explicitly through the API/UI, but this only stops the run; it does not change the semantics of completed runs.

- `explorer/server/index.ts`
  Artifact responses are now gzip-aware. If the browser accepts gzip and a precompressed artifact exists, the server returns the `.gz` payload directly with `Content-Encoding: gzip`.

- `explorer/server/simulation_worker.py`
  The warm worker reuses the canonical upstream simulation modules directly in-process. It maps the explorer controls onto the paper templates under `params/`, keeps the CSV inputs and parsed latency model in memory across runs, writes exact outputs into the shared `.simulation_cache`, and builds an explorer-specific manifest file so the frontend can render summaries before loading heavy artifacts.

- `explorer/server/simulation_worker.py`, `simulation.py`, `explorer/src/workers/simulationArtifactWorker.ts`
  Added exact derived summary artifacts for the frontend: `proposal_time_avg.json`, `attestation_sum.json`, and `top_regions_final.json`. These are generated from the existing raw slot traces and leave the original raw artifacts untouched. The UI now renders these smaller derived files instead of parsing the large raw nested arrays for those views.

- `explorer/src/pages/SimulationLabPage.tsx`, `explorer/src/lib/simulation-api.ts`, `explorer/src/main.tsx`
  The simulation lab is now backed by the exact async API. The frontend submits jobs, listens for SSE status updates, loads only the manifest first, and then requests artifacts on demand. React Query is used so exact runs, cache hits, and manifest/artifact state stay explicit in the UI.

- `explorer/src/workers/simulationArtifactWorker.ts`
  Added a browser Web Worker for artifact parsing. Heavy JSON parsing and block construction for timeseries and region-counter outputs now happen off the main thread, which keeps the page responsive while preserving the exact simulation artifacts unchanged.

- `explorer/src/App.tsx`
  The deep-dive and simulation lab tabs are now lazy-loaded code-split chunks instead of being bundled into the initial explorer payload. This reduces initial JavaScript cost for users who open the findings/history views only.

- `explorer/server/simulation-runtime.ts`, `explorer/server/simulation_worker.py`
  Canonicalized submitted simulation configs more aggressively before hashing/caching by normalizing numeric precision and exact integer fields on both the Node and Python sides. This improves exact cache reuse for equivalent requests that differ only in float formatting or object-key order.

- `models.py`, `validator_agent.py`, `distribution.py`
  Added more exact-preserving precomputation in the proposer threshold path. Per-slot attester-region latency vectors are now prepared once on the model, static relay/signal latency maps are precomputed by region, and the threshold solver now operates on prepared distribution inputs instead of rebuilding those NumPy arrays at every solve. These are cache/vectorization changes only; the underlying threshold math and target probability are unchanged.

- `simulation.py`
  Extended the lightweight export path with exact derived aggregates used directly by the website. This reduces frontend parsing overhead without changing the raw JSON/CSV outputs or the slot histories they came from.

- `dashboard/viewer.html`, `visualization.py`
  Country-border GeoJSON is now loaded from the checked-in local file instead of fetching GitHub at runtime. This improves reliability and first-load speed for the website without affecting simulation math or research outputs.

- `source_agent.py`
  Source offers are now queried lazily from `get_mev_offer_at_time(...)` instead of being stepped every 100 ms. `get_mev_offer()` now derives the current offer from model time, and `SourceAgent.step()` is a no-op.

- `models.py`
  The main loop now steps validators only, in a stable validator creation order, so relay/signal agents are no longer scheduled every tick. The model also keeps lightweight slot summaries (`slot_model_history`, `slot_validator_history`, `slot_proposer_history`) that preserve the JSON/CSV export contract without requiring the default Mesa `DataCollector` path. Full collector history is still available behind `--full-history`.

- `validator_agent.py`
  Attesters now exit early before any latency work when they have already attested or when the attestation cutoff has not been reached yet. Once the cutoff passes and no valid proposal exists, they finalize immediately without extra latency sampling.

- `validator_agent.py`
  Minimal-needed-time inputs are cached per slot and candidate region. This avoids rebuilding and resorting the same proposer-to-attester latency vectors during repeated migration evaluation in the same slot.

- `validator_agent.py`
  SSP validators now cache their relay-latency map by validator region instead of rebuilding it every slot when the validator has not moved. SSP optimal-latency proposers also stop querying relay offers on every 100 ms tick before the threshold-crossing condition is met.

- `validator_agent.py`
  The exact-mode threshold path now runs serially instead of through a Python `ThreadPoolExecutor`. Local profiling showed the thread-pool wait and shutdown overhead dominating the threshold solve on these workloads, so removing it improved exact-mode runtime without changing slot outcomes.

- `distribution.py`
  `find_min_threshold_fast()` now reuses precomputed lognormal parameter arrays for each latency/std tuple. The binary search still evaluates the same probability target and tolerance; it simply stops re-deriving the distribution parameters at every midpoint.

- `simulation.py`
  Output JSON/CSV files are now written from the lightweight slot summaries instead of building Pandas dataframes from `DataCollector` by default. The exported file names and shapes are preserved.

- `simulation.py`
  Added result caching under `.simulation_cache/<hash>`. The cache key includes the effective simulation drivers, relevant input-file hashes, and hashes of the core simulation source files so cached outputs are reused only when the run is truly identical.

- `simulation.py`
  Removed the unused `measure.py` import from the CLI entrypoint. That import was not used in this file and only added startup dependency overhead.

- `explorer/server/catalog.ts`, `explorer/server/index.ts`, `explorer/server/study-context.ts`
  Split the Claude tool layer into smaller atomic tools while keeping `render_blocks` as the final presentation step. The explorer can now explicitly search curated topic cards, retrieve curated cards by ID, search prior explorations, retrieve a prior exploration, suggest underexplored follow-up topics, and compose bounded exact-mode simulation configs without executing them. This changes the website's reasoning workflow and provenance control, but it does not alter the upstream simulation math or the `/api/explore` response shape consumed by the frontend.

## Flags

- `--full-history` / `--no-full-history`
  Controls whether Mesa `DataCollector` history is retained alongside the lightweight export path.

- `--cache-results` / `--no-cache-results`
  Controls reuse of cached output artifacts for identical runs.

- `--verbose` / `--no-verbose`
  Controls slot-level and migration debug logging without affecting output artifacts.

## Benchmarking

- `tests/simulation_benchmark.py`
  Added comparison support for benchmarking another repo root or local snapshot against the current tree. The harness now records both subprocess wall time and the simulation's own reported runtime so import/startup overhead can be separated from simulation-core performance.

- `tests/simulation_benchmark.py`
  Optional CLI flags are detected per target repo before invocation so older comparison snapshots can still be benchmarked even if they do not support every newer simulation flag.

## Explorer Hardening And Presentation

- `explorer/scripts/smoke-explorer.ts`
  Added a local explorer smoke harness that seeds temporary exploration history, boots the API server, checks `/api/health`, verifies curated-first Findings routing, verifies history reuse for a non-curated query, and checks exploration search. The launcher path was made Windows-safe by invoking the local `tsx` CLI through `node` rather than spawning `tsx.cmd` directly.

- `explorer/server/index.ts`
  The SPA fallback route now uses a regex catch-all for non-API paths instead of `app.get('*')`, which was not compatible with the installed Express 5 path matcher in this workspace. This is a server-startup compatibility fix only.

- `explorer/src/components/explore/BlockCanvas.tsx`, `explorer/src/components/blocks/ChartBlock.tsx`, `explorer/src/components/blocks/TimeSeriesBlock.tsx`, `explorer/src/components/blocks/MapBlock.tsx`
  Refined explorer animations and chart/map presentation without changing the block schemas, underlying values, or simulation outputs. The updates are intentionally presentation-only: stronger card hierarchy, slightly more deliberate reveal motion, clearer legends/labels, safer empty-state handling, and richer but still exact chart/map rendering from the same block data.

- `explorer/src/components/explore/QueryHistory.tsx`, `explorer/src/pages/FindingsPage.tsx`, `explorer/src/pages/DeepDivePage.tsx`
  Added page-level presentation polish for the Findings and Deep Dive flows: clearer framing of curated/history/generated provenance, more readable session-history recall, stronger section/card hierarchy, and more intentional paper-walkthrough affordances. This pass does not change query routing, block contents, or simulation-derived values; it only changes how the existing information is presented.

- `explorer/src/data/paper-sections.ts`, `explorer/src/pages/PaperReaderPage.tsx`, `explorer/src/pages/DeepDivePage.tsx`, `explorer/src/App.tsx`, `explorer/src/components/layout/TabNav.tsx`
  Added a dedicated editorial paper-reading route that reuses the same canonical section/block content as the Deep Dive page. The new page introduces a long-form reading layout, sticky table of contents, abstract/claim framing, section-by-section figure placement, active-section tracking, and copyable deep links, but it intentionally remains a presentation layer over the same paper facts and block data rather than a second source of research truth.

- `explorer/src/pages/PaperReaderPage.tsx`, `explorer/src/App.tsx`
  Extended the reader experience with a toggleable focus mode, better initial hash scrolling, live section progress, previous/next section affordances, and richer tab loading states. These changes are UX-only and do not alter any canonical paper content, block data, or simulation-backed outputs.

- `explorer/server/simulation-runtime.ts`, `explorer/package.json`, `explorer/.env.example`, `Dockerfile`, `.dockerignore`, `DEPLOYMENT_CHECKLIST.md`
  Hardened the production simulation path and deployment story. The runtime now defaults to `python3` on Linux, pre-warms worker slots, reports required runtime assets and worker errors in `/api/health`, and fails queued jobs with an explicit error if no simulation workers can be started instead of leaving them stuck indefinitely. A root Docker-based Railway deploy path was added so Node, Python, the simulation dependencies, and the required repo assets are packaged together. This is deployment/runtime plumbing only; it does not change the exact simulation logic or expected outputs.

## Notes

- These changes are intended to be exact-preserving for the existing simulation logic in non-fast mode.
- The explorer service changes are intentionally architectural rather than mathematical: they reduce process startup, duplicate work, and client-side parsing overhead while keeping the exact simulation codepath canonical.
- The deeper Python changes in this batch were limited to memoization, precomputation, and derived-output generation. More invasive model changes such as event-driven simulation or approximation modes were intentionally not introduced here.
- The dashboard/world-map note is separate from the simulation math: the viewer now uses the static checked-in GeoJSON for rendering country borders only. It is not live market data or live simulation data.

## Truth-First Summary

- The guiding constraint for this work was truth preservation over raw speed. Performance changes were only accepted when they could be defended as exact-preserving engineering changes rather than approximations.
- In practice this meant: no `--fast` defaulting, no timestep coarsening, no relaxed threshold math, no probabilistic shortcuts, and no architectural rewrites that would make the model harder to trust without a stronger regression story.
- The Python-side optimizations in this batch were limited to caching, precomputation, redundant-work removal, lighter export plumbing, and derived frontend artifacts computed from the same raw outputs.
- The explorer/runtime additions should be understood as a delivery layer around the same exact simulation engine, not as a replacement research model. They improve interactivity, concurrency, and web responsiveness, but they are not intended to redefine the canonical paper path.
- For highest-confidence research runs, prefer the direct Python CLI in exact mode. Use `--no-cache-results` if a fully fresh execution is desired, and `--full-history` if the legacy Mesa collector history is needed for inspection.
- The repeatable benchmark/regression harness was added specifically so performance work can be checked against fixed-seed outputs. The expectation is that any future optimization should continue to prove that published artifacts are unchanged before being trusted.
- A later website-focused pass prioritized product correctness before further caching work. The Findings tab now routes questions through curated topic cards first, then strong prior-history matches, and only falls through to a fresh model call when neither of those fits.
- Query/session provenance is now explicit in the UI: curated findings, reused public-history answers, fresh Claude generations, and exact simulation results are labeled differently so users can tell what is canonical versus generated.
- The local exploration store was upgraded with normalized-query metadata and stronger search semantics so the website can reuse prior answers in a principled way. It still uses local JSON persistence in this repo, but the structure now reflects the intended public-history behavior and can be swapped to a shared backing store later.
- The Simulation Lab UI was hardened for trust rather than speed: exact-run provenance, paper-scenario labels, and copyable config/run summaries make it easier to understand and reproduce what a given result actually represents.
- The later atomic-tool pass was also truth-preserving by design: it gives the model more explicit search/composition steps before presentation, but it does not change the paper facts, the exact simulation engine, or the frontend response contract. The goal was to make the website behave more like a careful research assistant, not to make it more improvisational.
## 2026-03-29: Upstream-Surface Alignment For Simulation Lab

- Restored the explorer baseline validator geography to the upstream `homogeneous` default instead of the website-only `uniform` alias that mapped to `homogeneous-gcp`.
- Removed the website-imposed slot floor so the API accepts `1..10000` slots, which is closer to the original CLI and allows the tiny regression scenarios again.
- Widened the public controls toward the upstream defaults: `1000` validators by default, numeric validator/slot inputs, and explicit distribution options for `homogeneous`, `homogeneous-gcp`, `heterogeneous`, and `random`.
- Kept backward compatibility for older `uniform` requests by normalizing them to `homogeneous-gcp` server-side rather than breaking existing clients.
- This change is about restoring fidelity to the original simulation surface, not speeding it up; the goal was to avoid website-specific engineering defaults from silently shifting research behavior.

## 2026-03-29: Simulation Copilot With Strict View Specs

- Added a dedicated simulation copilot path that keeps the exact engine and the UI registry separate.
- The new server route builds a constrained `render_simulation_view_spec` response rather than letting the model emit arbitrary blocks or UI code.
- The model can now:
  - suggest bounded exact configs,
  - choose supported metrics,
  - choose supported artifact views,
  - reorder charts and add narrative or caveat sections.
- The model cannot:
  - invent new metrics,
  - fabricate chart data,
  - mutate raw simulation semantics,
  - generate freeform frontend code.
- Artifact references are resolved server-side into real blocks from the exact manifest and artifact files, so the displayed charts still come from canonical outputs.
- The Simulation Lab frontend now includes a copilot panel that guides users toward supported questions and bounded runs, while leaving the existing manual artifact inspection flow intact.

## 2026-03-29: Truth Boundary And Derived Exact Views

- Added explicit truth-boundary handling for the simulation copilot so the UI distinguishes:
  - interpretation over exact outputs,
  - proposal-only responses,
  - guidance-only responses.
- Strengthened the copilot prompt so it must not present guidance, hypotheses, or proposed configurations as established truth.
- Added derived exact view capabilities that still stay inside the fixed visualization registry:
  - artifact bundles for core outcomes, timing/attestation, and geography,
  - summary charts built from exact manifest metrics,
  - continued server-side resolution of artifact references into real blocks.
- The intent is to give the website richer chart composition quickly, while keeping the source of truth anchored to exact manifests and artifact files rather than model-generated numbers.

## 2026-03-29: Simulation Copilot Schema Hardening

- Replaced the auto-generated JSON schema for `render_simulation_view_spec` with a strict hand-authored object schema in `explorer/server/catalog.ts`.
- Root cause: the previous `zod-to-json-schema` conversion for the simulation view spec collapsed to a top-level `$ref`/`OpenAiAnyType`, which Anthropic rejected at request time with `input_schema.type: Field required`.
- Added a startup assertion so every tool schema in the catalog must have an object root and cannot expose a top-level `$ref`. This turns similar regressions into local startup/build failures instead of live production surprises.
- Extended `explorer/scripts/smoke-explorer.ts` to run a canonical exact SSP simulation through the website API, check the saved benchmark-aligned summary values, and verify the simulation copilot returns a clean `503` when no Anthropic key is configured locally.
- Fixed two unrelated frontend compile blockers (`FindingsPage.tsx` string literal quoting and an unused `cn` import in `DeepDivePage.tsx`) so the explorer build reflects the real copilot/runtime state again.

## 2026-03-30: Research-Oriented Simulation Lab Visual Pass

- Refined the Simulation Lab and chart/map/stat blocks to feel more like a research instrument panel than a product demo.
- Added stronger stage hierarchy and subtle depth treatment around the simulation surfaces in `explorer/src/pages/SimulationLabPage.tsx` and `explorer/src/index.css`, while keeping the charts themselves flat and scale-faithful.
- The Simulation Lab now prefetches manifest-ready renderable artifacts in parallel and builds exact overview bundles from those exact outputs before the user drills into a single artifact. This is a UI responsiveness change over the same manifest/artifact files, not a change to simulation truth.
- Updated `BlockCanvas`, `ChartBlock`, `TimeSeriesBlock`, `MapBlock`, and `StatBlock` so motion is quieter, labels are clearer, and the panels read more like measured outputs than decorative cards.
- Re-ran the strict benchmark suite after the visual pass; hashes remained unchanged, confirming the simulation outputs were unaffected by these presentation changes.
