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
