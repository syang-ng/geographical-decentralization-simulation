# Deployment Checklist

This repo has two different kinds of runtime assets:

1. The exact simulation engine inputs needed for reproducible results.
2. The dashboard / explorer presentation assets needed for the original look and feel.

If the goal is to match the original project as closely as possible, deploy all of the items below.

## Exact Simulation Runtime

Required Python/runtime pieces:

- `requirements.txt`
- `simulation.py`
- `models.py`
- `validator_agent.py`
- `distribution.py`
- `source_agent.py`
- `consensus.py`
- `params/*.yaml`
- `explorer/server/simulation_worker.py`

Required input data:

- `data/gcp_latency.csv`
- `data/gcp_regions.csv`
- `data/validators.csv`
- `data/world_countries.geo.json`

Required environment/runtime expectations:

- Python available on PATH (`python3` on Linux is preferred)
- Node available for the explorer server
- `SIMULATION_REPO_ROOT` pointing at the repository root when the API is started from a nested working directory

## Original Dashboard Feel

Required static assets:

- `dashboard/data/gcp_regions.csv`
- `dashboard/data/world_countries.geo.json`
- `dashboard/assets/*`

Required precomputed example outputs:

- `dashboard/simulations/**/*.json`

Without these bundled simulation JSON files, the viewer may load but it will not feel like the original repo because the preset demo scenarios will be incomplete.

## Explorer Production

Required explorer runtime pieces:

- `explorer/package.json`
- `explorer/server/index.ts`
- `explorer/server/simulation-runtime.ts`
- `explorer/server/simulation_worker.py`
- `explorer/dist` generated during build

Recommended production checks:

1. `GET /api/health` should report `readyWorkers > 0`
2. `POST /api/simulations` should move jobs out of `queued`
3. The health payload should show all required runtime paths as present
4. The `Paper`, `Findings`, and `Simulation Lab` tabs should all load without missing static assets

## Railway

This repo now includes a root `Dockerfile` so Railway can build a single image with:

- Node
- Python
- pip-installed simulation dependencies
- built explorer frontend

The container defaults the simulation worker to `python3` and sets `SIMULATION_REPO_ROOT=/app`.
