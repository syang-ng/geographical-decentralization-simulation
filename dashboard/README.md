# Public Dashboard

## MVP Baseline

The local dashboard is a frozen copy of the published researcher demo.

- `dashboard/simulations/` contains the checked-in published `data.json` payloads.
- `dashboard/assets/research-catalog.js` is the frozen manifest that maps the public experiment labels to the local dataset paths.
- The dashboard does not depend on a live upstream website at runtime.

The published site is a static GitHub Pages app, so there is no live API or live simulation backend to stay connected to.

## Run Locally

```bash
cd dashboard
python3 -m http.server 12345
```

Then open `http://localhost:12345`.

## Regenerate Local Data

If you preprocess a new local simulation result, the preprocessor can now emit the same top-level metadata shape used by the published demo.

```bash
python3 preprocess_data.py \
  -d data \
  -o output/test \
  -m SSP \
  --cost 0.001 \
  --delta 12000 \
  --cutoff 4000 \
  --gamma 0.6667 \
  --description "Local experiment summary"
```

Notes:

- `v` is inferred automatically from the first slot unless `--validator-count` is provided.
- `sources` default to published-style labels such as `asia-east1-supplier` or `asia-east1-signal`.
- Use `--source-label-style raw` if you need the original source IDs instead.

## Maintenance Only

These scripts are not required for the MVP to run. They are only for refreshing or auditing the frozen baseline against the published site.

```bash
node sync-research-datasets.mjs
node verify-research-baseline.mjs
```
