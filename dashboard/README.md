# Public Dashboard

## Data Pre-processing

Preprocess data for the public dashboard.

```bash
cd ../
python3 preprocess_data.py -d data -o output/test -m SSP
```

After running the script, there will be a `preprocessed_data.json` file in the `output/test` folder. This file is used by the public dashboard.

## Research Dataset Sync

To make the local dashboard use the exact published researcher payloads, run:

```bash
node sync-research-datasets.mjs
```

This overwrites the local `dashboard/simulations` JSON files with the canonical data published at `https://geo-decentralization.github.io/`.

To verify that the local dashboard datasets still match the published researcher payloads, run:

```bash
node verify-research-baseline.mjs
```

## Test

Copy the preprocessed data to the folder `dashboard/simulations/test/data.json`. Then, run the following command to view the dashboard.

```bash
cd dashboard
python3 -m http.server 12345
```
