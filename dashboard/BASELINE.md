# Frozen Research Baseline

This dashboard should be understood as:

1. the published researcher demo, frozen locally for the MVP
2. a thin local compatibility layer that maps the public experiment labels to the repo's checked-in dataset files

What is frozen locally:

- launcher labels and intro copy from the published site
- viewer behavior aligned to the published site
- published `data.json` payloads checked into `dashboard/simulations/`

What is not live:

- no API
- no live simulation reruns
- no external data feed behind the page

If the team later wants a strict refresh from upstream, use the maintenance scripts in `dashboard/README.md`. They are intentionally outside the normal MVP flow.
