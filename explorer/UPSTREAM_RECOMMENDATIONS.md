# Upstream Recommendations

Things we encountered while building the explorer that we don't fully understand or would change. These are questions and suggestions for the researchers — not assumptions we act on.

**Rule: When in doubt, work around it. Don't "fix" what we don't understand. Log it here instead.**

---

## Questions (things we don't understand)

| # | What we found | Where | Our assumption (if any) | Question for researchers |
|---|--------------|-------|------------------------|------------------------|
| 1 | `measure.py` computes NNI, Moran's I, Geary's C but the paper uses Gini_g, HHI_g, CV_g, LC_g. Different metrics entirely. | `measure.py` vs paper Section 4 | We reference the paper's metrics in the explorer, not measure.py's | Are both sets computed somewhere? Which should users see? |
| 2 | `std_dev_ratio` defaults to 0.1 in `distribution.py` but appears to be called with 0.5 in some paths | `distribution.py`, `validator_agent.py` | We don't touch latency logic | Which value is canonical? |
| 3 | Migration cost in baseline YAMLs is 0.0001 but the paper analyzes 0.002 as the key threshold | `params/SSP-baseline.yaml` vs paper Section 5 | Use YAML values for simulation lab defaults | Is 0.0001 the intended default for user-facing experiments? |
| 4 | `homogeneous` distribution round-robins across macro-regions, grouping US into "northamerica" — but US has 9 regions vs Europe's 13. This means unequal per-region density. | `simulation.py` lines ~100-130 | Don't change distribution logic | Is this intentional weighting? |
| 5 | Visualization.py Dash app is standalone (port 8050). Unclear if it can be embedded or if it conflicts with other servers. | `visualization.py` | We link to it rather than embed | Is there a preferred way to integrate the 3D viewer? |

---

## Suggestions (things we'd change if it were ours)

| # | Suggestion | Why it would help | Difficulty |
|---|-----------|-------------------|-----------|
| 1 | `--json-output` flag on simulation.py — single JSON blob instead of 13 files | Simplifies API wrapping (no temp dir dance) | Low |
| 2 | `--progress` flag — emit `{"slot": N, "total": M}` to stderr periodically | Enables real progress bars in Simulation Lab | Low |
| 3 | `--config -` to accept YAML from stdin | Avoids temp file creation for programmatic configs | Low |
| 4 | Publish pre-computed results for all 6 experiments in a `results/` directory | We need actual numbers for the AI system prompt | None (just commit files) |
| 5 | Paper metrics (Gini_g, HHI_g, CV_g, LC_g) as functions in measure.py | Makes primary metrics programmatically accessible | Medium |

---

## Log

*Add entries as we encounter them during development.*

| Date | Context | What we found | Action taken |
|------|---------|--------------|-------------|
| 2026-03-28 | Initial code review | Questions #1-5 and suggestions #1-5 above | Logged, working around all of them |
