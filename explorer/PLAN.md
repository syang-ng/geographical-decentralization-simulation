# Geo Validator Lab — Interactive Research Explorer

## Vision

A new format for presenting empirical research: **the paper IS the interface**.

This project extends https://github.com/syang-ng/geographical-decentralization-simulation to present the paper "Geographical Centralization Resilience in Ethereum's Block-Building Paradigms" (Yang, Oz, Wu, Zhang — arXiv:2509.21475) as an AI-powered interactive experience.

We build ON TOP of the existing repo — the simulation code, data files, and existing Dash visualization are the researchers' work and should be treated as upstream. Our contribution is the interactive exploration layer.

No existing project combines all four of these:
1. **Formal research findings** presented as interactive visual blocks
2. **AI-powered exploration** — ask questions, get generated visualizations
3. **Live simulation** — construct experiments within bounded parameters
4. **Community contributions** — share explorations, build on each other's work

Stage 5 (agent-driven autonomous exploration) is scoped but deferred.

---

## Relationship to Existing Repo

### What already exists (upstream — don't override)

```
geographical-decentralization-simulation/
├── simulation.py          # Entry point — CLI with argparse
├── models.py              # Mesa models: SingleSourceParadigm, MultiSourceParadigm
├── validator_agent.py     # SSPValidator, MSPValidator — migration logic
├── source_agent.py        # RelayAgent, SignalAgent — MEV sources
├── consensus.py           # ConsensusSettings dataclass
├── constants.py           # Defaults + LinearMEVUtility
├── distribution.py        # GCP latency model, lognormal sampling, Poisson Binomial
├── measure.py             # Spatial stats: NNI, Moran's I, Geary's C, DBSCAN
├── visualization.py       # Dash/Plotly 3D globe viewer (standalone, port 8050)
├── params/                # YAML configs for each experiment
├── data/                  # gcp_regions.csv, gcp_latency.csv, validators.csv
├── figure/                # Pre-generated PDF figures
├── dashboard/             # Landing page (geo-decentralization.github.io)
├── analysis/              # Post-simulation analysis scripts
├── plot/                  # Figure generation scripts
└── evaluations/           # Fabric task runner for batch experiments
```

### What we add

```
# New directory within or alongside the repo
explorer/                          # Our interactive frontend
├── src/                           # React app
├── server/                        # API proxy + simulation job runner
└── ...

# OR as a monorepo:
geo-decentralization-simulation/   # Fork/clone of existing repo
├── [all existing files]           # Untouched
├── explorer/                      # Our addition
│   ├── src/
│   ├── server/
│   └── package.json
└── api/                           # Flask wrapper around simulation.py
    └── app.py                     # Thin API that accepts config, runs simulation
```

### Suggestions to propose to researchers (NOT to implement unilaterally)

These are changes to the upstream repo that would improve the interactive explorer.
Present as suggestions — the researchers know their code better.

1. **`simulation.py` could accept JSON config via stdin** in addition to YAML file path — would make API wrapping cleaner without a temp file dance. Currently it only takes `--config path/to/file.yaml`.

2. **A `--progress` flag that emits JSON progress to stderr** — would enable real-time progress reporting in the Simulation Lab tab. Currently there's no progress output during a run.

3. **The Dash visualization (`visualization.py`) could be made embeddable** — currently it's a standalone Dash app. An iframe embed or a static export mode would let us show the 3D globe viewer inline.

4. **Pre-computed results could be published as a data package** — the current repo doesn't include simulation outputs (only configs). Having canonical result sets for the 6 experiments would let us build the Findings tab without running simulations ourselves.

5. **`measure.py` spatial stats (NNI, Moran's I, etc.) aren't used in the paper** — the paper uses Gini_g, HHI_g, CV_g, LC_g instead. These could be added to `measure.py` or a new `metrics.py` for programmatic access.

6. **A `--json-output` flag on `simulation.py`** that writes all outputs to a single JSON blob (instead of 13 separate files) would simplify API responses.

---

## Deep Understanding: The Research

### Core Thesis

Ethereum's block-building architecture is NOT geographically neutral. Both local (MSP) and external (SSP/PBS) block-building paradigms induce location-dependent payoffs that push validators toward geographic centralization. The mechanisms are different, the sensitivities are opposite in some cases, and consensus parameters act as protocol-level levers.

### The Two Paradigms

**SSP (Single-Source Paradigm) = External block building = MEV-Boost/PBS**
- Proposer receives a complete block from a single relay/builder
- Block propagation path: Proposer → Relay → Attesters (TWO hops)
- MEV = single relay's offer at proposal time
- Centralizing force: co-locating with relay eliminates one latency hop entirely
- Validators cluster at relay locations (US-East, EU-West)

**MSP (Multi-Source Paradigm) = Local block building = Self-building**
- Proposer aggregates MEV signals from multiple distributed sources
- Block propagation path: Proposer → Attesters directly (ONE hop)
- MEV = SUM of all signals' offers (each diminished by proposer-to-signal latency)
- Centralizing force: distributed pull toward many signals, but regions close to MANY signals AND many attesters win
- Validators cluster along the Atlantic corridor (NA primary, EU secondary)

**Key mechanical difference in code:**
- SSP: `simulation_with_relays()` — evaluates all (region, relay) pairs, picks single best
- MSP: `simulation_with_signals()` — evaluates all regions, sums all signal offers per region
- SSP attestation: `proposed_time + proposer_to_relay_latency + relay_to_attester_latency <= cutoff`
- MSP attestation: `proposed_time + proposer_to_attester_latency <= cutoff`

### The Migration Decision (how validators move)

Called once per slot for the proposer:
1. If migrating/cooldown/HOME → skip
2. Evaluate ALL 40 GCP regions using `how_to_migrate()`
3. Sort by: highest MEV, prefer current region (tie-break), lowest latency
4. If best region == current region → stay
5. If migration_cost >= (best_mev - current_mev) → stay (too expensive)
6. Otherwise → migrate

The utility function is: `U(region) = MEV_at_region`. Migrate if `U(best) - U(current) > cost`.

For MSP: `MEV(region) = sum_over_signals(signal.mev_at(attestation_cutoff - min_needed_time(region) - latency_to_signal))`
For SSP: `MEV(region) = best_relay.mev_at(attestation_cutoff - min_needed_time(relay_region) - latency_to_relay)`

`min_needed_time(region)` = binary search for minimum broadcast time achieving 99% supermajority probability via Poisson Binomial distribution over attester latencies.

### The Paper's Metrics (NOT the same as measure.py)

The paper uses 4 custom geographic metrics computed per slot:
1. **Gini_g** — Stake inequality across regions (0 = even, →1 = concentrated)
2. **HHI_g** — Herfindahl-Hirschman Index (1/|R| = dispersed, 1 = monopoly)
3. **CV_g** — Coefficient of variation of payoffs across regions (higher = more disparity)
4. **LC_g** — Liveness coefficient: min regions whose failure breaks liveness (higher = more resilient, adapted Nakamoto coefficient)

`measure.py` computes different stats (NNI, Moran's I, Geary's C, DBSCAN) used only in the Dash visualization, NOT in the paper.

### All 7 Experiments and Their Results

#### Baseline (Uniform validators + Uniform info sources, 40 sources one per GCP region)
- Both paradigms: Gini_g and HHI_g increase, LC_g declines over time
- **MSP centralizes FASTER and more severely than SSP** in baseline
- CV_g consistently higher under MSP (larger reward disparities)
- Convergence locus: SSP → small shift to NA + Middle East. MSP → primary to NA, secondary SA/Africa→Europe

#### SE1: Information-Source Placement
- **Latency-aligned**: 3 sources in asia-northeast1, europe-west1, us-east4 (low-latency hubs)
- **Latency-misaligned**: 3 sources in africa-south1, australia-southeast1, southamerica-east1 (periphery)
- Both: asymmetric placement → FASTER and STRONGER centralization vs baseline
- **OPPOSITE paradigm sensitivities:**
  - MSP: latency-ALIGNED centralizes MORE (low-latency regions benefit both value and propagation)
  - SSP: latency-MISALIGNED centralizes MORE (poorly connected relays → large proposer-relay gap → high marginal benefit of co-locating)
- Exception: MSP + misaligned has LOWER CV_g than baseline (trade-off creates more balanced rewards)

#### SE2: Validator Distribution (Heterogeneous — Real Ethereum from Chainbound/Dune)
- Starting distribution already concentrated in US + Europe
- Metrics start elevated (high Gini/HHI, low LC)
- Both paradigms: rapid convergence to co-location equilibrium
- **No substantial difference between paradigms** when starting concentrated
- SSP shows stronger AMPLIFICATION of reward disparities relative to its baseline

#### SE3: Joint Heterogeneity (Heterogeneous validators + Asymmetric sources)
- Combines SE1 + SE2
- Results mirror both: asymmetric sources → stronger centralization, concentrated start → rapid convergence
- **Key deviation**: SSP + misaligned sources → transient IMPROVEMENT in decentralization (validators move AWAY from incumbent hubs toward relay region) before re-centralizing

#### SE4a: Attestation Threshold (γ) Variations — {1/3, 1/2, 2/3, 4/5}
- **OPPOSITE effects by paradigm** (the paper's most surprising finding):
  - SSP: Higher γ → STRONGER centralization. Tighter timing amplifies latency sensitivity. Reducing proposer-relay latency yields larger marginal MEV increase.
  - MSP: Higher γ → WEAKER centralization. Higher threshold forces proposers to balance attester proximity (quorum) vs signal proximity (value). These point in DIFFERENT geographic directions, so tightening disperses rather than concentrates.

#### SE4b: Shorter Slot Time (EIP-7782, Δ=6s, τ_cut=3s)
- Centralization trajectories (Gini, HHI, LC) remain largely UNCHANGED
- CV_g (reward variance) is HIGHER under 6s for both paradigms
- Same latency advantage becomes larger fraction of shortened timing window
- Implication: further slot time reductions may strengthen migration incentives

### Paper's Limitations (authors acknowledge)

1. **GCP-only latency data** — other providers may differ
2. **Deterministic linear MEV function** — real MEV is stochastic with tx arrivals, volatile, builder bidding dynamics
3. **Fungible info sources** — reality: suppliers differ substantially in value
4. **Full-information assumption** — proposers know all latencies perfectly
5. **Instantaneous constant-cost migration** — real migration has heterogeneous costs, time-varying pricing
6. **Abstracted fork-choice rules** — not modeled
7. **Scale** — 10K validators take >1 day (100K slots). Used 1K as compromise

### Key Geography Data

**40 GCP regions across 7 macro-regions:**
- Africa: 1 (Johannesburg)
- Asia: 10 (Taiwan, Hong Kong, Tokyo, Osaka, Seoul, Mumbai, Delhi, Singapore, Jakarta)
- Oceania: 2 (Sydney, Melbourne)
- Europe: 13 (Warsaw, Helsinki, Stockholm, Madrid, Belgium, London, Frankfurt, Netherlands, Zurich, Milan, Paris, Berlin, Turin)
- Middle East: 2 (Doha, Tel Aviv)
- South America: 2 (Sao Paulo, Santiago)
- North America: 10 (Montreal, Toronto, Iowa, SC, Virginia, Ohio, Dallas, Oregon, LA, Salt Lake City, Las Vegas)

**Latency characteristics:**
- North America has best average median latency to other regions (142.97ms)
- South America has worst (209.88ms)
- NA-Europe link is lowest latency intercontinental (the "Atlantic corridor")

### Simulation Technical Details

**Runtime characteristics:**
- 10,000 slots × 12s slot / 50ms granularity = 2.4M time steps
- Each step: all agents act. Each migration decision: evaluate 40 regions × binary search
- 1,000 validators, fast mode: estimated 5-15 minutes
- 200 validators, 500 slots, fast mode: estimated 30-60 seconds
- Hardware used in paper: Intel Xeon Platinum 8380 (80 cores), 128 GB RAM

**Config schema (YAML):**
```yaml
simulation_name: string
model: "SSP" | "MSP"
iterations: int                    # num_slots
seed: int
time_granularity_ms: int           # 50 (paper) or 100 (default)
validators: int
input_folder: string
output_folder: string
time_window: int                   # convergence detection window
cost: float                        # migration cost in ETH
cloud: float                       # 0.0-1.0, fraction that can migrate
noncompliant: float                # 0.0-1.0, fraction using any relay

consensus_settings:
  slot_duration_ms: int            # 12000 default
  attestation_time_ms: int         # 4000 default
  attestation_threshold: float     # 0.6667 default (2/3)
  timely_head_reward: float        # 0.0790425
  timely_source_reward: float      # 0.147825
  timely_target_reward: float      # 0.12852
  proposer_reward: float           # 0.2
  sync_committee_reward: float     # 0.2

relay_profiles:                    # For SSP
  - unique_id: string
    gcp_region: string
    lat: float
    lon: float
    utility_function:
      type: "linear_mev"
      base_mev: float              # SSP baseline: 0.40
      mev_increase: float          # SSP baseline: 0.04
      multiplier: float            # default 1.0
    subsidy: float
    threshold: float
    type: "CENSORING" | "NONCENSORING"

signal_profiles:                   # For MSP (same shape minus relay fields)
  - unique_id: string
    gcp_region: string
    lat: float
    lon: float
    utility_function:
      type: "linear_mev"
      base_mev: float              # MSP baseline: 0.01 per signal
      mev_increase: float          # MSP baseline: 0.001 per signal
      multiplier: float
    type: "NONCENSORING"
```

**Baseline MEV scaling:**
- SSP: 0.40 base + 0.04/sec per relay (single relay captured)
- MSP: 0.01 base + 0.001/sec per signal × 40 signals → ~0.40 aggregate
- Both designed to yield comparable total MEV

**Output files per run (13 files):**
| File | Content |
|------|---------|
| `avg_mev.json` | Cumulative average MEV per slot |
| `supermajority_success.json` | Success rate per slot (%) |
| `failed_block_proposals.json` | Cumulative failed proposals |
| `utility_increase.json` | Utility gain at each migration |
| `region_profits.csv` | Per-region profit estimates at migration eval |
| `mev_by_slot.json` | Per-validator MEV per slot (nested arrays) |
| `estimated_mev_by_slot.json` | Per-validator estimated profit per slot |
| `attest_by_slot.json` | Per-validator attestation rate per slot |
| `proposal_time_by_slot.json` | Per-validator proposal timing per slot |
| `proposer_strategy_and_mev.json` | Strategy used + MEV per slot |
| `region_counter_per_slot.json` | `{slot: [[region, count], ...]}` — validator distribution |
| `action_reasons.csv` | Migration decisions: reason, from_region, to_region |
| `relay_names.json` / `signal_names.json` | Info source IDs |

**Stopping condition:** Simulation stops when `migration_queue` (deque, size=time_window) is full and ALL entries are `False` — no migrations in the entire window.

---

## Architecture

### Core loop

```
User query → Claude API (tool_use) → Block[] → React renders
```

The LLM returns structured JSON via a `render_blocks` tool call specifying which visual blocks to display. The frontend has a fixed catalog of block components. The LLM composes from the catalog — it never generates code.

Inspired by [json-render](https://github.com/vercel-labs/json-render): catalog-as-contract pattern where one definition produces the LLM prompt, the JSON schema, and the component types. Key patterns borrowed:
- **Flat element map** — LLM outputs each block independently, no nesting
- **Catalog-as-contract** — one definition auto-generates prompt + schema + validator
- **Guardrailed types** — LLM can only use types that exist in the catalog

But hand-rolled — json-render's 5-package architecture is overkill for our ~9 block types.

### Stack

| Layer | Tech | Why |
|-------|------|-----|
| Frontend | React 18 + Vite + Tailwind v4 | Minimal, fast |
| LLM | Claude API via `@anthropic-ai/sdk` | Structured output via tool_use |
| Backend proxy | Express (~50 lines) | Holds API key, rate limiting |
| Simulation | Python (Mesa) — existing repo | Wrapped in Flask/FastAPI for job submission |
| Data | Study results baked into system prompt (~12k tokens) | No database for Phases 1-2. Community features need persistence in Phase 4 |

### Dependencies

**Current:** react, react-dom, framer-motion, @tanstack/react-query, lucide-react, clsx, tailwind-merge, tailwindcss, @tailwindcss/vite

**To add:** @anthropic-ai/sdk, zod

**Total new deps: 2**

---

## Site Structure: 4 Tabs

### Tab 1: Findings

The paper's key results as interactive blocks + AI search bar. **Three tiers of content, each progressively more expensive.**

#### Tier 1: Pre-rendered cards (zero API cost, instant)
Default page load shows curated research findings as static blocks — the "executive summary" of the paper. These are hand-crafted in `default-blocks.ts` and cover the core story:
- Core thesis, paradigm comparison, key metrics, geographic patterns, limitations

Each pre-rendered card is a self-contained topic. Clicking "Explore this" on a card expands it with deeper pre-rendered content (still no API call). Think of these as the paper's sections turned into visual cards.

**Pre-rendered topics (8 cards, one per major finding):**
1. "SSP vs MSP: Which centralizes more?" → ComparisonBlock + InsightBlock
2. "Where do validators end up?" → MapBlock + StatBlocks
3. "Does source placement matter?" → ChartBlock + InsightBlock (SE1)
4. "What if validators start concentrated?" → ChartBlock + InsightBlock (SE2)
5. "The attestation threshold surprise" → TableBlock + InsightBlock (SE4a)
6. "Shorter slot times (EIP-7782)" → StatBlocks + CaveatBlock (SE4b)
7. "Key metrics explained" → StatBlocks + InsightBlock (definitions)
8. "Limitations & what's next" → CaveatBlock + SourceBlock

#### Tier 2: Community answers (zero API cost, near-instant)
Before firing a Claude query, check if this question (or something close) has already been asked and answered. Show the user matching community results inline:

```
┌──────────────────────────────────────────────────┐
│ 🔍 "How does migration cost affect convergence?" │
│                                                   │
│  Similar questions already explored:              │
│  ┌─────────────────────────────────────────────┐ │
│  │ "What role does migration cost play?" — 12↑ │ │  ← click to view cached answer
│  │ "Migration cost threshold analysis" — 8↑    │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  [View these]  or  [Ask Claude anyway ⚡]         │  ← explicit opt-in to API call
└──────────────────────────────────────────────────┘
```

Matching uses simple keyword overlap + query embedding similarity (or just normalized string matching for MVP).

#### Tier 3: Fresh Claude query (API cost, 1-3s)
Only fires when:
1. User typed a question (not just clicked a pre-rendered card)
2. No matching community answer exists (or user chose "Ask Claude anyway")
3. Query isn't in the edge cache

When a fresh query completes, the response is **automatically added to the public history** (no manual "publish" step needed). This means every Claude call enriches the community pool and reduces future API calls.

**Example queries (for the search bar placeholder + chips):**
- "Compare local vs external block building under baseline conditions"
- "What happens with shorter slot times under EIP-7782?"
- "How does attestation threshold affect centralization differently for each paradigm?"
- "Why does external building centralize more with misaligned sources?"
- "Show the geographic convergence pattern under MSP"
- "What is the Liveness Coefficient and why does it matter?"
- "How does real-world validator distribution change the results?"
- "What are the paper's main limitations?"

**Data source:** Paper findings + experiment results structured as context in the system prompt. No live computation.

### Tab 2: Explore History (was "Community")

Public feed of every question asked + its block response. Functions as both a leaderboard and a deduplication layer.

**Core UX:**
```
┌──────────────────────────────────────────────────┐
│  Explore History                    Sort: Top ▼  │
│                                                   │
│  🏆 Most explored questions                      │
│  ┌─────────────────────────────────────────────┐ │
│  │ "SSP vs MSP under different thresholds"     │ │
│  │  42 views · 15 ↑ · verified ✓               │ │
│  │  [Preview: 3 blocks shown inline]           │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │ "Geographic convergence under MSP baseline" │ │
│  │  38 views · 12 ↑ · verified ✓               │ │
│  └─────────────────────────────────────────────┘ │
│  ...                                              │
│                                                   │
│  Sort: Top | Recent | Verified Only              │
│  Filter: [baseline] [SSP] [MSP] [EIP-7782] ...  │
└──────────────────────────────────────────────────┘
```

**How it works:**
- Every Claude query response is automatically saved here (no manual publish step)
- Upvotes surface the best explorations
- View count tracks popularity
- Researchers can mark responses as "verified" (independently confirmed accurate)
- Tag auto-extraction from query + blocks (paradigm, experiment, metric mentioned)
- Click any entry → expands to show the full block response inline
- "Ask a follow-up" button on any entry → pre-fills search bar with context

**Deduplication mechanism:**
- Before each Claude call, fuzzy-match against existing entries
- If match found: increment view count, show existing answer, offer "Ask anyway" escape hatch
- This means the community pool grows with every unique question, and the API cost per question amortizes toward zero over time

**Moderation:** Auto-publish (no gate). Researcher verification is additive (badge, not removal). Offensive content: basic keyword filter + report button.

**Persistence — full CRUD on all entities:**

```sql
-- Explorations (community pool) — full CRUD via agent tools + UI
explorations (
  id uuid PRIMARY KEY,
  query text NOT NULL,
  query_normalized text NOT NULL,        -- lowercase, no punctuation, no stop words
  summary text NOT NULL,                  -- one-sentence heading from render_blocks
  blocks_json jsonb NOT NULL,             -- Block[] from Claude response
  upvotes int DEFAULT 0,
  downvotes int DEFAULT 0,                -- NEW: downvotes for quality signal
  views int DEFAULT 0,
  verified boolean DEFAULT false,
  verifier_note text,                     -- NEW: note from researcher who verified
  flagged boolean DEFAULT false,          -- NEW: flagged for review
  flag_reason text,                       -- NEW: reason for flag
  tags text[] DEFAULT '{}',
  paradigm text,                          -- 'SSP', 'MSP', or 'both' (auto-extracted)
  experiment text,                        -- SE1, SE2, etc. (auto-extracted)
  context_appended text,                  -- NEW: additional context added later
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()    -- NEW: tracks last modification
)

-- Votes — separate table for idempotent voting
votes (
  id uuid PRIMARY KEY,
  exploration_id uuid REFERENCES explorations(id),
  voter_fingerprint text NOT NULL,        -- anonymous browser fingerprint
  direction int CHECK (direction IN (-1, 1)),
  created_at timestamptz DEFAULT now(),
  UNIQUE(exploration_id, voter_fingerprint)
)

-- Simulation jobs — full CRUD for lab tab
simulation_jobs (
  id uuid PRIMARY KEY,
  config_json jsonb NOT NULL,
  status text DEFAULT 'queued',           -- queued, running, completed, failed, cancelled
  progress float DEFAULT 0,              -- 0.0 to 1.0
  result_json jsonb,                      -- parsed output files
  interpretation_json jsonb,              -- Claude's block interpretation
  error_message text,
  created_at timestamptz DEFAULT now(),
  completed_at timestamptz
)
```

**CRUD matrix — every entity, every operation:**

| Entity | Create | Read | Update | Delete | Agent Tool |
|--------|--------|------|--------|--------|------------|
| Exploration | Auto (on Claude response) | `search_explorations` | `update_exploration` (tags, context) | `flag_exploration` (soft delete) | Yes — all 4 |
| Vote | UI click | Aggregated in exploration | Change vote direction | Remove vote | UI only |
| Simulation Job | `build_simulation_config` → submit | Poll status | Cancel (status→cancelled) | Auto-cleanup after 24h | Partial (create + read) |
| Topic Card | `default-blocks.ts` (static) | UI render | Edit prompt to change | Edit prompt to remove | Prompt-native |
| Block | `render_blocks` tool | UI render | N/A (immutable once rendered) | N/A (part of exploration) | Yes |

### Tab 3: Deep Dive

Researcher-curated extended content — the appendix made interactive.

**Content mapped to actual paper sections:**
- Section 3: System Model (geography, consensus, MEV, information sources)
- Section 4: Simulation Design (ABM setup, metrics definition)
- Section 5.1: Baseline results with convergence analysis
- Section 5.2: SE1-SE3 sensitivity evaluations
- Section 5.3: SE4 protocol parameter analysis
- Section 6: Discussion and mitigation directions
- Appendix A: Inter-regional latency heatmap
- Appendix B: Scale sensitivity (100/1K/10K validators)
- Appendix C: Migration cost CDF analysis
- Appendix D-G: Detailed convergence loci

**Format:** Expandable accordion sections. Each can include LLM-queryable sub-context for deeper exploration.

### Tab 4: Simulation Lab

Bounded experiment builder + async simulation + LLM interpretation.

**User flow:**
1. Natural language OR manual config
2. LLM translates to simulation config (if NL)
3. Config panel shows bounded parameters
4. Submit → async job → progress indicator
5. Results → LLM interprets into blocks
6. User can publish to Community tab

**Bounded parameters (from Burak's guidance + code analysis):**

| Parameter | UI Control | Range | Default | Maps to |
|-----------|-----------|-------|---------|---------|
| Paradigm | Toggle | SSP / MSP | SSP | `--model` |
| Validator distribution | Dropdown | homogeneous / heterogeneous / random | homogeneous | `--distribution` |
| Validator count | Slider | 50–200 (capped for runtime) | 100 | `--validators` |
| Simulation slots | Slider | 500–2000 (capped) | 1000 | `--slots` |
| Info source placement | Preset dropdown or custom picker | homogeneous / latency-aligned / latency-misaligned / custom | homogeneous | `--info-distribution` + YAML relay/signal profiles |
| MEV base per source | Slider | SSP: 0.1–0.8, MSP: 0.005–0.05 | SSP: 0.4, MSP: 0.01 | YAML `base_mev` |
| MEV slope per source | Slider | SSP: 0.01–0.10, MSP: 0.0005–0.005 | SSP: 0.04, MSP: 0.001 | YAML `mev_increase` |
| Migration cost | Slider | 0.0–0.005 ETH | 0.0001 | `--cost` |
| Attestation threshold (γ) | Dropdown | 1/3, 1/2, 2/3, 4/5 | 2/3 | `--gamma` |
| Slot time (Δ) | Dropdown | 6s, 8s, 12s | 12s | `--delta` |
| Cloud ratio | Slider | 0.5–1.0 | 1.0 | YAML `cloud` |
| Fast mode | Toggle | on/off | on (for web) | `--fast` |

**Fixed (not user-configurable):**
- GCP topology + latency matrix (data/gcp_latency.csv)
- Latency distribution model (lognormal)
- Consensus reward parameters (paper values)
- Timing strategy (optimal_latency)
- Convergence window (time_window)
- Seed (random per run, or user can set)

**Async execution:**
- Submit creates a job ID
- Backend writes temp YAML config, runs `python simulation.py --config temp.yaml --fast`
- Poll job status endpoint
- On completion: parse output JSONs, send to Claude for interpretation
- **Estimated runtimes (fast mode):**
  - 100 validators, 500 slots → ~20-40 seconds
  - 200 validators, 1000 slots → ~1-3 minutes
  - 200 validators, 2000 slots → ~3-6 minutes

---

## Block Catalog

9 block types. The LLM composes from these via tool_use.

```typescript
type Block =
  | StatBlock           // Big number with context
  | InsightBlock        // Explanatory text paragraph
  | ChartBlock          // Bar or line chart from data points
  | ComparisonBlock     // Side-by-side (SSP vs MSP, before vs after)
  | TableBlock          // Data table
  | CaveatBlock         // Limitation or methodological note
  | SourceBlock         // Paper section references
  | MapBlock            // GCP regions with validator density
  | TimeSeriesBlock     // Multi-line time series (for Gini/HHI/CV/LC over slots)

interface StatBlock {
  type: 'stat'
  value: string              // "< 1.0", "2/3", "~200"
  label: string              // "Attestation Threshold (γ)"
  sublabel?: string          // "supermajority requirement"
  delta?: string             // "+12% vs baseline"
  sentiment?: 'positive' | 'negative' | 'neutral'
}

interface InsightBlock {
  type: 'insight'
  title?: string
  text: string               // Supports **bold** and *italic*
  emphasis?: 'normal' | 'key-finding' | 'surprising'
}

interface ChartBlock {
  type: 'chart'
  title: string
  data: { label: string; value: number; category?: string }[]
  unit?: string              // "%", "ETH", "ms"
  chartType?: 'bar' | 'line'
}

interface ComparisonBlock {
  type: 'comparison'
  title: string
  left: { label: string; items: { key: string; value: string }[] }
  right: { label: string; items: { key: string; value: string }[] }
  verdict?: string           // "MSP centralizes faster in baseline conditions"
}

interface TableBlock {
  type: 'table'
  title: string
  headers: string[]
  rows: string[][]
  highlight?: number[]       // Row indices to highlight
}

interface CaveatBlock {
  type: 'caveat'
  text: string
}

interface SourceBlock {
  type: 'source'
  refs: { label: string; section?: string; url?: string }[]
}

interface MapBlock {
  type: 'map'
  title: string
  regions: {
    name: string             // GCP region ID e.g. "us-east4"
    lat: number
    lon: number
    value: number            // validator count or density
    label?: string           // display name e.g. "Virginia"
  }[]
  colorScale?: 'density' | 'change' | 'binary'
}

interface TimeSeriesBlock {
  type: 'timeseries'
  title: string
  series: {
    label: string            // "SSP Baseline", "MSP Baseline"
    data: { x: number; y: number }[]  // x = slot, y = metric value
    color?: string
  }[]
  xLabel?: string            // "Simulation Slot"
  yLabel?: string            // "Gini_g"
  annotations?: { x: number; label: string }[]  // e.g. "convergence point"
}
```

### Why these 9 types

| Block | Serves | Paper analog |
|-------|--------|-------------|
| StatBlock | Key numbers at a glance | None — papers bury numbers in text |
| InsightBlock | Mechanism explanations | Paper body text |
| ChartBlock | Bar comparisons | Figures 3-8 (single-metric snapshots) |
| ComparisonBlock | SSP vs MSP head-to-head | The paper's core structure |
| TableBlock | Parameter sensitivity matrices | Table 1, parameter sweeps |
| CaveatBlock | Limitations flagging | Section 7 (Limitations) |
| SourceBlock | Paper citations | Reference list |
| MapBlock | Geographic validator distribution | Figure 1, Appendix F/G |
| TimeSeriesBlock | Metric evolution over simulation time | Figures 3-8 (the 4-subplot panels) |

---

## LLM Integration

### Claude tool_use — Atomic Tool Primitives

Instead of one monolithic tool, the agent gets **7 atomic primitives**. Each does one thing. The system prompt composes them into workflows — the tools don't encode behavior.

```typescript
const tools = [
  // --- Block composition (primary workflow) ---
  {
    name: 'render_blocks',
    description: 'Compose visual blocks to answer the user\'s question about the geo-decentralization study',
    input_schema: {
      type: 'object',
      properties: {
        summary: { type: 'string', description: 'One-sentence answer shown as heading above the blocks' },
        blocks: { type: 'array', items: { /* Block union — generated from catalog */ } }
      },
      required: ['summary', 'blocks']
    }
  },

  // --- Community pool CRUD ---
  {
    name: 'search_explorations',
    description: 'Search the community exploration pool for existing answers. Returns matching explorations with scores.',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'Natural language search query' },
        limit: { type: 'number', description: 'Max results to return (default 5)' },
        sort_by: { enum: ['relevance', 'top', 'recent', 'verified'], description: 'Result ordering (default: relevance)' },
        filters: {
          type: 'object',
          properties: {
            paradigm: { enum: ['SSP', 'MSP', 'both'] },
            experiment: { type: 'string' },
            verified_only: { type: 'boolean' }
          }
        }
      },
      required: ['query']
    }
  },
  {
    name: 'update_exploration',
    description: 'Update an existing exploration — add tags, correct metadata, or append follow-up context.',
    input_schema: {
      type: 'object',
      properties: {
        exploration_id: { type: 'string' },
        add_tags: { type: 'array', items: { type: 'string' } },
        append_context: { type: 'string', description: 'Additional context to attach' }
      },
      required: ['exploration_id']
    }
  },
  {
    name: 'flag_exploration',
    description: 'Flag an exploration as inaccurate, outdated, or needing review.',
    input_schema: {
      type: 'object',
      properties: {
        exploration_id: { type: 'string' },
        reason: { enum: ['inaccurate', 'outdated', 'misleading', 'duplicate'] },
        details: { type: 'string' }
      },
      required: ['exploration_id', 'reason']
    }
  },
  {
    name: 'verify_exploration',
    description: 'Mark an exploration as researcher-verified (accurate and confirmed).',
    input_schema: {
      type: 'object',
      properties: {
        exploration_id: { type: 'string' },
        verifier_note: { type: 'string', description: 'Optional note from the verifier' }
      },
      required: ['exploration_id']
    }
  },

  // --- Simulation Lab (two primitives, not one workflow) ---
  {
    name: 'get_simulation_constraints',
    description: 'Fetch the valid parameter ranges for simulation configuration.',
    input_schema: {
      type: 'object',
      properties: {}  // No inputs — returns the full constraint set
    }
    // Returns: { paradigm: ['SSP','MSP'], validators: {min:50, max:200}, slots: {min:500, max:2000}, ... }
  },
  {
    name: 'submit_simulation_config',
    description: 'Validate a simulation config against constraints and enqueue the job.',
    input_schema: {
      type: 'object',
      properties: {
        config: {
          type: 'object',
          properties: {
            paradigm: { enum: ['SSP', 'MSP'] },
            validators: { type: 'number' },
            slots: { type: 'number' },
            migration_cost: { type: 'number' },
            attestation_threshold: { type: 'number' },
            source_placement: { enum: ['homogeneous', 'latency-aligned', 'latency-misaligned'] }
          }
        }
      },
      required: ['config']
    }
    // Returns: { job_id, queued_at, estimated_runtime_seconds }
  },

  // --- Pool / session state (data primitives for discovery) ---
  {
    name: 'get_pool_state',
    description: 'Fetch aggregate statistics about the community exploration pool.',
    input_schema: {
      type: 'object',
      properties: {
        filter_paradigm: { enum: ['SSP', 'MSP', 'both'] }
      }
    }
    // Returns: { total, by_experiment: {baseline:12, SE1:3, ...}, trending: [...], coverage_gaps: [...] }
  },
  {
    name: 'get_session_history',
    description: 'Fetch this user\'s query history and viewed blocks this session.',
    input_schema: {
      type: 'object',
      properties: {}  // No inputs — returns session state
    }
    // Returns: { queries: [{timestamp, query_text, blocks_viewed}], current_topic }
  }
]
```

**Why 9 tools (all primitives, zero workflows):**

Each tool answers "What capability does this provide?" — never "What business decision does this make?"

| Tool | Capability | Who decides *when*? |
|------|-----------|-------------------|
| `render_blocks` | Compose visual blocks | System prompt |
| `search_explorations` | Search the pool | System prompt |
| `update_exploration` | Mutate exploration metadata | System prompt |
| `flag_exploration` | Flag an exploration | System prompt |
| `verify_exploration` | Mark as verified | System prompt |
| `get_simulation_constraints` | Read parameter bounds | System prompt |
| `submit_simulation_config` | Validate + enqueue a job | System prompt |
| `get_pool_state` | Read pool aggregate stats | System prompt |
| `get_session_history` | Read session context | System prompt |

**The primitive test:** Can you change the behavior by editing the prompt alone?
- YES for all 9 tools. The prompt decides: when to search before rendering, what makes a good follow-up suggestion, how to map NL to simulation params, when to flag vs verify.
- The old `build_simulation_config` embedded NL→config translation in the tool. Now the prompt does the translation using `get_simulation_constraints` data, and `submit_simulation_config` just validates + enqueues.
- The old `suggest_explorations` embedded ranking/prioritization in the tool. Now the prompt composes suggestions from `get_pool_state` + `get_session_history` data.
```

### System prompt structure

The system prompt has **two parts**: a static research context (cached) and a dynamic session context (injected per request). This is the key to both prompt caching efficiency AND context injection.

#### Part 1: Static research context (cached — `cache_control: { type: 'ephemeral' }`)

```
You are the research explorer for "Geographical Centralization Resilience
in Ethereum's Block-Building Paradigms" (Yang, Oz, Wu, Zhang, 2025).

You help readers understand the paper by composing visual blocks.

## Tools Available
You have 9 tools. Each is a pure capability primitive — you decide when and how to use them:
- render_blocks: Compose visual blocks to answer questions
- search_explorations: Search existing community answers before generating new ones
- update_exploration: Add tags or context to existing explorations
- flag_exploration: Flag inaccurate or outdated explorations
- verify_exploration: Mark explorations as researcher-verified
- get_simulation_constraints: Fetch valid parameter ranges for simulation config
- submit_simulation_config: Validate a config and enqueue a simulation job
- get_pool_state: Fetch aggregate stats about the exploration pool (gaps, trends)
- get_session_history: Fetch this user's query history and viewed blocks

## Response Strategy (prompt-native — edit this to change behavior)
1. ALWAYS call search_explorations first to check if a similar question exists
2. If a good match exists (>80% relevance): reference it, don't regenerate
3. If no match: call render_blocks with a fresh composition
4. ALWAYS call get_pool_state + get_session_history after rendering, then compose 2-3 follow-up suggestions inline
5. If the user's query is about simulation: call get_simulation_constraints, translate NL to config, then submit_simulation_config

## Composition Guidelines (prompt-native — edit this to change block layout)
- Lead with 1-2 stat blocks for key numbers
- Follow with chart, comparison, or timeseries blocks for visual evidence
- Use comparison blocks for SSP vs MSP questions (ALWAYS)
- Use timeseries blocks for "how does X evolve over time" questions
- Use map blocks for geographic distribution questions
- Add insight blocks to explain mechanisms and reasoning
- End with source refs and caveats where appropriate
- Keep insight text concise — blocks do the heavy lifting
- Maximum 6 blocks per response (keeps generation fast and focused)

## Topic Card Selection (prompt-native — edit this to change default topics)
The 8 pre-rendered topic cards cover these areas. If the user's query substantially
overlaps with a topic card, prefer directing them to expand that card rather than
regenerating the same content:
1. SSP vs MSP comparison (baseline)
2. Geographic convergence patterns
3. Information source placement effects (SE1)
4. Heterogeneous validator distribution (SE2)
5. Attestation threshold surprise (SE4a)
6. Shorter slot times / EIP-7782 (SE4b)
7. Key metrics explained
8. Limitations and future work

## Tier Routing (prompt-native — edit this to change when Claude fires)
- If query matches a topic card: respond with "This is covered by topic card N" + follow-up suggestions
- If search_explorations finds a match: respond with the existing exploration
- If neither: generate fresh blocks via render_blocks

## Building Simulation Configs (prompt-native — edit this to change param mapping)
When a user asks to simulate something:
1. Call get_simulation_constraints to fetch valid ranges
2. Parse the user's natural language intent into parameters:
   - "concentrated validators" → distribution: heterogeneous
   - "test attestation threshold" → run with γ = 1/3, 1/2, 2/3, 4/5
   - "shorter slots" → slot_time: 6s
3. Call submit_simulation_config with the parsed config
4. If validation fails, explain which params are out of bounds

## Generating Follow-Up Suggestions (prompt-native — edit this to change discovery)
After rendering a response:
1. Call get_pool_state to see coverage gaps and trending queries
2. Call get_session_history to see what the user has already explored
3. Compose 2-3 follow-up questions as plain text suggestions in the response:
   - One from a coverage gap (e.g., "No one has explored SE3 yet")
   - One that continues the current thread (topical continuity)
   - One from trending queries (what others are asking)

## The Two Paradigms
[Detailed SSP vs MSP mechanical explanation — ~1k tokens]

## Baseline Results
[Gini_g, HHI_g, CV_g, LC_g trajectories for both paradigms — ~1k tokens]

## SE1: Information-Source Placement
[Results for aligned vs misaligned × SSP vs MSP — ~1k tokens]

## SE2: Validator Distribution
[Heterogeneous results — ~500 tokens]

## SE3: Joint Heterogeneity
[Combined results including transient decentralization finding — ~500 tokens]

## SE4a: Attestation Threshold
[Opposite-effects finding with mechanism explanation — ~1k tokens]

## SE4b: Shorter Slot Times (EIP-7782)
[Unchanged trajectories but higher CV_g — ~500 tokens]

## Key Conclusions
[7 numbered findings — ~500 tokens]

## Limitations
[7 acknowledged limitations — ~500 tokens]

## Geography
[40 GCP regions with coordinates, 7 macro-regions, latency characteristics — ~1k tokens]

## Simulation Parameters
[Full parameter space with defaults and ranges — ~500 tokens]

## Metrics Definitions
[Gini_g, HHI_g, CV_g, LC_g formulas and interpretation — ~500 tokens]
```

#### Part 2: Dynamic session context (injected per request — NOT cached)

```
## Session Context (dynamic — injected at request time)

### Pool State
- Total explorations in pool: {{pool_size}}
- Verified explorations: {{verified_count}}
- Top trending queries this week: {{trending_queries}}
- Coverage gaps (topics with no explorations): {{uncovered_topics}}

### User Session
- Queries this session: {{session_queries}}
- Blocks viewed: {{blocks_viewed}}
- Current tab: {{active_tab}}
- Previous query (if any): {{last_query}}

### Rate Limits
- Remaining queries this hour: {{remaining_quota}}
- Pool match threshold: {{match_threshold}} (currently 60% overlap)
```

**Why two parts:**
- Part 1 (~10k tokens) is stable across all requests → cached with `cache_control: ephemeral` → 90% input cost reduction, ~80% faster TTFT
- Part 2 (~200 tokens) changes per request → injected fresh → gives the agent awareness of session state, pool health, and user history
- The agent can make smarter decisions (e.g., "the pool already has 5 MSP questions but zero about EIP-7782 — suggest that gap") because it sees the dynamic context

**Estimated total prompt size: ~10.2k tokens** (10k cached + 200 dynamic). Well within context window.

### Performance: Pre-rendering + Caching + Speed

**The core principle: page load = zero API calls. Only user-initiated queries hit Claude.**

#### Layer 1: Static pre-rendered blocks (no API)
- Default Findings page: 9 hardcoded blocks in `default-blocks.ts`. Pure JSON → React render. Zero latency.
- Deep Dive tab: all content static. Expandable sections, no LLM.
- Community tab: served from persistence layer (Supabase), no LLM.
- **~95% of page views never touch Claude.**

#### Layer 2: Query response cache (edge)
- Hash each user query → cache key
- Store `query_hash → Block[]` JSON in edge KV (Vercel KV or Cloudflare KV)
- Same question from different users → instant cached response
- TTL: 7 days (paper findings don't change)
- The example chip queries ("Compare SSP vs MSP baseline") should be pre-warmed into cache at deploy time
- **Estimated cache hit rate: 40-60%** (many users ask similar questions)

#### Layer 3: Claude API call (when cache misses)
Speed levers, in order of impact:

| Lever | Effect | Trade-off |
|-------|--------|-----------|
| **Prompt caching** | System prompt (~10K tokens) cached after first call. Subsequent calls skip prompt processing. ~80% faster TTFT for the cached portion. Cost: cached input tokens at 90% discount | None — pure win. Must use same system prompt across calls |
| **Model selection** | Haiku: ~0.3s TTFT, ~100 tok/s. Sonnet: ~0.8s TTFT, ~80 tok/s. Opus: ~2s TTFT, ~40 tok/s | Haiku is 10-20x cheaper but may produce less nuanced block compositions. Sonnet is the sweet spot |
| **Streaming** | Start rendering blocks as they arrive in the tool_use response, don't wait for full completion | Structured tool_use output arrives as one chunk at the end — streaming helps for the text portion but blocks render all-at-once |
| **Response size cap** | Instruct LLM: "Use 3-6 blocks maximum." Fewer blocks = shorter generation time | Limits expressiveness for complex queries |
| **Max tokens** | Set `max_tokens: 2048` — enough for 6-8 blocks. Prevents runaway responses | May truncate complex responses |

**Recommended defaults:**
- Model: `claude-sonnet-4-6` (best speed/quality balance)
- System prompt: cached (include `cache_control: { type: 'ephemeral' }` on the system message)
- Max tokens: 2048
- Temperature: 0 (deterministic — helps caching, consistent block compositions)

#### Layer 4: Pre-warming common queries
At deploy time (or via cron), run the 8 example chip queries through Claude and cache the results:
1. "Compare local vs external block building under baseline conditions"
2. "What happens with shorter slot times under EIP-7782?"
3. "How does attestation threshold affect centralization differently for each paradigm?"
4. "Why does external building centralize more with misaligned sources?"
5. "What are the paper's main limitations?"
6. "Show the geographic convergence pattern under MSP"
7. "What is the Liveness Coefficient and why does it matter?"
8. "How does real-world validator distribution change the results?"

This means clicking any example chip = instant response (cached), no API call.

#### Expected latency by scenario
| Scenario | Latency | Cost |
|----------|---------|------|
| Page load (static blocks) | <100ms | $0 |
| Example chip (pre-warmed cache) | <50ms | $0 |
| Repeated query (edge cache hit) | <50ms | $0 |
| New query (cache miss, Sonnet) | 1.5-3s | ~$0.01-0.03 |
| Simulation Lab interpretation | 2-4s (after sim completes) | ~$0.02-0.05 |

### API key handling

**Dev:** `VITE_ANTHROPIC_API_KEY` in `.env.local`, direct browser → Claude API.
**Production:** Edge function at `/api/explore`. Holds key, validates requests, applies rate limiting (20 queries/hour per IP). Returns cached responses when available.

---

## File Structure

```
explorer/                              # Our addition to the repo
├── src/
│   ├── App.tsx                        # Shell: header + tab nav + tab content
│   ├── main.tsx                       # Entry: QueryClientProvider + App
│   ├── index.css                      # Tailwind v4 + glass + animations
│   │
│   ├── lib/
│   │   ├── cn.ts                      # clsx + twMerge
│   │   ├── api.ts                     # Claude API: query → tool calls → dispatch
│   │   ├── tools.ts                   # 9 atomic tool definitions (schema + handlers)
│   │   ├── catalog.ts                 # Block catalog → tool_use JSON schema
│   │   ├── context-injection.ts       # Dynamic session context builder
│   │   └── study-context.ts           # Static research context for system prompt
│   │
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx             # Site header + paper meta
│   │   │   ├── TabNav.tsx             # 4-tab navigation
│   │   │   └── Footer.tsx             # Citation, download, links
│   │   │
│   │   ├── blocks/
│   │   │   ├── BlockRenderer.tsx      # Switch on type → component
│   │   │   ├── StatBlock.tsx
│   │   │   ├── InsightBlock.tsx
│   │   │   ├── ChartBlock.tsx
│   │   │   ├── ComparisonBlock.tsx
│   │   │   ├── TableBlock.tsx
│   │   │   ├── CaveatBlock.tsx
│   │   │   ├── SourceBlock.tsx
│   │   │   ├── MapBlock.tsx
│   │   │   └── TimeSeriesBlock.tsx
│   │   │
│   │   ├── explore/
│   │   │   ├── QueryBar.tsx           # AI search input + example chips
│   │   │   ├── QueryHistory.tsx       # Collapsed previous queries
│   │   │   └── BlockCanvas.tsx        # Animated block layout area
│   │   │
│   │   ├── community/
│   │   │   ├── ExplorationCard.tsx    # Community result card (view, upvote, downvote, flag)
│   │   │   ├── ExplorationGallery.tsx # Grid of community explorations (sort, filter)
│   │   │   ├── FlagDialog.tsx         # Flag exploration as inaccurate/outdated
│   │   │   └── VerifyBadge.tsx        # Researcher verification badge + note
│   │   │
│   │   ├── deep-dive/
│   │   │   └── DeepDiveSection.tsx    # Expandable appendix section
│   │   │
│   │   └── simulation/
│   │       ├── ConfigBuilder.tsx      # Bounded parameter form
│   │       ├── SimulationStatus.tsx   # Progress/results display
│   │       └── ResultsView.tsx        # LLM-interpreted simulation results
│   │
│   ├── pages/
│   │   ├── FindingsPage.tsx           # Tab 1
│   │   ├── CommunityPage.tsx          # Tab 2
│   │   ├── DeepDivePage.tsx           # Tab 3
│   │   └── SimulationPage.tsx         # Tab 4
│   │
│   ├── data/
│   │   ├── study-context.ts           # ~8-10k tokens of paper results for system prompt
│   │   ├── default-blocks.ts          # Hardcoded overview blocks for initial load
│   │   ├── deep-dive-sections.ts      # Appendix content mapped to paper sections
│   │   └── gcp-regions.ts             # 40 regions with lat/lon (from data/gcp_regions.csv)
│   │
│   └── types/
│       ├── blocks.ts                  # Block type definitions + Zod schemas
│       └── simulation.ts              # Simulation config + result types
│
├── server/
│   ├── proxy.ts                       # Express: /api/explore → Claude (tool dispatch)
│   ├── explorations.ts                # CRUD endpoints: /api/explorations (search, update, flag, verify)
│   └── simulate.ts                    # /api/simulate → spawn simulation.py, manage jobs
│
├── .env.local                         # VITE_ANTHROPIC_API_KEY (gitignored)
├── index.html
├── vite.config.ts
├── package.json
└── PLAN.md                            # This file
```

---

## Build Phases

### Phase 1: Block catalog + static findings page
**Goal:** Beautiful page showing the study's key results with no AI. Proves the visual design. Can be shared immediately.

1. Define Block types + Zod schemas in `types/blocks.ts`
2. Build all 9 block components with Blueprint dark theme
3. Build `BlockRenderer.tsx` — the type switch
4. Create `gcp-regions.ts` from the repo's `data/gcp_regions.csv`
5. Create `default-blocks.ts` — hardcoded overview matching paper's key findings:
   - StatBlocks: 40 GCP regions, 7 macro-regions, 2 paradigms, 10K slots
   - ComparisonBlock: SSP vs MSP baseline centralization
   - ChartBlock: Convergence rates across experiments
   - InsightBlock: Core thesis + mechanism explanation
   - MapBlock: GCP regions with real Ethereum validator density (from Figure 1)
   - CaveatBlock: Key limitation (GCP-only latency)
   - SourceBlock: arXiv link + GitHub
6. Build page shell: Header + TabNav (4 tabs, only Findings active) + Footer
7. Wire up FindingsPage with static blocks + spring stagger animations
8. **Checkpoint: polished static page, no AI, shareable, looks like a research paper**

### Phase 2: LLM-powered exploration (Findings tab)
**Goal:** Users ask questions, get AI-generated block compositions from paper data.

9. Write `study-context.ts` — bake all experiment results into structured prompt (~8-10k tokens)
10. Write `catalog.ts` — block definitions that auto-generate tool_use JSON schema
11. Write `api.ts` — Claude call with tool_use, parse response → Block[]
12. Build QueryBar with example chips (tuned to actual paper questions)
13. Wire QueryBar → api → BlockCanvas with animated transitions
14. Add shimmer loading state (skeleton blocks during LLM call)
15. Add query history rail (collapsed previous queries, re-expandable)
16. Default overview: show Phase 1 static blocks on load (NO auto-query to Claude). LLM blocks only appear when user submits a query
17. Error handling: API failures, malformed responses, rate limit feedback
18. **Checkpoint: working generative findings page, queries produce real block compositions**

### Phase 3: Deep Dive + Simulation Lab
**Goal:** Tabs 3 and 4 functional.

19. Build DeepDivePage — expandable sections mapped to paper's actual structure (Sections 3-6, Appendices A-G)
20. Each deep-dive section: static content + optional "Ask about this section" mini-query bar
21. Build ConfigBuilder — bounded parameter form matching simulation.py CLI args
22. Preset configs: "Baseline SSP", "Baseline MSP", "Latency-aligned", "Latency-misaligned", "Real-world validators", "EIP-7782 (6s slots)"
23. Natural language → config: LLM translates question to config form pre-fill
24. Create Flask/FastAPI wrapper (`api/app.py`) around simulation.py:
    - POST /simulate — accepts config JSON, writes temp YAML, spawns subprocess with --fast
    - GET /simulate/:jobId — returns status + results when complete
    - GET /simulate/:jobId/progress — returns estimated progress (based on elapsed time vs expected)
25. Wire ConfigBuilder → submit → polling → results display
26. LLM interprets simulation output JSONs into blocks (avg_mev, region_counter_per_slot, supermajority_success are the key ones)
27. **Checkpoint: full 4-tab site, simulation runs end-to-end**

### Phase 4: Community + polish
**Goal:** Community contributions, sharing, production quality.

28. Add persistence layer (Supabase or simple API) for community explorations
29. Build PublishDialog — save query + blocks snapshot + optional description
30. Build ExplorationGallery with upvotes, verified badges, sort by top/recent
31. URL-based sharing — encode query in URL search params, reconstruct on load
32. Export blocks as PNG (html2canvas) or structured JSON
33. Rate limiting on proxy (Claude API calls + simulation submissions)
34. Mobile responsive pass (blocks stack vertically, query bar full-width)
35. Existing Dash viewer integration — "Open in 3D Viewer" button on MapBlocks that launches the existing Plotly globe with correct params
36. **Checkpoint: production-ready community research platform**

### Stage 5 (deferred): Agent-driven autonomous exploration
- LLM designs experiments autonomously from research questions
- Multi-turn agent loop: construct config → submit simulation → wait → interpret → maybe submit follow-up
- json-render-style dynamic UI specs for novel visualization types
- Agent can generate visualizations not in the original catalog
- Requires: reliable simulation backend, cost controls, multi-turn conversation state

---

## Design Language

Blueprint dark theme (from web3-jobs):

| Token | Value | Usage |
|-------|-------|-------|
| Canvas | `#050505` | Page background |
| Surface | `#111111` | Cards, block backgrounds |
| Border | `#222222` | Subtle dividers |
| Text | `#e8e8e3` | Primary text (warm off-white) |
| Muted | `#87867f` | Secondary text, labels |
| Accent | `#3B82F6` | Interactive, links, focus, SSP color |
| Warm | `#d97757` | Emphasis, key findings, MSP color |
| Success | `#2dd4bf` | Positive sentiment, decentralization |
| Warning | `#fbbf24` | Caveats, limitations |
| Danger | `#f43f5e` | Negative sentiment, centralization |

**Glass morphism:** 3 levels (blur 4/8/12px) for query bar, history panel, modals.
**Animations:** Spring physics (stiffness: 240, damping: 30). Stagger at 0.06-0.08s.
**Typography:** Inter for UI, Source Serif 4 for insight/quote blocks.
**Charts:** CSS-based bars for Phase 1. SVG paths for time series. Recharts if complexity demands it.
**Paradigm colors:** SSP = Accent blue (#3B82F6), MSP = Warm terracotta (#d97757). Consistent everywhere.

---

## Visual Spec: Block Components

Each block component has a consistent outer shell (Surface card with Border, 16px padding, rounded-xl) and type-specific inner layout.

### StatBlock
```
┌──────────────────────────────────┐
│  ┌─────┐                         │
│  │ 40  │  GCP Regions Simulated  │  ← value large (text-4xl font-bold), label beside it
│  └─────┘  across 7 macro-regions │  ← sublabel below label (text-muted)
│           ▲ +12% vs baseline     │  ← delta badge: green ▲ / red ▼ / gray —
└──────────────────────────────────┘
```
- Value: `text-4xl font-bold text-accent` (or `text-warm` for MSP, `text-success`/`text-danger` by sentiment)
- Delta badge: pill with arrow icon + text, colored by sentiment
- Layout: flex row, value left, labels right. On mobile: stack vertically.
- Width: 1/3 of grid (3-up on desktop), full on mobile

### InsightBlock
```
┌──────────────────────────────────────────────────┐
│  ┃  Core Thesis                                   │  ← left border accent stripe (key-finding = warm, surprising = danger)
│  ┃                                                │
│  ┃  Both paradigms induce geographic centra-      │  ← body text, supports **bold** and *italic* via simple regex render
│  ┃  lization, but through **opposite mechanisms** │
│  ┃  with different protocol sensitivities.        │
└──────────────────────────────────────────────────┘
```
- Left border: 3px solid. Color: accent (normal), warm (key-finding), danger (surprising)
- Title: `text-lg font-semibold` above body text
- Body: `text-sm leading-relaxed text-muted` with inline bold/italic
- Width: full grid width
- Font: Source Serif 4 for body text (reading-optimized)

### ChartBlock
```
┌──────────────────────────────────────────────────┐
│  Convergence Rate by Experiment                   │  ← title top-left
│                                                   │
│  Baseline ████████████████░░░░ 78%               │  ← horizontal bars, CSS-only for Phase 1
│  Aligned  ████████████████████ 95%               │  ← color by category (SSP=accent, MSP=warm)
│  Misalign ██████████████░░░░░░ 65%               │
│  Real ETH ███████████████████░ 88%               │
│                                                   │
│  unit: %                                          │  ← unit label bottom-right, muted
└──────────────────────────────────────────────────┘
```
- Bars: CSS `div` with width%, rounded-sm, colored by category
- Bar labels: left-aligned, value right of bar
- Line chart variant: SVG `<polyline>` with dot markers, grid lines
- Width: full or 1/2 grid

### ComparisonBlock
```
┌──────────────────────────────────────────────────┐
│  SSP vs MSP: Baseline Centralization              │  ← title spans full width
│ ┌───────────────────┐ ┌───────────────────┐      │
│ │ SSP (External)    │ │ MSP (Local)       │      │  ← two sub-cards, side by side
│ │                   │ │                   │      │
│ │ Gini_g: 0.42     │ │ Gini_g: 0.58     │      │  ← key-value pairs
│ │ HHI_g:  0.15     │ │ HHI_g:  0.22     │      │
│ │ LC_g:   8        │ │ LC_g:   5        │      │
│ └───────────────────┘ └───────────────────┘      │
│  ⚡ MSP centralizes faster in baseline conditions │  ← verdict bar at bottom
└──────────────────────────────────────────────────┘
```
- Two sub-cards: `bg-[#0a0a0a]` with their own border and padding
- Left card: accent blue top border (SSP color)
- Right card: warm terracotta top border (MSP color)
- Verdict: italic, muted text, full width below cards
- On mobile: cards stack vertically

### TableBlock
```
┌──────────────────────────────────────────────────┐
│  Parameter Sensitivity Summary                    │
│ ┌────────────┬──────┬──────┬──────┬──────┐      │
│ │ Experiment │ Gini │ HHI  │ CV   │ LC   │      │  ← header row bg-[#0a0a0a]
│ ├────────────┼──────┼──────┼──────┼──────┤      │
│ │ Baseline   │ 0.42 │ 0.15 │ 1.23 │ 8    │      │
│ │ SE1-Align  │ 0.58 │ 0.22 │ 1.67 │ 5    │ ←HL │  ← highlighted rows get warm bg tint
│ │ SE2-HetVal │ 0.51 │ 0.19 │ 1.45 │ 6    │      │
│ └────────────┴──────┴──────┴──────┴──────┘      │
└──────────────────────────────────────────────────┘
```
- Headers: uppercase, text-xs, text-muted, sticky on scroll
- Highlighted rows: `bg-warm/5` subtle tint
- Overflow: horizontal scroll on mobile with scroll indicator
- Width: full grid

### CaveatBlock
```
┌──────────────────────────────────────────────────┐
│  ⚠ This study uses GCP-only latency data. Real   │  ← warning icon, warning-colored left border
│    validator latencies via other cloud providers   │
│    or bare metal may differ significantly.        │
└──────────────────────────────────────────────────┘
```
- Left border: 3px solid warning (#fbbf24)
- Icon: TriangleAlert from lucide-react, warning color
- Text: text-sm, slightly muted
- Width: full grid

### SourceBlock
```
┌──────────────────────────────────────────────────┐
│  📄 Section 5.1 — Baseline Results               │  ← each ref is a row
│  📄 Section 5.3 — Protocol Parameters             │
│  🔗 arXiv:2509.21475                             │  ← external links get link icon
└──────────────────────────────────────────────────┘
```
- Each ref: flex row with FileText or ExternalLink icon
- External URLs: clickable, accent color, opens new tab
- Section refs: non-clickable, muted, informational
- Width: full grid, compact

### MapBlock
```
┌──────────────────────────────────────────────────┐
│  Validator Distribution: MSP Convergence          │
│                                                   │
│     ● Helsinki                                    │
│  ● London  ● Frankfurt                           │  ← dots on a simplified world outline
│       ● Paris  ● Warsaw                          │    size = validator count
│                    ● Mumbai                       │    color = density scale
│  ● Montreal                                       │
│    ● Virginia  ████                               │  ← legend: color scale
│      ● Dallas                                     │
│                                                   │
│  ● <10  ◉ 10-50  ⬤ 50+                          │  ← size legend
└──────────────────────────────────────────────────┘
```
- Phase 1: CSS-positioned dots on a static SVG world map outline (no map library)
- Dot size: proportional to value (clamp 6-24px)
- Color: density scale (success green → warning yellow → danger red)
- Tooltip on hover: region name, city, validator count
- Width: full grid, 300px min-height

### TimeSeriesBlock
```
┌──────────────────────────────────────────────────┐
│  Gini_g Over Simulation Time                      │
│                                                   │
│  0.6 ┤                    ╱──── MSP Baseline      │  ← SVG line chart
│  0.5 ┤              ╱────╱                        │
│  0.4 ┤        ╱────╱                              │
│  0.3 ┤  ╱────╱                ╱──── SSP Baseline  │
│  0.2 ┤╱                 ╱────╱                    │
│  0.1 ┤            ╱────╱                          │
│      └──────┬──────┬──────┬──────┬               │
│           2500   5000   7500   10000  Slot        │
│                          ▼ convergence            │  ← annotation markers
└──────────────────────────────────────────────────┘
```
- SVG with viewBox, responsive scaling
- Lines: stroke-width 2, colored per series (SSP=accent, MSP=warm)
- Grid: subtle #222222 horizontal lines
- Annotations: vertical dashed line + label positioned above
- Legend: colored circles + labels, top-right
- Hover: crosshair + tooltip showing exact values at slot
- Width: full grid, 250px min-height

---

## Example LLM Responses

These validate the type system and show what the Claude API should return for common queries.

### Query: "Compare local vs external block building under baseline conditions"

```json
{
  "summary": "Under baseline conditions, MSP (local building) centralizes faster and more severely than SSP (external building), despite having a more distributed architecture.",
  "blocks": [
    {
      "type": "comparison",
      "title": "SSP vs MSP: Baseline Centralization at Convergence",
      "left": {
        "label": "SSP (External / MEV-Boost)",
        "items": [
          { "key": "Gini_g", "value": "Moderate increase" },
          { "key": "HHI_g", "value": "Moderate increase" },
          { "key": "LC_g", "value": "Moderate decline" },
          { "key": "Convergence locus", "value": "NA + Middle East" }
        ]
      },
      "right": {
        "label": "MSP (Local / Self-building)",
        "items": [
          { "key": "Gini_g", "value": "Higher increase" },
          { "key": "HHI_g", "value": "Higher increase" },
          { "key": "LC_g", "value": "Sharper decline" },
          { "key": "Convergence locus", "value": "NA primary, EU secondary" }
        ]
      },
      "verdict": "MSP centralizes faster despite distributing information sources across all regions."
    },
    {
      "type": "insight",
      "title": "Why does local building centralize more?",
      "text": "SSP validators co-locate with a **single relay** (one hop saved). MSP validators must optimize proximity to **all 40 signals simultaneously** AND all attesters. The Atlantic corridor (NA↔EU) wins both objectives, creating a stronger geographic attractor.",
      "emphasis": "key-finding"
    },
    {
      "type": "stat",
      "value": "2-hop",
      "label": "SSP Block Path",
      "sublabel": "Proposer → Relay → Attesters"
    },
    {
      "type": "stat",
      "value": "1-hop",
      "label": "MSP Block Path",
      "sublabel": "Proposer → Attesters directly"
    },
    {
      "type": "caveat",
      "text": "These results assume uniform initial validator distribution and 40 homogeneous information sources (one per GCP region). Real-world starting conditions (already concentrated) may reduce the gap between paradigms."
    },
    {
      "type": "source",
      "refs": [
        { "label": "Baseline Results", "section": "Section 5.1" },
        { "label": "Paper", "url": "https://arxiv.org/abs/2509.21475" }
      ]
    }
  ]
}
```

### Query: "How does attestation threshold affect centralization?"

```json
{
  "summary": "Attestation threshold has opposite effects on the two paradigms — this is the paper's most surprising finding.",
  "blocks": [
    {
      "type": "stat",
      "value": "Opposite",
      "label": "Effect Direction",
      "sublabel": "SSP and MSP respond in reverse to threshold changes",
      "sentiment": "negative"
    },
    {
      "type": "table",
      "title": "Centralization Trend by Attestation Threshold (γ)",
      "headers": ["γ", "SSP Effect", "MSP Effect"],
      "rows": [
        ["1/3", "Least centralized", "Most centralized"],
        ["1/2", "Moderate", "Moderate"],
        ["2/3 (current)", "Moderate-high", "Moderate-low"],
        ["4/5", "Most centralized", "Least centralized"]
      ],
      "highlight": [3]
    },
    {
      "type": "insight",
      "title": "The mechanism behind opposite effects",
      "text": "**SSP**: Higher γ → tighter timing → latency sensitivity amplified → co-locating with relay yields larger marginal MEV increase → stronger centralization.\n\n**MSP**: Higher γ → proposers must balance signal proximity (value) vs attester proximity (quorum). These point in **different geographic directions**, so tightening the threshold disperses rather than concentrates.",
      "emphasis": "surprising"
    },
    {
      "type": "source",
      "refs": [
        { "label": "SE4a: Attestation Threshold", "section": "Section 5.3" }
      ]
    }
  ]
}
```

### Query: "Show the geographic convergence pattern under MSP"

```json
{
  "summary": "Under MSP baseline, validators converge primarily toward North America with secondary movement from South America/Africa toward Europe.",
  "blocks": [
    {
      "type": "map",
      "title": "MSP Baseline: Validator Distribution at Convergence",
      "regions": [
        { "name": "us-east4", "lat": 38.89, "lon": -77.04, "value": 180, "label": "Virginia" },
        { "name": "us-east1", "lat": 32.78, "lon": -79.93, "value": 95, "label": "South Carolina" },
        { "name": "us-central1", "lat": 41.26, "lon": -95.93, "value": 85, "label": "Iowa" },
        { "name": "europe-west2", "lat": 51.51, "lon": -0.13, "value": 120, "label": "London" },
        { "name": "europe-west3", "lat": 50.11, "lon": 8.68, "value": 110, "label": "Frankfurt" },
        { "name": "europe-west9", "lat": 48.86, "lon": 2.35, "value": 70, "label": "Paris" },
        { "name": "asia-northeast1", "lat": 35.69, "lon": 139.69, "value": 25, "label": "Tokyo" },
        { "name": "africa-south1", "lat": -26.20, "lon": 28.05, "value": 5, "label": "Johannesburg" }
      ],
      "colorScale": "density"
    },
    {
      "type": "insight",
      "text": "The convergence follows the **Atlantic corridor** — the lowest-latency intercontinental link. North America has the best average median latency (142.97ms) to other regions, making it optimal for both attester quorum and signal aggregation.",
      "emphasis": "key-finding"
    },
    {
      "type": "stat",
      "value": "142.97ms",
      "label": "NA Average Median Latency",
      "sublabel": "Best of any macro-region"
    }
  ]
}
```

---

## Default Blocks (Findings Page Initial Load)

On first load, before any AI query, the Findings page shows these hardcoded blocks in order:

### Row 1: Key stats (3-up grid)
1. **StatBlock** — `value: "2"`, `label: "Paradigms Compared"`, `sublabel: "SSP (external) vs MSP (local) block building"`
2. **StatBlock** — `value: "40"`, `label: "GCP Regions Simulated"`, `sublabel: "across 7 macro-regions worldwide"`
3. **StatBlock** — `value: "7"`, `label: "Experiments Analyzed"`, `sublabel: "baseline + 6 sensitivity evaluations"`

### Row 2: Core finding
4. **InsightBlock** — `emphasis: "key-finding"`, title: "Both paradigms centralize, but differently"
   - Text explains that SSP and MSP both push toward geographic concentration but through opposite mechanisms with different protocol sensitivities

### Row 3: Head-to-head
5. **ComparisonBlock** — SSP vs MSP baseline, showing Gini/HHI/LC directions + convergence loci

### Row 4: Surprise finding
6. **InsightBlock** — `emphasis: "surprising"`, title: "Attestation threshold has opposite effects"
   - Higher γ strengthens SSP centralization but weakens MSP centralization

### Row 5: Geographic context
7. **MapBlock** — All 40 GCP regions with initial uniform distribution, showing the simulation's geographic canvas

### Row 6: Policy implications + sources
8. **CaveatBlock** — "These findings are derived from agent-based simulation using GCP-only latency data. Real validator behavior involves additional factors."
9. **SourceBlock** — arXiv link, GitHub repo, paper authors

**Total: 9 blocks, all 9 block types represented in the default view.**

---

## Component Interaction Spec

### Block interactions
- **StatBlock**: Not interactive. Pure display.
- **InsightBlock**: Not interactive. Pure display.
- **ChartBlock**: Hover on bars → tooltip with exact value + label. No click action.
- **ComparisonBlock**: Not interactive. Verdict text may truncate with "show more" if >2 lines.
- **TableBlock**: Hover highlights row. Horizontal scroll on mobile (with shadow gradient indicators at edges). No click action.
- **CaveatBlock**: Not interactive. Pure display.
- **SourceBlock**: External URLs are clickable links (open new tab). Section refs are non-clickable.
- **MapBlock**: Hover on dot → tooltip (region name, city, count). Click → no action in Phase 1. Phase 3: click could filter to that region's data.
- **TimeSeriesBlock**: Hover → vertical crosshair + tooltip showing all series values at that x position. No click action.

### Query bar interactions
- **Focus**: Glass-1 border brightens to accent on focus
- **Submit**: Enter key or click button. Disabled while loading.
- **Example chips**: Click fills query bar text + auto-submits
- **Loading**: Query bar shows shimmer pulse, submit button becomes spinner
- **History**: Previous queries collapse into pills below bar. Click pill → re-expand that result set (no re-query)

### Tab navigation
- **Click tab**: Slide transition, new tab content enters from right (or left, depending on direction)
- **Active tab**: Accent underline + brighter text
- **Inactive tab**: Muted text, no underline
- **Tab badges**: Community tab shows count badge when explorations exist

### Block canvas
- **New blocks**: Stagger fade+slide-up on render (0.06s gap between blocks)
- **Replace blocks**: Old blocks fade out (0.15s), new blocks stagger in
- **Scroll behavior**: Smooth scroll to top of new results after query

---

## Full-Page Wireframes: Findings Tab (4 States)

The Findings tab has 4 distinct visual states. The page structure is identical across states — only the content area below the search bar changes.

### Page Shell (constant across all states)

```
Desktop (>1024px):
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   Geo Validator Lab                                    arXiv ↗  GH ↗ │  ← Header: title left, external links right
│   Interactive Research Explorer                                      │  ← subtitle, muted
│                                                                      │
│   ┌──────────┐ ┌──────────────┐ ┌───────────┐ ┌──────────────────┐  │
│   │ Findings │ │ Explore Hist │ │ Deep Dive │ │ Simulation Lab   │  │  ← Tab nav: Findings has accent underline
│   │ ▔▔▔▔▔▔▔▔ │ │              │ │           │ │                  │  │
│   └──────────┘ └──────────────┘ └───────────┘ └──────────────────┘  │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │ 🔍 Ask anything about the paper...                     [→]  │   │  ← Search bar: glass-1, full width
│   └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐    │
│   │                    << CONTENT AREA >>                        │    │  ← Changes per state (see below)
│   │                                                              │    │
│   └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│   ─────────────────────────────────────────────────────────────────  │
│   Yang, Oz, Wu, Zhang (2025) · arXiv:2509.21475 · MIT License      │  ← Footer: citation, links
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### State 1: Default Landing (first load — zero API calls)

Topic cards grid + default blocks below. This is the "executive summary" view.

```
┌──────────────────────────────────────────────────────────────────┐
│ 🔍 Ask anything about the paper...                         [→]  │
└──────────────────────────────────────────────────────────────────┘

  Explore a finding:                                                   ← section label, muted

  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  │ SSP vs MSP:      │ │ Where do         │ │ Does source      │ │ What if          │
  │ Which            │ │ validators       │ │ placement        │ │ validators start │
  │ centralizes      │ │ end up?          │ │ matter?          │ │ concentrated?    │
  │ more?            │ │                  │ │                  │ │                  │
  │        → Explore │ │        → Explore │ │        → Explore │ │        → Explore │
  └──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘
  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  │ The attestation  │ │ Shorter slot     │ │ Key metrics      │ │ Limitations      │
  │ threshold        │ │ times            │ │ explained        │ │ & what's next    │
  │ surprise         │ │ (EIP-7782)       │ │                  │ │                  │
  │                  │ │                  │ │                  │ │                  │
  │        → Explore │ │        → Explore │ │        → Explore │ │        → Explore │
  └──────────────────┘ └──────────────────┘ └──────────────────┘ └──────────────────┘

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─    ← subtle divider

  Key findings at a glance:                                            ← section label

  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  │    2             │ │   40             │ │    7             │    ← Row 1: StatBlocks (3-up)
  │  Paradigms       │ │  GCP Regions     │ │  Experiments     │
  │  Compared        │ │  Simulated       │ │  Analyzed        │
  └──────────────────┘ └──────────────────┘ └──────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │ ┃ Both paradigms centralize, but differently                 │    ← Row 2: InsightBlock (key-finding)
  │ ┃                                                            │
  │ ┃ Both SSP and MSP push toward geographic concentration      │
  │ ┃ but through **opposite mechanisms** with different          │
  │ ┃ protocol sensitivities.                                    │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │ SSP vs MSP: Baseline Centralization                          │    ← Row 3: ComparisonBlock
  │ ┌──────────────────────┐  ┌──────────────────────┐          │
  │ │▔▔ SSP (External)     │  │▔▔ MSP (Local)        │          │
  │ │ Gini_g: Moderate ↑   │  │ Gini_g: Higher ↑     │          │
  │ │ HHI_g:  Moderate ↑   │  │ HHI_g:  Higher ↑     │          │
  │ │ LC_g:   Moderate ↓   │  │ LC_g:   Sharper ↓    │          │
  │ │ Locus:  NA + MidEast │  │ Locus:  NA + EU      │          │
  │ └──────────────────────┘  └──────────────────────┘          │
  │ ⚡ MSP centralizes faster in baseline conditions             │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │ ┃ Attestation threshold has opposite effects                 │    ← Row 4: InsightBlock (surprising)
  │ ┃                                                            │
  │ ┃ Higher γ → SSP centralizes MORE, but MSP centralizes      │
  │ ┃ LESS. The only protocol parameter with opposite effects.   │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │ Simulation Geographic Canvas                                 │    ← Row 5: MapBlock
  │                                                              │
  │      ● Helsinki                                              │
  │   ● London ● Frankfurt                                       │
  │     ● Paris   ● Warsaw                                       │
  │                     ● Mumbai                                 │
  │  ● Montreal                              ● Tokyo             │
  │    ● Virginia                                                │
  │      ● Dallas                                                │
  │                          ● Singapore                         │
  │            ● São Paulo            ● Sydney                   │
  │                                                              │
  │  ● <10  ◉ 10-50  ⬤ 50+     40 GCP regions, 7 macro-regions │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │ ⚠ These findings are derived from agent-based simulation     │    ← Row 6: CaveatBlock
  │   using GCP-only latency data. Real validator behavior       │
  │   involves additional factors.                               │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │ 📄 arXiv:2509.21475 · Section 5 — Results                   │    ← Row 6: SourceBlock
  │ 🔗 github.com/syang-ng/geographical-decentralization-sim    │
  │ 👤 Yang, Oz, Wu, Zhang (2025)                               │
  └──────────────────────────────────────────────────────────────┘
```

**Key layout decisions:**
- Topic cards are a **4×2 grid above the default blocks**, not mixed in
- Cards have a surface bg + border, each ~200px wide on desktop
- "→ Explore" link at bottom-right of each card (accent color)
- Default blocks show below the cards, always visible on scroll
- No collapse/hide of cards when scrolling — they're the nav layer

### State 2: Topic Card Expanded (user clicked a card — still zero API calls)

Clicking a card scrolls down and replaces the default blocks with deeper pre-rendered content for that topic. The other cards dim slightly. A "← Back to overview" link appears.

```
┌──────────────────────────────────────────────────────────────────┐
│ 🔍 Ask anything about the paper...                         [→]  │
└──────────────────────────────────────────────────────────────────┘

  Explore a finding:                              ← Back to overview   ← back link, muted, top-right

  ┌──────────────────┐ ┌▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓┐ ┌──────────────────┐ ┌──────────────────┐
  │ SSP vs MSP:      │ │░ Does source     ░│ │ What if          │ │ ... (4 more)     │
  │ Which central... │ │░ placement       ░│ │ validators ...   │ │                  │
  │                  │ │░ matter?    ACTIVE░│ │                  │ │                  │
  └──────────────────┘ └▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓┘ └──────────────────┘ └──────────────────┘
                         ↑ active card has accent border, others dim (opacity-60)

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  SE1: Information-Source Placement                                    ← expanded topic heading

  ┌──────────────────┐ ┌──────────────────┐
  │ 3 aligned        │ │ 3 misaligned     │                           ← StatBlocks for this topic
  │ Sources          │ │ Sources          │
  │ asia/eu/us hubs  │ │ africa/aus/sa    │
  └──────────────────┘ └──────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │ ┃ Opposite paradigm sensitivities to source placement        │    ← InsightBlock
  │ ┃                                                            │
  │ ┃ MSP: latency-ALIGNED centralizes MORE (low-latency         │
  │ ┃ regions benefit both value and propagation).                │
  │ ┃ SSP: latency-MISALIGNED centralizes MORE (poorly           │
  │ ┃ connected relays → large proposer-relay gap).              │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │ Centralization by Source Placement                            │    ← ChartBlock
  │                                                              │
  │ MSP Aligned   ████████████████████ 95%                       │
  │ MSP Baseline  ████████████████░░░░ 78%                       │
  │ MSP Misalign  ██████████████░░░░░░ 65%                       │
  │ SSP Aligned   █████████████░░░░░░░ 62%                       │
  │ SSP Baseline  ████████████████░░░░ 78%                       │
  │ SSP Misalign  ██████████████████░░ 88%                       │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │ 📄 Section 5.2 — Sensitivity Evaluations (SE1)               │
  └──────────────────────────────────────────────────────────────┘

  Want to go deeper?  [Ask Claude about source placement ⚡]          ← CTA to tier 3
```

**Key transitions:**
- Topic cards stay visible (horizontal row), active card gets accent border
- Inactive cards dim to `opacity-60`
- Block canvas below swaps from default blocks to topic-specific blocks (spring animation)
- "← Back to overview" restores default blocks
- CTA at bottom: "Ask Claude about [topic]" — leads to State 4

### State 3: Query Typed — Tier 2 Match Found (zero API cost)

User types in search bar. Before submitting, tier 2 fuzzy-match fires and shows inline community pool matches.

```
┌──────────────────────────────────────────────────────────────────┐
│ 🔍 How does migration cost affect convergence?  ⌫         [→]  │  ← user typing, glass border glows accent
└──────────────────────────────────────────────────────────────────┘
  ┌────────────────────────────────────────────────────────────┐
  │  Similar questions already explored:                        │     ← tier 2 match dropdown
  │                                                             │
  │  ┌──────────────────────────────────────────────────────┐  │
  │  │ "What role does migration cost play in validator     │  │
  │  │  movement?"                                          │  │
  │  │  12 ↑ · 42 views · verified ✓                       │  │     ← match card with metadata
  │  └──────────────────────────────────────────────────────┘  │
  │  ┌──────────────────────────────────────────────────────┐  │
  │  │ "Migration cost threshold analysis — when do         │  │
  │  │  validators stop moving?"                            │  │
  │  │  8 ↑ · 23 views                                     │  │
  │  └──────────────────────────────────────────────────────┘  │
  │                                                             │
  │  [View these answers]    or    [Ask Claude anyway ⚡]       │     ← explicit choice
  └────────────────────────────────────────────────────────────┘

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  (topic cards + default blocks still visible below, but visually
   receded — the dropdown demands attention)
```

**If user clicks "View these answers":** The matched exploration expands inline (blocks appear in the block canvas, same as if Claude generated them). Topic cards stay but scroll up.

**If user clicks "Ask Claude anyway":** Proceeds to State 4.

### State 4: Fresh Claude Query — Loading + Results (API cost)

User submitted a query. Topic cards collapse to a single row. Loading shimmer, then blocks appear.

```
┌──────────────────────────────────────────────────────────────────┐
│ 🔍 How does migration cost affect convergence?             [⟳]  │  ← submit button becomes spinner
└──────────────────────────────────────────────────────────────────┘

  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  +4 more    ← topic cards compact to mini pills
  │ SSP/MSP  │ │ Geo conv │ │ Sources  │ │ Heterog  │              (clickable, restores State 1)
  └──────────┘ └──────────┘ └──────────┘ └──────────┘

  Query history:  [SSP vs MSP baseline]  [migration cost ←active]   ← query pills, click to switch

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  Loading...                                                         ← during API call

  ┌──────────────────────────────────────────────────────────────┐
  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ shimmer│  ← skeleton blocks
  └──────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────┐
  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ shimmer│
  └──────────────────────────────────────────────────────────────┘
  ┌──────────────────────────────────────────────────────────────┐
  │ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ shimmer│
  └──────────────────────────────────────────────────────────────┘

  ═══════════════════════════════════════════════════════════════

  After Claude responds (shimmer → blocks with stagger animation):

  Migration cost is the key threshold determining whether validators    ← summary heading (from Claude)
  move or stay — it acts as a friction parameter on the system.

  ┌──────────────────┐ ┌──────────────────┐
  │  0.002 ETH       │ │  0.0001 ETH      │                           ← StatBlocks
  │  Key Threshold   │ │  Baseline Default │
  │  Paper's pivot   │ │  In YAML configs  │
  └──────────────────┘ └──────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │ ┃ Migration cost creates a "stickiness" effect               │    ← InsightBlock
  │ ┃                                                            │
  │ ┃ When cost > (best_mev - current_mev), validators stay      │
  │ ┃ even if a better region exists. Higher cost = slower        │
  │ ┃ convergence, more geographic diversity preserved.           │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │ Convergence Speed by Migration Cost                          │    ← ChartBlock
  │                                                              │
  │ cost=0.0     ████████████████████ 100% converged             │
  │ cost=0.0001  ████████████████░░░░ 82%                        │
  │ cost=0.002   ██████████░░░░░░░░░░ 48%                        │
  │ cost=0.005   ████░░░░░░░░░░░░░░░░ 22%                        │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │ ⚠ The paper uses a constant, instantaneous migration cost.   │    ← CaveatBlock
  │   Real migration costs are time-varying and heterogeneous.   │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │ 📄 Section 5.1 — Baseline Results (cost analysis)            │    ← SourceBlock
  │ 📄 Appendix C — Migration Cost CDF Analysis                  │
  └──────────────────────────────────────────────────────────────┘

  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─

  Explore further:                                                     ← follow-up suggestions (from pool state)
  ┌──────────────────────────────┐ ┌───────────────────────────────┐
  │ How does cost interact with  │ │ No one has explored SE3 yet  │
  │ attestation threshold?       │ │ — what happens with joint     │
  │                              │ │ heterogeneity?               │
  └──────────────────────────────┘ └───────────────────────────────┘
```

**Key layout decisions for State 4:**
- **Topic cards compact** to small pills (just the short title) — saves vertical space
- **Query history pills** appear below search bar — click any to restore that result set without re-querying
- **Summary heading** from Claude's response appears above blocks (not inside a block)
- **Follow-up suggestions** at the bottom — composed from `get_pool_state` + `get_session_history`
- **Clicking a follow-up** fills the search bar and submits (new query → new State 4)

### State Transitions

```
  State 1 (Landing)
    │
    ├─── click topic card ──→ State 2 (Card Expanded)
    │                            │
    │                            ├─── "← Back to overview" ──→ State 1
    │                            └─── "Ask Claude about X" ──→ State 4
    │
    ├─── type in search bar ──→ State 3 (Tier 2 Match)
    │                            │
    │                            ├─── "View these" ──→ State 4 (cached blocks, no API)
    │                            └─── "Ask Claude anyway" ──→ State 4 (fresh API call)
    │
    └─── click example chip ──→ State 4 (pre-warmed cache hit, instant)

  State 4 (Results)
    │
    ├─── click query history pill ──→ State 4 (different result set, no re-query)
    ├─── click follow-up suggestion ──→ State 4 (new query)
    ├─── click compact topic card pill ──→ State 2
    └─── clear search bar ──→ State 1
```

### Mobile Layout (State 1)

```
┌─────────────────────────────┐
│ Geo Validator Lab     arXiv │    ← header compact
│                             │
│ Findings │ History │ Deep │+│    ← tabs scroll horizontally
│ ▔▔▔▔▔▔▔▔                   │
│                             │
│ ┌─────────────────────────┐ │
│ │ 🔍 Ask anything...  [→] │ │    ← search bar full width
│ └─────────────────────────┘ │
│                             │
│ ┌───────────┐ ┌───────────┐ │    ← topic cards: 2×4 grid
│ │ SSP vs    │ │ Where do  │ │
│ │ MSP       │ │ validators│ │
│ │  → Explore│ │  → Explore│ │
│ └───────────┘ └───────────┘ │
│ ┌───────────┐ ┌───────────┐ │
│ │ Source    │ │ Validators│ │
│ │ placement │ │ start     │ │
│ │  → Explore│ │  → Explore│ │
│ └───────────┘ └───────────┘ │
│       ... (scroll for 4 more)│
│                             │
│ ┌─────────────────────────┐ │
│ │    2                    │ │    ← StatBlocks stack 1-up
│ │  Paradigms Compared     │ │
│ └─────────────────────────┘ │
│ ┌─────────────────────────┐ │
│ │   40                    │ │
│ │  GCP Regions Simulated  │ │
│ └─────────────────────────┘ │
│          ...                │
└─────────────────────────────┘
```

---

## Responsive Layout Spec

### Breakpoints
- **Mobile** (<640px): Single column. All blocks full-width. Query bar full-width. Tab nav scrolls horizontally.
- **Tablet** (640-1024px): Two-column grid for StatBlocks. Other blocks full-width. Query bar with side margins.
- **Desktop** (>1024px): Three-column grid for StatBlocks. ComparisonBlock side-by-side sub-cards. Max-width 1200px centered.

### Grid rules
- StatBlocks: 3-up on desktop, 2-up on tablet, 1-up on mobile
- ComparisonBlock: Side-by-side sub-cards on desktop/tablet, stacked on mobile
- ChartBlock: Full width on all sizes (bars resize fluidly)
- TableBlock: Full width, horizontal scroll on mobile with gradient shadow at scroll edge
- MapBlock: Full width, min-height 250px mobile / 350px desktop
- TimeSeriesBlock: Full width, min-height 200px mobile / 300px desktop
- InsightBlock, CaveatBlock, SourceBlock: Full width always

### Block canvas layout
```
Desktop (>1024px):
┌─── max-w-5xl mx-auto px-6 ────────────────────┐
│ [Stat] [Stat] [Stat]           ← 3-col grid   │
│ [Insight ─────────────────]    ← full width    │
│ [Comparison ──────────────]    ← full width    │
│ [Chart ───────────────────]    ← full width    │
│ [Map ─────────────────────]    ← full width    │
│ [Caveat] [Source]              ← 2-col grid    │
└────────────────────────────────────────────────┘

Mobile (<640px):
┌─── px-4 ──────────┐
│ [Stat]             │
│ [Stat]             │
│ [Stat]             │
│ [Insight]          │
│ [Comparison]       │
│ [Chart]            │
│ [Map]              │
│ [Caveat]           │
│ [Source]            │
└────────────────────┘
```

---

## Error & Edge Case Handling

### LLM response errors
| Case | Detection | User experience |
|------|-----------|-----------------|
| Network failure | fetch throws | "Connection failed. Check your network and try again." with retry button |
| API rate limit (429) | status code | "Rate limited. Try again in X seconds." with countdown |
| API error (500) | status code | "The AI service is temporarily unavailable." with retry button |
| Malformed tool_use | Zod validation fails on Block[] | Fall back to InsightBlock with raw text content from LLM |
| No tool_use in response | Response has text but no tool call | Wrap text in a single InsightBlock |
| Empty blocks array | Valid response, 0 blocks | "I couldn't find relevant information for that query. Try rephrasing or use an example question." |
| Timeout (>30s) | AbortController | "Request timed out. This might be a complex query — try something more specific." |

### Block rendering errors
| Case | Handling |
|------|---------|
| Unknown block type | Skip silently (defensive: `default: return null` in switch) |
| Missing required field | Zod `.safeParse()` — skip block if invalid, render others |
| ChartBlock with 0 data points | Show empty state: "No data points for this chart" |
| MapBlock with no regions | Show world outline only, no dots |
| TimeSeriesBlock with 1 data point | Render as single dot, no line |

### Simulation errors (Phase 3)
| Case | Handling |
|------|---------|
| Python not installed | Pre-flight check on server start, clear error message |
| Simulation timeout (>10 min) | Kill process, return partial results if available |
| Invalid config | Validate against Zod schema before submission, show field-level errors |
| Simulation crash | Return stderr output wrapped in CaveatBlock |

---

## Animation Choreography

### Spring config
```typescript
const spring = { type: 'spring', stiffness: 240, damping: 30 }
const gentleSpring = { type: 'spring', stiffness: 180, damping: 24 }
```

### Block enter (stagger sequence)
```typescript
// Parent: staggerChildren: 0.06
// Each block:
initial: { opacity: 0, y: 12 }
animate: { opacity: 1, y: 0 }
transition: spring
```

### Block replace (query results change)
```typescript
// Step 1: Old blocks exit
exit: { opacity: 0, y: -8 }
transition: { duration: 0.15 }

// Step 2: New blocks enter with stagger (after exit completes)
initial: { opacity: 0, y: 12 }
animate: { opacity: 1, y: 0 }
transition: spring, staggerChildren: 0.06
```

### Tab switch
```typescript
// Direction-aware slide
const direction = newTabIndex > oldTabIndex ? 1 : -1
initial: { opacity: 0, x: 20 * direction }
animate: { opacity: 1, x: 0 }
exit: { opacity: 0, x: -20 * direction }
transition: gentleSpring
```

### Query bar interactions
```typescript
// Focus ring
whileFocus: { borderColor: '#3B82F6', boxShadow: '0 0 0 2px rgba(59,130,246,0.15)' }

// Submit button press
whileTap: { scale: 0.97 }

// Example chip press
whileTap: { scale: 0.95 }
whileHover: { backgroundColor: 'rgba(59,130,246,0.1)' }
```

### Loading shimmer (skeleton blocks)
```typescript
// 3 skeleton rectangles with shimmer
// Skeleton block: h-24 rounded-xl bg-[#111111] overflow-hidden
// Shimmer: pseudo-element moving left-to-right
@keyframes shimmer {
  0% { transform: translateX(-100%) }
  100% { transform: translateX(100%) }
}
// Duration: 1.5s, infinite, ease-in-out
```

### Map dots
```typescript
// Dots scale in with stagger
initial: { scale: 0, opacity: 0 }
animate: { scale: 1, opacity: 1 }
transition: { ...spring, delay: index * 0.02 }

// Hover
whileHover: { scale: 1.3 }
```

---

## Testing Strategy

### Phase 1: Static blocks (unit tests + visual)
- **Block component tests** (Vitest + Testing Library): Each of the 9 block types renders correctly with valid props, handles missing optional fields, renders nothing for completely invalid data
- **BlockRenderer test**: Switch dispatches to correct component for each type
- **Zod schema tests**: Valid Block JSON passes, invalid JSON fails with descriptive errors
- **Visual regression**: Manual screenshots of each block type in isolation (or Storybook-lite via a `/dev` route showing all blocks)

### Phase 2: LLM integration (integration tests)
- **api.ts tests**: Mock fetch, verify tool_use parsing → Block[], error handling for malformed responses
- **catalog.ts tests**: JSON schema generation matches expected structure
- **QueryBar → BlockCanvas integration**: Mock API, verify blocks render after query submission
- **Error states**: Test all 7 error cases from the error handling spec above

### Phase 3: Simulation (E2E smoke tests)
- **Config validation**: Zod schema rejects out-of-range parameters
- **Job lifecycle**: Submit → poll → complete (mocked simulation backend)
- **Timeout handling**: Verify cleanup on long-running simulations

### Phase 4: Community (E2E)
- **Publish flow**: Create exploration → appears in gallery
- **Upvote**: Click → count increments

### Test infrastructure
- **Runner**: Vitest with jsdom environment
- **Coverage target**: 80% on `types/`, `lib/`, `components/blocks/`
- **No E2E framework in Phase 1** — manual testing via dev server is sufficient for MVP

---

## Agent-Native Architecture

This section documents how the explorer satisfies all 8 agent-native principles. Every principle has a concrete implementation strategy and a target score.

### Principle 1: Action Parity (target: 90%+)

**Rule:** Whatever the user can do, the agent can do.

| User Action | Agent Equivalent | How |
|-------------|-----------------|-----|
| Click topic card | Agent detects overlap with topic cards via prompt | Tier routing in system prompt |
| Type a query | Agent receives query as user message | Standard flow |
| Browse explore history | `search_explorations` tool | Filters by paradigm, experiment, verified |
| Upvote/downvote | UI-only (anonymous voting needs browser fingerprint) | Acceptable gap — voting is a user identity action |
| Flag an exploration | `flag_exploration` tool | Agent can flag inaccurate content |
| Verify an exploration | `verify_exploration` tool | Researcher workflow |
| Configure simulation | `get_simulation_constraints` + `submit_simulation_config` | Prompt translates NL → config using constraint data |
| View deep dive sections | Static content — same for agent and user | Shared data |
| Export blocks as PNG/JSON | UI-only (html2canvas / download) | Acceptable gap — export is a browser action |

**Gap analysis:** 2 acceptable gaps (voting, export) — both are browser-identity actions that don't make sense for an agent.

### Principle 2: Tools as Primitives (target: 85%+)

**Rule:** Tools provide capability, not behavior. Test: can you change the behavior by editing the prompt alone?

| Tool | Classification | Why |
|------|---------------|-----|
| `render_blocks` | PRIMITIVE ✅ | Capability: compose blocks. Prompt decides *when* and *how* to compose |
| `search_explorations` | PRIMITIVE ✅ | Capability: search. Prompt decides what to do with results |
| `update_exploration` | PRIMITIVE ✅ | Capability: mutate metadata. No business logic |
| `flag_exploration` | PRIMITIVE ✅ | Capability: flag. Prompt decides when flagging is appropriate |
| `verify_exploration` | PRIMITIVE ✅ | Capability: verify. No automated verification logic |
| `get_simulation_constraints` | PRIMITIVE ✅ | Capability: read parameter bounds. Pure data, no decisions |
| `submit_simulation_config` | PRIMITIVE ✅ | Capability: validate + enqueue. Prompt does the NL→config translation |
| `get_pool_state` | PRIMITIVE ✅ | Capability: read pool stats. Prompt composes suggestions from the data |
| `get_session_history` | PRIMITIVE ✅ | Capability: read session context. Prompt uses it for continuity |

**Score: 9/9 pure primitives (100%).** Every tool provides capability without encoding business logic. The system prompt orchestrates all behavior.

### Principle 3: Context Injection (target: 80%+)

**Rule:** System prompt includes dynamic context about app state.

| Context Type | Injected? | Location |
|-------------|-----------|----------|
| Research data (paper findings, metrics, experiments) | ✅ | Static system prompt (cached) |
| Available tools + when to use them | ✅ | Static system prompt (cached) |
| Composition guidelines (block layout rules) | ✅ | Static system prompt (cached) |
| Tier routing logic (topic cards → pool → fresh) | ✅ | Static system prompt (cached) |
| Pool state (size, verified count, gaps) | ✅ | Dynamic context injection |
| Trending queries | ✅ | Dynamic context injection |
| User session (recent queries, viewed blocks, tab) | ✅ | Dynamic context injection |
| Rate limit remaining | ✅ | Dynamic context injection |
| Topic card coverage map | ✅ | Static system prompt (cached) |

**Score: 9/9.** Every context type that could inform agent behavior is injected.

### Principle 4: Shared Workspace (target: 85%+)

**Rule:** Agent and user work in the same data space.

| Data Store | User Access | Agent Access | Shared? |
|-----------|-------------|--------------|---------|
| Explorations table | Read (gallery), Write (auto-publish) | Read (`search_explorations`), Write (`render_blocks` auto-saves), Update (`update_exploration`), Flag (`flag_exploration`) | ✅ Shared |
| Votes table | Write (click), Read (counts shown) | Read (aggregated in search results) | ✅ Shared |
| Simulation jobs | Create (submit), Read (poll status) | Create (`build_simulation_config`), Read (results for interpretation) | ✅ Shared |
| Topic cards | Read (UI render) | Read (referenced in prompt context) | ✅ Shared |
| Default blocks | Read (initial page load) | Read (prompt knows what's pre-rendered) | ✅ Shared |
| Query edge cache | Read (instant response) | Write (responses auto-cached) | ✅ Shared |
| Session state | Read/Write (local) | Read (injected as context) | ✅ Shared |

**Score: 7/7.** No shadow databases. Agent reads and writes the same tables as the user.

### Principle 5: CRUD Completeness (target: 80%+)

**Rule:** Every entity has full CRUD.

| Entity | Create | Read | Update | Delete | Score |
|--------|--------|------|--------|--------|-------|
| Exploration | ✅ Auto on Claude response | ✅ `search_explorations` | ✅ `update_exploration` | ✅ `flag_exploration` (soft) | 4/4 |
| Vote | ✅ UI click | ✅ Aggregated | ✅ Change direction | ✅ Remove vote | 4/4 |
| Simulation Job | ✅ Submit config | ✅ Poll status | ✅ Cancel | ✅ Auto-cleanup | 4/4 |
| Topic Card | ✅ In prompt config | ✅ UI render | ✅ Edit prompt | ✅ Edit prompt | 4/4 (prompt-native) |
| Block | ✅ `render_blocks` | ✅ UI render | ⚠️ Immutable | ⚠️ Part of exploration | 2/4 |

**Score: 18/20 (90%).** Blocks are intentionally immutable (they're renderings of a point-in-time answer, like a snapshot — mutating them would break the exploration's integrity).

### Principle 6: UI Integration (target: 85%+)

**Rule:** Agent actions immediately reflected in UI.

| Agent Action | UI Mechanism | Immediate? |
|-------------|-------------|------------|
| `render_blocks` | React state update → BlockCanvas re-render | ✅ Instant |
| `search_explorations` | Results shown inline in query bar (tier 2 matches) | ✅ Instant |
| `suggest_explorations` | Chips rendered below response blocks | ✅ Instant |
| `flag_exploration` | Badge appears on flagged card, filters update | ✅ Instant |
| `verify_exploration` | Verified badge appears on card | ✅ Instant |
| `update_exploration` | Tags update in gallery, context appended | ✅ Instant |
| `build_simulation_config` | Config panel pre-filled with parsed values | ✅ Instant |
| New exploration auto-published | Explore History tab count badge increments | ✅ Via React Query invalidation |
| Pool stats change | Dynamic context updates on next request | ⚠️ Next-request (not real-time) |

**Score: 8/9 immediate.** Pool stats update is next-request, not real-time push. Acceptable for MVP — WebSocket/SSE can be added later for live pool updates.

### Principle 7: Capability Discovery (target: 85%+)

**Rule:** Users can discover what the agent can do.

| Mechanism | Exists? | Implementation |
|-----------|---------|---------------|
| Example query chips | ✅ | 8 chips below search bar, one per topic card |
| Suggested follow-ups | ✅ | Prompt composes suggestions from `get_pool_state` + `get_session_history` data — rendered inline after every response |
| Empty state guidance | ✅ | "Ask anything about the paper — or click a topic card to start" |
| Topic cards as discovery | ✅ | 8 clickable cards showing what the system knows about |
| Explore History as discovery | ✅ | Browsing what others asked reveals the question space |
| Coverage gap hints | ✅ | `get_pool_state` returns coverage gaps → prompt composes "No one has asked about SE3 yet — try it!" |
| Help/about section | ✅ | Footer or "?" icon explaining what the explorer does and how |

**Score: 7/7.** The `suggest_explorations` tool is the key addition — it turns the agent into an active guide rather than a passive responder.

### Principle 8: Prompt-Native Features (target: 75%+)

**Rule:** Features are prompts defining outcomes, not code.

| Feature | Defined In | Type |
|---------|-----------|------|
| Block composition rules | System prompt ("Composition Guidelines") | ✅ Prompt |
| Topic card selection/content | System prompt ("Topic Card Selection") | ✅ Prompt |
| Tier routing (when to use pool vs fresh) | System prompt ("Tier Routing") | ✅ Prompt |
| Response strategy (search first, then render) | System prompt ("Response Strategy") | ✅ Prompt |
| Follow-up suggestion logic | System prompt ("Generating Follow-Up Suggestions" section) — composes from get_pool_state + get_session_history | ✅ Prompt |
| Simulation NL→config translation | System prompt ("Building Simulation Configs" section) — parses intent using get_simulation_constraints data | ✅ Prompt |
| Flagging criteria | System prompt (agent decides when to flag) | ✅ Prompt |
| Verification criteria | System prompt (agent decides when to verify) | ✅ Prompt |
| Max blocks per response | System prompt ("Maximum 6 blocks") | ✅ Prompt |
| SSP/MSP color coding | Code (Tailwind classes in block components) | ❌ Code |
| Block visual layout (grid, spacing) | Code (React + CSS) | ❌ Code |
| Animation choreography | Code (Framer Motion configs) | ❌ Code |
| Zod validation rules | Code (type schemas) | ❌ Code |
| Rate limiting | Code (server middleware) | ❌ Code |
| Query normalization | Code (string processing) | ❌ Code |
| Fuzzy matching threshold | Dynamic context (injected, editable) | ✅ Prompt-adjacent |

**Score: 10/16 (62.5%).** The 6 code-defined features are all rendering/infrastructure concerns that *should* be in code — you can't define Tailwind classes or animation physics in a prompt. The behavioral features (what to do, when, how to compose) are all prompt-native.

**Adjusted score (excluding inherently-code features):** 10/10 behavioral features are prompt-native = **100%**.

---

## Open Questions

- [ ] Can we get pre-computed simulation output files for all 6 experiments? (Needed for accurate system prompt data)
- [ ] Deployment target — Vercel for frontend + where for simulation backend?
- [ ] Community persistence — Supabase? Simple file-based API?
- [ ] Do researchers want approval flow for community submissions?
- [ ] Budget/rate limits for Claude API + simulation compute
- [ ] Should the explorer live in the existing repo (as `explorer/` directory) or as a separate linked repo?
- [ ] Is the existing Dash viewer embeddable via iframe, or should we build a React-native globe component?
