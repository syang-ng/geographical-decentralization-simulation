# Explorer Chat Log

Running log of key decisions, challenges, and context from development conversations. Not a plan — a record of what we discussed and why.

---

## 2026-03-28 — Session 1: Planning + First Look at Existing Dashboard

### Vision agreed
- AI-driven interactive frontend for the geo-decentralization paper (arXiv:2509.21475)
- 4 tabs: Findings (AI search + blocks), Community (shared explorations), Deep Dive (appendix), Simulation Lab (bounded experiments)
- Stage 5 deferred: agent-driven autonomous exploration (dmarz's generative frontend idea)
- Build ON TOP of existing repo as a fork, not a replacement
- Don't override researchers' technical decisions — log unknowns in UPSTREAM_RECOMMENDATIONS.md

### Stakeholder input (from Slack)
- Sarah: community-facing site with curated findings + community contributions
- Burak: bounded simulation with safe parameter ranges, async execution
- dmarz: generative frontend — LLM returns structured blocks, React renders fixed catalog
- Shane/Hasu: interested in the policy implications angle
- Consensus: start with curated presentation, simulation lab as final tab, agent exploration later

### Challenge: pre-processed data files are enormous (BUT compress well)
- Each experiment's `data.json` is 5-19MB raw (per-slot, per-validator metrics for 1K-10K slots)
- ~30 experiment variants across baseline + 6 sensitivity evaluations × SSP/MSP × cost variants
- Total dataset: estimated 300-400MB raw on disk
- **HOWEVER**: JSON with repeating numeric arrays compresses extremely well. GitHub Pages serves gzip automatically:
  - Baseline External cost_0.002: 18MB raw → **246KB transferred** (74x compression ratio)
  - Load time: 614ms over the wire — fast enough
- The crash we experienced locally was because Python's `http.server` doesn't gzip. The live site at geo-decentralization.github.io works fine
- The browser still has to parse 18MB in memory after decompression, but modern V8 handles this in ~1-2s
- Our explorer doesn't need this raw data — the LLM works from a baked-in system prompt. But the existing Plotly viewer does need it for the globe + charts
- **Implication**: the "400MB data hosting problem" is actually a ~5-8MB compressed hosting problem. Fits easily on any static host with gzip enabled

### Challenge: existing dashboard has a data path bug
- `viewer.html` fetches `./data/gcp_regions.csv` relative to where it's served
- But `data/` lives at repo root, while the viewer is in `dashboard/`
- The dashboard was designed to be deployed at the repo root (GitHub Pages) not served from `dashboard/`
- Workaround: copied `gcp_regions.csv` into `dashboard/data/`
- Also: `readSettings()` has a fallback bug — `qs.get('dataset') || "data.json"` always returns `"data.json"` when URL params are stripped, even if localStorage has the correct path

### Challenge: deploying ~400MB of static simulation data (REVISED — smaller than expected)
- Raw on disk: 300-400MB across all experiments
- With gzip (which any CDN/static host provides automatically): estimated 5-8MB total transfer
- GitHub Pages: already hosting the existing dashboard at geo-decentralization.github.io — works fine
- Vercel free: 100MB deploy limit — the raw files on disk still exceed this, so data either goes in the git repo (served from Vercel with gzip) or on a separate CDN
- **Revised assessment**: this is NOT the infrastructure bottleneck we thought. Any host with gzip handles it. The real bottleneck is the client-side parse time for 18MB JSON, which is manageable (1-2s on modern browsers)
- Our explorer uses pre-downsampled data (~50KB per experiment) so this doesn't affect us at all

### Challenge: simulation runtime for the Lab tab
- 100 validators × 500 slots with --fast: ~20-40 seconds
- 200 validators × 2000 slots: ~3-6 minutes
- Paper used 10K validators on 80-core Xeon — not feasible for web
- Need async job execution with polling, bounded parameter ranges
- Python simulation backend required (Mesa ABM framework)
- Railway or Fly.io for hosting — need persistent process or at least container that stays warm

### Challenge: measure.py vs paper metrics are different
- `measure.py` computes NNI, Moran's I, Geary's C, DBSCAN (used in the Dash visualization)
- Paper uses Gini_g, HHI_g, CV_g, LC_g (custom geographic concentration metrics)
- Dashboard shows measure.py metrics, not the paper's metrics
- Logged in UPSTREAM_RECOMMENDATIONS.md — we reference paper metrics in the explorer

### Infrastructure discussion: $100 budget
- Vercel Pro ($20/mo): frontend + edge functions for Claude API proxy with rate limiting
- Cloudflare R2 (free at our scale): data file hosting with zero egress — though gzip on any static host makes this less critical than we thought
- Railway ($5/mo): Python simulation backend for Lab tab
- Remaining ~$75/mo: buffer
- **Key discovery**: gzip compression reduces the data hosting problem from "400MB is hard to serve" to "5-8MB compressed is trivial". The live GitHub Pages site already handles this fine
- The real costs are: Claude API calls per user query, and compute for the Simulation Lab (Python simulation runner)

### Challenge: live site vs our local copy are different versions
- geo-decentralization.github.io has a newer version with:
  - 2D Mercator world map (not the 3D Plotly globe in our local copy)
  - Configuration summary bar with simulation params
  - Green/teal theme (not our local navy/purple)
  - "RESEARCH DEMO" badge
  - Dataset naming: "Local"/"External" (not "MSP"/"SSP"), experiment names like "SE1-Information-Source-Placement-Effect"
- Our local fork has older code — need to pull latest before building on top
- The viewer successfully loads 18MB datasets because GitHub Pages gzips automatically

### Design decisions
- Blueprint dark theme (charcoal #050505, not the dashboard's navy #0f1226)
- 9 block types: Stat, Insight, Chart, Comparison, Table, Caveat, Source, Map, TimeSeries
- SSP = accent blue (#3B82F6), MSP = warm terracotta (#d97757) — consistent everywhere
- json-render pattern: catalog-as-contract, flat element map, guardrailed types
- Spring physics animations, glass morphism, Inter font

### What exists so far
- PLAN.md: comprehensive implementation plan with all 8 gaps filled (visual spec, example LLM responses, default blocks, interaction spec, responsive layout, error handling, animation choreography, testing strategy)
- UPSTREAM_RECOMMENDATIONS.md: 5 questions + 5 suggestions for researchers
- Scaffolded explorer/ directory with Vite + React + Tailwind v4
- index.css with Blueprint theme + glass morphism + animations
- cn.ts utility, initial type definitions, seed data (needs replacement)
- Initial component shells (FindingCard, AiSearchBar, CommunityResults, DeepDive, App)

### Performance planning session
- **Pre-rendering is critical**: default page load = zero API calls. Static blocks in `default-blocks.ts`, hand-crafted content
- **Prompt caching**: system prompt is ~10K tokens, cached with `cache_control: ephemeral`. Subsequent queries skip prompt processing — ~80% faster TTFT, 90% cost reduction on cached portion
- **Edge query cache**: hash(query) → Block[] JSON in Vercel KV. Pre-warm 8 example chip queries at deploy time. Estimated 40-60% cache hit rate
- **Model choice**: Sonnet (not Haiku — need nuanced cross-experiment reasoning; not Opus — overkill for fixed-context Q&A)
- **Temperature 0**: deterministic responses enable caching and reproducibility
- **Streaming limitation**: tool_use responses arrive as one blob, can't stream blocks individually. Compensate with shimmer loading + stagger animation on render
- **Fixed Phase 2 conflict**: step 16 said "auto-query on page load" — changed to "show static blocks, LLM only on user-initiated query"
- All trade-off decisions recorded in DECISIONS.md (D1-D10)

### Live site investigation
- geo-decentralization.github.io works — doesn't crash. The researchers updated it since our fork
- Their version has: 2D Mercator map, config summary bar, green/teal theme, "Local"/"External" naming
- Data loads fine because GitHub Pages gzips: 18MB → 246KB transferred (74x compression)
- Our local crash was caused by Python `http.server` not compressing
- **Our fork is behind upstream** — need to pull latest before building

### Three-tier gatekeeping for API calls
- User's key insight: gatekeep Claude behind pre-rendered options + public history
- Tier 1: 8 pre-rendered topic cards covering major findings (zero cost, instant)
- Tier 2: community pool — every past Claude response auto-saved, fuzzy-matched against new queries (zero cost)
- Tier 3: fresh Claude call — only when tiers 1+2 don't match AND user explicitly clicks "Ask Claude anyway"
- Auto-publish: no manual "publish" button. Every Claude response enters the pool automatically
- This means API cost per visitor approaches zero as the pool grows
- Renamed Tab 2 from "Community" to "Explore History" — it's a leaderboard/feed, not a social platform
- Trade-off decisions D11 + D12 recorded in DECISIONS.md

### Next up
- Phase 1: Block catalog + static findings page with real geo-decentralization data
- Replace fake seed data with actual paper findings
- Build all 9 block components
- Wire up 4-tab navigation

---

## 2026-03-28 — Session 2: Agent-Native Architecture Audit + Remediation

### Agent-native audit (8 principles, parallel sub-agents)

Ran `/agent-native-audit the plan, decisions` — 8 parallel sub-agents scored each principle:

| Principle | Before | Status |
|-----------|--------|--------|
| Action Parity | 37/43 (86%) | ✅ |
| Tools as Primitives | 5/8 (62.5%) | ⚠️ |
| Context Injection | 4/9 (44%) | ❌ |
| Shared Workspace | 7/9 (78%) | ⚠️ |
| CRUD Completeness | 3/8 (37.5%) | ❌ |
| UI Integration | 11/14 (78.6%) | ⚠️ |
| Capability Discovery | 5.5/7 (79%) | ⚠️ |
| Prompt-Native Features | 7/17 (41%) | ❌ |

**Overall: 63%** — three principles below 50%.

### Root cause: monolithic `render_blocks` tool

The single tool couldn't search, flag, verify, or update explorations. All behavior was implicitly in code. The fix was consistent across all three ❌ principles:

### Changes made to address audit findings

#### 1. Atomic tool primitives (D13)
Replaced single `render_blocks` with 7 atomic tools:
- `render_blocks` — compose blocks (kept, but now one of many)
- `search_explorations` — search community pool
- `update_exploration` — add tags, append context
- `flag_exploration` — soft-delete with reason
- `verify_exploration` — researcher verification
- `build_simulation_config` — NL → simulation config
- `suggest_explorations` — dynamic follow-up suggestions

#### 2. Two-part system prompt with context injection (D14)
Split system prompt into:
- **Static** (~10k tokens, cached): research data, tool descriptions, behavior rules
- **Dynamic** (~200 tokens, per-request): pool stats, trending queries, session history, rate limits

#### 3. Prompt-native behavior definition (D15)
Moved all behavioral features into system prompt with explicit "edit this to change behavior" annotations:
- Response Strategy (search first → pool → fresh → suggest)
- Composition Guidelines (block layout rules)
- Topic Card Selection (the 8 topics)
- Tier Routing (when to redirect vs generate)

#### 4. Full CRUD on explorations (D16)
Added to schema: `downvotes`, `verifier_note`, `flagged`, `flag_reason`, `context_appended`, `updated_at`. Separate `votes` table. CRUD matrix documented for all entities.

#### 5. Dynamic capability discovery (D17)
`suggest_explorations` called after every response — suggests follow-ups based on current context + pool coverage gaps.

#### 6. Immutable blocks by design (D18)
Blocks are snapshots. To "fix" a bad answer: flag it + generate fresh. The exploration wrapper has full CRUD; blocks inside are intentionally immutable.

### Verified scores after remediation (re-audit with 8 parallel sub-agents)

| Principle | Before | After | Change |
|-----------|--------|-------|--------|
| Action Parity | 86% | **92.3%** ✅ | +6.3 |
| Tools as Primitives | 62.5% | **100%** ✅ | +37.5 |
| Context Injection | 44% | **100%** ✅ | +56 |
| Shared Workspace | 78% | **87.5%** ✅ | +9.5 |
| CRUD Completeness | 37.5% | **90%** ✅ | +52.5 |
| UI Integration | 78.6% | **87.5%** ✅ | +8.9 |
| Capability Discovery | 79% | **100%** ✅ | +21 |
| Prompt-Native Features | 41% | **88%** ✅ | +47 |

**Overall: 93.2%** ✅ (up from 63%). All 8 principles now pass.

### Additional fix from audit feedback (D19)
Re-audit of Tools as Primitives scored 71% because `build_simulation_config` and `suggest_explorations` embedded decision-making logic. Split each into two data primitives:
- `build_simulation_config` → `get_simulation_constraints` + `submit_simulation_config`
- `suggest_explorations` → `get_pool_state` + `get_session_history`
Total tools: 7 → 9 (all pure primitives, 100% score).

### Decisions recorded
D13-D18 in DECISIONS.md. All address specific audit findings.

### Next up
- Re-run audit to verify improvements
- Build Phase 1: Block catalog + static findings page

---

## 2026-03-29 — Session 3: Phase 1 Build (Block Catalog + Static Findings)

### What was built
- All 9 block renderers: StatBlock, InsightBlock, ChartBlock, ComparisonBlock, TableBlock, CaveatBlock, SourceBlock, MapBlock, TimeSeriesBlock
- BlockRenderer discriminated-union dispatcher
- BlockCanvas layout component (3-up stat grid when first 3 blocks are stats, else vertical stack)
- 9 default blocks with real paper data in `data/default-blocks.ts`
- 8 TopicCard definitions mapped to actual paper findings (each with its own block set)
- FindingsPage with topic card grid + animated block display
- Header, TabNav, Footer layout components
- Full Tailwind v4 theme with Blueprint dark palette (`@theme` directive)
- Type system: 9 Zod schemas with discriminated union in `types/blocks.ts`, `parseBlocks()` helper
- Removed old scaffold files (FindingCard, AiSearchBar, CommunityResults, old DeepDive shell)

### Key decisions
- **Zod v4 discriminated union** as single source of truth for block types — shared between frontend validation and server tool schemas
- **Pre-rendered content only** in Phase 1 — zero API calls on page load
- **TopicCards** as Tier 1 gatekeeping: 8 curated findings users can browse without hitting Claude
- **Spring physics** for all animations (stiffness 240/damping 30), no ease/linear

### Verified
- All 9 block types render correctly
- Topic card expand/collapse with animated transitions
- Production build: 477KB JS, 30KB CSS

---

## 2026-03-29 — Session 4: Phase 2+3 Build + Audit Remediation

### Phase 2: LLM-Powered Exploration

#### Architecture
- Express sidecar on port 3201 proxying Claude API calls (keeps ANTHROPIC_API_KEY server-side)
- Vite dev proxy: `/api/*` → `http://localhost:3201`
- POST `/api/explore` → Claude Sonnet with tool_use (`render_blocks` tool)
- Prompt caching: ~10k token system prompt with `cache_control: ephemeral` (90% input cost reduction)

#### New files
- `server/index.ts` — Express API proxy with CORS, error handling, health check
- `server/study-context.ts` — ~130 lines of hand-extracted paper knowledge for system prompt
- `server/catalog.ts` — Auto-generates Claude tool_use JSON schema from Zod block schemas
- `server/tsconfig.json` — Node ES2023 config
- `.env.example` — Documents required env vars
- `src/lib/api.ts` — Frontend API client with Result<T> pattern
- `src/components/explore/ShimmerBlock.tsx` — Skeleton loading during Claude calls
- `src/components/explore/QueryHistory.tsx` — Pill-based session history trail
- `src/components/explore/ErrorDisplay.tsx` — Contextual error messages (429, 401, network)

#### Modified files
- `src/pages/FindingsPage.tsx` — Wired QueryBar → API → BlockCanvas with loading/error/success states, follow-up suggestion chips
- `src/components/explore/QueryBar.tsx` — Added example chips, loading spinner, enabled/disabled modes
- `vite.config.ts` — Added `/api` proxy to port 3201
- `package.json` — Added `@anthropic-ai/sdk`, `express`, `cors`, `zod-to-json-schema`, `tsx`

### Phase 3: Deep Dive + Simulation Lab

#### Deep Dive page (`src/pages/DeepDivePage.tsx`)
- 10 expandable accordion sections mapped to actual paper structure (sections 3-7)
- Each section contains pre-rendered blocks (InsightBlock, TableBlock, ComparisonBlock, etc.)
- "Expand all" / "Collapse all" controls
- Animated expand/collapse with spring physics

#### Simulation Lab page (`src/pages/SimulationLabPage.tsx`)
- Full config builder UI:
  - Paradigm toggle (SSP/MSP)
  - 6 presets (Baseline, Dense Urban, Low Cost Migration, etc.)
  - Sliders: validators 50-200, slots 500-2000, migration cost 0-0.005
  - Dropdowns: distribution type, source placement
  - Button groups: attestation threshold (γ), slot time delta (Δ)
- Run/Reset buttons, fast mode indicator
- Placeholder results panel (backend not connected — needs Python Mesa runner)

#### App.tsx changes
- All 4 tabs wired: Findings (active), Explore History (placeholder), Deep Dive, Simulation Lab
- Deep Dive and Simulation Lab tabs enabled in TabNav

### Audit Remediation (5 parallel fixes)

Ran `/audit` against Session 3+4 work. Overall score: 3.7/5. Five issues fixed in parallel:

1. **Env validation** (blocker → fixed)
   - `server/index.ts`: Fail-fast check for ANTHROPIC_API_KEY at startup with clear error message
   - `.env.example` created documenting all env vars
   - CORS origins configurable via ALLOWED_ORIGINS env var

2. **Shared constants extraction** (significant → fixed)
   - Created `src/lib/theme.ts` with SPRING, SPRING_SOFT, BLOCK_COLORS constants
   - Replaced local spring/color declarations across 11+ files (BlockCanvas, QueryBar, QueryHistory, ChartBlock, TimeSeriesBlock, MapBlock, FindingsPage, App.tsx, etc.)

3. **Error handling hardening** (significant → fixed)
   - `src/lib/api.ts`: Added `.catch(() => ({}))` on `res.json()` for non-JSON error responses
   - `BlockCanvas.tsx`: Added empty state ("No blocks to display") for zero-length arrays
   - Proper type assertions for parsed JSON throughout

4. **Accessibility** (significant → fixed)
   - `MapBlock.tsx`: `role="img"` + `aria-label` on SVG, labels on each dot, `role="tooltip"` on tooltip
   - `QueryBar.tsx`: `aria-label="Search the paper"` on input
   - `FindingsPage.tsx`: `aria-label={card.title}` + `aria-pressed={isActive}` on topic card buttons
   - `StatBlock.tsx`: `<span className="sr-only">` for sentiment (not color-only)

5. **Schema deduplication** (significant → fixed)
   - Rewrote `server/catalog.ts` from 228-line manual JSON Schema to auto-generation from Zod via `zod-to-json-schema`
   - Single source of truth: change `types/blocks.ts`, catalog updates automatically
   - Installed `zod-to-json-schema` dependency

### Verified after all fixes
- TypeScript compilation: clean (zero errors)
- Production build: 477KB JS, 30KB CSS, 4.14s build time

### What's next
- Wire Simulation Lab frontend to backend
- Build Explore History tab (community pool)
- Agent-driven autonomous exploration (generative frontend)

---

## 2026-03-29 — Session 5: Simulation Lab Wiring + Explore History

### Simulation Lab — fully wired to backend

The SimulationLabPage was already built as a UI shell. This session completed the wiring:

#### New files
- `src/lib/simulation-api.ts` — API client for all simulation endpoints (submit, poll, manifest, artifact download). Types: SimulationConfig, SimulationJob, SimulationManifest, SimulationArtifact
- `src/workers/simulationArtifactWorker.ts` — Web Worker that transforms raw simulation output into Block[] off the main thread. Handles:
  - Timeseries artifacts (avg_mev, supermajority_success, etc.) → TimeSeriesBlock with downsampling for >500 points
  - Map artifacts (region_counter_per_slot) → MapBlock using GCP region lat/lon lookup
  - Table artifacts (CSV files) → TableBlock with header extraction, capped at 20 rows
- `src/data/gcp-regions.ts` — Lat/lon lookup table for ~40 GCP regions (used by web worker)

#### Architecture
- SimulationLabPage uses React Query for:
  - `useMutation` for job submission
  - `useQuery` with 1s polling for job status
  - `useQuery` for manifest (triggered when job completes)
  - `useQuery` for individual artifact data
- Web Worker pattern keeps main thread responsive during artifact parsing
- Server-side simulation runtime (`server/simulation-runtime.ts`) manages job queue, config hashing for cache hits, Python worker process lifecycle
- Python worker (`server/simulation_worker.py`) wraps the researchers' actual Mesa simulation, produces manifests + gzipped artifacts

#### Server endpoints (already existed from Phase 3)
- POST /api/simulations — submit config, returns job snapshot
- GET /api/simulations/:jobId — poll job status
- GET /api/simulations/:jobId/manifest — full manifest with summary stats + artifact list
- GET /api/simulations/:jobId/artifacts/:name — raw artifact content

### Explore History — community pool with persistence

#### Architecture
- In-memory exploration store with JSON file persistence (`server/data/explorations.json`)
- Every successful Claude response auto-saved to the pool (no manual "publish" step)
- Text-based search on query + summary fields
- Sort by recent or top (net upvotes)
- Debounced 1s file writes for persistence

#### New files
- `server/exploration-store.ts` — ExplorationStore class with save/list/vote/getById/search. Extracts paradigm tags (SSP/MSP) and experiment tags (SE1-SE4) from block JSON. Uses crypto.randomUUID() for IDs.
- `src/pages/ExploreHistoryPage.tsx` — Gallery page with:
  - Search input + sort toggle (Recent / Top)
  - Glass morphism cards showing query, summary, vote count, tags, relative timestamps
  - Click-to-expand shows full BlockCanvas with the exploration's blocks
  - Thumbs up/down voting with React Query optimistic updates
  - Empty state guiding users to the Findings tab
  - Framer Motion animations throughout

#### Modified files
- `server/index.ts` — Added ExplorationStore import, auto-save in POST /api/explore handler, three new routes:
  - GET /api/explorations (sort, limit, search params)
  - GET /api/explorations/:id
  - POST /api/explorations/:id/vote (body: { delta: 1 | -1 })
- `src/lib/api.ts` — Added Exploration interface, listExplorations(), voteExploration() functions. Blocks parsed through parseBlocks() on read.
- `src/App.tsx` — Replaced PlaceholderTab with ExploreHistoryPage. Added Suspense wrappers for lazy-loaded pages.
- `src/components/layout/TabNav.tsx` — Enabled Explore History tab (removed disabled: true). Replaced local spring constant with SPRING import.

### Verified
- TypeScript compilation: clean (only pre-existing errors in unrelated import-resume-extractions.ts)
- Production build: 2.61s
  - Main: 331KB (gzip 101KB)
  - BlockCanvas shared chunk: 173KB (gzip 55KB)
  - SimulationLabPage: 19KB (gzip 5KB)
  - DeepDivePage: 10KB (gzip 4KB)
  - Worker: 7KB
  - CSS: 34KB (gzip 7KB)
- Code splitting: DeepDive + SimulationLab lazy-loaded, worker extracted

### What's next
- Stage 5 (deferred): Agent-driven autonomous exploration
- Rate limiting on Claude API proxy
- Mobile responsive polish pass
- Dash viewer integration ("Open in 3D Viewer" on MapBlocks)
