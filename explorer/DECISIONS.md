# Decision Log

Trade-offs we considered, what we chose, and why. Referenced by date so future sessions know the reasoning.

---

## D1: Pre-render vs generate on every page load (2026-03-28)

**Context:** The Findings page needs to show paper results. We could generate blocks via Claude on every page load, or pre-render them as static JSON.

**Options:**
1. Auto-query Claude on page load → every visitor costs ~$0.01-0.03 and waits 2-3s
2. Static hardcoded blocks → zero cost, instant render, Claude only for user-initiated queries
3. Hybrid: auto-query once, cache forever → first visitor waits, rest get cached

**Decision:** Option 2 (static blocks for default view).

**Why:** ~95% of visitors will see the default page and leave. Burning an API call for every one is wasteful and slow. The static blocks ARE the curated research presentation — that's the product. Claude is for exploration beyond the defaults, which is the power-user flow.

**Implication:** `default-blocks.ts` is hand-crafted content, not LLM-generated. This is editorial work, not engineering.

---

## D2: Sonnet vs Haiku for query responses (2026-03-28)

**Context:** When a user does ask a question, which Claude model generates the blocks?

**Options:**
1. Haiku — ~0.3s TTFT, ~$0.001/query, but simpler compositions
2. Sonnet — ~0.8s TTFT, ~$0.01-0.03/query, richer block combinations
3. Opus — ~2s TTFT, ~$0.05-0.10/query, best reasoning
4. Adaptive: Haiku for simple, Sonnet for complex (query classification overhead)

**Decision:** Sonnet as default, re-evaluate after seeing real query patterns.

**Why:** The system prompt is 8-10K tokens of dense research data. Haiku might miss nuance in cross-experiment comparisons or produce flat block compositions. Sonnet hits the sweet spot: fast enough (1.5-3s with caching), smart enough to compose meaningful multi-block responses. Opus is overkill for a fixed-context Q&A task.

**Cost math:** 1000 unique queries/month × $0.02 avg = $20/month. Affordable.

---

## D3: Prompt caching strategy (2026-03-28)

**Context:** The system prompt is ~10K tokens (paper findings, metrics, experiment results). This is the same for every query.

**Decision:** Use Anthropic's prompt caching with `cache_control: { type: 'ephemeral' }` on the system message.

**Why:** The system prompt is ~80% of the input tokens. Caching it means subsequent queries only pay for the user question (~20-50 tokens). This gives:
- ~80% reduction in TTFT for the cached portion
- 90% discount on cached input token cost
- Cache persists for 5 minutes (ephemeral) — covers burst usage within a session

**Risk:** Cache expires between sessions. First query of a new session pays full price. Acceptable — it's still only ~1s extra.

---

## D4: Edge query cache for common questions (2026-03-28)

**Context:** Many users will ask similar questions ("What's the main finding?", "SSP vs MSP?"). Should we cache responses?

**Decision:** Yes — hash query → cache Block[] JSON at the edge. Pre-warm the 8 example chip queries at deploy time.

**Why:** If 50% of queries are example chips or common questions, we halve our API costs and give those users instant responses. The paper findings don't change, so cached responses stay valid indefinitely.

**Implementation:** Vercel KV or Cloudflare KV. Simple key-value: `sha256(normalized_query) → { summary, blocks, cached_at }`. TTL 7 days.

---

## D5: Data file hosting — gzip changes everything (2026-03-28)

**Context:** Raw simulation data is 300-400MB across 30 experiment variants. We thought this was a hosting problem.

**Discovery:** GitHub Pages (and every CDN) serves gzip automatically. The 18MB baseline JSON compresses to 246KB on the wire (74x ratio). Total dataset: ~5-8MB compressed.

**Decision:** Not a separate infrastructure concern. Any static host handles this. Our explorer doesn't serve these files anyway — it uses pre-downsampled data or static blocks.

**Implication:** Killed the Cloudflare R2 requirement. The data can live in the git repo and be served by whatever hosts the site.

---

## D6: Our explorer vs the existing dashboard — complementary, not replacement (2026-03-28)

**Context:** The existing Plotly dashboard at geo-decentralization.github.io works. Why build something new?

**Decision:** The explorer is a different product for a different audience.

**Existing dashboard:**
- For researchers who want raw time-series data across all 13 metrics
- Requires choosing an experiment, understanding parameters, interpreting charts
- Shows measure.py metrics (NNI, Moran's I), not the paper's metrics (Gini_g, HHI_g)
- Interactive 2D map + playback — powerful but requires expertise

**Our explorer:**
- For anyone curious about the paper's findings
- AI explains what the results mean, not just what the numbers are
- Shows the paper's actual metrics and conclusions
- Zero-configuration: land on page → see the story
- Power users can ask deeper questions or run bounded simulations
- Community can share and build on explorations

**We link TO the dashboard** from MapBlock components ("Open in 3D Viewer") for users who want the raw data experience. We don't replace it.

---

## D7: Build on fork vs alongside (2026-03-28)

**Context:** Should the explorer live inside the researchers' repo or as a separate project?

**Decision:** Fork the repo, add `explorer/` as a subdirectory.

**Why:**
- Explorer can reference data files directly (`../data/gcp_regions.csv`)
- Single repo for deployment
- Researchers' code untouched — we only add, never modify
- Can submit upstream PRs for suggestions (UPSTREAM_RECOMMENDATIONS.md)

**Risk:** Our local fork diverges from upstream. Need to periodically pull. The live site already has newer code (2D Mercator map, different naming conventions) than our local copy.

---

## D8: Temperature 0 for deterministic responses (2026-03-28)

**Context:** Should Claude's responses vary for the same question?

**Decision:** Temperature 0.

**Why:**
- Makes edge caching reliable (same query → same response)
- Research findings are factual — no benefit from creative variation
- Reproducible: "I got this answer" → others get the same answer
- Enables the community tab: shared explorations are stable

**Trade-off:** Slightly less "conversational" feel. But the blocks format already provides the visual variety — we don't need the text to vary.

---

## D9: $100/month infrastructure allocation (2026-03-28)

**Context:** What's the optimal spend for a production deployment?

**Decision:**
| Item | Cost | Justification |
|------|------|---------------|
| Vercel Pro | $20/mo | 1TB bandwidth, 1M edge invocations/day, custom domain |
| Railway Hobby | $5/mo | Python simulation backend (Lab tab only) |
| Vercel KV | $0 (free tier: 3K reqs/day) | Query response cache |
| Domain | ~$1/mo amortized | Optional |
| **Total infra** | **~$26/mo** | |
| **Remaining** | **~$74/mo** | Claude API budget = ~3,700 Sonnet queries |

**Key insight:** Infrastructure is cheap. The real cost is Claude API calls, which scale with usage. At $0.02/query, $74/mo supports ~3,700 unique queries. With edge caching (estimated 50% hit rate), that's ~7,400 user queries/month — plenty for a research paper explorer.

---

## D10: Streaming tool_use responses (2026-03-28)

**Context:** Can we stream blocks to the user as Claude generates them?

**Reality check:** Claude's tool_use responses arrive as a single JSON blob when the tool call completes — not incrementally. Unlike text streaming where tokens flow one by one, the `render_blocks` tool input is emitted all at once.

**Decision:** Show shimmer/skeleton loading state during generation, then stagger-animate all blocks in at once.

**Why:** We can't meaningfully stream block-by-block because the tool_use payload arrives complete. The perceived speed comes from:
1. Fast TTFT via prompt caching (~0.3-0.8s)
2. Short generation via response size cap (3-6 blocks, max 2048 tokens)
3. Stagger animation on render (0.06s per block) gives a progressive reveal feel even though all data arrived simultaneously

**Alternative considered:** Multiple sequential tool calls (one per block). Rejected — adds latency from multiple round-trips and complicates the API contract.

---

## D11: Three-tier gatekeeping for Claude API calls (2026-03-28)

**Context:** User wants to minimize unnecessary API calls. Most visitors ask questions that have already been answered or are covered by the curated findings.

**Decision:** Three tiers, each checked in order. Claude only fires if the first two don't satisfy the query.

| Tier | What | API cost | Latency |
|------|------|----------|---------|
| 1. Pre-rendered cards | 8 curated topic cards covering the paper's major findings. Click to expand. | $0 | <100ms |
| 2. Community pool | Every past Claude response is auto-saved. Fuzzy-match incoming query against existing answers. Show matches with "View these" option | $0 | <200ms |
| 3. Fresh Claude call | Only if tiers 1+2 don't match, and user explicitly opts in ("Ask Claude anyway") | ~$0.02 | 1-3s |

**Why this compounds:**
- Day 1: all queries are tier 3 (fresh). ~$0.02 each
- Day 30: most queries match tier 2 (community pool has grown). ~$0.00 each
- Steady state: API cost per visitor approaches zero as the community pool saturates the question space

**Auto-publish, not manual publish:** Every Claude response automatically enters the community pool. No friction, no "publish" button to forget. This maximizes the deduplication effect.

**UX for tier 2 match:** When the user types a question and similar community answers exist, show them inline with an explicit escape hatch: "View these" or "Ask Claude anyway ⚡". The escape hatch is important — if someone asks a question that's *close* to an existing one but not exactly right, they need the option to get a fresh answer. That fresh answer then enriches the pool for the next person.

**Fuzzy matching (MVP):** Normalize query (lowercase, strip punctuation, remove stop words), compute token overlap with existing queries. Threshold: >60% overlap → show as match. Later: query embeddings for semantic similarity.

---

## D12: Rename "Community" tab to "Explore History" (2026-03-28)

**Context:** The original plan had a "Community" tab for manually published explorations. With auto-publishing, it's really a history/leaderboard.

**Decision:** Tab 2 is now "Explore History" — a public feed of every question asked, sorted by popularity/recency, with upvotes and researcher verification.

**Why:** "Community" implies manual curation and social features (profiles, comments). "Explore History" is what it actually is: a living record of what people have asked, surfacing the most useful explorations. It also doubles as the deduplication layer — if you can see what others have already asked, you're less likely to ask the same thing.

**This replaces Phase 4's community features.** No separate publish flow needed. The history builds itself from usage. Upvotes, verification badges, and filtering by topic/paradigm/experiment are the only social features needed.

---

## D13: Atomic tools instead of monolithic render_blocks (2026-03-28)

**Context:** The original plan had a single `render_blocks` tool. The agent-native audit scored Tools as Primitives at 62.5% and CRUD at 37.5% because one tool can't search, flag, verify, or update explorations.

**Decision:** 7 atomic tools: `render_blocks`, `search_explorations`, `update_exploration`, `flag_exploration`, `verify_exploration`, `build_simulation_config`, `suggest_explorations`.

**Why:**
- Each tool is a capability primitive — does one thing without business logic
- Full CRUD on explorations: create (auto on render), read (search), update (tags/context), delete (flag/soft-delete)
- The system prompt composes behavior from tools — changing *when* to search vs render = prompt edit, not code change
- `suggest_explorations` enables active capability discovery — the agent guides users to unexplored areas

**Trade-off:** More tool definitions = slightly larger tool schema in the API call (~500 more tokens). At $0.01/1M tokens for cached input, this costs essentially nothing.

---

## D14: Two-part system prompt — static cached + dynamic injected (2026-03-28)

**Context:** Agent-native audit scored Context Injection at 44% because the agent had no awareness of session state, pool health, or user history.

**Decision:** Split the system prompt into two parts:
1. **Static** (~10k tokens): Research data, tool descriptions, composition guidelines, tier routing. Cached with `cache_control: ephemeral`.
2. **Dynamic** (~200 tokens): Pool stats, trending queries, session history, rate limits. Injected fresh per request.

**Why:**
- The agent needs to know "there are 47 explorations in the pool, none about SE3" to make smart suggestions
- Session context (last query, blocks viewed, active tab) lets the agent avoid repeating itself
- Rate limit awareness prevents the agent from promising things it can't deliver
- The 200-token dynamic portion doesn't break prompt caching — only the static portion is cached

**Risk:** Dynamic context adds ~200 tokens per request. At current pricing this is negligible. The real risk is stale context if the pool changes between requests, but this is acceptable for a non-real-time research tool.

---

## D15: Prompt-native behavior definition (2026-03-28)

**Context:** Agent-native audit scored Prompt-Native Features at 41% because block composition rules, tier routing, and topic selection were implicitly assumed to be in code.

**Decision:** All behavioral features are defined in the system prompt with explicit section headers and "edit this to change behavior" annotations:
- **Response Strategy**: search first → pool match → fresh render → suggest follow-ups
- **Composition Guidelines**: block layout rules (lead with stats, use comparison for SSP vs MSP, max 6 blocks)
- **Topic Card Selection**: the 8 topics and overlap detection logic
- **Tier Routing**: when to redirect to a topic card vs serve from pool vs generate fresh

**Why:**
- Changing the explorer's behavior = editing prompt text, not deploying code
- Non-engineers (researchers) can adjust composition rules, topic cards, and routing without touching TypeScript
- Visual rendering (CSS, animations, grid layout) stays in code where it belongs — you can't prompt-define Tailwind classes
- Clear separation: *what to do* lives in the prompt, *how it looks* lives in code

**Implication:** The prompt IS the product spec. Versioning the prompt is as important as versioning the code.

---

## D16: Full CRUD on explorations via agent tools (2026-03-28)

**Context:** Agent-native audit scored CRUD Completeness at 37.5% because explorations could only be created and read, not updated, flagged, or verified by the agent.

**Decision:** Full CRUD:
- **Create**: Automatic — every `render_blocks` response is auto-saved to the pool
- **Read**: `search_explorations` with filters (paradigm, experiment, verified_only)
- **Update**: `update_exploration` — add tags, append context, correct metadata
- **Delete**: `flag_exploration` — soft delete with reason (inaccurate, outdated, misleading, duplicate)
- **Verify**: `verify_exploration` — researcher marks as confirmed accurate

**Schema additions:** `downvotes`, `verifier_note`, `flagged`, `flag_reason`, `context_appended`, `updated_at` columns on the explorations table. Separate `votes` table for idempotent up/downvoting.

**Why:**
- The agent can maintain pool quality (flag bad answers, verify good ones)
- Researchers get a verification workflow without building a separate admin panel
- Tags and context can be added after the fact as the pool evolves
- Soft delete preserves audit trail — flagged explorations are hidden, not destroyed

---

## D17: Dynamic capability discovery via suggest_explorations (2026-03-28)

**Context:** Agent-native audit scored Capability Discovery at 79%. Static example chips are good, but they don't adapt to what's already been explored.

**Decision:** The `suggest_explorations` tool is called after every response. The agent suggests 2-3 follow-up questions based on:
- What the user just asked (topical continuity)
- What's NOT in the pool yet (coverage gaps from dynamic context)
- What's trending (popular questions others are asking)

**Why:**
- Discovery becomes dynamic, not static — suggestions change as the pool grows
- The agent actively guides users toward unexplored areas of the paper
- Combined with pool state injection, the agent can say "No one has explored SE3 yet — try asking about joint heterogeneity"
- This is the key mechanism for broadening the pool's coverage over time

**Trade-off:** Every response now includes an extra tool call (`suggest_explorations`). Adds ~100 tokens to output. Negligible cost, significant UX improvement.

---

## D18: Blocks are intentionally immutable (2026-03-28)

**Context:** Agent-native audit flagged that blocks lack Update and Delete operations (only 2/4 CRUD).

**Decision:** Blocks within an exploration are intentionally immutable. They represent a point-in-time answer. To "update" an answer, you flag the old exploration and generate a new one.

**Why:**
- An exploration is a snapshot: "On this date, this question produced these blocks"
- Mutating blocks would break the integrity of shared explorations (someone upvoted specific content)
- The `update_exploration` tool handles the real need: adding tags, context, and corrections *around* the blocks without modifying the blocks themselves
- If a block-level answer is wrong, `flag_exploration` marks it and a fresh query generates the corrected version

**Implication:** The CRUD score for blocks (2/4) is by design, not a gap. The exploration entity wrapping the blocks has full CRUD.

---

## D19: Split workflow tools into pure primitives (2026-03-28)

**Context:** Re-audit of Tools as Primitives scored 71.4% (5/7). The auditor correctly identified that `build_simulation_config` and `suggest_explorations` embed decision-making logic that should live in the system prompt.

**The primitive test:** Can you change the behavior by editing the prompt alone? If no, the tool has leaked business logic.

**Decision:** Split both workflow tools into data primitives:

**`build_simulation_config` → 2 primitives:**
- `get_simulation_constraints` — pure read: returns valid parameter ranges
- `submit_simulation_config` — pure write: validates config against constraints, enqueues job
- NL→config translation moves to system prompt ("Building Simulation Configs" section)

**`suggest_explorations` → 2 primitives:**
- `get_pool_state` — pure read: returns pool stats, coverage gaps, trending queries
- `get_session_history` — pure read: returns user's session queries and viewed blocks
- Suggestion composition moves to system prompt ("Generating Follow-Up Suggestions" section)

**Why:**
- All 9 tools are now pure capability primitives — zero business logic
- Changing how NL maps to simulation params = prompt edit (researchers can adjust)
- Changing suggestion ranking = prompt edit (prioritize gaps vs trending vs continuity)
- Tool definitions are stable; behavioral evolution happens entirely in the prompt

**Total tool count:** 7 → 9 (two workflows each became two primitives). Net +2 tools, but 100% primitive score vs 71.4% before.
