/**
 * Express API proxy for Claude calls.
 * Keeps the API key server-side. Frontend calls /api/explore -> curated/history routing -> Claude -> Block[].
 *
 * Start: npx tsx server/index.ts
 * Env:   ANTHROPIC_API_KEY=sk-ant-...
 */

import express from 'express'
import cors from 'cors'
import path from 'node:path'
import { existsSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import Anthropic from '@anthropic-ai/sdk'
import { STUDY_CONTEXT } from './study-context.ts'
import { buildTools } from './catalog.ts'
import { SimulationRuntime, parseSimulationRequest, type SimulationRequest } from './simulation-runtime.ts'
import { ExplorationStore, normalizeQuery, type ListOptions } from './exploration-store.ts'
import { OVERVIEW_CARD, TOPIC_CARDS, type TopicCard } from '../src/data/default-blocks.ts'

const STOP_WORDS = new Set([
  'a',
  'an',
  'and',
  'are',
  'does',
  'do',
  'for',
  'how',
  'in',
  'is',
  'it',
  'of',
  'on',
  'or',
  'the',
  'to',
  'what',
  'when',
  'where',
  'which',
  'why',
  'with',
])

const CURATED_CARDS = [OVERVIEW_CARD, ...TOPIC_CARDS]
const ALL_EXPLORATION_TOPICS = TOPIC_CARDS

const DEFAULT_SIMULATION_CONFIG: SimulationRequest = {
  paradigm: 'SSP',
  validators: 100,
  slots: 1000,
  distribution: 'uniform',
  sourcePlacement: 'homogeneous',
  migrationCost: 0.0001,
  attestationThreshold: 2 / 3,
  slotTime: 12,
  seed: 25873,
}

const SIMULATION_PRESETS = {
  'baseline-ssp': {
    paradigm: 'SSP',
    distribution: 'uniform',
    sourcePlacement: 'homogeneous',
  },
  'baseline-msp': {
    paradigm: 'MSP',
    distribution: 'uniform',
    sourcePlacement: 'homogeneous',
  },
  'latency-aligned': {
    sourcePlacement: 'latency-aligned',
  },
  'latency-misaligned': {
    sourcePlacement: 'latency-misaligned',
  },
  'heterogeneous-start': {
    distribution: 'heterogeneous',
  },
  'eip-7782': {
    slotTime: 6,
  },
} satisfies Record<string, Partial<SimulationRequest>>

const apiKey = process.env.ANTHROPIC_API_KEY
const allowedOrigins = (process.env.ALLOWED_ORIGINS ?? 'http://localhost:3200,http://localhost:5180').split(',')

const app = express()
app.use(cors({ origin: allowedOrigins }))
app.use(express.json({ limit: '1mb' }))

const client = apiKey ? new Anthropic({ apiKey }) : null
const tools = buildTools()
const simulationRuntime = new SimulationRuntime()
const explorationStore = new ExplorationStore()

interface ExploreRequest {
  query: string
  history?: Array<{ query: string; summary: string }>
}

type ExploreSource = 'curated' | 'history' | 'generated'

interface ExploreProvenance {
  source: ExploreSource
  label: string
  detail: string
  canonical: boolean
  topicId?: string
  explorationId?: string
  similarityScore?: number
}

interface ExploreResponse {
  summary: string
  blocks: unknown[]
  followUps: string[]
  model: string
  cached: boolean
  provenance: ExploreProvenance
}

interface CuratedMatch {
  card: TopicCard
  score: number
}

function tokenize(text: string): string[] {
  const normalized = normalizeQuery(text)
  if (!normalized) return []
  return normalized
    .split(' ')
    .filter(token => token.length > 1 && !STOP_WORDS.has(token))
}

function overlapScore(left: readonly string[], right: readonly string[]): number {
  if (left.length === 0 || right.length === 0) return 0
  const rightSet = new Set(right)
  let intersection = 0
  for (const token of left) {
    if (rightSet.has(token)) intersection += 1
  }
  return intersection / Math.max(left.length, right.length)
}

function scoreTopicCard(query: string, card: TopicCard): number {
  const normalizedQuery = normalizeQuery(query)
  if (!normalizedQuery) return 0

  const candidates = [card.title, card.description, ...card.prompts]
  const queryTokens = tokenize(query)
  let best = 0

  for (const candidate of candidates) {
    const normalizedCandidate = normalizeQuery(candidate)
    if (!normalizedCandidate) continue
    if (normalizedCandidate === normalizedQuery) return 1
    if (
      normalizedCandidate.includes(normalizedQuery)
      || normalizedQuery.includes(normalizedCandidate)
    ) {
      best = Math.max(best, 0.9)
    }
    best = Math.max(best, overlapScore(queryTokens, tokenize(candidate)))
  }

  return best
}

function findCuratedMatch(query: string): CuratedMatch | null {
  let best: CuratedMatch | null = null

  for (const card of CURATED_CARDS) {
    const score = scoreTopicCard(query, card)
    if (score < 0.62) continue
    if (!best || score > best.score) {
      best = { card, score }
    }
  }

  return best
}

function buildCuratedFollowUps(cardId: string): string[] {
  return TOPIC_CARDS
    .filter(card => card.id !== cardId)
    .slice(0, 3)
    .map(card => card.prompts[0] ?? card.title)
}

function buildCuratedResponse(match: CuratedMatch): ExploreResponse {
  const isOverview = match.card.id === OVERVIEW_CARD.id
  return {
    summary: match.card.title,
    blocks: match.card.blocks,
    followUps: buildCuratedFollowUps(match.card.id),
    model: '',
    cached: false,
    provenance: {
      source: 'curated',
      label: isOverview ? 'Curated overview' : 'Curated topic card',
      detail: isOverview
        ? 'Matched a canonical editorial overview of the paper.'
        : 'Matched a curated paper finding without needing a fresh model call.',
      canonical: true,
      topicId: match.card.id,
      similarityScore: Number(match.score.toFixed(2)),
    },
  }
}

function buildHistoryResponse(match: ReturnType<ExplorationStore['findBestMatch']>): ExploreResponse | null {
  if (!match) return null
  const exact = match.reason === 'exact'
  return {
    summary: match.exploration.summary,
    blocks: match.exploration.blocks,
    followUps: match.exploration.followUps,
    model: match.exploration.model,
    cached: match.exploration.cached,
    provenance: {
      source: 'history',
      label: exact ? 'Prior exploration' : 'Matched prior exploration',
      detail: exact
        ? 'Reused an exact prior exploration from the public history.'
        : 'Reused a closely related prior exploration from the public history.',
      canonical: false,
      explorationId: match.exploration.id,
      similarityScore: Number(match.score.toFixed(2)),
    },
  }
}

function findTopicMatches(query: string, limit = 5): Array<{
  id: string
  title: string
  description: string
  prompts: readonly string[]
  score: number
}> {
  const normalizedQuery = normalizeQuery(query)
  const pool = normalizedQuery ? CURATED_CARDS : ALL_EXPLORATION_TOPICS

  return pool
    .map(card => ({
      id: card.id,
      title: card.title,
      description: card.description,
      prompts: card.prompts,
      score: normalizedQuery ? scoreTopicCard(query, card) : 1,
    }))
    .filter(entry => !normalizedQuery || entry.score >= 0.22)
    .toSorted((left, right) => right.score - left.score)
    .slice(0, Math.max(1, limit))
}

function getTopicCardById(id: string): TopicCard | null {
  return CURATED_CARDS.find(card => card.id === id) ?? null
}

function explorationCoverageForCard(card: TopicCard): number {
  const cardTokens = tokenize(`${card.title} ${card.description} ${card.prompts.join(' ')}`)
  if (cardTokens.length === 0) return 0

  let matches = 0
  for (const exploration of explorationStore.list({ limit: 500 })) {
    const explorationTokens = tokenize(`${exploration.query} ${exploration.summary}`)
    if (overlapScore(cardTokens, explorationTokens) >= 0.22) {
      matches += 1
    }
  }
  return matches
}

function suggestUnderexploredTopics(query: string | undefined, limit = 3) {
  const queryTokens = query ? tokenize(query) : []

  return ALL_EXPLORATION_TOPICS
    .map(card => {
      const coverage = explorationCoverageForCard(card)
      const relevance = queryTokens.length === 0
        ? 0
        : overlapScore(queryTokens, tokenize(`${card.title} ${card.description} ${card.prompts.join(' ')}`))
      return {
        id: card.id,
        title: card.title,
        description: card.description,
        suggestedQuery: card.prompts[0] ?? card.title,
        explorationCount: coverage,
        relevance,
      }
    })
    .toSorted((left, right) => {
      if (right.relevance !== left.relevance) return right.relevance - left.relevance
      if (left.explorationCount !== right.explorationCount) return left.explorationCount - right.explorationCount
      return left.title.localeCompare(right.title)
    })
    .slice(0, Math.max(1, limit))
    .map(entry => ({
      ...entry,
      reason: entry.explorationCount === 0
        ? 'No closely matching public explorations yet.'
        : `${entry.explorationCount} closely matching public exploration${entry.explorationCount === 1 ? '' : 's'} so far.`,
    }))
}

function attestationCutoffMs(slotTime: number): number {
  if (slotTime === 6) return 3000
  if (slotTime === 8) return 4000
  return 4000
}

function paperScenarioLabels(config: SimulationRequest): string[] {
  const labels: string[] = []

  if (config.distribution === 'heterogeneous' && config.sourcePlacement !== 'homogeneous') {
    labels.push('SE3 joint heterogeneity')
  } else if (config.distribution === 'heterogeneous') {
    labels.push('SE2 heterogeneous validators')
  } else if (config.sourcePlacement === 'latency-aligned') {
    labels.push('SE1 latency-aligned sources')
  } else if (config.sourcePlacement === 'latency-misaligned') {
    labels.push('SE1 latency-misaligned sources')
  } else {
    labels.push('Baseline geography/source setup')
  }

  if (config.slotTime === 6) {
    labels.push('SE4b shorter slots')
  } else if (Math.abs(config.attestationThreshold - 2 / 3) > 0.01) {
    labels.push('SE4a gamma variation')
  }

  labels.push(config.paradigm === 'SSP' ? 'SSP exact mode' : 'MSP exact mode')
  return labels
}

function coerceNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return undefined
}

function buildSimulationConfig(input: Record<string, unknown>) {
  const presetName = typeof input.preset === 'string' ? input.preset : null
  const preset = presetName ? SIMULATION_PRESETS[presetName as keyof typeof SIMULATION_PRESETS] : undefined

  const candidate = {
    ...DEFAULT_SIMULATION_CONFIG,
    ...(preset ?? {}),
  } satisfies SimulationRequest

  const paradigm = input.paradigm
  if (paradigm === 'SSP' || paradigm === 'MSP') {
    candidate.paradigm = paradigm
  }

  const distribution = input.distribution
  if (distribution === 'uniform' || distribution === 'heterogeneous' || distribution === 'random') {
    candidate.distribution = distribution
  }

  const sourcePlacement = input.sourcePlacement
  if (
    sourcePlacement === 'homogeneous'
    || sourcePlacement === 'latency-aligned'
    || sourcePlacement === 'latency-misaligned'
  ) {
    candidate.sourcePlacement = sourcePlacement
  }

  const validators = coerceNumber(input.validators)
  if (validators !== undefined) candidate.validators = validators

  const slots = coerceNumber(input.slots)
  if (slots !== undefined) candidate.slots = slots

  const migrationCost = coerceNumber(input.migrationCost)
  if (migrationCost !== undefined) candidate.migrationCost = migrationCost

  const attestationThreshold = coerceNumber(input.attestationThreshold)
  if (attestationThreshold !== undefined) candidate.attestationThreshold = attestationThreshold

  const slotTime = coerceNumber(input.slotTime)
  if (slotTime !== undefined) candidate.slotTime = slotTime

  const seed = coerceNumber(input.seed)
  if (seed !== undefined) candidate.seed = seed

  const parsed = parseSimulationRequest(candidate)
  if (!parsed) {
    return {
      valid: false,
      error: 'The requested config is outside the exact-mode bounds.',
      defaults: DEFAULT_SIMULATION_CONFIG,
      presets: Object.keys(SIMULATION_PRESETS),
      bounds: {
        validators: '25-1000',
        slots: '50-10000',
        migrationCost: '0-0.02 ETH',
        attestationThreshold: '0 < gamma < 1',
        slotTime: [6, 8, 12],
      },
    }
  }

  return {
    valid: true,
    exactMode: true,
    config: parsed,
    preset: presetName,
    attestationCutoffMs: attestationCutoffMs(parsed.slotTime),
    scenarioLabels: paperScenarioLabels(parsed),
    notes: [
      'This config stays within the bounded exact-mode Simulation Lab limits.',
      'It composes a run configuration only; it does not execute the simulation.',
    ],
  }
}

// --- Tool executor for multi-turn Claude calls ---

function executeToolCall(
  name: string,
  input: Record<string, unknown>,
): unknown {
  if (name === 'search_topic_cards') {
    const query = typeof input.query === 'string' ? input.query : ''
    const limit = typeof input.limit === 'number' ? input.limit : 5
    return findTopicMatches(query, limit)
  }

  if (name === 'get_topic_card') {
    const id = typeof input.id === 'string' ? input.id : ''
    const card = getTopicCardById(id)
    if (!card) return { error: 'Topic card not found' }
    return {
      id: card.id,
      title: card.title,
      description: card.description,
      prompts: card.prompts,
      blocks: card.blocks,
    }
  }

  if (name === 'search_explorations') {
    const options: ListOptions = {
      search: typeof input.query === 'string' ? input.query : undefined,
      paradigm: input.paradigm as ListOptions['paradigm'],
      experiment: typeof input.experiment === 'string' ? input.experiment : undefined,
      verifiedOnly: typeof input.verified_only === 'boolean' ? input.verified_only : undefined,
      sort: (input.sort as 'recent' | 'top') ?? 'recent',
      limit: typeof input.limit === 'number' ? input.limit : 10,
    }
    const results = explorationStore.list(options)
    return results.map(e => ({
      id: e.id,
      query: e.query,
      summary: e.summary,
      votes: e.votes,
      verified: e.verified,
      paradigmTags: e.paradigmTags,
      experimentTags: e.experimentTags,
      createdAt: e.createdAt,
    }))
  }

  if (name === 'get_exploration') {
    const id = typeof input.id === 'string' ? input.id : ''
    const exploration = explorationStore.getById(id)
    if (!exploration) return { error: 'Exploration not found' }
    return exploration
  }

  if (name === 'suggest_underexplored_topics') {
    const query = typeof input.query === 'string' ? input.query : undefined
    const limit = typeof input.limit === 'number' ? input.limit : 3
    return suggestUnderexploredTopics(query, limit)
  }

  if (name === 'build_simulation_config') {
    return buildSimulationConfig(input)
  }

  if (name === 'verify_exploration') {
    const id = typeof input.id === 'string' ? input.id : ''
    const verified = typeof input.verified === 'boolean' ? input.verified : true
    const updated = explorationStore.verify(id, verified)
    if (!updated) return { error: 'Exploration not found' }
    return { id: updated.id, verified: updated.verified }
  }

  return { error: `Unknown tool: ${name}` }
}

app.post('/api/explore', async (req, res) => {
  const { query, history } = req.body as ExploreRequest
  const trimmedQuery = query?.trim()

  if (!trimmedQuery) {
    res.status(400).json({ error: 'Query is required' })
    return
  }

  const curatedMatch = findCuratedMatch(trimmedQuery)
  if (curatedMatch) {
    res.json(buildCuratedResponse(curatedMatch))
    return
  }

  const historyMatch = explorationStore.findBestMatch(trimmedQuery, 0.82)
  const reusedHistory = buildHistoryResponse(historyMatch)
  if (reusedHistory) {
    res.json(reusedHistory)
    return
  }

  if (!client) {
    res.status(503).json({ error: 'Anthropic API is not configured on this server.' })
    return
  }

  try {
    const sessionContext = history?.length
      ? `\n\n## Session Context\nPrevious queries this session:\n${history.map((entry, index) => `${index + 1}. "${entry.query}" -> ${entry.summary}`).join('\n')}\n\nBuild on prior context where relevant.`
      : ''

    // Multi-turn tool execution loop
    const messages: Anthropic.Messages.MessageParam[] = [
      { role: 'user', content: trimmedQuery },
    ]
    const MAX_TOOL_ROUNDS = 3
    let finalResponse: Anthropic.Messages.Message | null = null

    for (let round = 0; round <= MAX_TOOL_ROUNDS; round++) {
      const response = await client.messages.create({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 4096,
        system: [
          {
            type: 'text',
            text: STUDY_CONTEXT + sessionContext,
            cache_control: history?.length ? undefined : { type: 'ephemeral' },
          },
        ],
        tools,
        tool_choice: round === MAX_TOOL_ROUNDS
          ? { type: 'tool', name: 'render_blocks' }
          : { type: 'auto' },
        messages,
      })

      // Check if Claude called render_blocks (the terminal tool)
      const renderCall = response.content.find(
        (block): block is Anthropic.Messages.ToolUseBlock =>
          block.type === 'tool_use' && block.name === 'render_blocks',
      )

      if (renderCall || response.stop_reason === 'end_turn') {
        finalResponse = response
        break
      }

      // Execute intermediate tool calls and continue the loop
      const toolCalls = response.content.filter(
        (block): block is Anthropic.Messages.ToolUseBlock => block.type === 'tool_use',
      )

      if (toolCalls.length === 0) {
        finalResponse = response
        break
      }

      // Add assistant message with tool calls
      messages.push({ role: 'assistant', content: response.content })

      // Add tool results
      const toolResults: Anthropic.Messages.ToolResultBlockParam[] = toolCalls.map(call => ({
        type: 'tool_result' as const,
        tool_use_id: call.id,
        content: JSON.stringify(executeToolCall(call.name, call.input as Record<string, unknown>)),
      }))
      messages.push({ role: 'user', content: toolResults })
    }

    if (!finalResponse) {
      res.status(500).json({ error: 'Tool execution loop exhausted without a final response' })
      return
    }

    const toolUse = finalResponse.content.find(
      (block): block is Anthropic.Messages.ToolUseBlock => block.type === 'tool_use',
    )

    if (!toolUse) {
      res.status(500).json({ error: 'No tool_use in response' })
      return
    }

    const input = toolUse.input as {
      summary?: string
      blocks?: unknown[]
      follow_ups?: string[]
    }

    const result: ExploreResponse = {
      summary: input.summary ?? 'Here are the findings:',
      blocks: input.blocks ?? [],
      followUps: input.follow_ups ?? [],
      model: finalResponse.model,
      cached: finalResponse.usage?.cache_read_input_tokens
        ? finalResponse.usage.cache_read_input_tokens > 0
        : false,
      provenance: {
        source: 'generated',
        label: 'Fresh Claude response',
        detail: 'Generated a new block composition from the study context and the current question.',
        canonical: false,
      },
    }

    res.json(result)

    explorationStore.save({
      query: trimmedQuery,
      summary: result.summary,
      blocks: result.blocks,
      followUps: result.followUps,
      model: result.model,
      cached: result.cached,
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error'
    const status =
      message.includes('rate_limit') ? 429 :
      message.includes('authentication') ? 401 :
      500

    res.status(status).json({ error: message })
  }
})

app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    tools: tools.length,
    anthropicEnabled: Boolean(client),
    simulations: simulationRuntime.health(),
  })
})

app.post('/api/simulations', (req, res) => {
  const config = parseSimulationRequest(req.body)
  if (!config) {
    res.status(400).json({ error: 'Invalid simulation request payload.' })
    return
  }

  const rawClientId = typeof req.body?.clientId === 'string' ? req.body.clientId : null
  const job = simulationRuntime.submit(config, { clientId: rawClientId })
  res.status(job.status === 'completed' ? 200 : 202).json(job)
})

app.get('/api/simulations/:jobId', (req, res) => {
  const job = simulationRuntime.getJob(req.params.jobId)
  if (!job) {
    res.status(404).json({ error: 'Simulation job not found.' })
    return
  }

  res.json(job)
})

app.post('/api/simulations/:jobId/cancel', (req, res) => {
  const job = simulationRuntime.cancel(req.params.jobId)
  if (!job) {
    res.status(404).json({ error: 'Simulation job not found.' })
    return
  }

  res.json(job)
})

app.get('/api/simulations/:jobId/events', (req, res) => {
  const initialSnapshot = simulationRuntime.getJob(req.params.jobId)
  if (!initialSnapshot) {
    res.status(404).json({ error: 'Simulation job not found.' })
    return
  }

  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache, no-transform')
  res.setHeader('Connection', 'keep-alive')
  res.flushHeaders?.()

  const heartbeat = setInterval(() => {
    res.write(`: keep-alive\n\n`)
  }, 15000)

  res.write(`event: snapshot\n`)
  res.write(`data: ${JSON.stringify(initialSnapshot)}\n\n`)

  if (
    initialSnapshot.status === 'completed'
    || initialSnapshot.status === 'failed'
    || initialSnapshot.status === 'cancelled'
  ) {
    clearInterval(heartbeat)
    res.end()
    return
  }

  const unsubscribe = simulationRuntime.subscribe(req.params.jobId, snapshot => {
    res.write(`event: snapshot\n`)
    res.write(`data: ${JSON.stringify(snapshot)}\n\n`)

    if (
      snapshot.status === 'completed'
      || snapshot.status === 'failed'
      || snapshot.status === 'cancelled'
    ) {
      clearInterval(heartbeat)
      unsubscribe?.()
      res.end()
    }
  })

  req.on('close', () => {
    clearInterval(heartbeat)
    unsubscribe()
  })
})

app.get('/api/simulations/:jobId/manifest', (req, res) => {
  const manifest = simulationRuntime.getManifest(req.params.jobId)
  if (!manifest) {
    res.status(409).json({ error: 'Simulation job is not completed yet.' })
    return
  }

  res.json(manifest)
})

app.get('/api/simulations/:jobId/artifacts/:artifactName', async (req, res) => {
  try {
    const acceptEncoding = req.header('accept-encoding') ?? ''
    const { artifact, body, contentEncoding } = await simulationRuntime.readArtifact(
      req.params.jobId,
      req.params.artifactName,
      { preferGzip: /\bgzip\b/i.test(acceptEncoding) },
    )
    res.setHeader('Content-Type', `${artifact.contentType}; charset=utf-8`)
    res.setHeader('ETag', artifact.sha256)
    res.setHeader('Vary', 'Accept-Encoding')
    if (contentEncoding) {
      res.setHeader('Content-Encoding', contentEncoding)
    }
    res.send(body)
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unable to read artifact.'
    const status = message.includes('not available') ? 409 :
      message.includes('Unknown artifact') ? 404 :
      500
    res.status(status).json({ error: message })
  }
})

app.get('/api/explorations', (req, res) => {
  const sort = (req.query.sort as 'recent' | 'top') ?? 'recent'
  const limit = req.query.limit ? Number(req.query.limit) : undefined
  const search = (req.query.search as string) ?? undefined

  const explorations = explorationStore.list({ sort, limit, search })
  res.json(explorations)
})

app.get('/api/explorations/:id', (req, res) => {
  const exploration = explorationStore.getById(req.params.id)
  if (!exploration) {
    res.status(404).json({ error: 'Exploration not found' })
    return
  }
  res.json(exploration)
})

app.post('/api/explorations/:id/vote', (req, res) => {
  const delta = (req.body as { delta?: number }).delta
  if (delta !== 1 && delta !== -1) {
    res.status(400).json({ error: 'delta must be 1 or -1' })
    return
  }

  const updated = explorationStore.vote(req.params.id, delta)
  if (!updated) {
    res.status(404).json({ error: 'Exploration not found' })
    return
  }
  res.json(updated)
})

// --- Static file serving for production (Railway serves both API + SPA) ---

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const DIST_DIR = path.join(__dirname, '..', 'dist')

if (existsSync(DIST_DIR)) {
  app.use(express.static(DIST_DIR))
  // SPA fallback: serve index.html for any non-API route
  app.get('*', (_req, res) => {
    res.sendFile(path.join(DIST_DIR, 'index.html'))
  })
}

const PORT = Number(process.env.PORT ?? 3201)
app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Explorer API listening on http://localhost:${PORT}`)
})
