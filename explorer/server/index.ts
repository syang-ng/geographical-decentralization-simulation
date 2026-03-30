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
import fs from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import Anthropic from '@anthropic-ai/sdk'
import { STUDY_CONTEXT, SIMULATION_COPILOT_CONTEXT } from './study-context.ts'
import { buildTools } from './catalog.ts'
import { SimulationRuntime, parseSimulationRequest, type SimulationRequest } from './simulation-runtime.ts'
import { ExplorationStore, normalizeQuery, type ListOptions } from './exploration-store.ts'
import { OVERVIEW_CARD, TOPIC_CARDS, type TopicCard } from '../src/data/default-blocks.ts'
import {
  buildSimulationArtifactBundle,
  buildSimulationSummaryChart,
  parseSimulationArtifactToBlocks,
  parseSimulationBlockBundle,
  type SimulationRenderableArtifact,
} from '../src/lib/simulation-artifact-blocks.ts'
import { parseBlocks, type Block } from '../src/types/blocks.ts'
import {
  type SimulationArtifactBundle,
  type SimulationChartMetricKey,
  simulationViewSpecSchema,
  type SimulationViewSection,
  type SimulationViewSpec,
} from '../src/types/simulation-view.ts'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const EXPLORER_ROOT = path.resolve(__dirname, '..')
const DEFAULT_ANTHROPIC_MODEL = 'claude-sonnet-4-20250514'
const MISSING_ANTHROPIC_CONFIG_MESSAGE = 'Anthropic API is not configured on this server. Add ANTHROPIC_API_KEY to explorer/.env or your shell environment.'

const envFile = path.join(EXPLORER_ROOT, '.env')
if (existsSync(envFile)) {
  process.loadEnvFile(envFile)
}

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
const MAX_GENERATED_BLOCKS = 6
const MAX_GENERATED_FOLLOW_UPS = 3
const DEFAULT_GENERATED_SOURCE_BLOCK: Block = {
  type: 'source',
  refs: [
    {
      label: 'arXiv:2509.21475',
      section: 'Geo-decentralization study',
      url: 'https://arxiv.org/abs/2509.21475',
    },
    {
      label: 'Simulation repository',
      section: 'Code and datasets',
      url: 'https://github.com/syang-ng/geographical-decentralization-simulation',
    },
  ],
}
const DEFAULT_GENERATED_CAVEAT_BLOCK: Block = {
  type: 'caveat',
  text: 'Assistant framing is secondary to the cited paper context and any exact simulation outputs shown alongside it.',
}

const DEFAULT_SIMULATION_CONFIG: SimulationRequest = {
  paradigm: 'SSP',
  validators: 1000,
  slots: 1000,
  distribution: 'homogeneous',
  sourcePlacement: 'homogeneous',
  migrationCost: 0.0001,
  attestationThreshold: 2 / 3,
  slotTime: 12,
  seed: 25873,
}

const PAPER_REFERENCE_OVERRIDES = {
  validators: 1000,
  slots: 10000,
  distribution: 'homogeneous',
  sourcePlacement: 'homogeneous',
  migrationCost: 0.002,
  attestationThreshold: 2 / 3,
  slotTime: 12,
} satisfies Partial<SimulationRequest>

const SIMULATION_PRESETS = {
  'baseline-ssp': {
    ...PAPER_REFERENCE_OVERRIDES,
    paradigm: 'SSP',
  },
  'baseline-msp': {
    ...PAPER_REFERENCE_OVERRIDES,
    paradigm: 'MSP',
  },
  'latency-aligned': {
    ...PAPER_REFERENCE_OVERRIDES,
    sourcePlacement: 'latency-aligned',
  },
  'latency-misaligned': {
    ...PAPER_REFERENCE_OVERRIDES,
    sourcePlacement: 'latency-misaligned',
  },
  'heterogeneous-start': {
    ...PAPER_REFERENCE_OVERRIDES,
    distribution: 'heterogeneous',
  },
  'eip-7782': {
    ...PAPER_REFERENCE_OVERRIDES,
    slotTime: 6,
  },
} satisfies Record<string, Partial<SimulationRequest>>

const apiKey = process.env.ANTHROPIC_API_KEY
const anthropicModel = process.env.ANTHROPIC_MODEL?.trim() || DEFAULT_ANTHROPIC_MODEL
const allowedOrigins = (process.env.ALLOWED_ORIGINS ?? 'http://localhost:3200,http://localhost:5180').split(',')
const MAX_EXPLORE_QUERY_LENGTH = Math.max(120, Number(process.env.MAX_EXPLORE_QUERY_LENGTH ?? 600))
const MAX_SIMULATION_QUESTION_LENGTH = Math.max(160, Number(process.env.MAX_SIMULATION_QUESTION_LENGTH ?? 900))
const MAX_SESSION_HISTORY_ENTRIES = Math.max(1, Number(process.env.MAX_SESSION_HISTORY_ENTRIES ?? 6))
const MAX_SESSION_SUMMARY_LENGTH = Math.max(80, Number(process.env.MAX_SESSION_SUMMARY_LENGTH ?? 280))
const MAX_CLIENT_ID_LENGTH = Math.max(16, Number(process.env.MAX_CLIENT_ID_LENGTH ?? 128))

function getRequesterId(req: express.Request): string {
  const forwarded = req.headers['x-forwarded-for']
  if (typeof forwarded === 'string' && forwarded.trim()) {
    return forwarded.split(',')[0]!.trim()
  }
  return req.ip || 'unknown'
}

function createRateLimitMiddleware(
  label: string,
  windowMs: number,
  maxRequests: number,
): express.RequestHandler {
  const buckets = new Map<string, RateLimitBucket>()

  return (req, res, next) => {
    const now = Date.now()
    const requesterId = `${label}:${getRequesterId(req)}`
    const bucket = buckets.get(requesterId)

    if (!bucket || bucket.resetAt <= now) {
      buckets.set(requesterId, { count: 1, resetAt: now + windowMs })
      next()
      return
    }

    if (bucket.count >= maxRequests) {
      const retryAfterSeconds = Math.max(1, Math.ceil((bucket.resetAt - now) / 1000))
      res.setHeader('Retry-After', String(retryAfterSeconds))
      res.status(429).json({
        error: `Too many ${label} requests from this client. Retry in about ${retryAfterSeconds} seconds.`,
      })
      return
    }

    bucket.count += 1
    next()
  }
}

const exploreRateLimit = createRateLimitMiddleware('explore', 60_000, 24)
const simulationCopilotRateLimit = createRateLimitMiddleware('simulation-copilot', 60_000, 18)
const simulationSubmitRateLimit = createRateLimitMiddleware('simulations', 60_000, 10)

const app = express()
app.set('trust proxy', true)
app.use(cors({ origin: allowedOrigins }))
app.use(express.json({ limit: '1mb' }))

const client = apiKey ? new Anthropic({ apiKey }) : null
const allTools = buildTools()
const exploreTools = allTools.filter(tool => tool.name !== 'render_simulation_view_spec')
const simulationCopilotTools = allTools.filter(
  tool => tool.name !== 'render_blocks' && tool.name !== 'verify_exploration',
)
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

interface SimulationCopilotRequest {
  question: string
  currentJobId?: string | null
  currentConfig?: SimulationRequest | null
}

interface SimulationCopilotResponse {
  summary: string
  mode: SimulationViewSpec['mode']
  guidance?: string
  truthBoundary: {
    label: string
    detail: string
  }
  suggestedPrompts: string[]
  proposedConfig?: SimulationRequest
  viewSpec: SimulationViewSpec
  blocks: readonly Block[]
  model: string
  cached: boolean
}

interface RateLimitBucket {
  count: number
  resetAt: number
}

interface CuratedMatch {
  card: TopicCard
  score: number
}

const TOPIC_HINTS: Partial<Record<TopicCard['id'], readonly string[]>> = {
  'ssp-vs-msp': ['ssp', 'msp', 'external', 'local', 'compare', 'comparison', 'paradigm'],
  'geographic-convergence': ['where', 'geographic', 'geography', 'concentrate', 'concentration', 'convergence', 'regions'],
  'source-placement': ['source', 'sources', 'placement', 'relay', 'aligned', 'misaligned', 'latency aligned', 'latency misaligned'],
  'initial-distribution': ['start', 'starting', 'initial', 'begin', 'begins', 'heterogeneous', 'distribution', 'real ethereum'],
  'attestation-threshold': ['attestation', 'threshold', 'gamma'],
  'shorter-slots': ['shorter', 'slot', 'slots', '6s', '6 second', '6-second', 'eip-7782'],
  'metrics-explained': ['metric', 'metrics', 'gini', 'hhi', 'cv', 'lc'],
  limitations: ['limitation', 'limitations', 'caveat', 'caveats', 'assumption', 'assumptions', 'next', 'future'],
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

function hintScore(query: string, card: TopicCard): number {
  const normalizedQuery = normalizeQuery(query)
  if (!normalizedQuery) return 0

  const hints = TOPIC_HINTS[card.id]
  if (!hints?.length) return 0

  let matches = 0
  for (const hint of hints) {
    const normalizedHint = normalizeQuery(hint)
    if (!normalizedHint) continue
    if (normalizedQuery.includes(normalizedHint)) {
      matches += 1
    }
  }

  if (matches === 0) return 0
  return Math.min(0.36, 0.18 + (matches - 1) * 0.08)
}

function scoreTopicCard(query: string, card: TopicCard): number {
  const normalizedQuery = normalizeQuery(query)
  if (!normalizedQuery) return 0

  const candidates = [card.title, card.description, ...card.prompts]
  const queryTokens = tokenize(query)
  let best = hintScore(query, card)

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
    const minimumScore = hintScore(query, card) > 0 ? 0.54 : 0.62
    if (score < minimumScore) continue
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

function collapseWhitespace(value: string | undefined | null): string {
  return value?.replace(/\s+/g, ' ').trim() ?? ''
}

function limitText(value: string | undefined | null, maxChars: number): string {
  const normalized = collapseWhitespace(value)
  if (normalized.length <= maxChars) return normalized
  return `${normalized.slice(0, Math.max(0, maxChars - 1)).trimEnd()}…`
}

function trimTrailingPunctuation(value: string): string {
  return value.replace(/[.?!:;,]+$/g, '').trim()
}

function normalizeInterpretiveTitle(rawTitle: string | undefined): string {
  const title = limitText(rawTitle, 72)
  if (!title) return 'Guide interpretation'
  if (/^(guide interpretation|interpretation|assistant framing|reading note)/i.test(title)) {
    return title
  }
  return limitText(`Guide interpretation: ${title}`, 72)
}

function qualifyPaperSummary(rawSummary: string | undefined, fallback = ''): string {
  const summary = limitText(rawSummary, 140)
  const safeFallback = limitText(fallback, 140)
  if (!summary) return safeFallback ?? ''
  if (
    /^(paper-backed reading:|from the paper\b|based on the paper\b|what the paper suggests:)/i.test(summary)
    || /\b(shows|suggests|indicates|points to)\b/i.test(summary)
  ) {
    return summary
  }
  return limitText(`Paper-backed reading: ${summary}`, 140) || safeFallback || 'Paper-backed exploration'
}

function stablePriority(type: Block['type']): number {
  switch (type) {
    case 'stat':
      return 0
    case 'comparison':
    case 'chart':
    case 'timeseries':
    case 'map':
    case 'table':
      return 1
    case 'insight':
      return 2
    case 'caveat':
      return 3
    case 'source':
      return 4
    default:
      return 5
  }
}

function blockSignature(block: Block): string {
  return JSON.stringify(block)
}

function orderBlocksEvidenceFirst(blocks: readonly Block[]): Block[] {
  return blocks
    .map((block, index) => ({
      block: block.type === 'insight'
        ? { ...block, title: normalizeInterpretiveTitle(block.title) }
        : block,
      index,
    }))
    .toSorted((left, right) => {
      const priorityGap = stablePriority(left.block.type) - stablePriority(right.block.type)
      return priorityGap !== 0 ? priorityGap : left.index - right.index
    })
    .map(entry => entry.block)
}

function normalizeGeneratedBlock(block: Block): Block | null {
  switch (block.type) {
    case 'stat': {
      const value = limitText(block.value, 24)
      const label = limitText(block.label, 54)
      if (!value || !label) return null
      return {
        ...block,
        value,
        label,
        sublabel: limitText(block.sublabel, 96) || undefined,
        delta: limitText(block.delta, 56) || undefined,
      }
    }
    case 'insight': {
      const text = limitText(block.text, 380)
      if (!text) return null
      return {
        ...block,
        title: normalizeInterpretiveTitle(block.title),
        text,
      }
    }
    case 'comparison': {
      const leftItems = block.left.items
        .map(item => ({
          key: limitText(item.key, 32),
          value: limitText(item.value, 52),
        }))
        .filter(item => item.key && item.value)
        .slice(0, 6)
      const rightItems = block.right.items
        .map(item => ({
          key: limitText(item.key, 32),
          value: limitText(item.value, 52),
        }))
        .filter(item => item.key && item.value)
        .slice(0, 6)
      if (leftItems.length === 0 || rightItems.length === 0) return null
      return {
        ...block,
        title: limitText(block.title, 72),
        left: {
          ...block.left,
          label: limitText(block.left.label, 36),
          items: leftItems,
        },
        right: {
          ...block.right,
          label: limitText(block.right.label, 36),
          items: rightItems,
        },
        verdict: limitText(block.verdict, 180) || undefined,
      }
    }
    case 'chart': {
      const data = block.data
        .map(entry => ({
          ...entry,
          label: limitText(entry.label, 30),
          category: limitText(entry.category, 24) || undefined,
        }))
        .filter(entry => entry.label)
        .slice(0, 8)
      if (data.length === 0) return null
      return {
        ...block,
        title: limitText(block.title, 72),
        data,
        unit: limitText(block.unit, 12) || undefined,
      }
    }
    case 'table': {
      const headers = block.headers.map(header => limitText(header, 28)).filter(Boolean).slice(0, 6)
      const rows = block.rows
        .slice(0, 8)
        .map(row => row.slice(0, headers.length || 6).map(cell => limitText(cell, 42)))
        .filter(row => row.length > 0)
      if (headers.length === 0 || rows.length === 0) return null
      return {
        ...block,
        title: limitText(block.title, 72),
        headers,
        rows,
        highlight: block.highlight?.filter(index => index >= 0 && index < rows.length).slice(0, 3),
      }
    }
    case 'caveat': {
      const text = limitText(block.text, 240)
      return text ? { ...block, text } : null
    }
    case 'source': {
      const refs = block.refs
        .map(ref => ({
          label: limitText(ref.label, 64),
          section: limitText(ref.section, 40) || undefined,
          url: limitText(ref.url, 200) || undefined,
        }))
        .filter(ref => ref.label)
        .filter((ref, index, all) =>
          all.findIndex(candidate =>
            candidate.label === ref.label
            && candidate.section === ref.section
            && candidate.url === ref.url,
          ) === index,
        )
        .slice(0, 4)
      return refs.length > 0 ? { ...block, refs } : null
    }
    case 'map': {
      const regions = block.regions
        .map(region => ({
          ...region,
          name: limitText(region.name, 32),
          label: limitText(region.label, 28) || undefined,
        }))
        .filter(region => region.name)
        .toSorted((left, right) => right.value - left.value)
        .slice(0, 12)
      return regions.length >= 2
        ? {
            ...block,
            title: limitText(block.title, 72),
            regions,
          }
        : null
    }
    case 'timeseries': {
      const series = block.series
        .map(item => ({
          ...item,
          label: limitText(item.label, 32),
          data: item.data.slice(0, 16),
          color: limitText(item.color, 16) || undefined,
        }))
        .filter(item => item.label && item.data.length > 0)
        .slice(0, 4)
      if (series.length === 0) return null
      return {
        ...block,
        title: limitText(block.title, 72),
        series,
        xLabel: limitText(block.xLabel, 24) || undefined,
        yLabel: limitText(block.yLabel, 24) || undefined,
        annotations: block.annotations
          ?.map(annotation => ({
            ...annotation,
            label: limitText(annotation.label, 28),
          }))
          .filter(annotation => annotation.label)
          .slice(0, 4),
      }
    }
    default:
      return block
  }
}

function normalizeGeneratedSummary(
  rawSummary: string | undefined,
  query: string,
  blocks: readonly Block[],
): string {
  const summary = qualifyPaperSummary(rawSummary, '')
  if (summary) return summary

  const firstInsight = blocks.find((block): block is Extract<Block, { type: 'insight' }> => block.type === 'insight')
  if (firstInsight?.title) {
    return qualifyPaperSummary(firstInsight.title, '')
  }

  return qualifyPaperSummary(undefined, trimTrailingPunctuation(query))
}

function fallbackFollowUps(query: string): string[] {
  const fromTopics = findTopicMatches(query, 4)
    .flatMap(match => match.prompts)
    .filter(prompt => normalizeQuery(prompt) !== normalizeQuery(query))

  const fromCoverage = suggestUnderexploredTopics(query, 3).map(topic => topic.suggestedQuery)
  const combined = [...fromTopics, ...fromCoverage]
  const unique: string[] = []

  for (const prompt of combined) {
    const cleaned = limitText(prompt, 110)
    if (!cleaned) continue
    if (unique.some(existing => normalizeQuery(existing) === normalizeQuery(cleaned))) continue
    unique.push(cleaned)
    if (unique.length >= MAX_GENERATED_FOLLOW_UPS) break
  }

  return unique
}

function normalizeGeneratedFollowUps(
  query: string,
  rawFollowUps: readonly string[] | undefined,
): string[] {
  const normalized: string[] = []
  for (const followUp of rawFollowUps ?? []) {
    const cleaned = limitText(followUp, 110)
    if (!cleaned) continue
    if (normalizeQuery(cleaned) === normalizeQuery(query)) continue
    if (normalized.some(existing => normalizeQuery(existing) === normalizeQuery(cleaned))) continue
    normalized.push(cleaned)
    if (normalized.length >= MAX_GENERATED_FOLLOW_UPS) break
  }

  if (normalized.length >= 2) return normalized

  for (const fallback of fallbackFollowUps(query)) {
    if (normalized.some(existing => normalizeQuery(existing) === normalizeQuery(fallback))) continue
    normalized.push(fallback)
    if (normalized.length >= MAX_GENERATED_FOLLOW_UPS) break
  }

  return normalized
}

function normalizeGeneratedBlocks(rawBlocks: unknown[] | undefined): Block[] {
  const normalized = parseBlocks(rawBlocks ?? [])
    .map(normalizeGeneratedBlock)
    .filter((block): block is Block => block !== null)

  const deduped = normalized.filter((block, index, all) =>
    all.findIndex(candidate => blockSignature(candidate) === blockSignature(block)) === index,
  )

  const ordered = orderBlocksEvidenceFirst(deduped)

  const hasInsight = ordered.some(block => block.type === 'insight' || block.type === 'comparison')
  const hasSource = ordered.some(block => block.type === 'source')
  const hasCaveat = ordered.some(block => block.type === 'caveat')
  const polished = [...ordered]

  if (polished.length === 0) {
    return [
      {
        type: 'insight',
        emphasis: 'normal',
        title: 'Paper-backed note',
        text: 'The explorer is falling back to a conservative paper-backed note because the model did not return a safe structured visualization.',
      },
      DEFAULT_GENERATED_CAVEAT_BLOCK,
      DEFAULT_GENERATED_SOURCE_BLOCK,
    ]
  }

  if (!hasInsight && polished.length > 0 && polished.length < MAX_GENERATED_BLOCKS) {
    polished.splice(Math.min(1, polished.length), 0, {
      type: 'insight',
      emphasis: 'normal',
      title: 'Assistant framing',
      text: 'This composition is a compact guide to the supplied evidence. Treat the underlying paper context, artifact labels, and exact outputs as primary.',
    })
  }

  if (!hasCaveat && polished.length < MAX_GENERATED_BLOCKS) {
    polished.push(DEFAULT_GENERATED_CAVEAT_BLOCK)
  }

  if (!hasSource && polished.length < MAX_GENERATED_BLOCKS) {
    polished.push(DEFAULT_GENERATED_SOURCE_BLOCK)
  }

  return polished.slice(0, MAX_GENERATED_BLOCKS)
}

function normalizeSimulationPrompts(
  question: string,
  rawPrompts: readonly string[] | undefined,
  fallbacks: readonly string[],
): string[] {
  const normalized: string[] = []

  for (const prompt of rawPrompts ?? []) {
    const cleaned = limitText(prompt, 110)
    if (!cleaned) continue
    if (normalizeQuery(cleaned) === normalizeQuery(question)) continue
    if (normalized.some(existing => normalizeQuery(existing) === normalizeQuery(cleaned))) continue
    normalized.push(cleaned)
    if (normalized.length >= 4) break
  }

  for (const prompt of fallbacks) {
    const cleaned = limitText(prompt, 110)
    if (!cleaned) continue
    if (normalized.some(existing => normalizeQuery(existing) === normalizeQuery(cleaned))) continue
    normalized.push(cleaned)
    if (normalized.length >= 4) break
  }

  return normalized
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
    labels.push('Reference: SE3 joint heterogeneity')
  } else if (config.distribution === 'heterogeneous') {
    labels.push('Reference: SE2 heterogeneous validators')
  } else if (config.distribution === 'homogeneous-gcp') {
    labels.push('Equal per-GCP validator start')
  } else if (config.sourcePlacement === 'latency-aligned') {
    labels.push('Reference: SE1 latency-aligned sources')
  } else if (config.sourcePlacement === 'latency-misaligned') {
    labels.push('Reference: SE1 latency-misaligned sources')
  } else {
    labels.push('Reference: baseline geography/source setup')
  }

  if (config.slotTime === 6) {
    labels.push('Reference: SE4b shorter slots')
  } else if (Math.abs(config.attestationThreshold - 2 / 3) > 0.01) {
    labels.push('Reference: SE4a gamma variation')
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
  if (
    distribution === 'homogeneous'
    || distribution === 'homogeneous-gcp'
    || distribution === 'heterogeneous'
    || distribution === 'random'
  ) {
    candidate.distribution = distribution
  } else if (distribution === 'uniform') {
    candidate.distribution = 'homogeneous-gcp'
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
        validators: '1-1000',
        slots: '1-10000',
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
      'Named study presets use the paper-style 10,000-slot and 0.002 ETH reference setup unless you override fields.',
      'It composes a run configuration only; it does not execute the simulation.',
      'Scenario labels are paper references for orientation, not standalone evidence.',
    ],
  }
}

function formatMetricNumber(value: number, digits = 2): string {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  }).format(value)
}

function defaultSimulationPrompts(manifest: { config: SimulationRequest } | null): string[] {
  if (!manifest) {
    return [
      'Set up the paper baseline SSP run (10,000 slots, 0.002 ETH).',
      'Mirror that paper baseline for MSP so I can compare paradigms.',
      'Hold the paradigm fixed and switch from latency-aligned to latency-misaligned sources.',
      'Load the real Ethereum validator start and explain what should change.',
    ]
  }

  return [
    'Build a core outcomes overview from this exact run.',
    'Explain which regions dominate in this run and why.',
    'What is the nearest paper-backed follow-up to run next?',
  ]
}

function buildTruthBoundary(
  mode: SimulationViewSpec['mode'],
  hasManifest: boolean,
): { label: string; detail: string } {
  if (mode === 'proposed-run') {
    return {
      label: 'Proposal, not a result',
      detail: 'This response suggests a bounded exact run configuration. It is not simulation evidence until the exact engine executes it.',
    }
  }

  if (mode === 'guidance' || !hasManifest) {
    return {
      label: 'Guide interpretation, not evidence',
      detail: 'This response is guidance about the supported model surface. It should not be read as an exact simulation result.',
    }
  }

  return {
    label: 'Assistant framing over exact outputs',
    detail: 'The numbers and charts come from the exact manifest and artifacts for the loaded run. The ordering, emphasis, and narrative are secondary model framing.',
  }
}

function metricNumericValue(
  metric: SimulationChartMetricKey,
  manifest: { summary: Record<string, number>; config: SimulationRequest } | null,
  fallbackConfig: SimulationRequest,
): number | null {
  const config = manifest?.config ?? fallbackConfig
  switch (metric) {
    case 'finalAverageMev':
      return manifest ? manifest.summary.finalAverageMev : null
    case 'finalSupermajoritySuccess':
      return manifest ? manifest.summary.finalSupermajoritySuccess : null
    case 'finalFailedBlockProposals':
      return manifest ? manifest.summary.finalFailedBlockProposals : null
    case 'finalUtilityIncrease':
      return manifest ? manifest.summary.finalUtilityIncrease : null
    case 'validators':
      return config.validators
    case 'slots':
      return config.slots
    case 'migrationCost':
      return config.migrationCost
    case 'attestationThreshold':
      return config.attestationThreshold
    case 'slotTime':
      return config.slotTime
    case 'attestationCutoffMs':
      return manifest ? manifest.summary.attestationCutoffMs : attestationCutoffMs(config.slotTime)
    default:
      return null
  }
}

function metricDisplayLabel(metric: SimulationChartMetricKey): string {
  switch (metric) {
    case 'finalAverageMev':
      return 'Final Avg MEV'
    case 'finalSupermajoritySuccess':
      return 'Supermajority Success'
    case 'finalFailedBlockProposals':
      return 'Failed Proposals'
    case 'finalUtilityIncrease':
      return 'Utility Increase'
    case 'validators':
      return 'Validators'
    case 'slots':
      return 'Slots'
    case 'migrationCost':
      return 'Migration Cost'
    case 'attestationThreshold':
      return 'Gamma'
    case 'slotTime':
      return 'Slot Time'
    case 'attestationCutoffMs':
      return 'Cutoff (ms)'
    default:
      return metric
  }
}

async function loadArtifactTexts(
  manifest: {
    outputDir: string
    artifacts: ReadonlyArray<{ name: string; renderable: boolean }>
  },
  names: readonly string[],
): Promise<Partial<Record<string, string>>> {
  const loaded: Partial<Record<string, string>> = {}
  for (const name of names) {
    const artifact = manifest.artifacts.find(candidate => candidate.name === name)
    if (!artifact || !artifact.renderable) continue
    try {
      loaded[name] = await fs.readFile(path.join(manifest.outputDir, name), 'utf8')
    } catch {
      // Ignore unreadable artifacts and let the caller decide how to degrade.
    }
  }
  return loaded
}

function bundleArtifactNames(bundle: SimulationArtifactBundle): readonly string[] {
  switch (bundle) {
    case 'core-outcomes':
      return ['avg_mev.json', 'supermajority_success.json', 'failed_block_proposals.json']
    case 'timing-and-attestation':
      return ['proposal_time_avg.json', 'attestation_sum.json']
    case 'geography-overview':
      return ['top_regions_final.json']
    default:
      return []
  }
}

function metricBlockForSection(
  section: Extract<SimulationViewSection, { kind: 'metric' }>,
  manifest: { summary: Record<string, number>; config: SimulationRequest } | null,
  fallbackConfig: SimulationRequest,
): Block | null {
  const config = manifest?.config ?? fallbackConfig
  let label = section.label
  let value = ''
  let sublabel = section.sublabel

  switch (section.metric) {
    case 'finalAverageMev':
      if (!manifest) return null
      label ??= 'Final Average MEV'
      sublabel ??= 'Exact run summary'
      value = `${formatMetricNumber(manifest.summary.finalAverageMev, 4)} ETH`
      break
    case 'finalSupermajoritySuccess':
      if (!manifest) return null
      label ??= 'Final Supermajority Success'
      sublabel ??= 'Exact run summary'
      value = `${formatMetricNumber(manifest.summary.finalSupermajoritySuccess, 2)}%`
      break
    case 'finalFailedBlockProposals':
      if (!manifest) return null
      label ??= 'Failed Block Proposals'
      sublabel ??= 'Exact run summary'
      value = formatMetricNumber(manifest.summary.finalFailedBlockProposals, 0)
      break
    case 'finalUtilityIncrease':
      if (!manifest) return null
      label ??= 'Final Utility Increase'
      sublabel ??= 'Exact run summary'
      value = `${formatMetricNumber(manifest.summary.finalUtilityIncrease, 6)} ETH`
      break
    case 'slotsRecorded':
      if (!manifest) return null
      label ??= 'Slots Recorded'
      sublabel ??= 'Exact run summary'
      value = formatMetricNumber(manifest.summary.slotsRecorded, 0)
      break
    case 'attestationCutoffMs':
      if (!manifest) return null
      label ??= 'Attestation Cutoff'
      sublabel ??= 'Consensus timing'
      value = `${formatMetricNumber(manifest.summary.attestationCutoffMs, 0)} ms`
      break
    case 'validators':
      label ??= 'Validators'
      sublabel ??= 'Simulation config'
      value = formatMetricNumber(config.validators, 0)
      break
    case 'slots':
      label ??= 'Slots'
      sublabel ??= 'Simulation config'
      value = formatMetricNumber(config.slots, 0)
      break
    case 'migrationCost':
      label ??= 'Migration Cost'
      sublabel ??= 'Simulation config'
      value = `${formatMetricNumber(config.migrationCost, 6)} ETH`
      break
    case 'attestationThreshold':
      label ??= 'Attestation Threshold'
      sublabel ??= 'Simulation config'
      value = formatMetricNumber(config.attestationThreshold, 4)
      break
    case 'slotTime':
      label ??= 'Slot Time'
      sublabel ??= 'Simulation config'
      value = `${formatMetricNumber(config.slotTime, 0)} s`
      break
    case 'seed':
      label ??= 'Seed'
      sublabel ??= 'Simulation config'
      value = formatMetricNumber(config.seed, 0)
      break
    default:
      return null
  }

  return {
    type: 'stat',
    value,
    label,
    sublabel,
    sentiment: section.sentiment,
  }
}

function withArtifactPresentation(
  blocks: readonly Block[],
  title: string | undefined,
  note: string | undefined,
): readonly Block[] {
  const nextBlocks = blocks.map((block, index) => {
    if (index !== 0 || !title) return block
    if ('title' in block && typeof block.title === 'string') {
      return { ...block, title }
    }
    return block
  })

  if (!note) return nextBlocks

  return [
    {
      type: 'insight',
      title: title ? 'Why this view' : undefined,
      text: note,
    },
    ...nextBlocks,
  ]
}

async function summarizeRenderableArtifacts(
  manifest: { outputDir: string; artifacts: ReadonlyArray<{ name: string; renderable: boolean }> },
): Promise<string> {
  const lines: string[] = []

  for (const artifact of manifest.artifacts) {
    if (!artifact.renderable) continue
    const artifactPath = path.join(manifest.outputDir, artifact.name)

    try {
      const rawText = await fs.readFile(artifactPath, 'utf8')

      if (artifact.name === 'top_regions_final.json') {
        const rows = JSON.parse(rawText) as Array<[string, number]>
        const summary = rows.slice(0, 5).map(([region, value]) => `${region}=${value}`).join(', ')
        lines.push(`- ${artifact.name}: top regions ${summary}`)
        continue
      }

      const values = JSON.parse(rawText) as unknown
      if (Array.isArray(values) && values.every(value => typeof value === 'number')) {
        const numeric = values as number[]
        const first = numeric[0] ?? 0
        const last = numeric[numeric.length - 1] ?? 0
        const min = Math.min(...numeric)
        const max = Math.max(...numeric)
        lines.push(
          `- ${artifact.name}: first=${formatMetricNumber(first, 4)}, last=${formatMetricNumber(last, 4)}, min=${formatMetricNumber(min, 4)}, max=${formatMetricNumber(max, 4)}`,
        )
      }
    } catch {
      // Ignore unreadable digests and keep the copilot working with the remaining context.
    }
  }

  return lines.join('\n')
}

async function loadOverviewBundleBlocks(
  manifest: {
    outputDir: string
    overviewBundles?: ReadonlyArray<{
      bundle: SimulationArtifactBundle
      name: string
    }>
  },
  bundle: SimulationArtifactBundle,
): Promise<readonly Block[] | null> {
  const overviewBundle = manifest.overviewBundles?.find(candidate => candidate.bundle === bundle)
  if (!overviewBundle) {
    return null
  }

  try {
    const rawText = await fs.readFile(path.join(manifest.outputDir, overviewBundle.name), 'utf8')
    return parseSimulationBlockBundle(rawText)
  } catch {
    return null
  }
}

async function resolveSimulationViewSpec(
  viewSpec: SimulationViewSpec,
  manifest: {
    outputDir: string
    summary: Record<string, number>
    artifacts: ReadonlyArray<{
      name: string
      label: string
      kind: 'timeseries' | 'map' | 'table' | 'raw'
      renderable: boolean
    }>
    overviewBundles?: ReadonlyArray<{
      bundle: SimulationArtifactBundle
      name: string
    }>
    config: SimulationRequest
  } | null,
  fallbackConfig: SimulationRequest,
): Promise<readonly Block[]> {
  const blocks: Block[] = []
  const artifactTextCache = new Map<string, string>()

  const loadArtifactText = async (artifactName: string): Promise<string | null> => {
    if (!manifest) return null
    if (artifactTextCache.has(artifactName)) {
      return artifactTextCache.get(artifactName) ?? null
    }
    try {
      const rawText = await fs.readFile(path.join(manifest.outputDir, artifactName), 'utf8')
      artifactTextCache.set(artifactName, rawText)
      return rawText
    } catch {
      return null
    }
  }

  for (const section of viewSpec.sections) {
    if (section.kind === 'metric') {
      const block = metricBlockForSection(section, manifest, fallbackConfig)
      if (block) blocks.push(block)
      continue
    }

    if (section.kind === 'summary-chart') {
      const entries = section.metrics
        .map(metric => {
          const value = metricNumericValue(metric, manifest, fallbackConfig)
          return value == null
            ? null
            : {
                metric,
                label: metricDisplayLabel(metric),
                value,
              }
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null)

      if (section.note) {
        blocks.push({
          type: 'insight',
          title: 'Interpretation note',
          text: section.note,
        })
      }

      if (entries.length >= 2) {
        blocks.push(buildSimulationSummaryChart(section.title, entries, section.unit))
      } else {
        blocks.push({
          type: 'caveat',
          text: `${section.title} needs at least two supported numeric metrics to render.`,
        })
      }
      continue
    }

    if (section.kind === 'insight') {
      blocks.push({
        type: 'insight',
        title: section.title,
        text: section.text,
        emphasis: section.emphasis,
      })
      continue
    }

    if (section.kind === 'caveat') {
      blocks.push({
        type: 'caveat',
        text: section.text,
      })
      continue
    }

    if (section.kind === 'source') {
      blocks.push({
        type: 'source',
        refs: section.refs,
      })
      continue
    }

    if (section.kind === 'artifact-bundle') {
      if (!manifest) {
        blocks.push({
          type: 'caveat',
          text: 'This bundle references exact simulation artifacts, but there is no completed run loaded yet.',
        })
        continue
      }

      const prebuiltBundleBlocks = await loadOverviewBundleBlocks(manifest, section.bundle)
      const bundleBlocks = prebuiltBundleBlocks ?? await (async () => {
        const artifactTexts = await loadArtifactTexts(manifest, bundleArtifactNames(section.bundle))
        return buildSimulationArtifactBundle(
          section.bundle,
          artifactTexts as Partial<Record<SimulationRenderableArtifact['name'], string>>,
        )
      })()

      if (section.note) {
        blocks.push({
          type: 'insight',
          title: section.title ? 'Why this bundle' : undefined,
          text: section.note,
        })
      }

      if (bundleBlocks.length > 0) {
        blocks.push(...bundleBlocks)
      } else {
        blocks.push({
          type: 'caveat',
          text: `The ${section.bundle} bundle could not be assembled from the available exact artifacts.`,
        })
      }
      continue
    }

    if (!manifest) {
      blocks.push({
        type: 'caveat',
        text: 'This view references simulation artifacts, but there is no completed exact run loaded yet.',
      })
      continue
    }

    const artifact = manifest.artifacts.find(candidate => candidate.name === section.artifactName)
    if (!artifact || !artifact.renderable) {
      blocks.push({
        type: 'caveat',
        text: `${section.artifactName} is not available as a renderable artifact for this run.`,
      })
      continue
    }

    try {
      const rawText = await loadArtifactText(artifact.name)
      if (!rawText) {
        throw new Error('Artifact missing')
      }
      const artifactBlocks = parseSimulationArtifactToBlocks(
        artifact as SimulationRenderableArtifact,
        rawText,
      )
      blocks.push(...withArtifactPresentation(artifactBlocks, section.title, section.note))
    } catch {
      blocks.push({
        type: 'caveat',
        text: `The exact artifact ${artifact.name} could not be loaded for this response.`,
      })
    }
  }

  if (blocks.length > 0) return orderBlocksEvidenceFirst(blocks)

  return [
    {
      type: 'caveat',
      text: viewSpec.guidance ?? 'Ask about a paper finding, propose a bounded run, or load an exact result first.',
    },
  ]
}

async function buildSimulationCopilotContext(
  currentConfig: SimulationRequest,
  currentJob:
    | {
        status: string
        error: string | null
        manifest?: {
          outputDir: string
          config: SimulationRequest
          summary: Record<string, number>
          artifacts: ReadonlyArray<{
            name: string
            label: string
            kind: 'timeseries' | 'map' | 'table' | 'raw'
            renderable: boolean
          }>
        }
      }
    | null,
): Promise<string> {
  const manifest = currentJob?.manifest
  const parts = [
    '## Current Simulation Config',
    `- paradigm: ${currentConfig.paradigm}`,
    `- validators: ${currentConfig.validators}`,
    `- slots: ${currentConfig.slots}`,
    `- distribution: ${currentConfig.distribution}`,
    `- sourcePlacement: ${currentConfig.sourcePlacement}`,
    `- migrationCost: ${currentConfig.migrationCost}`,
    `- attestationThreshold: ${currentConfig.attestationThreshold}`,
    `- slotTime: ${currentConfig.slotTime}`,
    `- seed: ${currentConfig.seed}`,
  ]

  if (!currentJob) {
    parts.push('## Current Exact Result', 'No simulation job is currently loaded.')
    return parts.join('\n')
  }

  parts.push('## Current Job Status', `- status: ${currentJob.status}`)
  if (currentJob.error) {
    parts.push(`- error: ${currentJob.error}`)
  }

  if (!manifest) {
    parts.push('## Current Exact Result', 'There is no completed exact result available yet.')
    return parts.join('\n')
  }

  parts.push(
    '## Current Exact Result',
    `- finalAverageMev: ${manifest.summary.finalAverageMev}`,
    `- finalSupermajoritySuccess: ${manifest.summary.finalSupermajoritySuccess}`,
    `- finalFailedBlockProposals: ${manifest.summary.finalFailedBlockProposals}`,
    `- finalUtilityIncrease: ${manifest.summary.finalUtilityIncrease}`,
    `- slotsRecorded: ${manifest.summary.slotsRecorded}`,
    `- attestationCutoffMs: ${manifest.summary.attestationCutoffMs}`,
    '## Available Renderable Artifacts',
    ...manifest.artifacts
      .filter(artifact => artifact.renderable)
      .map(artifact => `- ${artifact.name}: ${artifact.label}`),
    '## Reusable Exact Chart Bundles',
    '- core-outcomes: avg_mev + supermajority_success + failed_block_proposals',
    '- timing-and-attestation: proposal_time_avg + attestation_sum',
    '- geography-overview: top_regions_final map + table',
  )

  const digest = await summarizeRenderableArtifacts(manifest)
  if (digest) {
    parts.push('## Artifact Digests', digest)
  }

  return parts.join('\n')
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

app.post('/api/explore', exploreRateLimit, async (req, res) => {
  const { query, history } = req.body as ExploreRequest
  const trimmedQuery = query?.trim()

  if (!trimmedQuery) {
    res.status(400).json({ error: 'Query is required' })
    return
  }
  if (trimmedQuery.length > MAX_EXPLORE_QUERY_LENGTH) {
    res.status(400).json({
      error: `Query is too long. Keep it under ${MAX_EXPLORE_QUERY_LENGTH} characters and narrow the ask.`,
    })
    return
  }

  const sessionHistory = Array.isArray(history)
    ? history
      .filter((entry): entry is { query: string; summary: string } =>
        Boolean(entry)
        && typeof entry.query === 'string'
        && typeof entry.summary === 'string',
      )
      .slice(-MAX_SESSION_HISTORY_ENTRIES)
      .map(entry => ({
        query: limitText(entry.query.trim(), MAX_EXPLORE_QUERY_LENGTH),
        summary: limitText(entry.summary.trim(), MAX_SESSION_SUMMARY_LENGTH),
      }))
      .filter(entry => entry.query.length > 0 && entry.summary.length > 0)
    : []

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
    res.status(503).json({ error: MISSING_ANTHROPIC_CONFIG_MESSAGE })
    return
  }

  try {
    const sessionContext = sessionHistory.length
      ? `\n\n## Session Context\nPrevious queries this session:\n${sessionHistory.map((entry, index) => `${index + 1}. "${entry.query}" -> ${entry.summary}`).join('\n')}\n\nBuild on prior context where relevant.`
      : ''

    // Multi-turn tool execution loop
    const messages: Anthropic.Messages.MessageParam[] = [
      { role: 'user', content: trimmedQuery },
    ]
    const MAX_TOOL_ROUNDS = 3
    let finalResponse: Anthropic.Messages.Message | null = null

    for (let round = 0; round <= MAX_TOOL_ROUNDS; round++) {
      const response = await client.messages.create({
        model: anthropicModel,
        max_tokens: 4096,
        system: [
          {
            type: 'text',
            text: STUDY_CONTEXT + sessionContext,
            cache_control: sessionHistory.length ? undefined : { type: 'ephemeral' },
          },
        ],
        tools: exploreTools,
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
    const normalizedBlocks = normalizeGeneratedBlocks(input.blocks)
    const normalizedSummary = normalizeGeneratedSummary(input.summary, trimmedQuery, normalizedBlocks)
    const normalizedFollowUps = normalizeGeneratedFollowUps(trimmedQuery, input.follow_ups)

    const result: ExploreResponse = {
      summary: normalizedSummary,
      blocks: normalizedBlocks,
      followUps: normalizedFollowUps,
      model: finalResponse.model,
      cached: finalResponse.usage?.cache_read_input_tokens
        ? finalResponse.usage.cache_read_input_tokens > 0
        : false,
      provenance: {
        source: 'generated',
        label: 'Fresh interpretation',
        detail: 'Generated a new structured reading from the study context and the current question.',
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

app.post('/api/simulation-copilot', simulationCopilotRateLimit, async (req, res) => {
  const { question, currentJobId, currentConfig } = req.body as SimulationCopilotRequest
  const trimmedQuestion = question?.trim()

  if (!trimmedQuestion) {
    res.status(400).json({ error: 'Question is required.' })
    return
  }
  if (trimmedQuestion.length > MAX_SIMULATION_QUESTION_LENGTH) {
    res.status(400).json({
      error: `Question is too long. Keep it under ${MAX_SIMULATION_QUESTION_LENGTH} characters and focus on one comparison or hypothesis at a time.`,
    })
    return
  }

  if (!client) {
    res.status(503).json({ error: MISSING_ANTHROPIC_CONFIG_MESSAGE })
    return
  }

  const currentJob = typeof currentJobId === 'string' && currentJobId
    ? simulationRuntime.getJob(currentJobId)
    : null
  const parsedCurrentConfig = currentConfig ? parseSimulationRequest(currentConfig) : null
  const effectiveConfig = currentJob?.manifest?.config ?? currentJob?.config ?? parsedCurrentConfig ?? DEFAULT_SIMULATION_CONFIG
  const currentContext = await buildSimulationCopilotContext(effectiveConfig, currentJob)

  try {
    const messages: Anthropic.Messages.MessageParam[] = [
      { role: 'user', content: trimmedQuestion },
    ]
    const MAX_TOOL_ROUNDS = 4
    let finalResponse: Anthropic.Messages.Message | null = null

    for (let round = 0; round <= MAX_TOOL_ROUNDS; round++) {
      const response = await client.messages.create({
        model: anthropicModel,
        max_tokens: 4096,
        system: [
          {
            type: 'text',
            text: `${SIMULATION_COPILOT_CONTEXT}\n\n${currentContext}`,
            cache_control: { type: 'ephemeral' },
          },
        ],
        tools: simulationCopilotTools,
        tool_choice: round === MAX_TOOL_ROUNDS
          ? { type: 'tool', name: 'render_simulation_view_spec' }
          : { type: 'auto' },
        messages,
      })

      const finalTool = response.content.find(
        (block): block is Anthropic.Messages.ToolUseBlock =>
          block.type === 'tool_use' && block.name === 'render_simulation_view_spec',
      )

      if (finalTool || response.stop_reason === 'end_turn') {
        finalResponse = response
        break
      }

      const toolCalls = response.content.filter(
        (block): block is Anthropic.Messages.ToolUseBlock => block.type === 'tool_use',
      )

      if (toolCalls.length === 0) {
        finalResponse = response
        break
      }

      messages.push({ role: 'assistant', content: response.content })
      const toolResults: Anthropic.Messages.ToolResultBlockParam[] = toolCalls.map(call => ({
        type: 'tool_result' as const,
        tool_use_id: call.id,
        content: JSON.stringify(executeToolCall(call.name, call.input as Record<string, unknown>)),
      }))
      messages.push({ role: 'user', content: toolResults })
    }

    if (!finalResponse) {
      res.status(500).json({ error: 'Simulation copilot tool loop exhausted without a final response.' })
      return
    }

    const toolUse = finalResponse.content.find(
      (block): block is Anthropic.Messages.ToolUseBlock =>
        block.type === 'tool_use' && block.name === 'render_simulation_view_spec',
    )

    if (!toolUse) {
      res.status(500).json({ error: 'Simulation copilot did not return a view spec.' })
      return
    }

    const parsedViewSpec = simulationViewSpecSchema.safeParse(toolUse.input)
    if (!parsedViewSpec.success) {
      res.status(500).json({ error: 'Simulation copilot returned an invalid view spec.' })
      return
    }

    const proposedConfig = parsedViewSpec.data.proposedConfig
      ? parseSimulationRequest(parsedViewSpec.data.proposedConfig)
      : null
    const sanitizedViewSpec: SimulationViewSpec = proposedConfig
      ? { ...parsedViewSpec.data, proposedConfig }
      : { ...parsedViewSpec.data, proposedConfig: undefined }

    const blocks = await resolveSimulationViewSpec(
      sanitizedViewSpec,
      currentJob?.manifest ?? null,
      effectiveConfig,
    )

    const result: SimulationCopilotResponse = {
      summary: limitText(sanitizedViewSpec.summary, 180) || 'Simulation guidance',
      mode: sanitizedViewSpec.mode,
      guidance: limitText(sanitizedViewSpec.guidance, 240) || undefined,
      truthBoundary: buildTruthBoundary(sanitizedViewSpec.mode, Boolean(currentJob?.manifest)),
      suggestedPrompts: normalizeSimulationPrompts(
        trimmedQuestion,
        sanitizedViewSpec.suggestedPrompts,
        defaultSimulationPrompts(currentJob?.manifest ?? null),
      ),
      proposedConfig: proposedConfig ?? undefined,
      viewSpec: sanitizedViewSpec,
      blocks,
      model: finalResponse.model,
      cached: finalResponse.usage?.cache_read_input_tokens
        ? finalResponse.usage.cache_read_input_tokens > 0
        : false,
    }

    res.json(result)
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
    tools: exploreTools.length,
    simulationCopilotTools: simulationCopilotTools.length,
    anthropicEnabled: Boolean(client),
    anthropicModel: client ? anthropicModel : null,
    envFileLoaded: existsSync(envFile),
    simulations: simulationRuntime.health(),
  })
})

app.post('/api/simulations', simulationSubmitRateLimit, (req, res) => {
  const config = parseSimulationRequest(req.body)
  if (!config) {
    res.status(400).json({ error: 'Invalid simulation request payload.' })
    return
  }

  const rawClientId = typeof req.body?.clientId === 'string' ? req.body.clientId : null
  const clientId = rawClientId?.trim()
  if (clientId && clientId.length > MAX_CLIENT_ID_LENGTH) {
    res.status(400).json({ error: `clientId is too long. Keep it under ${MAX_CLIENT_ID_LENGTH} characters.` })
    return
  }

  try {
    const job = simulationRuntime.submit(config, { clientId })
    res.status(job.status === 'completed' ? 200 : 202).json(job)
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Simulation submission failed.'
    const status = message.includes('queue is full') || message.includes('Too many active simulation jobs')
      ? 429
      : 500
    res.status(status).json({ error: message })
  }
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

function clientAcceptsEncoding(acceptEncoding: string, encoding: 'br' | 'gzip'): boolean {
  const pattern = encoding === 'br' ? /\bbr\b/i : /\bgzip\b/i
  return pattern.test(acceptEncoding)
}

app.get('/api/simulations/:jobId/artifacts/:artifactName', async (req, res) => {
  try {
    const acceptEncoding = req.header('accept-encoding') ?? ''
    const { artifact, body, contentEncoding } = await simulationRuntime.readArtifact(
      req.params.jobId,
      req.params.artifactName,
      {
        preferBrotli: clientAcceptsEncoding(acceptEncoding, 'br'),
        preferGzip: clientAcceptsEncoding(acceptEncoding, 'gzip'),
      },
    )
    res.setHeader('Content-Type', `${artifact.contentType}; charset=utf-8`)
    res.setHeader('ETag', artifact.sha256)
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable')
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

app.get('/api/simulations/:jobId/overview-bundles/:bundleName', async (req, res) => {
  try {
    const acceptEncoding = req.header('accept-encoding') ?? ''
    const { overviewBundle, body, contentEncoding } = await simulationRuntime.readOverviewBundle(
      req.params.jobId,
      req.params.bundleName,
      {
        preferBrotli: clientAcceptsEncoding(acceptEncoding, 'br'),
        preferGzip: clientAcceptsEncoding(acceptEncoding, 'gzip'),
      },
    )
    res.setHeader('Content-Type', 'application/json; charset=utf-8')
    res.setHeader('ETag', overviewBundle.sha256)
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable')
    res.setHeader('Vary', 'Accept-Encoding')
    if (contentEncoding) {
      res.setHeader('Content-Encoding', contentEncoding)
    }
    res.send(body)
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unable to read overview bundle.'
    const status = message.includes('not available') ? 409 :
      message.includes('Unknown overview bundle') ? 404 :
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

const DIST_DIR = path.join(__dirname, '..', 'dist')
const DASHBOARD_DIR = path.resolve(EXPLORER_ROOT, '..', 'dashboard')

if (existsSync(DASHBOARD_DIR)) {
  app.use('/research-demo', express.static(DASHBOARD_DIR))
  app.get('/research-demo', (_req, res) => {
    res.sendFile(path.join(DASHBOARD_DIR, 'index.html'))
  })
}

if (existsSync(DIST_DIR)) {
  app.use(express.static(DIST_DIR))
  // SPA fallback: serve index.html for any non-API route
  app.get(/^\/(?!api(?:\/|$)).*/, (_req, res) => {
    res.sendFile(path.join(DIST_DIR, 'index.html'))
  })
}

const PORT = Number(process.env.PORT ?? 3201)
app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Explorer API listening on http://localhost:${PORT}`)
})
