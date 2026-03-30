/**
 * Durable exploration store with JSON file persistence.
 * This keeps enough metadata for query reuse and provenance, even though the
 * current backend still uses a local JSON file rather than an external DB.
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync, renameSync, rmSync } from 'node:fs'
import { join, dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { randomUUID } from 'node:crypto'

const __dirname = dirname(fileURLToPath(import.meta.url))
const DATA_DIR = join(__dirname, 'data')
const DATA_FILE = process.env.EXPLORATION_STORE_FILE
  ? resolve(process.env.EXPLORATION_STORE_FILE)
  : join(DATA_DIR, 'explorations.json')
const DATA_FILE_DIR = dirname(DATA_FILE)
const MAX_EXPLORATIONS = Math.max(200, Number(process.env.EXPLORATION_STORE_MAX_ITEMS ?? 4000))

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

export interface Exploration {
  readonly id: string
  readonly query: string
  readonly normalizedQuery: string
  readonly summary: string
  readonly blocks: unknown[]
  readonly followUps: string[]
  readonly model: string
  readonly cached: boolean
  readonly source: 'generated'
  readonly votes: number
  readonly createdAt: string
  readonly paradigmTags: string[]
  readonly experimentTags: string[]
  readonly verified: boolean
}

export interface ExplorationMatch {
  readonly exploration: Exploration
  readonly score: number
  readonly reason: 'exact' | 'similar'
}

type SortOption = 'recent' | 'top'

export interface ListOptions {
  readonly sort?: SortOption
  readonly limit?: number
  readonly search?: string
  readonly paradigm?: 'SSP' | 'MSP'
  readonly experiment?: string
  readonly verifiedOnly?: boolean
}

function extractParadigmTags(blocks: unknown[]): string[] {
  const text = JSON.stringify(blocks)
  const tags: string[] = []
  if (/\bSSP\b/.test(text)) tags.push('SSP')
  if (/\bMSP\b/.test(text)) tags.push('MSP')
  return tags
}

function extractExperimentTags(blocks: unknown[]): string[] {
  const text = JSON.stringify(blocks)
  const matches = text.match(/\bSE[1-4]\b/g)
  if (!matches) return []
  return [...new Set(matches)].sort()
}

export function normalizeQuery(query: string): string {
  return query
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function tokenizeNormalized(normalized: string): string[] {
  if (!normalized) return []
  return normalized
    .split(' ')
    .filter(token => token.length > 1 && !STOP_WORDS.has(token))
}

function uniqueTokens(text: string): string[] {
  return [...new Set(tokenizeNormalized(normalizeQuery(text)))]
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

function buildSearchScore(
  exploration: Exploration,
  normalizedSearch: string,
  searchTokens: readonly string[],
): number {
  if (!normalizedSearch) return 0
  if (exploration.normalizedQuery === normalizedSearch) return 1

  const queryTokens = uniqueTokens(exploration.query)
  const summaryTokens = uniqueTokens(exploration.summary)
  const includesScore = (
    exploration.normalizedQuery.includes(normalizedSearch)
    || normalizedSearch.includes(exploration.normalizedQuery)
  ) ? 0.88 : 0

  return Math.max(
    includesScore,
    overlapScore(searchTokens, queryTokens),
    overlapScore(searchTokens, summaryTokens) * 0.8,
  )
}

function hydrateExploration(raw: Partial<Exploration> & Pick<Exploration, 'query' | 'summary' | 'blocks'>): Exploration {
  return {
    id: raw.id ?? randomUUID(),
    query: raw.query,
    normalizedQuery: raw.normalizedQuery ?? normalizeQuery(raw.query),
    summary: raw.summary,
    blocks: raw.blocks,
    followUps: raw.followUps ?? [],
    model: raw.model ?? '',
    cached: raw.cached ?? false,
    source: 'generated',
    votes: raw.votes ?? 0,
    createdAt: raw.createdAt ?? new Date().toISOString(),
    paradigmTags: raw.paradigmTags ?? extractParadigmTags(raw.blocks),
    experimentTags: raw.experimentTags ?? extractExperimentTags(raw.blocks),
    verified: raw.verified ?? false,
  }
}

export class ExplorationStore {
  private explorations: Exploration[] = []
  private persistTimer: ReturnType<typeof setTimeout> | null = null
  private byId = new Map<string, Exploration>()
  private byNormalizedQuery = new Map<string, Exploration>()

  constructor() {
    this.loadFromDisk()
  }

  save(data: {
    readonly query: string
    readonly summary: string
    readonly blocks: unknown[]
    readonly followUps: string[]
    readonly model: string
    readonly cached: boolean
  }): Exploration {
    const normalizedQuery = normalizeQuery(data.query)
    const existing = this.findExactQuery(data.query)
    if (existing) {
      return existing
    }

    const exploration = hydrateExploration({
      query: data.query,
      normalizedQuery,
      summary: data.summary,
      blocks: data.blocks,
      followUps: data.followUps,
      model: data.model,
      cached: data.cached,
    })

    this.setExplorations([exploration, ...this.explorations].slice(0, MAX_EXPLORATIONS))
    this.schedulePersist()
    return exploration
  }

  list(options?: ListOptions): Exploration[] {
    const sort = options?.sort ?? 'recent'
    const limit = options?.limit ?? 50
    const search = options?.search?.trim()

    // Apply tag and verified filters first
    let pool = this.explorations
    if (options?.paradigm) {
      pool = pool.filter(e => e.paradigmTags.includes(options.paradigm!))
    }
    if (options?.experiment) {
      pool = pool.filter(e => e.experimentTags.includes(options.experiment!))
    }
    if (options?.verifiedOnly) {
      pool = pool.filter(e => e.verified)
    }

    if (search) {
      const normalizedSearch = normalizeQuery(search)
      const searchTokens = tokenizeNormalized(normalizedSearch)
      const scored = pool
        .map(exploration => ({
          exploration,
          score: buildSearchScore(exploration, normalizedSearch, searchTokens),
        }))
        .filter(entry => entry.score >= 0.26)
        .toSorted((left, right) => {
          if (right.score !== left.score) return right.score - left.score
          if (sort === 'top' && right.exploration.votes !== left.exploration.votes) {
            return right.exploration.votes - left.exploration.votes
          }
          return new Date(right.exploration.createdAt).getTime() - new Date(left.exploration.createdAt).getTime()
        })
      return scored.slice(0, limit).map(entry => entry.exploration)
    }

    const results = [...pool]
    if (sort === 'top') {
      return results
        .toSorted((a, b) => b.votes - a.votes)
        .slice(0, limit)
    }

    return results
      .toSorted((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
      .slice(0, limit)
  }

  findExactQuery(query: string): Exploration | null {
    const normalized = normalizeQuery(query)
    return this.byNormalizedQuery.get(normalized) ?? null
  }

  findBestMatch(query: string, minimumScore = 0.72): ExplorationMatch | null {
    const normalizedQuery = normalizeQuery(query)
    if (!normalizedQuery) return null

    const exact = this.findExactQuery(query)
    if (exact) {
      return { exploration: exact, score: 1, reason: 'exact' }
    }

    const queryTokens = tokenizeNormalized(normalizedQuery)
    let best: ExplorationMatch | null = null

    for (const exploration of this.explorations) {
      const score = buildSearchScore(exploration, normalizedQuery, queryTokens)
      if (score < minimumScore) continue
      if (!best || score > best.score) {
        best = { exploration, score, reason: 'similar' }
      }
    }

    return best
  }

  vote(id: string, delta: 1 | -1): Exploration | null {
    const index = this.explorations.findIndex(exploration => exploration.id === id)
    if (index === -1) return null

    const updated: Exploration = {
      ...this.explorations[index],
      votes: this.explorations[index].votes + delta,
    }

    this.setExplorations(this.explorations.map((exploration, currentIndex) =>
      currentIndex === index ? updated : exploration,
    ))
    this.schedulePersist()
    return updated
  }

  verify(id: string, verified: boolean): Exploration | null {
    const index = this.explorations.findIndex(e => e.id === id)
    if (index === -1) return null

    const updated: Exploration = { ...this.explorations[index], verified }
    this.setExplorations(this.explorations.map((e, i) => (i === index ? updated : e)))
    this.schedulePersist()
    return updated
  }

  getById(id: string): Exploration | null {
    return this.byId.get(id) ?? null
  }

  private loadFromDisk(): void {
    try {
      if (!existsSync(DATA_FILE)) return
      const raw = readFileSync(DATA_FILE, 'utf-8')
      const parsed = JSON.parse(raw) as unknown
      if (Array.isArray(parsed)) {
        this.setExplorations(
          parsed
            .map(item => hydrateExploration(item as Exploration))
            .slice(0, MAX_EXPLORATIONS),
        )
      }
    } catch (error) {
      // eslint-disable-next-line no-console
      console.warn('Failed to load exploration history from disk, starting with an empty store.', error)
      this.setExplorations([])
    }
  }

  private schedulePersist(): void {
    if (this.persistTimer) clearTimeout(this.persistTimer)
    this.persistTimer = setTimeout(() => this.persistToDisk(), 1000)
  }

  private persistToDisk(): void {
    try {
      if (!existsSync(DATA_FILE_DIR)) {
        mkdirSync(DATA_FILE_DIR, { recursive: true })
      }
      const tempFile = `${DATA_FILE}.${process.pid}.tmp`
      writeFileSync(tempFile, JSON.stringify(this.explorations, null, 2), 'utf-8')
      rmSync(DATA_FILE, { force: true })
      renameSync(tempFile, DATA_FILE)
    } catch (error) {
      // eslint-disable-next-line no-console
      console.warn('Failed to persist exploration history to disk.', error)
    }
  }

  private setExplorations(explorations: Exploration[]): void {
    this.explorations = explorations
    this.byId = new Map(explorations.map(exploration => [exploration.id, exploration]))
    this.byNormalizedQuery = new Map(
      explorations.map(exploration => [exploration.normalizedQuery, exploration]),
    )
  }
}
