/**
 * In-memory exploration store with JSON file persistence.
 * Auto-saves to server/data/explorations.json after each mutation (debounced 1s).
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const DATA_DIR = join(__dirname, 'data')
const DATA_FILE = join(DATA_DIR, 'explorations.json')

export interface Exploration {
  readonly id: string
  readonly query: string
  readonly summary: string
  readonly blocks: unknown[]
  readonly followUps: string[]
  readonly model: string
  readonly cached: boolean
  readonly votes: number
  readonly createdAt: string
  readonly paradigmTags: string[]
  readonly experimentTags: string[]
}

type SortOption = 'recent' | 'top'

interface ListOptions {
  readonly sort?: SortOption
  readonly limit?: number
  readonly search?: string
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

function matchesSearch(exploration: Exploration, query: string): boolean {
  const lower = query.toLowerCase()
  return (
    exploration.query.toLowerCase().includes(lower) ||
    exploration.summary.toLowerCase().includes(lower)
  )
}

export class ExplorationStore {
  private explorations: Exploration[] = []
  private persistTimer: ReturnType<typeof setTimeout> | null = null

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
    const exploration: Exploration = {
      id: crypto.randomUUID(),
      query: data.query,
      summary: data.summary,
      blocks: data.blocks,
      followUps: data.followUps,
      model: data.model,
      cached: data.cached,
      votes: 0,
      createdAt: new Date().toISOString(),
      paradigmTags: extractParadigmTags(data.blocks),
      experimentTags: extractExperimentTags(data.blocks),
    }

    this.explorations = [exploration, ...this.explorations]
    this.schedulePersist()
    return exploration
  }

  list(options?: ListOptions): Exploration[] {
    const sort = options?.sort ?? 'recent'
    const limit = options?.limit ?? 50
    const search = options?.search

    let results = search
      ? this.explorations.filter(e => matchesSearch(e, search))
      : [...this.explorations]

    if (sort === 'top') {
      results = results.toSorted((a, b) => b.votes - a.votes)
    } else {
      results = results.toSorted(
        (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
      )
    }

    return results.slice(0, limit)
  }

  vote(id: string, delta: 1 | -1): Exploration | null {
    const index = this.explorations.findIndex(e => e.id === id)
    if (index === -1) return null

    const updated: Exploration = {
      ...this.explorations[index],
      votes: this.explorations[index].votes + delta,
    }

    this.explorations = this.explorations.map((e, i) => (i === index ? updated : e))
    this.schedulePersist()
    return updated
  }

  getById(id: string): Exploration | null {
    return this.explorations.find(e => e.id === id) ?? null
  }

  private loadFromDisk(): void {
    try {
      if (!existsSync(DATA_FILE)) return
      const raw = readFileSync(DATA_FILE, 'utf-8')
      const parsed = JSON.parse(raw) as unknown
      if (Array.isArray(parsed)) {
        this.explorations = parsed as Exploration[]
      }
    } catch {
      // Start fresh if file is corrupt
      this.explorations = []
    }
  }

  private schedulePersist(): void {
    if (this.persistTimer) clearTimeout(this.persistTimer)
    this.persistTimer = setTimeout(() => this.persistToDisk(), 1000)
  }

  private persistToDisk(): void {
    try {
      if (!existsSync(DATA_DIR)) {
        mkdirSync(DATA_DIR, { recursive: true })
      }
      writeFileSync(DATA_FILE, JSON.stringify(this.explorations, null, 2), 'utf-8')
    } catch {
      // Silently fail — in-memory store remains authoritative
    }
  }
}
