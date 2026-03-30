import type { Block } from '../types/blocks'
import { parseBlocks } from '../types/blocks'

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? `${import.meta.env.VITE_API_BASE_URL}/api`
  : '/api'

export type ExploreSource = 'curated' | 'history' | 'generated'

export interface ExploreProvenance {
  readonly source: ExploreSource
  readonly label: string
  readonly detail: string
  readonly canonical: boolean
  readonly topicId?: string
  readonly explorationId?: string
  readonly similarityScore?: number
}

export interface ExploreResponse {
  readonly summary: string
  readonly blocks: readonly Block[]
  readonly followUps: readonly string[]
  readonly model: string
  readonly cached: boolean
  readonly provenance: ExploreProvenance
}

export interface ExploreError {
  readonly error: string
  readonly status: number
}

export interface ApiHealth {
  readonly status: 'ok'
  readonly tools: number
  readonly simulationCopilotTools: number
  readonly anthropicEnabled: boolean
  readonly anthropicModel: string | null
  readonly envFileLoaded: boolean
  readonly simulations: {
    readonly readyWorkers: number
    readonly busyWorkers: number
    readonly queuedJobs: number
    readonly cacheEntries: number
    readonly prewarm?: {
      readonly enabled: boolean
      readonly running: boolean
      readonly startedAt: string | null
      readonly finishedAt: string | null
      readonly completed: number
      readonly total: number
      readonly lastError: string | null
    }
  }
}

export type ExploreResult =
  | { ok: true; data: ExploreResponse }
  | { ok: false; error: ExploreError }

interface HistoryEntry {
  readonly query: string
  readonly summary: string
}

export async function explore(
  query: string,
  history: readonly HistoryEntry[] = [],
): Promise<ExploreResult> {
  try {
    const res = await fetch(`${API_BASE}/explore`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, history }),
    })

    if (!res.ok) {
      const body = await res.json().catch(() => ({ error: res.statusText })) as Record<string, unknown>
      return {
        ok: false,
        error: { error: (body.error as string) ?? res.statusText, status: res.status },
      }
    }

    const raw = await res.json().catch(() => ({})) as Record<string, unknown>
    const blocks = parseBlocks((raw.blocks as unknown[]) ?? [])

    return {
      ok: true,
      data: {
        summary: typeof raw.summary === 'string' ? raw.summary : '',
        blocks,
        followUps: Array.isArray(raw.followUps)
          ? raw.followUps.filter((value): value is string => typeof value === 'string')
          : [],
        model: typeof raw.model === 'string' ? raw.model : '',
        cached: typeof raw.cached === 'boolean' ? raw.cached : false,
        provenance: {
          source: raw.provenance && typeof raw.provenance === 'object' && typeof (raw.provenance as Record<string, unknown>).source === 'string'
            ? ((raw.provenance as Record<string, unknown>).source as ExploreSource)
            : 'generated',
          label: raw.provenance && typeof raw.provenance === 'object' && typeof (raw.provenance as Record<string, unknown>).label === 'string'
            ? ((raw.provenance as Record<string, unknown>).label as string)
            : 'Fresh response',
          detail: raw.provenance && typeof raw.provenance === 'object' && typeof (raw.provenance as Record<string, unknown>).detail === 'string'
            ? ((raw.provenance as Record<string, unknown>).detail as string)
            : '',
          canonical: raw.provenance && typeof raw.provenance === 'object' && typeof (raw.provenance as Record<string, unknown>).canonical === 'boolean'
            ? ((raw.provenance as Record<string, unknown>).canonical as boolean)
            : false,
          topicId: raw.provenance && typeof raw.provenance === 'object' && typeof (raw.provenance as Record<string, unknown>).topicId === 'string'
            ? ((raw.provenance as Record<string, unknown>).topicId as string)
            : undefined,
          explorationId: raw.provenance && typeof raw.provenance === 'object' && typeof (raw.provenance as Record<string, unknown>).explorationId === 'string'
            ? ((raw.provenance as Record<string, unknown>).explorationId as string)
            : undefined,
          similarityScore: raw.provenance && typeof raw.provenance === 'object' && typeof (raw.provenance as Record<string, unknown>).similarityScore === 'number'
            ? ((raw.provenance as Record<string, unknown>).similarityScore as number)
            : undefined,
        },
      },
    }
  } catch (err) {
    return {
      ok: false,
      error: {
        error: err instanceof Error ? err.message : 'Network error',
        status: 0,
      },
    }
  }
}

export async function getApiHealth(): Promise<ApiHealth> {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) {
    throw new Error(`Failed to fetch API health: ${res.statusText}`)
  }

  return await res.json() as ApiHealth
}

export async function healthCheck(): Promise<boolean> {
  try {
    await getApiHealth()
    return true
  } catch {
    return false
  }
}

// --- Exploration history ---

export interface Exploration {
  readonly id: string
  readonly query: string
  readonly summary: string
  readonly blocks: readonly Block[]
  readonly followUps: readonly string[]
  readonly model: string
  readonly cached: boolean
  readonly source: 'generated'
  readonly votes: number
  readonly createdAt: string
  readonly paradigmTags: readonly string[]
  readonly experimentTags: readonly string[]
  readonly verified: boolean
}

interface RawExploration {
  readonly id: string
  readonly query: string
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

function parseExploration(raw: RawExploration): Exploration {
  return {
    ...raw,
    blocks: parseBlocks(raw.blocks ?? []),
  }
}

export async function listExplorations(options?: {
  sort?: 'recent' | 'top'
  limit?: number
  search?: string
}): Promise<Exploration[]> {
  const params = new URLSearchParams()
  if (options?.sort) params.set('sort', options.sort)
  if (options?.limit) params.set('limit', String(options.limit))
  if (options?.search) params.set('search', options.search)

  const qs = params.toString()
  const url = `${API_BASE}/explorations${qs ? `?${qs}` : ''}`
  const res = await fetch(url)

  if (!res.ok) {
    throw new Error(`Failed to list explorations: ${res.statusText}`)
  }

  const raw = (await res.json()) as RawExploration[]
  return raw.map(parseExploration)
}

export async function getExploration(id: string): Promise<Exploration> {
  const res = await fetch(`${API_BASE}/explorations/${encodeURIComponent(id)}`)
  if (!res.ok) {
    throw new Error(`Failed to fetch exploration: ${res.statusText}`)
  }
  const raw = (await res.json()) as RawExploration
  return parseExploration(raw)
}

export async function voteExploration(id: string, delta: 1 | -1): Promise<Exploration> {
  const res = await fetch(`${API_BASE}/explorations/${id}/vote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ delta }),
  })

  if (!res.ok) {
    throw new Error(`Failed to vote: ${res.statusText}`)
  }

  const raw = (await res.json()) as RawExploration
  return parseExploration(raw)
}
