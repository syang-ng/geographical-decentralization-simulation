import type { Block } from '../types/blocks'
import type { SimulationViewSpec } from '../types/simulation-view'

const API_BASE = import.meta.env.VITE_API_BASE_URL
  ? `${import.meta.env.VITE_API_BASE_URL}/api`
  : '/api'

export interface SimulationConfig {
  readonly paradigm: 'SSP' | 'MSP'
  readonly validators: number
  readonly slots: number
  readonly distribution: 'homogeneous' | 'homogeneous-gcp' | 'heterogeneous' | 'random'
  readonly sourcePlacement: 'homogeneous' | 'latency-aligned' | 'latency-misaligned'
  readonly migrationCost: number
  readonly attestationThreshold: number
  readonly slotTime: number
  readonly seed: number
}

export interface SimulationSummary {
  readonly slotsRecorded: number
  readonly attestationCutoffMs: number
  readonly finalAverageMev: number
  readonly finalSupermajoritySuccess: number
  readonly finalFailedBlockProposals: number
  readonly finalUtilityIncrease: number
  readonly topRegions: ReadonlyArray<{
    readonly name: string
    readonly count: number
  }>
}

export interface SimulationArtifact {
  readonly name: string
  readonly label: string
  readonly kind: 'timeseries' | 'map' | 'table' | 'raw'
  readonly description: string
  readonly contentType: string
  readonly bytes: number
  readonly gzipBytes: number | null
  readonly sha256: string
  readonly lazy: boolean
  readonly renderable: boolean
}

export interface SimulationManifest {
  readonly jobId: string
  readonly configHash: string
  readonly cacheKey: string
  readonly cacheHit: boolean
  readonly runtimeSeconds: number
  readonly outputDir: string
  readonly config: SimulationConfig
  readonly summary: SimulationSummary
  readonly artifacts: ReadonlyArray<SimulationArtifact>
}

export interface SimulationJob {
  readonly id: string
  readonly status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'
  readonly createdAt: string
  readonly updatedAt: string
  readonly configHash: string
  readonly queuePosition: number | null
  readonly cacheHit: boolean | null
  readonly error: string | null
  readonly config: SimulationConfig
  readonly manifest?: SimulationManifest
}

export interface SimulationCopilotResponse {
  readonly summary: string
  readonly mode: SimulationViewSpec['mode']
  readonly guidance?: string
  readonly truthBoundary: {
    readonly label: string
    readonly detail: string
  }
  readonly suggestedPrompts: readonly string[]
  readonly proposedConfig?: SimulationConfig
  readonly viewSpec: SimulationViewSpec
  readonly blocks: readonly Block[]
  readonly model: string
  readonly cached: boolean
}

interface ApiErrorBody {
  readonly error?: string
}

async function parseError(res: Response): Promise<Error> {
  const body = await res.json().catch(() => ({ error: res.statusText })) as ApiErrorBody
  return new Error(body.error ?? res.statusText)
}

export async function submitSimulation(config: SimulationConfig): Promise<SimulationJob> {
  return submitSimulationForClient(config, null)
}

export async function submitSimulationForClient(
  config: SimulationConfig,
  clientId: string | null,
): Promise<SimulationJob> {
  const res = await fetch(`${API_BASE}/simulations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...config, clientId }),
  })

  if (!res.ok) {
    throw await parseError(res)
  }

  return await res.json() as SimulationJob
}

export async function cancelSimulationJob(jobId: string): Promise<SimulationJob> {
  const res = await fetch(`${API_BASE}/simulations/${jobId}/cancel`, {
    method: 'POST',
  })
  if (!res.ok) {
    throw await parseError(res)
  }
  return await res.json() as SimulationJob
}

export async function getSimulationJob(jobId: string): Promise<SimulationJob> {
  const res = await fetch(`${API_BASE}/simulations/${jobId}`)
  if (!res.ok) {
    throw await parseError(res)
  }
  return await res.json() as SimulationJob
}

export async function getSimulationManifest(jobId: string): Promise<SimulationManifest> {
  const res = await fetch(`${API_BASE}/simulations/${jobId}/manifest`)
  if (!res.ok) {
    throw await parseError(res)
  }
  return await res.json() as SimulationManifest
}

export async function getSimulationArtifact(jobId: string, artifactName: string): Promise<string> {
  const res = await fetch(`${API_BASE}/simulations/${jobId}/artifacts/${encodeURIComponent(artifactName)}`)
  if (!res.ok) {
    throw await parseError(res)
  }
  return await res.text()
}

export async function submitSimulationCopilot(request: {
  question: string
  currentJobId?: string | null
  currentConfig?: SimulationConfig | null
}): Promise<SimulationCopilotResponse> {
  const res = await fetch(`${API_BASE}/simulation-copilot`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })

  if (!res.ok) {
    throw await parseError(res)
  }

  return await res.json() as SimulationCopilotResponse
}
