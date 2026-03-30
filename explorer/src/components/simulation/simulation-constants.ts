import type { SimulationConfig } from '../../lib/simulation-api'
import { parseBlocks, type Block } from '../../types/blocks'
import type { SimulationArtifactBundle } from '../../types/simulation-view'

export const DEFAULT_CONFIG: SimulationConfig = {
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

const PAPER_BASELINE_PRESET: Partial<SimulationConfig> = {
  validators: 1000,
  slots: 10000,
  distribution: 'homogeneous',
  sourcePlacement: 'homogeneous',
  migrationCost: 0.002,
  attestationThreshold: 2 / 3,
  slotTime: 12,
}

export const PRESETS: ReadonlyArray<{
  readonly label: string
  readonly description: string
  readonly config: Partial<SimulationConfig>
}> = [
  {
    label: 'Paper SSP',
    description: 'SSP with the paper-style 10,000-slot and 0.002 ETH reference setup.',
    config: { ...PAPER_BASELINE_PRESET, paradigm: 'SSP' },
  },
  {
    label: 'Paper MSP',
    description: 'MSP with the same paper-style reference setup for direct comparison.',
    config: { ...PAPER_BASELINE_PRESET, paradigm: 'MSP' },
  },
  {
    label: 'SE1 Aligned',
    description: 'Paper-style run with latency-aligned sources.',
    config: { ...PAPER_BASELINE_PRESET, sourcePlacement: 'latency-aligned' },
  },
  {
    label: 'SE1 Misaligned',
    description: 'Paper-style run with latency-misaligned sources.',
    config: { ...PAPER_BASELINE_PRESET, sourcePlacement: 'latency-misaligned' },
  },
  {
    label: 'SE2 Real ETH',
    description: 'Paper-style run with the heterogeneous Ethereum validator start.',
    config: { ...PAPER_BASELINE_PRESET, distribution: 'heterogeneous' },
  },
  {
    label: 'EIP-7782',
    description: 'Paper-style run with 6-second slots.',
    config: { ...PAPER_BASELINE_PRESET, slotTime: 6 },
  },
]

export const OVERVIEW_BUNDLES: ReadonlyArray<{
  readonly bundle: SimulationArtifactBundle
  readonly label: string
  readonly description: string
}> = [
  {
    bundle: 'core-outcomes',
    label: 'Core outcomes',
    description: 'MEV, supermajority success, and failed proposal trends.',
  },
  {
    bundle: 'timing-and-attestation',
    label: 'Timing and attestations',
    description: 'Proposal latency and aggregate attestation behavior.',
  },
  {
    bundle: 'geography-overview',
    label: 'Geography',
    description: 'Final regional concentration and top-region table.',
  },
]

export const COPY_RESET_DELAY_MS = 1600
const PARSED_ARTIFACT_CACHE_PREFIX = 'simulation_lab_parsed_artifact:'

export function formatNumber(value: number, digits = 2): string {
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${formatNumber(bytes / 1024, 1)} KB`
  return `${formatNumber(bytes / (1024 * 1024), 1)} MB`
}

export function paperScenarioLabels(config: SimulationConfig): string[] {
  const labels: string[] = []

  if (config.distribution === 'heterogeneous' && config.sourcePlacement !== 'homogeneous') {
    labels.push('SE3 joint heterogeneity')
  } else if (config.distribution === 'heterogeneous') {
    labels.push('SE2 heterogeneous validators')
  } else if (config.distribution === 'homogeneous-gcp') {
    labels.push('Equal per-GCP validator start')
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

export function readOrCreateClientId(): string {
  const storageKey = 'simulation_lab_client_id'
  const existing = window.localStorage.getItem(storageKey)
  if (existing) return existing
  const created = window.crypto.randomUUID()
  window.localStorage.setItem(storageKey, created)
  return created
}

export function readSessionArtifactBlocks(cacheKey: string): readonly Block[] | null {
  try {
    const stored = window.sessionStorage.getItem(`${PARSED_ARTIFACT_CACHE_PREFIX}${cacheKey}`)
    if (!stored) return null
    const parsed = JSON.parse(stored) as unknown
    return Array.isArray(parsed) ? parseBlocks(parsed) : null
  } catch {
    return null
  }
}

export function writeSessionArtifactBlocks(cacheKey: string, blocks: readonly Block[]): void {
  try {
    window.sessionStorage.setItem(
      `${PARSED_ARTIFACT_CACHE_PREFIX}${cacheKey}`,
      JSON.stringify(blocks),
    )
  } catch {
    // Ignore storage exhaustion and keep the in-memory cache path.
  }
}
