import { useEffect, useMemo, useRef, useState, startTransition } from 'react'
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Ban, Check, Copy, Play, RotateCcw, Sparkles } from 'lucide-react'
import { BlockCanvas } from '../components/explore/BlockCanvas'
import { cn } from '../lib/cn'
import { getApiHealth } from '../lib/api'
import {
  cancelSimulationJob,
  getSimulationArtifact,
  getSimulationManifest,
  getSimulationJob,
  submitSimulationCopilot,
  submitSimulationForClient,
  type SimulationArtifact,
  type SimulationCopilotResponse,
  type SimulationConfig,
  type SimulationJob,
} from '../lib/simulation-api'
import { buildSimulationArtifactBundle } from '../lib/simulation-artifact-blocks'
import { SPRING } from '../lib/theme'
import type { Block } from '../types/blocks'
import type { SimulationArtifactBundle } from '../types/simulation-view'

interface WorkerSuccess {
  readonly id: number
  readonly ok: true
  readonly blocks: readonly Block[]
}

interface WorkerFailure {
  readonly id: number
  readonly ok: false
  readonly error: string
}

const DEFAULT_CONFIG: SimulationConfig = {
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

const PRESETS: Array<{ label: string; description: string; config: Partial<SimulationConfig> }> = [
  {
    label: 'Baseline SSP',
    description: 'External block building with the upstream homogeneous baseline.',
    config: { paradigm: 'SSP', distribution: 'homogeneous', sourcePlacement: 'homogeneous' },
  },
  {
    label: 'Baseline MSP',
    description: 'Local block building with the same upstream baseline geography.',
    config: { paradigm: 'MSP', distribution: 'homogeneous', sourcePlacement: 'homogeneous' },
  },
  {
    label: 'Latency-aligned',
    description: 'SE1 exact-mode aligned information sources.',
    config: { sourcePlacement: 'latency-aligned' },
  },
  {
    label: 'Latency-misaligned',
    description: 'SE1 exact-mode peripheral information sources.',
    config: { sourcePlacement: 'latency-misaligned' },
  },
  {
    label: 'Real ETH Start',
    description: 'Heterogeneous validator distribution from the paper dataset.',
    config: { distribution: 'heterogeneous' },
  },
  {
    label: 'EIP-7782',
    description: 'Shorter slots with the exact path and paper cutoff.',
    config: { slotTime: 6 },
  },
]

const THRESHOLD_OPTIONS = [
  { label: '1/3', value: 1 / 3 },
  { label: '1/2', value: 1 / 2 },
  { label: '2/3', value: 2 / 3 },
  { label: '4/5', value: 4 / 5 },
]

const SLOT_OPTIONS = [
  { label: '6s', value: 6 },
  { label: '8s', value: 8 },
  { label: '12s', value: 12 },
]

const OVERVIEW_BUNDLES: ReadonlyArray<{
  bundle: SimulationArtifactBundle
  label: string
  description: string
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

const BUNDLE_ARTIFACT_NAMES: Record<SimulationArtifactBundle, readonly string[]> = {
  'core-outcomes': [
    'avg_mev.json',
    'supermajority_success.json',
    'failed_block_proposals.json',
  ],
  'timing-and-attestation': [
    'proposal_time_avg.json',
    'attestation_sum.json',
  ],
  'geography-overview': [
    'top_regions_final.json',
  ],
}

const COPY_RESET_DELAY_MS = 1600

function formatNumber(value: number, digits = 2): string {
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${formatNumber(bytes / 1024, 1)} KB`
  return `${formatNumber(bytes / (1024 * 1024), 1)} MB`
}

function attestationCutoffMs(slotTime: number): number {
  if (slotTime === 6) return 3000
  if (slotTime === 8) return 4000
  return 4000
}

function paperScenarioLabels(config: SimulationConfig): string[] {
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

function buildRunSummary(manifest: { cacheHit: boolean; cacheKey: string; config: SimulationConfig; runtimeSeconds: number; summary: { finalAverageMev: number; finalSupermajoritySuccess: number } }): string {
  return [
    `Exact simulation run`,
    `Paradigm: ${manifest.config.paradigm}`,
    `Scenario: ${paperScenarioLabels(manifest.config).join(' | ')}`,
    `Seed: ${manifest.config.seed}`,
    `Validators: ${manifest.config.validators}`,
    `Slots: ${manifest.config.slots}`,
    `Runtime: ${formatNumber(manifest.runtimeSeconds, 2)}s`,
    `Final average MEV: ${formatNumber(manifest.summary.finalAverageMev, 4)} ETH`,
    `Final supermajority success: ${formatNumber(manifest.summary.finalSupermajoritySuccess, 2)}%`,
    `Execution: ${manifest.cacheHit ? 'Exact cache hit' : 'Fresh exact run'}`,
    `Cache key: ${manifest.cacheKey}`,
  ].join('\n')
}

function selectDefaultArtifact(artifacts: readonly SimulationArtifact[]): string | null {
  const preferred = artifacts.find(artifact => artifact.renderable && !artifact.lazy)
  if (preferred) return preferred.name
  return artifacts.find(artifact => artifact.renderable)?.name ?? null
}

function readOrCreateClientId(): string {
  const storageKey = 'simulation_lab_client_id'
  const existing = window.localStorage.getItem(storageKey)
  if (existing) return existing
  const created = window.crypto.randomUUID()
  window.localStorage.setItem(storageKey, created)
  return created
}

export function SimulationLabPage() {
  const queryClient = useQueryClient()
  const [config, setConfig] = useState<SimulationConfig>({ ...DEFAULT_CONFIG })
  const [clientId] = useState(readOrCreateClientId)
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)
  const [selectedArtifactName, setSelectedArtifactName] = useState<string | null>(null)
  const [selectedBundle, setSelectedBundle] = useState<SimulationArtifactBundle>('core-outcomes')
  const [parsedBlocks, setParsedBlocks] = useState<readonly Block[]>([])
  const [parsedArtifactCache, setParsedArtifactCache] = useState<Record<string, readonly Block[]>>({})
  const [parseError, setParseError] = useState<string | null>(null)
  const [isParsing, setIsParsing] = useState(false)
  const [copyState, setCopyState] = useState<'config' | 'run' | null>(null)
  const [copilotQuestion, setCopilotQuestion] = useState('')
  const [copilotResponse, setCopilotResponse] = useState<SimulationCopilotResponse | null>(null)

  const workerRef = useRef<Worker | null>(null)
  const workerRequestIdRef = useRef(0)

  useEffect(() => {
    const worker = new Worker(new URL('../workers/simulationArtifactWorker.ts', import.meta.url), { type: 'module' })
    workerRef.current = worker
    return () => {
      worker.terminate()
      workerRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!currentJobId) return

    const stream = new EventSource(`/api/simulations/${currentJobId}/events`)
    const handleSnapshot = (event: MessageEvent<string>) => {
      const snapshot = JSON.parse(event.data) as SimulationJob
      queryClient.setQueryData(['simulation-job', currentJobId], snapshot)
      if (snapshot.manifest) {
        queryClient.setQueryData(['simulation-manifest', currentJobId], snapshot.manifest)
      }
    }

    stream.addEventListener('snapshot', handleSnapshot as EventListener)

    return () => {
      stream.removeEventListener('snapshot', handleSnapshot as EventListener)
      stream.close()
    }
  }, [currentJobId, queryClient])

  useEffect(() => {
    setCopilotResponse(null)
  }, [currentJobId])

  const updateConfig = <K extends keyof SimulationConfig>(key: K, value: SimulationConfig[K]) => {
    setConfig(previous => ({ ...previous, [key]: value }))
  }

  const applyPreset = (preset: Partial<SimulationConfig>) => {
    setConfig(previous => ({ ...previous, ...preset }))
  }

  const resetConfig = () => {
    setConfig({ ...DEFAULT_CONFIG })
  }

  const submitMutation = useMutation({
    mutationFn: (nextConfig: SimulationConfig) => submitSimulationForClient(nextConfig, clientId),
    onSuccess: job => {
      queryClient.setQueryData(['simulation-job', job.id], job)
      if (job.manifest) {
        queryClient.setQueryData(['simulation-manifest', job.id], job.manifest)
      }
      setCurrentJobId(job.id)
      setSelectedBundle('core-outcomes')
      setSelectedArtifactName(null)
      setParsedBlocks([])
      setParsedArtifactCache({})
      setParseError(null)
    },
  })

  const cancelMutation = useMutation({
    mutationFn: cancelSimulationJob,
    onSuccess: job => {
      queryClient.setQueryData(['simulation-job', job.id], job)
    },
  })

  const copilotMutation = useMutation({
    mutationFn: (question: string) => submitSimulationCopilot({
      question,
      currentJobId,
      currentConfig: manifest?.config ?? config,
    }),
    onSuccess: response => {
      setCopilotResponse(response)
    },
  })

  const jobQuery = useQuery({
    queryKey: ['simulation-job', currentJobId],
    queryFn: () => getSimulationJob(currentJobId!),
    enabled: Boolean(currentJobId),
  })

  const apiHealthQuery = useQuery({
    queryKey: ['api-health'],
    queryFn: getApiHealth,
    staleTime: 30_000,
  })

  const manifestQuery = useQuery({
    queryKey: ['simulation-manifest', currentJobId],
    queryFn: () => getSimulationManifest(currentJobId!),
    enabled: jobQuery.data?.status === 'completed' && !jobQuery.data?.manifest,
  })

  const manifest = jobQuery.data?.manifest ?? manifestQuery.data ?? null
  const eagerArtifacts = useMemo(
    () => manifest?.artifacts.filter(artifact => artifact.renderable && !artifact.lazy) ?? [],
    [manifest],
  )

  const eagerArtifactQueries = useQueries({
    queries: eagerArtifacts.map(artifact => ({
      queryKey: ['simulation-artifact', currentJobId, artifact.name],
      queryFn: () => getSimulationArtifact(currentJobId!, artifact.name),
      enabled: Boolean(currentJobId),
      staleTime: Infinity,
    })),
  })

  const eagerArtifactTexts = useMemo(() => {
    const entries = eagerArtifacts.flatMap((artifact, index) => {
      const rawText = eagerArtifactQueries[index]?.data
      return typeof rawText === 'string'
        ? [[artifact.name, rawText] as const]
        : []
    })

    return Object.fromEntries(entries) as Partial<Record<SimulationArtifact['name'], string>>
  }, [eagerArtifactQueries, eagerArtifacts])

  const overviewBlocks = useMemo(
    () => buildSimulationArtifactBundle(selectedBundle, eagerArtifactTexts),
    [eagerArtifactTexts, selectedBundle],
  )

  const isOverviewLoading = useMemo(() => {
    const relevantArtifacts = BUNDLE_ARTIFACT_NAMES[selectedBundle]
    return relevantArtifacts.some(artifactName => {
      const artifactIndex = eagerArtifacts.findIndex(artifact => artifact.name === artifactName)
      if (artifactIndex < 0) return false
      return eagerArtifactQueries[artifactIndex]?.isFetching ?? false
    })
  }, [eagerArtifactQueries, eagerArtifacts, selectedBundle])

  useEffect(() => {
    if (!manifest) return
    if (selectedArtifactName) return
    const nextArtifact = selectDefaultArtifact(manifest.artifacts)
    if (nextArtifact) {
      setSelectedArtifactName(nextArtifact)
    }
  }, [manifest, selectedArtifactName])

  const selectedArtifact = useMemo(
    () => manifest?.artifacts.find(artifact => artifact.name === selectedArtifactName) ?? null,
    [manifest, selectedArtifactName],
  )

  const artifactQuery = useQuery({
    queryKey: ['simulation-artifact', currentJobId, selectedArtifactName],
    queryFn: () => getSimulationArtifact(currentJobId!, selectedArtifactName!),
    enabled: Boolean(currentJobId && selectedArtifactName && selectedArtifact?.renderable),
  })
  const selectedArtifactRawText = selectedArtifactName
    ? eagerArtifactTexts[selectedArtifactName] ?? artifactQuery.data ?? null
    : null

  useEffect(() => {
    if (!selectedArtifact || !selectedArtifactRawText || !workerRef.current) {
      return
    }

    const cachedBlocks = parsedArtifactCache[selectedArtifact.name]
    if (cachedBlocks) {
      setParsedBlocks(cachedBlocks)
      setParseError(null)
      setIsParsing(false)
      return
    }

    const worker = workerRef.current
    const requestId = ++workerRequestIdRef.current
    setIsParsing(true)
    setParseError(null)

    const handleMessage = (event: MessageEvent<WorkerSuccess | WorkerFailure>) => {
      if (event.data.id !== requestId) return
      worker.removeEventListener('message', handleMessage as EventListener)
      if (event.data.ok) {
        const nextBlocks = event.data.blocks
        setParsedBlocks(nextBlocks)
        setParsedArtifactCache(previous => ({
          ...previous,
          [selectedArtifact.name]: nextBlocks,
        }))
        setParseError(null)
      } else {
        setParsedBlocks([])
        setParseError(event.data.error)
      }
      setIsParsing(false)
    }

    worker.addEventListener('message', handleMessage as EventListener)
    worker.postMessage({
      id: requestId,
      artifact: {
        name: selectedArtifact.name,
        label: selectedArtifact.label,
        kind: selectedArtifact.kind,
      },
      rawText: selectedArtifactRawText,
    })

    return () => {
      worker.removeEventListener('message', handleMessage as EventListener)
    }
  }, [parsedArtifactCache, selectedArtifact, selectedArtifactRawText])

  const status = submitMutation.isPending
    ? 'submitting'
    : jobQuery.data?.status ?? 'idle'

  const onSubmit = () => {
    submitMutation.mutate(config)
  }

  const onCancel = () => {
    if (!currentJobId) return
    cancelMutation.mutate(currentJobId)
  }

  const onSelectArtifact = (artifactName: string) => {
    startTransition(() => {
      setSelectedArtifactName(artifactName)
      setParsedBlocks([])
      setParseError(null)
    })
  }

  const copyToClipboard = async (text: string, kind: 'config' | 'run') => {
    await navigator.clipboard.writeText(text)
    setCopyState(kind)
    window.setTimeout(() => {
      setCopyState(previous => (previous === kind ? null : previous))
    }, COPY_RESET_DELAY_MS)
  }

  const canCancel = jobQuery.data?.status === 'queued' || jobQuery.data?.status === 'running'
  const copilotAvailable = apiHealthQuery.data?.anthropicEnabled ?? false
  const copilotDisabled = apiHealthQuery.isLoading || !copilotAvailable || copilotMutation.isPending
  const copilotPromptSuggestions = copilotResponse?.suggestedPrompts?.length
    ? copilotResponse.suggestedPrompts
    : manifest
      ? [
          'Show the MEV and supermajority charts from this run.',
          'Explain why these regions dominate in this exact result.',
          'What is the nearest paper-aligned follow-up to run next?',
        ]
      : [
          'Set up the baseline SSP run from the paper.',
          'Suggest an MSP comparison that stays within exact bounds.',
          'What can I vary without leaving the supported model?',
        ]

  return (
    <div>
      <div className="lab-stage mb-8 px-6 py-6">
        <div className="flex items-center gap-2 mb-2">
          <span className="w-2 h-2 rounded-full bg-accent" />
          <span className="text-xs text-muted">Simulation lab</span>
        </div>
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(260px,0.8fr)]">
          <div className="relative z-[1]">
            <h1 className="text-xl font-semibold text-text-primary">
              Run exact-mode simulations with an instrument-panel workflow.
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-muted">
              Configure parameters, pick a preset, and inspect exact replications of paper scenarios through precomputed overview bundles and raw artifacts.
            </p>
          </div>

          <div className="relative z-[1] grid gap-2 sm:grid-cols-3 lg:grid-cols-1">
            <div className="rounded-xl border border-border-subtle bg-white/88 px-3 py-3 backdrop-blur-sm">
              <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Default surface</div>
              <div className="mt-1 text-sm font-medium text-text-primary">Upstream aligned</div>
              <div className="mt-1 text-xs text-muted">1,000 validators, homogeneous baseline, exact mode.</div>
            </div>
            <div className="rounded-xl border border-border-subtle bg-white/88 px-3 py-3 backdrop-blur-sm">
              <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Rendering</div>
              <div className="mt-1 text-sm font-medium text-text-primary">Parallel overview bundles</div>
              <div className="mt-1 text-xs text-muted">Manifest-ready artifacts are prefetched and staged together.</div>
            </div>
            <div className="rounded-xl border border-border-subtle bg-white/88 px-3 py-3 backdrop-blur-sm">
              <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Truth boundary</div>
              <div className="mt-1 text-sm font-medium text-text-primary">Animation frames, not meaning</div>
              <div className="mt-1 text-xs text-muted">Motion is limited to entry and navigation, not data scaling.</div>
            </div>
          </div>
        </div>
      </div>

      <div className="mb-6">
        <span className="text-xs text-muted mb-2 block">
          Quick presets
        </span>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {PRESETS.map(preset => (
            <button
              key={preset.label}
              onClick={() => applyPreset(preset.config)}
              className="text-left bg-white border border-border-subtle rounded-lg p-2.5 hover:border-border-hover transition-all"
            >
              <div className="text-xs font-medium text-text-primary">{preset.label}</div>
              <div className="text-xs text-muted">{preset.description}</div>
            </button>
          ))}
        </div>
      </div>

      <div className="lab-stage p-5 mb-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="text-xs text-muted mb-1.5 block">
              Paradigm
            </label>
            <div className="flex gap-1">
              {(['SSP', 'MSP'] as const).map(paradigm => (
                <button
                  key={paradigm}
                  onClick={() => updateConfig('paradigm', paradigm)}
                  className={cn(
                    'flex-1 py-1.5 rounded-lg text-xs font-medium transition-all',
                    config.paradigm === paradigm
                      ? paradigm === 'SSP'
                        ? 'bg-white text-accent border border-accent'
                        : 'bg-white text-accent-warm border border-accent-warm'
                      : 'bg-white text-muted border border-border-subtle hover:border-border-hover',
                  )}
                >
                  {paradigm}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs text-muted mb-1.5 block">
              Validator Distribution
            </label>
            <select
              value={config.distribution}
              onChange={event => updateConfig('distribution', event.target.value as SimulationConfig['distribution'])}
              className="w-full bg-white border border-border-subtle rounded-lg px-3 py-1.5 text-xs text-text-primary outline-none focus:ring-1 focus:ring-accent"
            >
              <option value="homogeneous">Homogeneous (upstream baseline default)</option>
              <option value="homogeneous-gcp">Homogeneous per GCP region</option>
              <option value="heterogeneous">Heterogeneous (real ETH data)</option>
              <option value="random">Random</option>
            </select>
          </div>

          <div>
            <label className="text-xs text-muted mb-1.5 block">
              Information Source Placement
            </label>
            <select
              value={config.sourcePlacement}
              onChange={event => updateConfig('sourcePlacement', event.target.value as SimulationConfig['sourcePlacement'])}
              className="w-full bg-white border border-border-subtle rounded-lg px-3 py-1.5 text-xs text-text-primary outline-none focus:ring-1 focus:ring-accent"
            >
              <option value="homogeneous">Homogeneous</option>
              <option value="latency-aligned">Latency-aligned</option>
              <option value="latency-misaligned">Latency-misaligned</option>
            </select>
          </div>

          <div>
            <label className="text-xs text-muted mb-1.5 block">
              Seed
            </label>
            <input
              type="number"
              value={config.seed}
              min={0}
              max={2147483647}
              onChange={event => updateConfig('seed', Number(event.target.value))}
              className="w-full bg-white border border-border-subtle rounded-lg px-3 py-1.5 text-xs text-text-primary outline-none focus:ring-1 focus:ring-accent"
            />
          </div>

          <div>
            <label className="text-xs text-muted mb-1.5 block">
              Validators
            </label>
            <input
              type="number"
              min={1}
              max={1000}
              step={1}
              value={config.validators}
              onChange={event => updateConfig('validators', Number(event.target.value))}
              className="w-full bg-white border border-border-subtle rounded-lg px-3 py-1.5 text-xs text-text-primary outline-none focus:ring-1 focus:ring-accent"
            />
            <div className="mt-1 text-xs text-muted">Upstream defaults and paper baselines use 1,000 validators.</div>
          </div>

          <div>
            <label className="text-xs text-muted mb-1.5 block">
              Slots
            </label>
            <input
              type="number"
              min={1}
              max={10000}
              step={1}
              value={config.slots}
              onChange={event => updateConfig('slots', Number(event.target.value))}
              className="w-full bg-white border border-border-subtle rounded-lg px-3 py-1.5 text-xs text-text-primary outline-none focus:ring-1 focus:ring-accent"
            />
            <div className="mt-1 text-xs text-muted">The upstream presets run up to 10,000 slots; shorter runs remain exact but are noisier.</div>
          </div>

          <div>
            <label className="text-xs text-muted mb-1.5 block">
              Migration Cost: {config.migrationCost.toFixed(4)} ETH
            </label>
            <input
              type="range"
              min={0}
              max={0.005}
              step={0.0001}
              value={config.migrationCost}
              onChange={event => updateConfig('migrationCost', Number(event.target.value))}
              className="w-full accent-accent"
            />
            <div className="flex justify-between text-xs text-text-faint">
              <span>0</span>
              <span>0.005</span>
            </div>
          </div>

          <div>
            <label className="text-xs text-muted mb-1.5 block">
              Attestation Threshold (γ)
            </label>
            <div className="flex gap-1">
              {THRESHOLD_OPTIONS.map(option => (
                <button
                  key={option.label}
                  onClick={() => updateConfig('attestationThreshold', option.value)}
                  className={cn(
                    'flex-1 py-1.5 rounded-lg text-xs font-medium transition-all',
                    Math.abs(config.attestationThreshold - option.value) < 0.01
                      ? 'bg-white text-accent border border-accent'
                      : 'bg-white text-muted border border-border-subtle hover:border-border-hover',
                  )}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-xs text-muted mb-1.5 block">
              Slot Time (Δ)
            </label>
            <div className="flex gap-1">
              {SLOT_OPTIONS.map(option => (
                <button
                  key={option.label}
                  onClick={() => updateConfig('slotTime', option.value)}
                  className={cn(
                    'flex-1 py-1.5 rounded-lg text-xs font-medium transition-all',
                    config.slotTime === option.value
                      ? 'bg-white text-accent border border-accent'
                      : 'bg-white text-muted border border-border-subtle hover:border-border-hover',
                  )}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3 mt-5 pt-4 border-t border-border-subtle/50">
          <motion.button
            onClick={onSubmit}
            whileTap={{ scale: 0.98 }}
            transition={SPRING}
            disabled={submitMutation.isPending}
            className={cn(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all',
              'bg-accent text-white hover:bg-accent/80 disabled:opacity-60 disabled:cursor-not-allowed',
            )}
          >
            <Play className="w-3 h-3" />
            {submitMutation.isPending ? 'Submitting…' : 'Run Exact Simulation'}
          </motion.button>

          <button
            onClick={resetConfig}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs text-muted hover:text-text-primary transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
            Reset
          </button>

          {canCancel && (
            <button
              onClick={onCancel}
              disabled={cancelMutation.isPending}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs text-danger hover:text-danger transition-colors"
            >
              <Ban className="w-3 h-3" />
              Cancel
            </button>
          )}

          <div className="flex-1" />

          <div className="text-xs text-text-faint text-right">
            <div className="flex items-center gap-1 justify-end">
              <Sparkles className="w-3 h-3" />
              Exact mode only
            </div>
            <div>Slot cutoff: {attestationCutoffMs(config.slotTime)} ms</div>
            <div>{paperScenarioLabels(config).join(' · ')}</div>
          </div>
        </div>
      </div>

      {(currentJobId || submitMutation.isError) && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={SPRING}
          className="lab-panel rounded-xl p-5 mb-6"
        >
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div>
              <div className="text-xs text-muted mb-1">
                Job status
              </div>
              <div className="text-sm font-medium text-text-primary">
                {status === 'idle' && 'Ready'}
                {status === 'submitting' && 'Submitting configuration'}
                {status === 'queued' && 'Queued for exact execution'}
                {status === 'running' && 'Running exact simulation'}
                {status === 'completed' && 'Completed'}
                {status === 'failed' && 'Failed'}
                {status === 'cancelled' && 'Cancelled'}
              </div>
            </div>

            {jobQuery.data && (
              <div className="grid grid-cols-2 gap-3 text-xs text-muted min-w-[220px]">
                <div>
                  <span className="block text-xs text-text-faint">Queue</span>
                  {jobQuery.data.queuePosition ?? 'live'}
                </div>
                <div>
                  <span className="block text-xs text-text-faint">Cache</span>
                  {jobQuery.data.cacheHit ? 'hit' : 'fresh'}
                </div>
              </div>
            )}
          </div>

          {(submitMutation.error || jobQuery.data?.error || cancelMutation.error) && (
            <div className="mt-4 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
              {(submitMutation.error as Error | null)?.message
                ?? (cancelMutation.error as Error | null)?.message
                ?? jobQuery.data?.error}
            </div>
          )}
        </motion.div>
      )}

      <div className="lab-stage p-5 mb-6">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="text-xs text-muted mb-1">
              Simulation copilot
            </div>
            <div className="text-sm text-text-primary">
              {manifest
                ? 'Ask about the current exact run, or ask for a bounded next experiment.'
                : 'Ask for a paper-aligned exact run, or get redirected toward what the simulator can actually answer.'}
            </div>
          </div>
          <div className="max-w-xl text-xs text-muted">
            The copilot can reorder supported charts, add narrative, and suggest bounded configs.
            It cannot invent metrics or change the exact engine.
          </div>
        </div>

        <div className="mt-3 text-xs text-muted">
          {apiHealthQuery.isLoading
            ? 'Checking live copilot availability...'
            : copilotAvailable
              ? `Live model: ${apiHealthQuery.data?.anthropicModel}. Best prompts name a metric, artifact, scenario, or next experimental decision.`
              : 'Live copilot is offline. Add ANTHROPIC_API_KEY to explorer/.env to enable bounded Sonnet guidance.'}
        </div>

        <div className="mt-4 flex flex-col gap-3 lg:flex-row">
          <textarea
            value={copilotQuestion}
            onChange={event => setCopilotQuestion(event.target.value)}
            rows={3}
            placeholder={manifest
              ? 'Example: Show avg_mev, then supermajority_success, then explain the top regions.'
              : 'Example: Propose an exact SSP vs MSP comparison under shorter slots that stays within bounds.'}
            disabled={apiHealthQuery.isLoading || !copilotAvailable}
            className="min-h-[92px] flex-1 resize-y bg-white border border-border-subtle rounded-lg px-3 py-2 text-sm text-text-primary outline-none focus:ring-1 focus:ring-accent"
          />

          <div className="flex flex-col gap-2 lg:w-48">
            <button
              onClick={() => copilotMutation.mutate(copilotQuestion.trim())}
              disabled={!copilotQuestion.trim() || copilotDisabled}
              className={cn(
                'flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-xs font-medium transition-all',
                'bg-accent text-white hover:bg-accent/80 disabled:opacity-60 disabled:cursor-not-allowed',
              )}
            >
              <Sparkles className="w-3 h-3" />
              {apiHealthQuery.isLoading
                ? 'Checking...'
                : copilotMutation.isPending
                  ? 'Thinking...'
                  : 'Ask copilot'}
            </button>

            {copilotResponse?.proposedConfig && (
              <button
                onClick={() => setConfig({ ...copilotResponse.proposedConfig! })}
                className="rounded-lg border border-border-subtle bg-white px-3 py-2 text-xs text-text-primary hover:border-border-hover transition-colors"
              >
                Apply proposed config
              </button>
            )}
          </div>
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          {copilotPromptSuggestions.map(prompt => (
            <button
              key={prompt}
              onClick={() => {
                if (!copilotAvailable) return
                setCopilotQuestion(prompt)
                copilotMutation.mutate(prompt)
              }}
              disabled={!copilotAvailable}
              className="text-xs text-muted hover:text-text-primary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {prompt}
            </button>
          ))}
        </div>

        {copilotMutation.error && (
          <div className="mt-4 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
            {(copilotMutation.error as Error).message}
          </div>
        )}

        {copilotResponse && (
          <div className="mt-5 space-y-4">
            <div className="border-l-2 border-warning pl-4 py-2">
              <div className="flex items-center gap-1.5 text-xs font-medium text-text-primary mb-1">
                <span className="w-1.5 h-1.5 rounded-full bg-warning" />
                {copilotResponse.truthBoundary.label}
              </div>
              <div className="text-xs text-muted">
                {copilotResponse.truthBoundary.detail}
              </div>
            </div>

            <div className="rounded-xl border border-border-subtle bg-white/90 px-4 py-3">
              <div className="text-xs text-muted mb-1">
                Copilot summary
              </div>
              <div className="text-sm text-text-primary">{copilotResponse.summary}</div>
              {copilotResponse.guidance && (
                <div className="mt-2 text-xs text-muted">{copilotResponse.guidance}</div>
              )}
              <div className="mt-2 text-xs text-muted">
                {copilotResponse.mode === 'proposed-run'
                  ? 'Proposed bounded run'
                  : copilotResponse.mode === 'guidance'
                    ? 'Guidance only'
                    : 'Current exact result'}
                {` · ${copilotResponse.model}`}
                {copilotResponse.cached ? ' · prompt cache hit' : ' · fresh call'}
              </div>
            </div>

            {copilotResponse.proposedConfig && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs text-muted">
                <div>
                  <span className="block text-xs text-text-faint">Paradigm</span>
                  {copilotResponse.proposedConfig.paradigm}
                </div>
                <div>
                  <span className="block text-xs text-text-faint">Distribution</span>
                  {copilotResponse.proposedConfig.distribution}
                </div>
                <div>
                  <span className="block text-xs text-text-faint">Validators</span>
                  {copilotResponse.proposedConfig.validators.toLocaleString()}
                </div>
                <div>
                  <span className="block text-xs text-text-faint">Slots</span>
                  {copilotResponse.proposedConfig.slots.toLocaleString()}
                </div>
              </div>
            )}

            <BlockCanvas blocks={copilotResponse.blocks} />
          </div>
        )}
      </div>

      {manifest && (
        <>
          <div className="grid grid-cols-2 xl:grid-cols-4 gap-3 mb-6">
            <div className="bg-white border border-border-subtle rounded-lg p-4">
              <div className="text-xs text-muted mb-2">Avg MEV</div>
              <div className="text-2xl font-semibold text-text-primary">
                {formatNumber(manifest.summary.finalAverageMev, 4)}
              </div>
              <div className="text-xs text-muted mt-1">ETH</div>
            </div>

            <div className="bg-white border border-border-subtle rounded-lg p-4">
              <div className="text-xs text-muted mb-2">Supermajority</div>
              <div className="text-2xl font-semibold text-text-primary">
                {formatNumber(manifest.summary.finalSupermajoritySuccess, 2)}%
              </div>
            </div>

            <div className="bg-white border border-border-subtle rounded-lg p-4">
              <div className="text-xs text-muted mb-2">Runtime</div>
              <div className="text-2xl font-semibold text-text-primary">
                {formatNumber(manifest.runtimeSeconds, 2)}s
              </div>
            </div>

            <div className="bg-white border border-border-subtle rounded-lg p-4">
              <div className="text-xs text-muted mb-2">Slots</div>
              <div className="text-2xl font-semibold text-text-primary">
                {manifest.summary.slotsRecorded.toLocaleString()}
              </div>
            </div>
          </div>

          <div className="lab-stage p-5 mb-6">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
              <div>
                <div className="text-xs text-muted mb-1">
                  Run provenance
                </div>
                <div className="text-sm text-text-primary">
                  {manifest.cacheHit ? 'Exact cache hit' : 'Fresh exact execution'}
                </div>
                <div className="text-xs text-muted mt-1 max-w-2xl">
                  {manifest.cacheHit
                    ? 'Reused an identical exact run from the shared exact cache. Outputs are unchanged for the same inputs.'
                    : 'Executed the canonical exact simulator with the current configuration and seed.'}
                </div>
                <div className="flex flex-wrap gap-2 mt-3">
                  {paperScenarioLabels(manifest.config).map(label => (
                    <span
                      key={label}
                      className="inline-flex items-center gap-1.5 text-xs text-muted"
                    >
                      <span className="w-1.5 h-1.5 rounded-full bg-accent" />
                      {label}
                    </span>
                  ))}
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                <button
                  onClick={() => copyToClipboard(JSON.stringify(manifest.config, null, 2), 'config')}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-border-subtle bg-white px-3 py-2 text-xs text-text-primary hover:border-border-hover transition-colors"
                >
                  {copyState === 'config' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                  {copyState === 'config' ? 'Copied config' : 'Copy config JSON'}
                </button>
                <button
                  onClick={() => copyToClipboard(buildRunSummary(manifest), 'run')}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-border-subtle bg-white px-3 py-2 text-xs text-text-primary hover:border-border-hover transition-colors"
                >
                  {copyState === 'run' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                  {copyState === 'run' ? 'Copied summary' : 'Copy run summary'}
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 text-xs text-muted">
              <div>
                <span className="block text-xs text-text-faint">Seed</span>
                {manifest.config.seed}
              </div>
              <div>
                <span className="block text-xs text-text-faint">Validators</span>
                {manifest.config.validators.toLocaleString()}
              </div>
              <div>
                <span className="block text-xs text-text-faint">Slots</span>
                {manifest.config.slots.toLocaleString()}
              </div>
              <div>
                <span className="block text-xs text-text-faint">Cache key</span>
                {manifest.cacheKey.slice(0, 12)}
              </div>
            </div>
          </div>

          <div className="lab-stage p-5 mb-6">
            <div className="flex items-center justify-between gap-3 mb-4">
              <div>
                <div className="text-xs text-muted mb-1">
                  Exact overview
                </div>
                <div className="text-sm text-text-primary">
                  Multi-chart bundles built from manifest-ready exact artifacts.
                </div>
              </div>
              <div className="text-xs text-muted">
                Prefetched in parallel from the current exact run
              </div>
            </div>

            <div className="flex flex-wrap gap-2 mb-4">
              {OVERVIEW_BUNDLES.map(option => (
                <button
                  key={option.bundle}
                  onClick={() => startTransition(() => setSelectedBundle(option.bundle))}
                  className={cn(
                    'rounded-lg border px-3 py-2 text-left transition-colors',
                    selectedBundle === option.bundle
                      ? 'border-accent bg-white'
                      : 'border-border-subtle bg-white hover:border-border-hover',
                  )}
                >
                  <div className="text-xs font-medium text-text-primary">{option.label}</div>
                  <div className="text-xs text-muted">{option.description}</div>
                </button>
              ))}
            </div>

            {isOverviewLoading && overviewBlocks.length === 0 && (
              <div className="py-12 text-sm text-muted text-center">
                Preparing exact overview charts…
              </div>
            )}

            {!isOverviewLoading && overviewBlocks.length > 0 && (
              <BlockCanvas blocks={overviewBlocks} />
            )}

            {!isOverviewLoading && overviewBlocks.length === 0 && (
              <div className="py-12 text-sm text-muted text-center">
                This overview bundle is still waiting on exact artifacts from the current run.
              </div>
            )}
          </div>

          <div className="lab-stage p-5 mb-6">
            <div className="flex items-center justify-between gap-3 mb-4">
              <div>
                <div className="text-xs text-muted mb-1">
                  Artifact manifest
                </div>
                <div className="text-sm text-text-primary">
                  Summary data is already loaded. Pick a compact derived artifact or a raw export.
                </div>
              </div>
              <div className="text-xs text-muted text-right">
                {manifest.cacheHit ? 'Served from exact cache' : 'Fresh exact run'}
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              {manifest.artifacts.map(artifact => (
                <button
                  key={artifact.name}
                  onClick={() => onSelectArtifact(artifact.name)}
                  disabled={!artifact.renderable}
                  className={cn(
                    'text-left rounded-lg border px-4 py-3 transition-all',
                    selectedArtifactName === artifact.name
                      ? 'border-accent bg-white'
                      : 'border-border-subtle bg-white hover:border-border-hover',
                    !artifact.renderable && 'opacity-60 cursor-not-allowed',
                  )}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-sm font-medium text-text-primary">{artifact.label}</div>
                      <div className="text-xs text-muted mt-1">{artifact.description}</div>
                    </div>
                    <div className="text-xs text-muted whitespace-nowrap">
                      {artifact.lazy ? 'lazy' : 'ready'}
                    </div>
                  </div>

                  <div className="flex items-center gap-3 mt-3 text-xs text-muted">
                    <span>{formatBytes(artifact.bytes)}</span>
                    {artifact.gzipBytes != null && <span>gzip {formatBytes(artifact.gzipBytes)}</span>}
                    <span>{artifact.kind}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="lab-stage p-5">
            <div className="flex items-center justify-between gap-3 mb-4">
              <div>
                <div className="text-xs text-muted mb-1">
                  Rendered artifact
                </div>
                <div className="text-sm text-text-primary">
                  {selectedArtifact?.label ?? 'Select an artifact to render'}
                </div>
              </div>
              {selectedArtifact && (
                <div className="text-xs text-muted">
                  {selectedArtifact.kind} · {selectedArtifact.lazy ? 'lazy-loaded' : 'manifest-ready'}
                </div>
              )}
            </div>

            {((artifactQuery.isFetching && !selectedArtifactRawText) || isParsing) && (
              <div className="py-12 text-sm text-muted text-center">
                Parsing artifact in a browser worker…
              </div>
            )}

            {parseError && (
              <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
                {parseError}
              </div>
            )}

            {!artifactQuery.isFetching && !isParsing && !parseError && parsedBlocks.length > 0 && (
              <BlockCanvas blocks={parsedBlocks} />
            )}

            {!artifactQuery.isFetching && !isParsing && !parseError && parsedBlocks.length === 0 && (
              <div className="py-12 text-sm text-muted text-center">
                Pick a renderable artifact to inspect the exact run.
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
