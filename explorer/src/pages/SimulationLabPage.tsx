import { useEffect, useMemo, useRef, useState, startTransition } from 'react'
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { SimConfigPanel } from '../components/simulation/SimConfigPanel'
import { SimCopilotPanel } from '../components/simulation/SimCopilotPanel'
import { SimJobStatus } from '../components/simulation/SimJobStatus'
import { SimResultsPanel } from '../components/simulation/SimResultsPanel'
import { getApiHealth } from '../lib/api'
import {
  cancelSimulationJob,
  getSimulationArtifact,
  getSimulationManifest,
  getSimulationJob,
  getSimulationOverviewBundle,
  submitSimulationCopilot,
  submitSimulationForClient,
  type SimulationArtifact,
  type SimulationCopilotResponse,
  type SimulationConfig,
  type SimulationJob,
  type SimulationOverviewBundle,
} from '../lib/simulation-api'
import { parseBlocks, type Block } from '../types/blocks'
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

const PAPER_BASELINE_PRESET: Partial<SimulationConfig> = {
  validators: 1000,
  slots: 10000,
  distribution: 'homogeneous',
  sourcePlacement: 'homogeneous',
  migrationCost: 0.002,
  attestationThreshold: 2 / 3,
  slotTime: 12,
}

const PRESETS: Array<{ label: string; description: string; config: Partial<SimulationConfig> }> = [
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

const COPY_RESET_DELAY_MS = 1600
const PARSED_ARTIFACT_CACHE_PREFIX = 'simulation_lab_parsed_artifact:'

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

function readSessionArtifactBlocks(cacheKey: string): readonly Block[] | null {
  try {
    const stored = window.sessionStorage.getItem(`${PARSED_ARTIFACT_CACHE_PREFIX}${cacheKey}`)
    if (!stored) return null
    const parsed = JSON.parse(stored) as unknown
    return Array.isArray(parsed) ? parseBlocks(parsed) : null
  } catch {
    return null
  }
}

function writeSessionArtifactBlocks(cacheKey: string, blocks: readonly Block[]): void {
  try {
    window.sessionStorage.setItem(
      `${PARSED_ARTIFACT_CACHE_PREFIX}${cacheKey}`,
      JSON.stringify(blocks),
    )
  } catch {
    // Ignore storage exhaustion and keep the in-memory cache path.
  }
}

function isManifestOverviewBundle(
  bundle: (typeof OVERVIEW_BUNDLES)[number] | SimulationOverviewBundle | null,
): bundle is SimulationOverviewBundle {
  return Boolean(bundle && 'bytes' in bundle)
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
  const availableOverviewBundles = useMemo(
    () => manifest?.overviewBundles ?? [],
    [manifest],
  )
  const overviewBundleOptions = useMemo(
    () => availableOverviewBundles.length > 0 ? availableOverviewBundles : OVERVIEW_BUNDLES,
    [availableOverviewBundles],
  )

  const overviewBundleQueries = useQueries({
    queries: availableOverviewBundles.map(bundle => ({
      queryKey: ['simulation-overview-bundle', currentJobId, bundle.bundle, bundle.sha256],
      queryFn: () => getSimulationOverviewBundle(currentJobId!, bundle.bundle),
      enabled: Boolean(currentJobId),
      staleTime: Infinity,
    })),
  })

  const selectedOverviewBundleIndex = availableOverviewBundles.findIndex(bundle => bundle.bundle === selectedBundle)
  const selectedOverviewBundleInfo = overviewBundleOptions.find(bundle => bundle.bundle === selectedBundle) ?? null
  const selectedOverviewBundleMetrics = isManifestOverviewBundle(selectedOverviewBundleInfo)
    ? selectedOverviewBundleInfo
    : null
  const overviewBlocks = selectedOverviewBundleIndex >= 0
    ? overviewBundleQueries[selectedOverviewBundleIndex]?.data ?? []
    : []
  const isOverviewLoading = selectedOverviewBundleIndex >= 0
    ? (overviewBundleQueries[selectedOverviewBundleIndex]?.isFetching ?? false)
    : false

  useEffect(() => {
    if (!manifest) return
    if (selectedArtifactName) return
    const nextArtifact = selectDefaultArtifact(manifest.artifacts)
    if (nextArtifact) {
      setSelectedArtifactName(nextArtifact)
    }
  }, [manifest, selectedArtifactName])

  useEffect(() => {
    if (!manifest?.overviewBundles?.length) return
    if (manifest.overviewBundles.some(bundle => bundle.bundle === selectedBundle)) return
    startTransition(() => {
      setSelectedBundle(manifest.overviewBundles[0]!.bundle)
    })
  }, [manifest, selectedBundle])

  const selectedArtifact = useMemo(
    () => manifest?.artifacts.find(artifact => artifact.name === selectedArtifactName) ?? null,
    [manifest, selectedArtifactName],
  )

  const artifactQuery = useQuery({
    queryKey: ['simulation-artifact', currentJobId, selectedArtifactName],
    queryFn: () => getSimulationArtifact(currentJobId!, selectedArtifactName!),
    enabled: Boolean(currentJobId && selectedArtifactName && selectedArtifact?.renderable),
    staleTime: Infinity,
  })
  const selectedArtifactRawText = artifactQuery.data ?? null

  useEffect(() => {
    if (!selectedArtifact || !selectedArtifactRawText || !workerRef.current) {
      return
    }

    const cacheKey = selectedArtifact.sha256
    const cachedBlocks = parsedArtifactCache[cacheKey] ?? readSessionArtifactBlocks(cacheKey)
    if (cachedBlocks) {
      if (!parsedArtifactCache[cacheKey]) {
        setParsedArtifactCache(previous => ({
          ...previous,
          [cacheKey]: cachedBlocks,
        }))
      }
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
          [cacheKey]: nextBlocks,
        }))
        writeSessionArtifactBlocks(cacheKey, nextBlocks)
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
  const prewarmState = apiHealthQuery.data?.simulations.prewarm ?? null
  const copilotPromptSuggestions = copilotResponse?.suggestedPrompts?.length
    ? copilotResponse.suggestedPrompts
    : manifest
      ? [
          'Show the core outcomes bundle from this exact run.',
          'Explain why these regions dominate in this exact result.',
          'What is the nearest paper-backed follow-up to run next?',
        ]
      : [
          'Set up the paper baseline SSP run (10,000 slots, 0.002 ETH).',
          'Mirror that paper baseline for MSP so I can compare the paradigms.',
          'Hold the paradigm fixed and switch from latency-aligned to latency-misaligned sources.',
          'Load the real Ethereum validator start and explain what should change.',
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
              Run exact simulations with a bounded lab workflow.
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-muted">
              The page opens on a lighter interactive default. Use the research presets when you want the paper-style 10,000-slot and 0.002 ETH scenario family, then inspect exact outputs through overview bundles, raw artifacts, and bounded interpretation.
            </p>
          </div>

          <div className="relative z-[1] grid gap-2 sm:grid-cols-3 lg:grid-cols-1">
            <div className="rounded-xl border border-border-subtle bg-white/88 px-3 py-3 backdrop-blur-sm">
              <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Default surface</div>
              <div className="mt-1 text-sm font-medium text-text-primary">Interactive default</div>
              <div className="mt-1 text-xs text-muted">1,000 validators, 1,000 slots, 0.0001 ETH migration cost, exact mode.</div>
            </div>
            <div className="rounded-xl border border-border-subtle bg-white/88 px-3 py-3 backdrop-blur-sm">
              <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Research presets</div>
              <div className="mt-1 text-sm font-medium text-text-primary">Paper-style scenarios</div>
              <div className="mt-1 text-xs text-muted">Named presets load the 10,000-slot and 0.002 ETH reference family unless the scenario changes it.</div>
            </div>
            <div className="rounded-xl border border-border-subtle bg-white/88 px-3 py-3 backdrop-blur-sm">
              <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Exact delivery</div>
              <div className="mt-1 text-sm font-medium text-text-primary">
                {apiHealthQuery.isLoading
                  ? 'Checking runtime state'
                  : prewarmState?.running
                    ? `Warming canonical cache (${prewarmState.completed}/${prewarmState.total})`
                    : prewarmState?.finishedAt
                      ? 'Canonical cache warmed'
                      : 'Manifest-ready sidecars'}
              </div>
              <div className="mt-1 text-xs text-muted">
                {prewarmState?.running
                  ? 'Common exact presets warm in the background while live runs stay on the main queue.'
                  : 'Overview bundles plus gzip and Brotli sidecars are served directly from exact outputs.'}
              </div>
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
        <div className="mt-2 text-xs text-muted">
          Presets load the paper-style scenario family. The page still opens on the lighter interactive default for faster iteration.
        </div>
      </div>

      <SimConfigPanel
        config={config}
        onConfigChange={updateConfig}
        onSubmit={onSubmit}
        onReset={resetConfig}
        isSubmitting={submitMutation.isPending}
        canCancel={canCancel}
        onCancel={onCancel}
        paperScenarioLabels={paperScenarioLabels(config)}
      />

      {(currentJobId || submitMutation.isError) && (
        <SimJobStatus
          status={status}
          jobData={jobQuery.data ?? null}
          submitError={(submitMutation.error as Error | null) ?? null}
          cancelError={(cancelMutation.error as Error | null) ?? null}
        />
      )}

      <SimCopilotPanel
        copilotQuestion={copilotQuestion}
        onQuestionChange={setCopilotQuestion}
        onAsk={question => copilotMutation.mutate(question)}
        onApplyConfig={setConfig}
        copilotResponse={copilotResponse}
        copilotAvailable={copilotAvailable}
        isHealthLoading={apiHealthQuery.isLoading}
        isMutating={copilotMutation.isPending}
        mutationError={(copilotMutation.error as Error | null) ?? null}
        hasManifest={Boolean(manifest)}
        promptSuggestions={copilotPromptSuggestions}
      />

      {manifest && (
        <SimResultsPanel
          manifest={manifest}
          overviewBundleOptions={overviewBundleOptions}
          selectedBundle={selectedBundle}
          onSelectBundle={setSelectedBundle}
          selectedOverviewBundleMetrics={selectedOverviewBundleMetrics}
          overviewBlocks={overviewBlocks}
          isOverviewLoading={isOverviewLoading}
          selectedArtifact={selectedArtifact}
          selectedArtifactName={selectedArtifactName}
          onSelectArtifact={onSelectArtifact}
          isArtifactFetching={artifactQuery.isFetching}
          isParsing={isParsing}
          parseError={parseError}
          parsedBlocks={parsedBlocks}
          copyState={copyState}
          onCopy={copyToClipboard}
        />
      )}
    </div>
  )
}
