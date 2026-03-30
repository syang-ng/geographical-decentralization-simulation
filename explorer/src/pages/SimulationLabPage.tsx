import { useEffect, useMemo, useRef, useState, startTransition } from 'react'
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { SimConfigPanel } from '../components/simulation/SimConfigPanel'
import { SimCopilotPanel } from '../components/simulation/SimCopilotPanel'
import { SimJobStatus } from '../components/simulation/SimJobStatus'
import { ResearchDemoSurface } from '../components/simulation/ResearchDemoSurface'
import { SimResultsPanel } from '../components/simulation/SimResultsPanel'
import {
  COPY_RESET_DELAY_MS,
  DEFAULT_CONFIG,
  OVERVIEW_BUNDLES,
  PRESETS,
  paperScenarioLabels,
  readOrCreateClientId,
  readSessionArtifactBlocks,
  writeSessionArtifactBlocks,
} from '../components/simulation/simulation-constants'
import { getApiHealth } from '../lib/api'
import { Wayfinder } from '../components/layout/Wayfinder'
import type { TabId } from '../components/layout/TabNav'
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

function selectDefaultArtifact(artifacts: readonly SimulationArtifact[]): string | null {
  const preferred = artifacts.find(artifact => artifact.renderable && !artifact.lazy)
  if (preferred) return preferred.name
  return artifacts.find(artifact => artifact.renderable)?.name ?? null
}

function isManifestOverviewBundle(
  bundle: (typeof OVERVIEW_BUNDLES)[number] | SimulationOverviewBundle | null,
): bundle is SimulationOverviewBundle {
  return Boolean(bundle && 'bytes' in bundle)
}

const APP_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, '') ?? ''

export function SimulationLabPage({ onTabChange }: { onTabChange?: (tab: TabId) => void } = {}) {
  const queryClient = useQueryClient()
  const [surfaceMode, setSurfaceMode] = useState<'research' | 'lab'>('research')
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
      <div className="flex items-center justify-between gap-4 mb-5">
        <div className="flex items-center gap-2.5 min-w-0">
          <span className="w-2 h-2 rounded-full bg-accent shrink-0" />
          <h1 className="text-base font-semibold text-text-primary truncate">Simulation Lab</h1>
          <span className="text-xs text-muted hidden sm:inline">Exact runs with bounded workflow</span>
        </div>

        <div className="inline-flex rounded-full border border-border-subtle bg-white p-0.5 shrink-0">
          <button
            onClick={() => setSurfaceMode('research')}
            className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${surfaceMode === 'research' ? 'bg-accent text-white' : 'text-text-primary hover:bg-surface-active'}`}
          >
            Published
          </button>
          <button
            onClick={() => setSurfaceMode('lab')}
            className={`rounded-full px-3 py-1.5 text-xs font-medium transition-colors ${surfaceMode === 'lab' ? 'bg-accent text-white' : 'text-text-primary hover:bg-surface-active'}`}
          >
            Lab
          </button>
        </div>
      </div>

      {surfaceMode === 'research' ? (
        <ResearchDemoSurface
          catalogScriptUrl={`${APP_BASE_URL}/research-demo/assets/research-catalog.js`}
          viewerBaseUrl={`${APP_BASE_URL}/research-demo`}
        />
      ) : (
        <>

      <div className="grid gap-3 mb-5 sm:grid-cols-3">
        <div className="rounded-xl border border-border-subtle bg-white px-4 py-3">
          <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Surface</div>
          <div className="mt-1 text-sm font-medium text-text-primary">Interactive exact run</div>
          <div className="mt-1 text-xs text-muted">
            This side runs fresh exact simulations instead of switching among frozen published outputs.
          </div>
        </div>
        <div className="rounded-xl border border-border-subtle bg-white px-4 py-3">
          <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Paper-scale max</div>
          <div className="mt-1 text-sm font-medium text-text-primary">1,000 validators · 10,000 slots</div>
          <div className="mt-1 text-xs text-muted">
            That matches the top-end scale of the main researcher precomputed runs.
          </div>
        </div>
        <div className="rounded-xl border border-border-subtle bg-white px-4 py-3">
          <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Current default</div>
          <div className="mt-1 text-sm font-medium text-text-primary">Reduced for speed</div>
          <div className="mt-1 text-xs text-muted">
            The lab opens at `1,000` validators, `1,000` slots, and `0.0001 ETH` migration cost so iteration stays faster.
          </div>
        </div>
      </div>

      <div className="mb-4">
        <span className="text-xs text-muted mb-1.5 block">Quick presets</span>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map(preset => (
            <button
              key={preset.label}
              onClick={() => applyPreset(preset.config)}
              className="text-left bg-white border border-border-subtle rounded-lg px-3 py-2 hover:border-border-hover transition-colors"
            >
              <div className="text-xs font-medium text-text-primary">{preset.label}</div>
              <div className="text-[11px] text-muted">{preset.description}</div>
            </button>
          ))}
        </div>
        <div className="mt-2 text-xs text-muted">
          Presets jump to the paper-style scenario family. The default surface is intentionally smaller than the frozen `10,000`-slot baseline for faster exact iteration.
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
        </>
      )}

      {onTabChange && (
        <Wayfinder links={[
          { label: 'Explore findings', hint: 'Curated lenses & AI interpretation', onClick: () => onTabChange('findings') },
          { label: 'Read the paper', hint: 'Full editorial reading guide', onClick: () => onTabChange('paper') },
        ]} />
      )}
    </div>
  )
}
