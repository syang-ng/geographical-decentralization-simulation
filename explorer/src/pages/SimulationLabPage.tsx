import { useEffect, useMemo, useRef, useState, startTransition } from 'react'
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { SimConfigPanel } from '../components/simulation/SimConfigPanel'
import { SimCopilotPanel } from '../components/simulation/SimCopilotPanel'
import { SimJobStatus } from '../components/simulation/SimJobStatus'
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

export function SimulationLabPage() {
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
  const researchDemoUrl = `${APP_BASE_URL}/research-demo/`
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
        <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="w-2 h-2 rounded-full bg-accent" />
              <span className="text-xs text-muted">Simulation surfaces</span>
            </div>
            <h1 className="text-xl font-semibold text-text-primary">
              Switch between the published research demo and our exact lab.
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-muted">
              Use the research side when you want the canonical paper-style launcher with dataset, Local/External, and published results. Switch to our side when you want live exact runs, deeper controls, and analysis tooling.
            </p>
          </div>

          <div className="inline-flex rounded-full border border-border-subtle bg-white/88 p-1 backdrop-blur-sm">
            <button
              onClick={() => setSurfaceMode('research')}
              className={`rounded-full px-4 py-2 text-xs font-medium transition-colors ${surfaceMode === 'research' ? 'bg-accent text-white' : 'text-text-primary hover:bg-white'}`}
            >
              Theirs
            </button>
            <button
              onClick={() => setSurfaceMode('lab')}
              className={`rounded-full px-4 py-2 text-xs font-medium transition-colors ${surfaceMode === 'lab' ? 'bg-accent text-white' : 'text-text-primary hover:bg-white'}`}
            >
              Ours
            </button>
          </div>
        </div>
      </div>

      {surfaceMode === 'research' ? (
        <div className="lab-stage p-4 sm:p-5">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <div className="text-xs text-muted mb-1">
                Published baseline
              </div>
              <div className="text-sm text-text-primary">
                This is the frozen researcher-style launcher with the canonical dataset family and `Local` / `External` selector flow.
              </div>
              <div className="mt-2 text-xs text-muted max-w-2xl">
                It keeps the paper-facing options separate from our live simulation controls, which avoids mixing viewer navigation with engine parameters like slots.
              </div>
            </div>

            <a
              href={researchDemoUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center justify-center rounded-lg border border-border-subtle bg-white px-4 py-2 text-xs font-medium text-text-primary hover:border-border-hover transition-colors"
            >
              Open Full Demo
            </a>
          </div>

          <div className="mt-4 overflow-hidden rounded-2xl border border-border-subtle bg-white/80 shadow-sm">
            <iframe
              title="Published research demo"
              src={researchDemoUrl}
              className="block h-[1120px] w-full border-0 bg-white"
            />
          </div>
        </div>
      ) : (
        <>

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
        </>
      )}
    </div>
  )
}
