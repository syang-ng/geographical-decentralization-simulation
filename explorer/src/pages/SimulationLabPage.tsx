import { useEffect, useMemo, useRef, useState, startTransition } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import { Activity, Ban, Clock3, Database, FlaskConical, Play, RotateCcw, Rows3, Sparkles } from 'lucide-react'
import { BlockCanvas } from '../components/explore/BlockCanvas'
import { cn } from '../lib/cn'
import {
  cancelSimulationJob,
  getSimulationArtifact,
  getSimulationManifest,
  getSimulationJob,
  submitSimulationForClient,
  type SimulationArtifact,
  type SimulationConfig,
  type SimulationJob,
} from '../lib/simulation-api'
import { SPRING } from '../lib/theme'
import type { Block } from '../types/blocks'

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
  validators: 100,
  slots: 1000,
  distribution: 'uniform',
  sourcePlacement: 'homogeneous',
  migrationCost: 0.0001,
  attestationThreshold: 2 / 3,
  slotTime: 12,
  seed: 25873,
}

const PRESETS: Array<{ label: string; description: string; config: Partial<SimulationConfig> }> = [
  {
    label: 'Baseline SSP',
    description: 'External block building with uniform validator geography.',
    config: { paradigm: 'SSP', distribution: 'uniform', sourcePlacement: 'homogeneous' },
  },
  {
    label: 'Baseline MSP',
    description: 'Local block building with the same exact engine.',
    config: { paradigm: 'MSP', distribution: 'uniform', sourcePlacement: 'homogeneous' },
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
  const [parsedBlocks, setParsedBlocks] = useState<readonly Block[]>([])
  const [parseError, setParseError] = useState<string | null>(null)
  const [isParsing, setIsParsing] = useState(false)

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
      setSelectedArtifactName(null)
      setParsedBlocks([])
      setParseError(null)
    },
  })

  const cancelMutation = useMutation({
    mutationFn: cancelSimulationJob,
    onSuccess: job => {
      queryClient.setQueryData(['simulation-job', job.id], job)
    },
  })

  const jobQuery = useQuery({
    queryKey: ['simulation-job', currentJobId],
    queryFn: () => getSimulationJob(currentJobId!),
    enabled: Boolean(currentJobId),
  })

  const manifestQuery = useQuery({
    queryKey: ['simulation-manifest', currentJobId],
    queryFn: () => getSimulationManifest(currentJobId!),
    enabled: jobQuery.data?.status === 'completed' && !jobQuery.data?.manifest,
  })

  const manifest = jobQuery.data?.manifest ?? manifestQuery.data ?? null

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

  useEffect(() => {
    if (!artifactQuery.data || !selectedArtifact || !workerRef.current) {
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
        setParsedBlocks(event.data.blocks)
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
      rawText: artifactQuery.data,
    })

    return () => {
      worker.removeEventListener('message', handleMessage as EventListener)
    }
  }, [artifactQuery.data, selectedArtifact])

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

  const canCancel = jobQuery.data?.status === 'queued' || jobQuery.data?.status === 'running'

  return (
    <div>
      <div className="flex items-center gap-2 mb-6">
        <FlaskConical className="w-4 h-4 text-accent" />
        <span className="text-xs text-muted uppercase tracking-wider font-medium">
          Simulation Lab — exact mode with async caching
        </span>
      </div>

      <div className="mb-6">
        <span className="text-[10px] text-muted uppercase tracking-wider font-medium mb-2 block">
          Quick presets
        </span>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {PRESETS.map(preset => (
            <button
              key={preset.label}
              onClick={() => applyPreset(preset.config)}
              className="text-left glass-1 rounded-lg p-2.5 hover:border-accent/30 transition-all"
            >
              <div className="text-xs font-medium text-text-primary">{preset.label}</div>
              <div className="text-[10px] text-muted">{preset.description}</div>
            </button>
          ))}
        </div>
      </div>

      <div className="glass-1 rounded-lg p-5 mb-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium mb-1.5 block">
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
                        ? 'bg-accent/20 text-accent border border-accent/30'
                        : 'bg-accent-warm/20 text-accent-warm border border-accent-warm/30'
                      : 'bg-surface/50 text-muted border border-border-subtle hover:border-white/10',
                  )}
                >
                  {paradigm}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium mb-1.5 block">
              Validator Distribution
            </label>
            <select
              value={config.distribution}
              onChange={event => updateConfig('distribution', event.target.value as SimulationConfig['distribution'])}
              className="w-full bg-surface/50 border border-border-subtle rounded-lg px-3 py-1.5 text-xs text-text-primary outline-none focus:border-accent/30"
            >
              <option value="uniform">Uniform (40 regions equal)</option>
              <option value="heterogeneous">Heterogeneous (real ETH data)</option>
              <option value="random">Random</option>
            </select>
          </div>

          <div>
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium mb-1.5 block">
              Source Placement
            </label>
            <select
              value={config.sourcePlacement}
              onChange={event => updateConfig('sourcePlacement', event.target.value as SimulationConfig['sourcePlacement'])}
              className="w-full bg-surface/50 border border-border-subtle rounded-lg px-3 py-1.5 text-xs text-text-primary outline-none focus:border-accent/30"
            >
              <option value="homogeneous">Homogeneous</option>
              <option value="latency-aligned">Latency-aligned</option>
              <option value="latency-misaligned">Latency-misaligned</option>
            </select>
          </div>

          <div>
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium mb-1.5 block">
              Seed
            </label>
            <input
              type="number"
              value={config.seed}
              min={0}
              max={2147483647}
              onChange={event => updateConfig('seed', Number(event.target.value))}
              className="w-full bg-surface/50 border border-border-subtle rounded-lg px-3 py-1.5 text-xs text-text-primary outline-none focus:border-accent/30"
            />
          </div>

          <div>
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium mb-1.5 block">
              Validators: {config.validators}
            </label>
            <input
              type="range"
              min={50}
              max={250}
              step={10}
              value={config.validators}
              onChange={event => updateConfig('validators', Number(event.target.value))}
              className="w-full accent-accent"
            />
            <div className="flex justify-between text-[9px] text-muted/40">
              <span>50</span>
              <span>250</span>
            </div>
          </div>

          <div>
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium mb-1.5 block">
              Slots: {config.slots}
            </label>
            <input
              type="range"
              min={500}
              max={3000}
              step={100}
              value={config.slots}
              onChange={event => updateConfig('slots', Number(event.target.value))}
              className="w-full accent-accent"
            />
            <div className="flex justify-between text-[9px] text-muted/40">
              <span>500</span>
              <span>3000</span>
            </div>
          </div>

          <div>
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium mb-1.5 block">
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
            <div className="flex justify-between text-[9px] text-muted/40">
              <span>0</span>
              <span>0.005</span>
            </div>
          </div>

          <div>
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium mb-1.5 block">
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
                      ? 'bg-accent/20 text-accent border border-accent/30'
                      : 'bg-surface/50 text-muted border border-border-subtle hover:border-white/10',
                  )}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-[10px] text-muted uppercase tracking-wider font-medium mb-1.5 block">
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
                      ? 'bg-accent/20 text-accent border border-accent/30'
                      : 'bg-surface/50 text-muted border border-border-subtle hover:border-white/10',
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

          <div className="text-[10px] text-muted/70 text-right">
            <div className="flex items-center gap-1 justify-end">
              <Sparkles className="w-3 h-3" />
              Exact mode only
            </div>
            <div>Slot cutoff: {attestationCutoffMs(config.slotTime)} ms</div>
          </div>
        </div>
      </div>

      {(currentJobId || submitMutation.isError) && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={SPRING}
          className="glass-1 rounded-lg p-5 mb-6"
        >
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
            <div>
              <div className="text-[10px] text-muted uppercase tracking-wider font-medium mb-1">
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
                  <span className="block text-[10px] uppercase tracking-wider text-muted/70">Queue</span>
                  {jobQuery.data.queuePosition ?? 'live'}
                </div>
                <div>
                  <span className="block text-[10px] uppercase tracking-wider text-muted/70">Cache</span>
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

      {manifest && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3 mb-6">
            <div className="bg-surface border border-border-subtle rounded-xl p-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-muted mb-2">
                <Activity className="w-3 h-3" />
                Avg MEV
              </div>
              <div className="text-2xl font-semibold text-accent">
                {formatNumber(manifest.summary.finalAverageMev, 4)} ETH
              </div>
            </div>

            <div className="bg-surface border border-border-subtle rounded-xl p-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-muted mb-2">
                <Rows3 className="w-3 h-3" />
                Supermajority
              </div>
              <div className="text-2xl font-semibold text-text-primary">
                {formatNumber(manifest.summary.finalSupermajoritySuccess, 2)}%
              </div>
            </div>

            <div className="bg-surface border border-border-subtle rounded-xl p-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-muted mb-2">
                <Clock3 className="w-3 h-3" />
                Runtime
              </div>
              <div className="text-2xl font-semibold text-text-primary">
                {formatNumber(manifest.runtimeSeconds, 2)}s
              </div>
            </div>

            <div className="bg-surface border border-border-subtle rounded-xl p-4">
              <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider text-muted mb-2">
                <Database className="w-3 h-3" />
                Slots
              </div>
              <div className="text-2xl font-semibold text-text-primary">
                {manifest.summary.slotsRecorded.toLocaleString()}
              </div>
            </div>
          </div>

          <div className="glass-1 rounded-lg p-5 mb-6">
            <div className="flex items-center justify-between gap-3 mb-4">
              <div>
                <div className="text-[10px] text-muted uppercase tracking-wider font-medium mb-1">
                  Artifact manifest
                </div>
                <div className="text-sm text-text-primary">
                  Summary data is already loaded. Pick a compact derived artifact or a raw export.
                </div>
              </div>
              <div className="text-[10px] text-muted text-right">
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
                    'text-left rounded-xl border px-4 py-3 transition-all',
                    selectedArtifactName === artifact.name
                      ? 'border-accent/40 bg-accent/10'
                      : 'border-border-subtle bg-surface/60 hover:border-white/10',
                    !artifact.renderable && 'opacity-60 cursor-not-allowed',
                  )}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-sm font-medium text-text-primary">{artifact.label}</div>
                      <div className="text-xs text-muted mt-1">{artifact.description}</div>
                    </div>
                    <div className="text-[10px] text-muted whitespace-nowrap">
                      {artifact.lazy ? 'lazy' : 'ready'}
                    </div>
                  </div>

                  <div className="flex items-center gap-3 mt-3 text-[10px] text-muted">
                    <span>{formatBytes(artifact.bytes)}</span>
                    {artifact.gzipBytes != null && <span>gzip {formatBytes(artifact.gzipBytes)}</span>}
                    <span>{artifact.kind}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="glass-1 rounded-lg p-5">
            <div className="flex items-center justify-between gap-3 mb-4">
              <div>
                <div className="text-[10px] text-muted uppercase tracking-wider font-medium mb-1">
                  Rendered artifact
                </div>
                <div className="text-sm text-text-primary">
                  {selectedArtifact?.label ?? 'Select an artifact to render'}
                </div>
              </div>
              {selectedArtifact && (
                <div className="text-[10px] text-muted">
                  {selectedArtifact.kind} · {selectedArtifact.lazy ? 'lazy-loaded' : 'manifest-ready'}
                </div>
              )}
            </div>

            {(artifactQuery.isFetching || isParsing) && (
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
