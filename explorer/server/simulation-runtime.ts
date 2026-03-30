import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import crypto from 'node:crypto'
import { existsSync, readdirSync } from 'node:fs'
import fs from 'node:fs/promises'
import { availableParallelism } from 'node:os'
import path from 'node:path'
import readline from 'node:readline'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'
import { brotliCompress as brotliCompressCallback, constants as zlibConstants } from 'node:zlib'

const SERVER_DIR = path.dirname(fileURLToPath(import.meta.url))
const EXPLORER_ROOT = path.resolve(SERVER_DIR, '..')
const REPO_ROOT = process.env.SIMULATION_REPO_ROOT
  ? path.resolve(process.env.SIMULATION_REPO_ROOT)
  : path.resolve(EXPLORER_ROOT, '..')
const WORKER_PATH = path.join(SERVER_DIR, 'simulation_worker.py')
const SIMULATION_CACHE_ROOT = path.join(REPO_ROOT, '.simulation_cache')
const DEFAULT_PYTHON_EXECUTABLE = process.platform === 'win32' ? 'python' : 'python3'
const DEFAULT_WORKER_POOL_SIZE = Math.max(
  1,
  Math.min(
    Number(process.env.SIMULATION_WORKERS ?? Math.max(2, Math.ceil(availableParallelism() / 2))),
    8,
  ),
)
const DEFAULT_QUEUED_JOB_TTL_MS = Math.max(60_000, Number(process.env.SIMULATION_QUEUE_TTL_MS ?? 10 * 60_000))
const ENABLE_CANONICAL_PREWARM = !/^false$/i.test(process.env.SIMULATION_PREWARM ?? 'true')
const CANONICAL_PREWARM_CONFIGS: ReadonlyArray<SimulationRequest> = [
  {
    paradigm: 'SSP',
    validators: 1000,
    slots: 1000,
    distribution: 'homogeneous',
    sourcePlacement: 'homogeneous',
    migrationCost: 0.0001,
    attestationThreshold: roundTo(2 / 3, 6),
    slotTime: 12,
    seed: 25873,
  },
  {
    paradigm: 'MSP',
    validators: 1000,
    slots: 1000,
    distribution: 'homogeneous',
    sourcePlacement: 'homogeneous',
    migrationCost: 0.0001,
    attestationThreshold: roundTo(2 / 3, 6),
    slotTime: 12,
    seed: 25873,
  },
  {
    paradigm: 'SSP',
    validators: 1000,
    slots: 1000,
    distribution: 'homogeneous',
    sourcePlacement: 'latency-aligned',
    migrationCost: 0.0001,
    attestationThreshold: roundTo(2 / 3, 6),
    slotTime: 12,
    seed: 25873,
  },
  {
    paradigm: 'MSP',
    validators: 1000,
    slots: 1000,
    distribution: 'homogeneous',
    sourcePlacement: 'latency-aligned',
    migrationCost: 0.0001,
    attestationThreshold: roundTo(2 / 3, 6),
    slotTime: 12,
    seed: 25873,
  },
  {
    paradigm: 'SSP',
    validators: 1000,
    slots: 1000,
    distribution: 'homogeneous',
    sourcePlacement: 'latency-misaligned',
    migrationCost: 0.0001,
    attestationThreshold: roundTo(2 / 3, 6),
    slotTime: 12,
    seed: 25873,
  },
  {
    paradigm: 'MSP',
    validators: 1000,
    slots: 1000,
    distribution: 'homogeneous',
    sourcePlacement: 'latency-misaligned',
    migrationCost: 0.0001,
    attestationThreshold: roundTo(2 / 3, 6),
    slotTime: 12,
    seed: 25873,
  },
  {
    paradigm: 'SSP',
    validators: 1000,
    slots: 1000,
    distribution: 'heterogeneous',
    sourcePlacement: 'homogeneous',
    migrationCost: 0.0001,
    attestationThreshold: roundTo(2 / 3, 6),
    slotTime: 12,
    seed: 25873,
  },
  {
    paradigm: 'MSP',
    validators: 1000,
    slots: 1000,
    distribution: 'heterogeneous',
    sourcePlacement: 'homogeneous',
    migrationCost: 0.0001,
    attestationThreshold: roundTo(2 / 3, 6),
    slotTime: 12,
    seed: 25873,
  },
  {
    paradigm: 'SSP',
    validators: 1000,
    slots: 1000,
    distribution: 'homogeneous',
    sourcePlacement: 'homogeneous',
    migrationCost: 0.0001,
    attestationThreshold: roundTo(2 / 3, 6),
    slotTime: 6,
    seed: 25873,
  },
  {
    paradigm: 'MSP',
    validators: 1000,
    slots: 1000,
    distribution: 'homogeneous',
    sourcePlacement: 'homogeneous',
    migrationCost: 0.0001,
    attestationThreshold: roundTo(2 / 3, 6),
    slotTime: 6,
    seed: 25873,
  },
] as const
const brotliCompress = promisify(brotliCompressCallback)
const DEFAULT_COMPLETED_JOB_RETENTION_MS = Math.max(10 * 60_000, Number(process.env.SIMULATION_COMPLETED_RETENTION_MS ?? 6 * 60 * 60_000))
const DEFAULT_MAX_STORED_JOBS = Math.max(100, Number(process.env.SIMULATION_MAX_STORED_JOBS ?? 500))
const DEFAULT_MAX_QUEUE_LENGTH = Math.max(DEFAULT_WORKER_POOL_SIZE, Number(process.env.SIMULATION_MAX_QUEUE_LENGTH ?? 48))
const DEFAULT_MAX_ACTIVE_JOBS_PER_CLIENT = Math.max(1, Number(process.env.SIMULATION_MAX_ACTIVE_JOBS_PER_CLIENT ?? 4))
const REQUIRED_RUNTIME_PATHS = [
  'data/gcp_regions.csv',
  'data/gcp_latency.csv',
  'data/validators.csv',
  'data/world_countries.geo.json',
  'params/SSP-baseline.yaml',
  'params/SSP-latency-aligned.yaml',
  'params/SSP-latency-misaligned.yaml',
  'params/MSP-baseline.yaml',
  'params/MSP-latency-aligned.yaml',
  'params/MSP-latency-misaligned.yaml',
  'explorer/server/simulation_worker.py',
] as const

export interface SimulationRequest {
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
  readonly brotliBytes: number | null
  readonly sha256: string
  readonly lazy: boolean
  readonly renderable: boolean
}

export interface SimulationOverviewBundle {
  readonly bundle: 'core-outcomes' | 'timing-and-attestation' | 'geography-overview'
  readonly name: string
  readonly label: string
  readonly description: string
  readonly bytes: number
  readonly gzipBytes: number | null
  readonly brotliBytes: number | null
  readonly sha256: string
}

export interface SimulationManifest {
  readonly jobId: string
  readonly configHash: string
  readonly cacheKey: string
  readonly cacheHit: boolean
  readonly runtimeSeconds: number
  readonly outputDir: string
  readonly config: SimulationRequest
  readonly summary: SimulationSummary
  readonly artifacts: ReadonlyArray<SimulationArtifact>
  readonly overviewBundles: ReadonlyArray<SimulationOverviewBundle>
}

export type SimulationJobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface SimulationJobSnapshot {
  readonly id: string
  readonly status: SimulationJobStatus
  readonly createdAt: string
  readonly updatedAt: string
  readonly configHash: string
  readonly queuePosition: number | null
  readonly cacheHit: boolean | null
  readonly error: string | null
  readonly config: SimulationRequest
  readonly manifest?: SimulationManifest
}

interface JobRecord {
  id: string
  createdAt: string
  createdAtMs: number
  updatedAt: string
  updatedAtMs: number
  status: SimulationJobStatus
  configHash: string
  queuePosition: number | null
  cacheHit: boolean | null
  error: string | null
  config: SimulationRequest
  manifest?: SimulationManifest
  clientId: string | null
  workerSlot: number | null
}

interface WorkerResponse {
  readonly id: number
  readonly ok: boolean
  readonly result?: SimulationManifest
  readonly error?: {
    readonly message: string
    readonly traceback?: string
  }
}

interface PendingRequest {
  readonly jobId: string
  readonly workerSlot: number
  readonly resolve: (manifest: SimulationManifest) => void
  readonly reject: (error: Error) => void
}

interface PrewarmState {
  enabled: boolean
  running: boolean
  startedAt: string | null
  finishedAt: string | null
  completed: number
  total: number
  lastError: string | null
}

interface WorkerSlot {
  readonly index: number
  process: ChildProcessWithoutNullStreams | null
  currentRequestId: number | null
  currentJobId: string | null
  lastError: string | null
  lastStartedAt: string | null
}

type JobListener = (snapshot: SimulationJobSnapshot) => void
type CompressionEncoding = 'br' | 'gzip' | null

function isWithinDirectory(directoryPath: string, candidatePath: string): boolean {
  const relative = path.relative(directoryPath, candidatePath)
  return relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative))
}

function stableStringify(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(',')}]`
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, nested]) => `${JSON.stringify(key)}:${stableStringify(nested)}`)
    return `{${entries.join(',')}}`
  }
  return JSON.stringify(value)
}

function nowIso(): string {
  return new Date().toISOString()
}

function roundTo(value: number, digits: number): number {
  return Number(value.toFixed(digits))
}

function normalizeDistribution(raw: unknown): SimulationRequest['distribution'] | null {
  if (raw === 'uniform') return 'homogeneous-gcp'
  if (
    raw === 'homogeneous'
    || raw === 'homogeneous-gcp'
    || raw === 'heterogeneous'
    || raw === 'random'
  ) {
    return raw
  }
  return null
}

function normalizeConfig(config: SimulationRequest): SimulationRequest {
  return {
    paradigm: config.paradigm,
    validators: Math.trunc(config.validators),
    slots: Math.trunc(config.slots),
    distribution: config.distribution,
    sourcePlacement: config.sourcePlacement,
    migrationCost: roundTo(config.migrationCost, 6),
    attestationThreshold: roundTo(config.attestationThreshold, 6),
    slotTime: Math.trunc(config.slotTime),
    seed: Math.trunc(config.seed),
  }
}

function configHash(config: SimulationRequest): string {
  return crypto.createHash('sha256').update(stableStringify(config)).digest('hex')
}

export function parseSimulationRequest(raw: unknown): SimulationRequest | null {
  if (!raw || typeof raw !== 'object') return null
  const body = raw as Record<string, unknown>

  const paradigm = body.paradigm
  const distribution = normalizeDistribution(body.distribution)
  const sourcePlacement = body.sourcePlacement
  const validators = Number(body.validators)
  const slots = Number(body.slots)
  const migrationCost = Number(body.migrationCost)
  const attestationThreshold = Number(body.attestationThreshold)
  const slotTime = Number(body.slotTime)
  const seed = Number(body.seed)

  if (paradigm !== 'SSP' && paradigm !== 'MSP') return null
  if (!distribution) return null
  if (sourcePlacement !== 'homogeneous' && sourcePlacement !== 'latency-aligned' && sourcePlacement !== 'latency-misaligned') return null
  if (!Number.isInteger(validators) || validators < 1 || validators > 1000) return null
  if (!Number.isInteger(slots) || slots < 1 || slots > 10000) return null
  if (!Number.isFinite(migrationCost) || migrationCost < 0 || migrationCost > 0.02) return null
  if (!Number.isFinite(attestationThreshold) || attestationThreshold <= 0 || attestationThreshold >= 1) return null
  if (!Number.isInteger(slotTime) || ![6, 8, 12].includes(slotTime)) return null
  if (!Number.isInteger(seed) || seed < 0 || seed > 0x7fffffff) return null

  return normalizeConfig({
    paradigm,
    validators,
    slots,
    distribution,
    sourcePlacement,
    migrationCost,
    attestationThreshold,
    slotTime,
    seed,
  })
}

export class SimulationRuntime {
  private readonly pythonExecutable = process.env.PYTHON_EXECUTABLE ?? DEFAULT_PYTHON_EXECUTABLE
  private readonly workerPoolSize = DEFAULT_WORKER_POOL_SIZE
  private readonly queuedJobTtlMs = DEFAULT_QUEUED_JOB_TTL_MS
  private readonly completedJobRetentionMs = DEFAULT_COMPLETED_JOB_RETENTION_MS
  private readonly maxStoredJobs = DEFAULT_MAX_STORED_JOBS
  private readonly maxQueueLength = DEFAULT_MAX_QUEUE_LENGTH
  private readonly maxActiveJobsPerClient = DEFAULT_MAX_ACTIVE_JOBS_PER_CLIENT
  private readonly jobs = new Map<string, JobRecord>()
  private readonly configToJob = new Map<string, string>()
  private readonly queue: string[] = []
  private readonly pending = new Map<number, PendingRequest>()
  private readonly listeners = new Map<string, Set<JobListener>>()
  private readonly workers: WorkerSlot[]
  private readonly prewarmState: PrewarmState
  private requestCounter = 0
  private jobCounter = 0
  private pumping = false

  constructor() {
    this.workers = Array.from({ length: this.workerPoolSize }, (_, index) => ({
      index,
      process: null,
      currentRequestId: null,
      currentJobId: null,
      lastError: null,
      lastStartedAt: null,
    }))
    this.prewarmState = {
      enabled: ENABLE_CANONICAL_PREWARM,
      running: false,
      startedAt: null,
      finishedAt: null,
      completed: 0,
      total: CANONICAL_PREWARM_CONFIGS.length,
      lastError: null,
    }
    void this.initialize()
  }

  submit(config: SimulationRequest, options: { clientId?: string | null } = {}): SimulationJobSnapshot {
    this.pruneJobs()
    const normalized = normalizeConfig(config)
    const hash = configHash(normalized)
    const existingJobId = this.configToJob.get(hash)
    if (existingJobId) {
      const existing = this.jobs.get(existingJobId)
      if (existing && existing.status !== 'failed' && existing.status !== 'cancelled') {
        return this.snapshot(existing)
      }
      this.configToJob.delete(hash)
    }

    const clientId = options.clientId?.trim() ? options.clientId.trim() : null
    if (clientId) {
      this.cancelSupersededQueuedJobs(clientId)
      if (this.countActiveJobsForClient(clientId) >= this.maxActiveJobsPerClient) {
        throw new Error(
          `Too many active simulation jobs for this client. Wait for a running job to finish before submitting another.`,
        )
      }
    }
    this.dropStaleQueuedJobs()
    if (this.queue.length >= this.maxQueueLength) {
      throw new Error('Simulation queue is full. Try again once a queued job has started or completed.')
    }

    const id = `sim-${Date.now()}-${++this.jobCounter}`
    const createdAt = nowIso()
    const createdAtMs = Date.now()
    const job: JobRecord = {
      id,
      createdAt,
      createdAtMs,
      updatedAt: createdAt,
      updatedAtMs: createdAtMs,
      status: 'queued',
      configHash: hash,
      queuePosition: this.queue.length + 1,
      cacheHit: null,
      error: null,
      config: normalized,
      clientId,
      workerSlot: null,
    }

    this.jobs.set(id, job)
    this.configToJob.set(hash, id)
    this.queue.push(id)
    this.notify(job)
    void this.pumpQueue()
    return this.snapshot(job)
  }

  cancel(jobId: string, reason = 'Cancelled by user.'): SimulationJobSnapshot | null {
    const job = this.jobs.get(jobId)
    if (!job) return null
    if (job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled') {
      return this.snapshot(job)
    }

    if (job.status === 'queued') {
      const queueIndex = this.queue.indexOf(jobId)
      if (queueIndex >= 0) {
        this.queue.splice(queueIndex, 1)
      }
      this.finalizeCancelled(job, reason)
      this.refreshQueuePositions()
      void this.pumpQueue()
      return this.snapshot(job)
    }

    if (job.status === 'running' && job.workerSlot != null) {
      const worker = this.workers[job.workerSlot]
      if (worker.currentRequestId != null) {
        const pending = this.pending.get(worker.currentRequestId)
        if (pending) {
          pending.reject(new Error(reason))
          this.pending.delete(worker.currentRequestId)
        }
      }
      worker.currentRequestId = null
      worker.currentJobId = null
      this.finalizeCancelled(job, reason)
      worker.process?.kill()
      worker.process = null
      void this.pumpQueue()
      return this.snapshot(job)
    }

    return this.snapshot(job)
  }

  getJob(jobId: string): SimulationJobSnapshot | null {
    this.pruneJobs()
    const job = this.jobs.get(jobId)
    return job ? this.snapshot(job) : null
  }

  getManifest(jobId: string): SimulationManifest | null {
    this.pruneJobs()
    return this.jobs.get(jobId)?.manifest ?? null
  }

  subscribe(jobId: string, listener: JobListener): (() => void) | null {
    const job = this.jobs.get(jobId)
    if (!job) {
      return null
    }

    let listeners = this.listeners.get(jobId)
    if (!listeners) {
      listeners = new Set<JobListener>()
      this.listeners.set(jobId, listeners)
    }
    listeners.add(listener)

    return () => {
      const registered = this.listeners.get(jobId)
      if (!registered) return
      registered.delete(listener)
      if (registered.size === 0) {
        this.listeners.delete(jobId)
      }
    }
  }

  async readArtifact(
    jobId: string,
    artifactName: string,
    options: { preferGzip?: boolean; preferBrotli?: boolean } = {},
  ): Promise<{ artifact: SimulationArtifact; body: Buffer | string; contentEncoding: CompressionEncoding }> {
    const manifest = this.getManifest(jobId)
    if (!manifest) {
      throw new Error('Manifest not available for this job yet.')
    }

    const artifact = manifest.artifacts.find(item => item.name === artifactName)
    if (!artifact) {
      throw new Error(`Unknown artifact: ${artifactName}`)
    }

    return await this.readManifestAsset(manifest, artifact, options)
  }

  async readOverviewBundle(
    jobId: string,
    bundleName: string,
    options: { preferGzip?: boolean; preferBrotli?: boolean } = {},
  ): Promise<{ overviewBundle: SimulationOverviewBundle; body: Buffer | string; contentEncoding: CompressionEncoding }> {
    const manifest = this.getManifest(jobId)
    if (!manifest) {
      throw new Error('Manifest not available for this job yet.')
    }

    const overviewBundle = manifest.overviewBundles.find(item =>
      item.bundle === bundleName || item.name === bundleName,
    )
    if (!overviewBundle) {
      throw new Error(`Unknown overview bundle: ${bundleName}`)
    }

    const { asset, body, contentEncoding } = await this.readManifestAsset(manifest, overviewBundle, options)
    return { overviewBundle: asset, body, contentEncoding }
  }

  health() {
    this.pruneJobs()
    const readyWorkers = this.workers.filter(worker => worker.process !== null).length
    const busyWorkers = this.workers.filter(worker => worker.currentJobId !== null).length
    return {
      workerPoolSize: this.workerPoolSize,
      maxQueueLength: this.maxQueueLength,
      maxStoredJobs: this.maxStoredJobs,
      completedJobRetentionMs: this.completedJobRetentionMs,
      readyWorkers,
      busyWorkers,
      queuedJobs: this.queue.length,
      totalJobs: this.jobs.size,
      cacheEntries: this.countCacheEntries(),
      pythonExecutable: this.pythonExecutable,
      repoRoot: REPO_ROOT,
      prewarm: { ...this.prewarmState },
      requiredPaths: REQUIRED_RUNTIME_PATHS.map(relativePath => ({
        path: relativePath,
        exists: existsSync(path.join(REPO_ROOT, relativePath)),
      })),
      workerStates: this.workers.map(worker => ({
        index: worker.index,
        pid: worker.process?.pid ?? null,
        ready: worker.process !== null,
        busy: worker.currentJobId !== null,
        currentJobId: worker.currentJobId,
        lastStartedAt: worker.lastStartedAt,
        lastError: worker.lastError,
      })),
    }
  }

  private snapshot(job: JobRecord): SimulationJobSnapshot {
    return {
      id: job.id,
      status: job.status,
      createdAt: job.createdAt,
      updatedAt: job.updatedAt,
      configHash: job.configHash,
      queuePosition: job.queuePosition,
      cacheHit: job.cacheHit,
      error: job.error,
      config: job.config,
      manifest: job.manifest,
    }
  }

  private notify(job: JobRecord): void {
    const listeners = this.listeners.get(job.id)
    if (!listeners || listeners.size === 0) {
      return
    }

    const snapshot = this.snapshot(job)
    for (const listener of listeners) {
      listener(snapshot)
    }
  }

  private touch(job: JobRecord): void {
    job.updatedAt = nowIso()
    job.updatedAtMs = Date.now()
    this.notify(job)
  }

  private finalizeCancelled(job: JobRecord, reason: string): void {
    job.status = 'cancelled'
    job.error = reason
    job.queuePosition = null
    job.workerSlot = null
    if (this.configToJob.get(job.configHash) === job.id) {
      this.configToJob.delete(job.configHash)
    }
    this.touch(job)
    this.pruneJobs()
  }

  private cancelSupersededQueuedJobs(clientId: string): void {
    for (const jobId of [...this.queue]) {
      const job = this.jobs.get(jobId)
      if (!job || job.clientId !== clientId) continue
      this.queue.splice(this.queue.indexOf(jobId), 1)
      this.finalizeCancelled(job, 'Superseded by a newer submission from the same client.')
    }
    this.refreshQueuePositions()
  }

  private dropStaleQueuedJobs(): void {
    const cutoff = Date.now() - this.queuedJobTtlMs
    for (const jobId of [...this.queue]) {
      const job = this.jobs.get(jobId)
      if (!job || job.createdAtMs >= cutoff) continue
      this.queue.splice(this.queue.indexOf(jobId), 1)
      this.finalizeCancelled(job, 'Dropped from the queue after exceeding the stale-job timeout.')
    }
    this.refreshQueuePositions()
  }

  private countActiveJobsForClient(clientId: string): number {
    let count = 0
    for (const job of this.jobs.values()) {
      if (job.clientId !== clientId) continue
      if (job.status === 'queued' || job.status === 'running') {
        count += 1
      }
    }
    return count
  }

  private isTerminal(job: JobRecord): boolean {
    return job.status === 'completed' || job.status === 'failed' || job.status === 'cancelled'
  }

  private evictJob(job: JobRecord): void {
    if (this.configToJob.get(job.configHash) === job.id) {
      this.configToJob.delete(job.configHash)
    }
    this.listeners.delete(job.id)
    this.jobs.delete(job.id)
  }

  private pruneJobs(): void {
    const now = Date.now()
    const retentionCutoff = now - this.completedJobRetentionMs

    const expiredJobs = [...this.jobs.values()]
      .filter(job =>
        this.isTerminal(job)
        && job.updatedAtMs < retentionCutoff
        && job.workerSlot == null
        && !this.listeners.has(job.id),
      )
      .sort((left, right) => left.updatedAtMs - right.updatedAtMs)

    for (const job of expiredJobs) {
      this.evictJob(job)
    }

    if (this.jobs.size <= this.maxStoredJobs) {
      return
    }

    const overflowCandidates = [...this.jobs.values()]
      .filter(job =>
        this.isTerminal(job)
        && job.workerSlot == null
        && !this.listeners.has(job.id),
      )
      .sort((left, right) => left.updatedAtMs - right.updatedAtMs)

    for (const job of overflowCandidates) {
      if (this.jobs.size <= this.maxStoredJobs) break
      this.evictJob(job)
    }
  }

  private refreshQueuePositions(): void {
    for (let index = 0; index < this.queue.length; index += 1) {
      const job = this.jobs.get(this.queue[index])
      if (!job) continue
      if (job.queuePosition !== index + 1) {
        job.queuePosition = index + 1
        this.touch(job)
      }
    }
  }

  private async pumpQueue(): Promise<void> {
    if (this.pumping) return
    this.pumping = true

    try {
      this.dropStaleQueuedJobs()

      while (this.queue.length > 0) {
        const worker = await this.findIdleWorker()
        if (!worker) {
          if (!this.workers.some(currentWorker => currentWorker.currentJobId !== null)) {
            this.failQueuedJobs(this.describeWorkerUnavailableReason())
          }
          break
        }

        const nextJobId = this.queue.shift()
        if (!nextJobId) break

        const job = this.jobs.get(nextJobId)
        if (!job || job.status !== 'queued') {
          continue
        }

      this.dispatchToWorker(worker, job)
      }

      this.refreshQueuePositions()
    } finally {
      this.pumping = false
      if (this.queue.length > 0 && this.workers.some(worker => worker.currentJobId === null)) {
        void this.pumpQueue()
      }
    }
  }

  private async findIdleWorker(): Promise<WorkerSlot | null> {
    for (const worker of this.workers) {
      if (worker.currentJobId !== null) continue
      await this.ensureWorker(worker)
      if (worker.currentJobId === null) {
        return worker
      }
    }
    return null
  }

  private dispatchToWorker(worker: WorkerSlot, job: JobRecord): void {
    worker.currentJobId = job.id
    job.workerSlot = worker.index
    job.status = 'running'
    job.queuePosition = null
    this.touch(job)

    void this.sendToWorker(worker, job.id, job.config)
      .then(async manifest => {
        if (job.status === 'cancelled') {
          return
        }
        const primedManifest = await this.primeManifestAssets(manifest)
        job.status = 'completed'
        job.cacheHit = primedManifest.cacheHit
        job.manifest = primedManifest
        job.workerSlot = null
        this.touch(job)
        this.pruneJobs()
      })
      .catch(error => {
        if (job.status === 'cancelled') {
          return
        }
        job.status = 'failed'
        job.error = error instanceof Error ? error.message : 'Simulation worker failed'
        job.workerSlot = null
        if (this.configToJob.get(job.configHash) === job.id) {
          this.configToJob.delete(job.configHash)
        }
        this.touch(job)
        this.pruneJobs()
      })
      .finally(() => {
        if (worker.currentJobId === job.id) {
          worker.currentJobId = null
          worker.currentRequestId = null
        }
        void this.pumpQueue()
      })
  }

  private async sendToWorker(worker: WorkerSlot, jobId: string, config: SimulationRequest): Promise<SimulationManifest> {
    await this.ensureWorker(worker)
    if (!worker.process) {
      throw new Error('Simulation worker is not available.')
    }

    const requestId = ++this.requestCounter
    const payload = JSON.stringify({
      id: requestId,
      type: 'run',
      payload: { job_id: jobId, config },
    })

    worker.currentRequestId = requestId

    return await new Promise<SimulationManifest>((resolve, reject) => {
      this.pending.set(requestId, {
        jobId,
        workerSlot: worker.index,
        resolve,
        reject,
      })

      worker.process!.stdin.write(`${payload}\n`, 'utf8', error => {
        if (error) {
          this.pending.delete(requestId)
          reject(error)
        }
      })
    })
  }

  private async initialize(): Promise<void> {
    await this.warmWorkers()
    if (this.prewarmState.enabled) {
      void this.runCanonicalPrewarm()
    }
  }

  private async ensureWorker(worker: WorkerSlot): Promise<void> {
    if (worker.process) {
      return
    }

    const processHandle = spawn(this.pythonExecutable, [WORKER_PATH], {
      cwd: REPO_ROOT,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
      },
    })

    const stdout = readline.createInterface({ input: processHandle.stdout })
    stdout.on('line', line => {
      this.handleWorkerLine(worker, line)
    })

    processHandle.stderr.on('data', chunk => {
      const message = chunk.toString('utf8').trim()
      if (message) {
        // eslint-disable-next-line no-console
        console.error(`[simulation-worker:${worker.index}] ${message}`)
      }
    })

    processHandle.on('exit', (code, signal) => {
      this.handleWorkerExit(worker, code, signal)
    })

    processHandle.on('error', error => {
      worker.lastError = error instanceof Error ? error.message : String(error)
      worker.process = null
      // eslint-disable-next-line no-console
      console.error(`[simulation-worker:${worker.index}] failed to start`, error)
    })

    worker.lastError = null
    worker.lastStartedAt = nowIso()
    worker.process = processHandle
  }

  private handleWorkerLine(worker: WorkerSlot, line: string): void {
    if (!line.trim()) return

    let message: WorkerResponse
    try {
      message = JSON.parse(line) as WorkerResponse
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error(`[simulation-worker:${worker.index}] invalid JSON line`, error, line)
      return
    }

    const pending = this.pending.get(message.id)
    if (!pending) {
      return
    }

    this.pending.delete(message.id)
    worker.currentRequestId = null

    if (message.ok && message.result) {
      pending.resolve(message.result)
      return
    }

    const description = [
      message.error?.message ?? 'Unknown simulation worker error',
      message.error?.traceback,
    ]
      .filter(Boolean)
      .join('\n')
    pending.reject(new Error(description))
  }

  private handleWorkerExit(worker: WorkerSlot, code: number | null, signal: NodeJS.Signals | null): void {
    const requestId = worker.currentRequestId
    const jobId = worker.currentJobId

    worker.process = null
    worker.currentRequestId = null
    worker.currentJobId = null
    worker.lastError = `Simulation worker exited unexpectedly (code=${code ?? 'null'}, signal=${signal ?? 'null'}).`

    if (requestId != null) {
      const pending = this.pending.get(requestId)
      if (pending) {
        this.pending.delete(requestId)
        pending.reject(
          new Error(
            `Simulation worker exited unexpectedly (code=${code ?? 'null'}, signal=${signal ?? 'null'}).`,
          ),
        )
      }
    }

    if (jobId) {
      const job = this.jobs.get(jobId)
      if (job && job.status === 'running') {
        job.workerSlot = null
      }
    }
  }

  private async warmWorkers(): Promise<void> {
    await Promise.all(this.workers.map(worker => this.ensureWorker(worker).catch(() => undefined)))
  }

  private countCacheEntries(): number {
    try {
      return readdirSync(SIMULATION_CACHE_ROOT, { withFileTypes: true })
        .filter(entry => entry.isDirectory())
        .length
    } catch {
      return 0
    }
  }

  private async readManifestAsset<T extends { name: string }>(
    manifest: SimulationManifest,
    asset: T,
    options: { preferGzip?: boolean; preferBrotli?: boolean } = {},
  ): Promise<{ asset: T; body: Buffer | string; contentEncoding: CompressionEncoding }> {
    const outputDir = path.resolve(manifest.outputDir)
    const assetPath = path.resolve(outputDir, asset.name)
    if (!isWithinDirectory(outputDir, assetPath)) {
      throw new Error('Artifact path escapes the output directory.')
    }

    if (options.preferBrotli) {
      try {
        const body = await fs.readFile(`${assetPath}.br`)
        return { asset, body, contentEncoding: 'br' }
      } catch {
        // Fall through to the next encoding.
      }
    }

    if (options.preferGzip) {
      try {
        const body = await fs.readFile(`${assetPath}.gz`)
        return { asset, body, contentEncoding: 'gzip' }
      } catch {
        // Fall through to the raw file.
      }
    }

    const body = await fs.readFile(assetPath, 'utf8')
    return { asset, body, contentEncoding: null }
  }

  private async ensureBrotliFile(filePath: string): Promise<number | null> {
    let sourceStat
    try {
      sourceStat = await fs.stat(filePath)
    } catch (error) {
      const code = error && typeof error === 'object' && 'code' in error ? String(error.code) : null
      if (code === 'ENOENT') {
        return null
      }
      throw error
    }

    const brotliPath = `${filePath}.br`
    try {
      const brotliStat = await fs.stat(brotliPath)
      if (brotliStat.mtimeMs >= sourceStat.mtimeMs) {
        return brotliStat.size
      }
    } catch {
      // Rebuild the Brotli sidecar below.
    }

    const body = await fs.readFile(filePath)
    const compressed = await brotliCompress(body, {
      params: {
        [zlibConstants.BROTLI_PARAM_QUALITY]: 5,
      },
    })
    await fs.writeFile(brotliPath, compressed)
    return compressed.length
  }

  private async primeManifestAssets(manifest: SimulationManifest): Promise<SimulationManifest> {
    const outputDir = path.resolve(manifest.outputDir)

    const nextArtifacts = await Promise.all(manifest.artifacts.map(async artifact => {
      const assetPath = path.resolve(outputDir, artifact.name)
      if (!isWithinDirectory(outputDir, assetPath)) {
        return artifact
      }

      try {
        const brotliBytes = await this.ensureBrotliFile(assetPath)
        return brotliBytes === artifact.brotliBytes
          ? artifact
          : { ...artifact, brotliBytes }
      } catch (error) {
        // eslint-disable-next-line no-console
        console.warn(`[simulation-runtime] failed to prime Brotli artifact ${artifact.name}`, error)
        return artifact
      }
    }))

    const nextOverviewBundles = await Promise.all(manifest.overviewBundles.map(async overviewBundle => {
      const assetPath = path.resolve(outputDir, overviewBundle.name)
      if (!isWithinDirectory(outputDir, assetPath)) {
        return overviewBundle
      }

      try {
        const brotliBytes = await this.ensureBrotliFile(assetPath)
        return brotliBytes === overviewBundle.brotliBytes
          ? overviewBundle
          : { ...overviewBundle, brotliBytes }
      } catch (error) {
        // eslint-disable-next-line no-console
        console.warn(`[simulation-runtime] failed to prime Brotli overview bundle ${overviewBundle.name}`, error)
        return overviewBundle
      }
    }))

    const changed = nextArtifacts.some((artifact, index) => artifact !== manifest.artifacts[index])
      || nextOverviewBundles.some((bundle, index) => bundle !== manifest.overviewBundles[index])
    if (!changed) {
      return manifest
    }

    const nextManifest: SimulationManifest = {
      ...manifest,
      artifacts: nextArtifacts,
      overviewBundles: nextOverviewBundles,
    }

    try {
      await fs.writeFile(
        path.join(outputDir, 'explorer_manifest.json'),
        JSON.stringify(nextManifest, null, 2),
        'utf8',
      )
    } catch (error) {
      // eslint-disable-next-line no-console
      console.warn('[simulation-runtime] failed to rewrite primed manifest', error)
    }

    return nextManifest
  }

  private async runCanonicalPrewarm(): Promise<void> {
    if (!this.prewarmState.enabled || this.prewarmState.running || CANONICAL_PREWARM_CONFIGS.length === 0) {
      return
    }

    this.prewarmState.running = true
    this.prewarmState.startedAt = nowIso()
    this.prewarmState.finishedAt = null
    this.prewarmState.completed = 0
    this.prewarmState.lastError = null

    const processHandle = spawn(this.pythonExecutable, [WORKER_PATH], {
      cwd: REPO_ROOT,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
      },
    })

    const stdout = readline.createInterface({ input: processHandle.stdout })
    const pending = new Map<number, { resolve: (manifest: SimulationManifest) => void; reject: (error: Error) => void }>()
    let exited = false

    const rejectPending = (error: Error) => {
      for (const request of pending.values()) {
        request.reject(error)
      }
      pending.clear()
    }

    stdout.on('line', line => {
      if (!line.trim()) return

      let message: WorkerResponse
      try {
        message = JSON.parse(line) as WorkerResponse
      } catch (error) {
        rejectPending(new Error(`Canonical prewarm worker returned invalid JSON: ${String(error)}`))
        return
      }

      const request = pending.get(message.id)
      if (!request) {
        return
      }

      pending.delete(message.id)
      if (message.ok && message.result) {
        request.resolve(message.result)
        return
      }

      const description = [
        message.error?.message ?? 'Unknown simulation worker error',
        message.error?.traceback,
      ]
        .filter(Boolean)
        .join('\n')
      request.reject(new Error(description))
    })

    processHandle.stderr.on('data', chunk => {
      const message = chunk.toString('utf8').trim()
      if (message) {
        // eslint-disable-next-line no-console
        console.error(`[simulation-prewarm] ${message}`)
      }
    })

    processHandle.on('error', error => {
      exited = true
      rejectPending(error instanceof Error ? error : new Error(String(error)))
    })

    processHandle.on('exit', (code, signal) => {
      if (exited) return
      exited = true
      rejectPending(
        new Error(`Canonical prewarm worker exited unexpectedly (code=${code ?? 'null'}, signal=${signal ?? 'null'}).`),
      )
    })

    const sendRequest = async (requestId: number, config: SimulationRequest): Promise<SimulationManifest> => await new Promise((resolve, reject) => {
      pending.set(requestId, { resolve, reject })
      const payload = JSON.stringify({
        id: requestId,
        type: 'run',
        payload: {
          job_id: `prewarm-${configHash(config).slice(0, 12)}`,
          config,
        },
      })

      processHandle.stdin.write(`${payload}\n`, 'utf8', error => {
        if (!error) {
          return
        }
        pending.delete(requestId)
        reject(error)
      })
    })

    try {
      for (const [index, config] of CANONICAL_PREWARM_CONFIGS.entries()) {
        const requestId = index + 1
        const manifest = await sendRequest(requestId, config)
        await this.primeManifestAssets(manifest)
        this.prewarmState.completed = requestId
      }
    } catch (error) {
      this.prewarmState.lastError = error instanceof Error ? error.message : 'Canonical prewarm failed.'
    } finally {
      this.prewarmState.running = false
      this.prewarmState.finishedAt = nowIso()
      stdout.close()
      if (!exited) {
        processHandle.kill()
      }
    }
  }

  private describeWorkerUnavailableReason(): string {
    const missingPaths = REQUIRED_RUNTIME_PATHS
      .filter(relativePath => !existsSync(path.join(REPO_ROOT, relativePath)))
      .map(relativePath => path.join(REPO_ROOT, relativePath))

    const workerErrors = this.workers
      .map(worker => worker.lastError)
      .filter((message): message is string => Boolean(message))

    const parts = ['Simulation workers are unavailable.']
    if (missingPaths.length > 0) {
      parts.push(`Missing runtime assets: ${missingPaths.join(', ')}`)
    }
    if (workerErrors.length > 0) {
      parts.push(`Worker errors: ${workerErrors.join(' | ')}`)
    }
    parts.push(`Python executable: ${this.pythonExecutable}`)
    return parts.join(' ')
  }

  private failQueuedJobs(reason: string): void {
    while (this.queue.length > 0) {
      const jobId = this.queue.shift()
      if (!jobId) continue
      const job = this.jobs.get(jobId)
      if (!job || job.status !== 'queued') continue
      job.status = 'failed'
      job.error = reason
      job.queuePosition = null
      job.workerSlot = null
      if (this.configToJob.get(job.configHash) === job.id) {
        this.configToJob.delete(job.configHash)
      }
      this.touch(job)
    }
  }

  /**
   * Pre-warm the simulation cache by submitting a list of configs.
   * Each config is submitted as a background job — cache hits return instantly,
   * cache misses run the full simulation so future users get instant results.
   */
  prewarm(
    configs: ReadonlyArray<SimulationRequest>,
    label = 'prewarm',
  ): { submitted: number; alreadyCached: number; jobs: ReadonlyArray<SimulationJobSnapshot> } {
    let alreadyCached = 0
    const snapshots: SimulationJobSnapshot[] = []

    for (const config of configs) {
      const snapshot = this.submit(config, { clientId: `__${label}__` })
      snapshots.push(snapshot)
      if (snapshot.cacheHit) {
        alreadyCached++
      }
    }

    return { submitted: snapshots.length, alreadyCached, jobs: snapshots }
  }
}
