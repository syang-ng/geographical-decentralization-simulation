import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs/promises'
import path from 'node:path'
import readline from 'node:readline'
import { fileURLToPath } from 'node:url'

const SERVER_DIR = path.dirname(fileURLToPath(import.meta.url))
const EXPLORER_ROOT = path.resolve(SERVER_DIR, '..')
const REPO_ROOT = process.env.SIMULATION_REPO_ROOT
  ? path.resolve(process.env.SIMULATION_REPO_ROOT)
  : path.resolve(EXPLORER_ROOT, '..')
const WORKER_PATH = path.join(SERVER_DIR, 'simulation_worker.py')
const DEFAULT_WORKER_POOL_SIZE = Math.max(1, Math.min(Number(process.env.SIMULATION_WORKERS ?? 2), 4))
const DEFAULT_QUEUED_JOB_TTL_MS = Math.max(60_000, Number(process.env.SIMULATION_QUEUE_TTL_MS ?? 10 * 60_000))

export interface SimulationRequest {
  readonly paradigm: 'SSP' | 'MSP'
  readonly validators: number
  readonly slots: number
  readonly distribution: 'uniform' | 'heterogeneous' | 'random'
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
  readonly config: SimulationRequest
  readonly summary: SimulationSummary
  readonly artifacts: ReadonlyArray<SimulationArtifact>
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

interface WorkerSlot {
  readonly index: number
  process: ChildProcessWithoutNullStreams | null
  currentRequestId: number | null
  currentJobId: string | null
}

type JobListener = (snapshot: SimulationJobSnapshot) => void

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
  const distribution = body.distribution
  const sourcePlacement = body.sourcePlacement
  const validators = Number(body.validators)
  const slots = Number(body.slots)
  const migrationCost = Number(body.migrationCost)
  const attestationThreshold = Number(body.attestationThreshold)
  const slotTime = Number(body.slotTime)
  const seed = Number(body.seed)

  if (paradigm !== 'SSP' && paradigm !== 'MSP') return null
  if (distribution !== 'uniform' && distribution !== 'heterogeneous' && distribution !== 'random') return null
  if (sourcePlacement !== 'homogeneous' && sourcePlacement !== 'latency-aligned' && sourcePlacement !== 'latency-misaligned') return null
  if (!Number.isInteger(validators) || validators < 25 || validators > 1000) return null
  if (!Number.isInteger(slots) || slots < 50 || slots > 10000) return null
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
  private readonly pythonExecutable = process.env.PYTHON_EXECUTABLE ?? 'python'
  private readonly workerPoolSize = DEFAULT_WORKER_POOL_SIZE
  private readonly queuedJobTtlMs = DEFAULT_QUEUED_JOB_TTL_MS
  private readonly jobs = new Map<string, JobRecord>()
  private readonly configToJob = new Map<string, string>()
  private readonly queue: string[] = []
  private readonly pending = new Map<number, PendingRequest>()
  private readonly listeners = new Map<string, Set<JobListener>>()
  private readonly workers: WorkerSlot[]
  private requestCounter = 0
  private jobCounter = 0
  private pumping = false

  constructor() {
    this.workers = Array.from({ length: this.workerPoolSize }, (_, index) => ({
      index,
      process: null,
      currentRequestId: null,
      currentJobId: null,
    }))
  }

  submit(config: SimulationRequest, options: { clientId?: string | null } = {}): SimulationJobSnapshot {
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
    }
    this.dropStaleQueuedJobs()

    const id = `sim-${Date.now()}-${++this.jobCounter}`
    const createdAt = nowIso()
    const createdAtMs = Date.now()
    const job: JobRecord = {
      id,
      createdAt,
      createdAtMs,
      updatedAt: createdAt,
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
    const job = this.jobs.get(jobId)
    return job ? this.snapshot(job) : null
  }

  getManifest(jobId: string): SimulationManifest | null {
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
    options: { preferGzip?: boolean } = {},
  ): Promise<{ artifact: SimulationArtifact; body: Buffer | string; contentEncoding: 'gzip' | null }> {
    const manifest = this.getManifest(jobId)
    if (!manifest) {
      throw new Error('Manifest not available for this job yet.')
    }

    const artifact = manifest.artifacts.find(item => item.name === artifactName)
    if (!artifact) {
      throw new Error(`Unknown artifact: ${artifactName}`)
    }

    const outputDir = path.resolve(manifest.outputDir)
    const artifactPath = path.resolve(outputDir, artifact.name)
    if (!artifactPath.startsWith(outputDir)) {
      throw new Error('Artifact path escapes the output directory.')
    }

    if (options.preferGzip) {
      const gzipPath = `${artifactPath}.gz`
      try {
        const body = await fs.readFile(gzipPath)
        return { artifact, body, contentEncoding: 'gzip' }
      } catch {
        // Fall through to the raw file.
      }
    }

    const body = await fs.readFile(artifactPath, 'utf8')
    return { artifact, body, contentEncoding: null }
  }

  health() {
    const readyWorkers = this.workers.filter(worker => worker.process !== null).length
    const busyWorkers = this.workers.filter(worker => worker.currentJobId !== null).length
    return {
      workerPoolSize: this.workerPoolSize,
      readyWorkers,
      busyWorkers,
      queuedJobs: this.queue.length,
      totalJobs: this.jobs.size,
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
      .then(manifest => {
        if (job.status === 'cancelled') {
          return
        }
        job.status = 'completed'
        job.cacheHit = manifest.cacheHit
        job.manifest = manifest
        job.workerSlot = null
        this.touch(job)
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
      // eslint-disable-next-line no-console
      console.error(`[simulation-worker:${worker.index}] failed to start`, error)
    })

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
}
