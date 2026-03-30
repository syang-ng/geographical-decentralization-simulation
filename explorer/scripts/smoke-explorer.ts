import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const EXPLORER_ROOT = path.resolve(__dirname, '..')
const DATA_DIR = path.join(EXPLORER_ROOT, 'server', 'data')
const DATA_FILE = path.join(DATA_DIR, 'explorations.json')
const PORT = 3219
const BASE_URL = `http://127.0.0.1:${PORT}`
const TSX_CLI = path.join(EXPLORER_ROOT, 'node_modules', 'tsx', 'dist', 'cli.mjs')

interface ExplorationSeed {
  readonly id: string
  readonly query: string
  readonly normalizedQuery: string
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

function normalizeQuery(query: string): string {
  return query
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message)
  }
}

async function waitForSimulation(
  jobId: string,
  attempts = 60,
): Promise<Record<string, unknown>> {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const response = await fetch(`${BASE_URL}/api/simulations/${jobId}`)
    assert(response.ok, `Expected simulation job ${jobId} lookup to succeed`)
    const payload = await response.json() as Record<string, unknown>
    const status = payload.status

    if (
      status === 'completed'
      || status === 'failed'
      || status === 'cancelled'
    ) {
      return payload
    }

    await new Promise(resolve => setTimeout(resolve, 250))
  }

  throw new Error(`Timed out waiting for simulation job ${jobId}`)
}

async function waitForServer(url: string, attempts = 40): Promise<void> {
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    try {
      const response = await fetch(url)
      if (response.ok) return
    } catch {
      // Server not ready yet.
    }

    await new Promise(resolve => setTimeout(resolve, 250))
  }

  throw new Error(`Timed out waiting for ${url}`)
}

async function withSeededHistory(seed: ExplorationSeed, run: () => Promise<void>) {
  await mkdir(DATA_DIR, { recursive: true })
  const hadOriginal = existsSync(DATA_FILE)
  const original = hadOriginal ? await readFile(DATA_FILE, 'utf8') : null

  try {
    await writeFile(DATA_FILE, JSON.stringify([seed], null, 2), 'utf8')
    await run()
  } finally {
    if (original !== null) {
      await writeFile(DATA_FILE, original, 'utf8')
    } else {
      await rm(DATA_FILE, { force: true })
    }
  }
}

function startServer(): ChildProcessWithoutNullStreams {
  const child = spawn(
    process.execPath,
    [TSX_CLI, 'server/index.ts'],
    {
      cwd: EXPLORER_ROOT,
      env: {
        ...process.env,
        PORT: String(PORT),
      },
      stdio: 'pipe',
    },
  )

  child.stdout.on('data', chunk => {
    process.stdout.write(`[explorer] ${chunk}`)
  })
  child.stderr.on('data', chunk => {
    process.stderr.write(`[explorer:err] ${chunk}`)
  })

  return child
}

async function stopServer(child: ChildProcessWithoutNullStreams) {
  if (child.killed || child.exitCode !== null) return

  const exited = new Promise<void>(resolve => {
    child.once('exit', () => resolve())
  })

  child.kill()
  const timeout = new Promise<void>(resolve => {
    setTimeout(() => {
      if (child.exitCode === null) {
        child.kill('SIGKILL')
      }
      resolve()
    }, 1500)
  })

  await Promise.race([exited, timeout])
}

async function main() {
  const seededExploration: ExplorationSeed = {
    id: randomUUID(),
    query: 'How do asymmetric relay latencies change proposer migration after convergence?',
    normalizedQuery: normalizeQuery('How do asymmetric relay latencies change proposer migration after convergence?'),
    summary: 'Asymmetric relay paths bias late-stage proposer moves toward the lowest-latency relay corridor.',
    blocks: [
      {
        type: 'insight',
        emphasis: 'surprising',
        title: 'Opposite gamma sensitivity',
        text: 'Once most validators have converged, asymmetric relay paths still pull proposers toward the lowest-latency relay corridor.',
      },
    ],
    followUps: ['Does this change the final concentration ranking?'],
    model: 'smoke-seed',
    cached: true,
    source: 'generated',
    votes: 3,
    createdAt: new Date().toISOString(),
    paradigmTags: ['SSP', 'MSP'],
    experimentTags: ['SE4'],
    verified: true,
  }

  await withSeededHistory(seededExploration, async () => {
    const server = startServer()

    try {
      await waitForServer(`${BASE_URL}/api/health`)

      const healthResponse = await fetch(`${BASE_URL}/api/health`)
      assert(healthResponse.ok, 'Expected /api/health to succeed')
      const health = await healthResponse.json() as Record<string, unknown>
      assert(typeof health.tools === 'number' && health.tools >= 7, 'Expected expanded tool catalog in /api/health')
      const simulations = health.simulations as Record<string, unknown> | undefined
      assert(typeof simulations?.readyWorkers === 'number' && simulations.readyWorkers >= 1, 'Expected at least one ready simulation worker')

      const curatedResponse = await fetch(`${BASE_URL}/api/explore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'How does SSP compare to MSP?' }),
      })
      assert(curatedResponse.ok, 'Expected curated /api/explore request to succeed')
      const curatedPayload = await curatedResponse.json() as Record<string, unknown>
      const curatedProvenance = curatedPayload.provenance as Record<string, unknown> | undefined
      assert(curatedProvenance?.source === 'curated', 'Expected curated query to return curated provenance')
      assert(Array.isArray(curatedPayload.blocks) && curatedPayload.blocks.length > 0, 'Expected curated query to return blocks')

      const historyResponse = await fetch(`${BASE_URL}/api/explore`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: seededExploration.query }),
      })
      assert(historyResponse.ok, 'Expected history /api/explore request to succeed')
      const historyPayload = await historyResponse.json() as Record<string, unknown>
      const historyProvenance = historyPayload.provenance as Record<string, unknown> | undefined
      assert(historyProvenance?.source === 'history', 'Expected seeded history query to reuse public history')
      assert(historyProvenance?.explorationId === seededExploration.id, 'Expected history query to reference seeded exploration ID')

      const listResponse = await fetch(`${BASE_URL}/api/explorations?search=relay latencies`)
      assert(listResponse.ok, 'Expected /api/explorations search to succeed')
      const listPayload = await listResponse.json() as Array<Record<string, unknown>>
      assert(listPayload.some(entry => entry.id === seededExploration.id), 'Expected seeded exploration to appear in search results')

      const simulationResponse = await fetch(`${BASE_URL}/api/simulations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          paradigm: 'SSP',
          validators: 48,
          slots: 4,
          distribution: 'homogeneous',
          sourcePlacement: 'homogeneous',
          migrationCost: 0.0001,
          attestationThreshold: 2 / 3,
          slotTime: 12,
          seed: 123,
        }),
      })
      assert(simulationResponse.ok, 'Expected /api/simulations submission to succeed')
      const simulationJob = await simulationResponse.json() as Record<string, unknown>
      assert(typeof simulationJob.id === 'string', 'Expected simulation job to include an ID')
      const simulationJobId = simulationJob.id

      const finalJob = await waitForSimulation(simulationJobId)
      assert(finalJob.status === 'completed', 'Expected smoke simulation to complete')

      const manifestResponse = await fetch(`${BASE_URL}/api/simulations/${simulationJobId}/manifest`)
      assert(manifestResponse.ok, 'Expected completed simulation manifest to be available')
      const manifest = await manifestResponse.json() as Record<string, unknown>
      const summary = manifest.summary as Record<string, unknown> | undefined
      assert(summary?.finalAverageMev === 0.555879, 'Expected canonical SSP smoke avg MEV to match the saved benchmark')
      assert(summary?.finalSupermajoritySuccess === 100, 'Expected canonical SSP smoke supermajority success to match the saved benchmark')
      assert(summary?.finalFailedBlockProposals === 0, 'Expected canonical SSP smoke failed block proposals to match the saved benchmark')

      if (health.anthropicEnabled === false) {
        const copilotResponse = await fetch(`${BASE_URL}/api/simulation-copilot`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            question: 'What should I look at first in this run?',
            currentJobId: simulationJobId,
          }),
        })
        assert(copilotResponse.status === 503, 'Expected simulation copilot to return 503 when Anthropic is not configured')
      }

      process.stdout.write('Explorer smoke checks passed.\n')
    } finally {
      await stopServer(server)
    }
  })
}

main().catch(error => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error)
  process.stderr.write(`${message}\n`)
  process.exitCode = 1
})
