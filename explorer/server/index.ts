/**
 * Express API proxy for Claude calls.
 * Keeps the API key server-side. Frontend calls /api/explore → Claude tool_use → Block[].
 *
 * Start: npx tsx server/index.ts
 * Env:   ANTHROPIC_API_KEY=sk-ant-...
 */

import express from 'express'
import cors from 'cors'
import Anthropic from '@anthropic-ai/sdk'
import { STUDY_CONTEXT } from './study-context.ts'
import { buildTools } from './catalog.ts'
import { SimulationRuntime, parseSimulationRequest } from './simulation-runtime.ts'
import { ExplorationStore } from './exploration-store.ts'

const apiKey = process.env.ANTHROPIC_API_KEY
const allowedOrigins = (process.env.ALLOWED_ORIGINS ?? 'http://localhost:3200,http://localhost:5180').split(',')

const app = express()
app.use(cors({ origin: allowedOrigins }))
app.use(express.json({ limit: '1mb' }))

// Simple in-memory rate limiter for /api/explore (10 requests/min per IP)
const rateLimitMap = new Map<string, { count: number; resetAt: number }>()
const RATE_LIMIT_MAX = 10
const RATE_LIMIT_WINDOW_MS = 60_000

function rateLimitExplore(
  req: express.Request,
  res: express.Response,
  next: express.NextFunction,
) {
  const ip = req.ip ?? req.socket.remoteAddress ?? 'unknown'
  const now = Date.now()
  const entry = rateLimitMap.get(ip)

  if (!entry || now > entry.resetAt) {
    rateLimitMap.set(ip, { count: 1, resetAt: now + RATE_LIMIT_WINDOW_MS })
    next()
    return
  }

  if (entry.count >= RATE_LIMIT_MAX) {
    const retryAfter = Math.ceil((entry.resetAt - now) / 1000)
    res.setHeader('Retry-After', String(retryAfter))
    res.status(429).json({ error: `Rate limit exceeded. Try again in ${retryAfter}s.` })
    return
  }

  entry.count++
  next()
}

const client = apiKey ? new Anthropic({ apiKey }) : null
const tools = buildTools()
const simulationRuntime = new SimulationRuntime()
const explorationStore = new ExplorationStore()

interface ExploreRequest {
  query: string
  history?: Array<{ query: string; summary: string }>
}

interface ExploreResponse {
  summary: string
  blocks: unknown[]
  followUps: string[]
  model: string
  cached: boolean
}

app.post('/api/explore', rateLimitExplore, async (req, res) => {
  if (!client) {
    res.status(503).json({ error: 'Anthropic API is not configured on this server.' })
    return
  }

  const { query, history } = req.body as ExploreRequest

  if (!query?.trim()) {
    res.status(400).json({ error: 'Query is required' })
    return
  }

  try {
    // Build dynamic session context
    const sessionContext = history?.length
      ? `\n\n## Session Context\nPrevious queries this session:\n${history.map((h, i) => `${i + 1}. "${h.query}" → ${h.summary}`).join('\n')}\n\nBuild on prior context where relevant.`
      : ''

    const response = await client.messages.create({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 4096,
      system: [
        {
          type: 'text',
          text: STUDY_CONTEXT + sessionContext,
          cache_control: history?.length ? undefined : { type: 'ephemeral' },
        },
      ],
      tools,
      tool_choice: { type: 'tool', name: 'render_blocks' },
      messages: [
        {
          role: 'user',
          content: query,
        },
      ],
    })

    // Extract tool_use result
    const toolUse = response.content.find(
      (block): block is Anthropic.Messages.ToolUseBlock => block.type === 'tool_use'
    )

    if (!toolUse) {
      res.status(500).json({ error: 'No tool_use in response' })
      return
    }

    const input = toolUse.input as {
      summary?: string
      blocks?: unknown[]
      follow_ups?: string[]
    }

    const result: ExploreResponse = {
      summary: input.summary ?? 'Here are the findings:',
      blocks: input.blocks ?? [],
      followUps: input.follow_ups ?? [],
      model: response.model,
      cached: response.usage?.cache_read_input_tokens
        ? response.usage.cache_read_input_tokens > 0
        : false,
    }

    res.json(result)

    // Auto-save exploration to store
    explorationStore.save({
      query,
      summary: result.summary,
      blocks: result.blocks,
      followUps: result.followUps,
      model: result.model,
      cached: result.cached,
    })
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error'
    const status =
      message.includes('rate_limit') ? 429 :
      message.includes('authentication') ? 401 :
      500

    res.status(status).json({ error: message })
  }
})

app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    tools: tools.length,
    anthropicEnabled: Boolean(client),
    simulations: simulationRuntime.health(),
  })
})

app.post('/api/simulations', (req, res) => {
  const config = parseSimulationRequest(req.body)
  if (!config) {
    res.status(400).json({ error: 'Invalid simulation request payload.' })
    return
  }

  const rawClientId = typeof req.body?.clientId === 'string' ? req.body.clientId : null
  const job = simulationRuntime.submit(config, { clientId: rawClientId })
  res.status(job.status === 'completed' ? 200 : 202).json(job)
})

app.get('/api/simulations/:jobId', (req, res) => {
  const job = simulationRuntime.getJob(req.params.jobId)
  if (!job) {
    res.status(404).json({ error: 'Simulation job not found.' })
    return
  }

  res.json(job)
})

app.post('/api/simulations/:jobId/cancel', (req, res) => {
  const job = simulationRuntime.cancel(req.params.jobId)
  if (!job) {
    res.status(404).json({ error: 'Simulation job not found.' })
    return
  }

  res.json(job)
})

app.get('/api/simulations/:jobId/events', (req, res) => {
  const initialSnapshot = simulationRuntime.getJob(req.params.jobId)
  if (!initialSnapshot) {
    res.status(404).json({ error: 'Simulation job not found.' })
    return
  }

  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache, no-transform')
  res.setHeader('Connection', 'keep-alive')
  res.flushHeaders?.()

  const heartbeat = setInterval(() => {
    res.write(`: keep-alive\n\n`)
  }, 15000)

  res.write(`event: snapshot\n`)
  res.write(`data: ${JSON.stringify(initialSnapshot)}\n\n`)

  if (
    initialSnapshot.status === 'completed'
    || initialSnapshot.status === 'failed'
    || initialSnapshot.status === 'cancelled'
  ) {
    clearInterval(heartbeat)
    res.end()
    return
  }

  const unsubscribe = simulationRuntime.subscribe(req.params.jobId, snapshot => {
    res.write(`event: snapshot\n`)
    res.write(`data: ${JSON.stringify(snapshot)}\n\n`)

    if (
      snapshot.status === 'completed'
      || snapshot.status === 'failed'
      || snapshot.status === 'cancelled'
    ) {
      clearInterval(heartbeat)
      unsubscribe?.()
      res.end()
    }
  })

  req.on('close', () => {
    clearInterval(heartbeat)
    unsubscribe()
  })
})

app.get('/api/simulations/:jobId/manifest', (req, res) => {
  const manifest = simulationRuntime.getManifest(req.params.jobId)
  if (!manifest) {
    res.status(409).json({ error: 'Simulation job is not completed yet.' })
    return
  }

  res.json(manifest)
})

app.get('/api/simulations/:jobId/artifacts/:artifactName', async (req, res) => {
  try {
    const acceptEncoding = req.header('accept-encoding') ?? ''
    const { artifact, body, contentEncoding } = await simulationRuntime.readArtifact(
      req.params.jobId,
      req.params.artifactName,
      { preferGzip: /\bgzip\b/i.test(acceptEncoding) },
    )
    res.setHeader('Content-Type', `${artifact.contentType}; charset=utf-8`)
    res.setHeader('ETag', artifact.sha256)
    res.setHeader('Vary', 'Accept-Encoding')
    if (contentEncoding) {
      res.setHeader('Content-Encoding', contentEncoding)
    }
    res.send(body)
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unable to read artifact.'
    const status = message.includes('not available') ? 409 :
      message.includes('Unknown artifact') ? 404 :
      500
    res.status(status).json({ error: message })
  }
})

// --- Exploration history routes ---

app.get('/api/explorations', (_req, res) => {
  const sort = (_req.query.sort as 'recent' | 'top') ?? 'recent'
  const limit = _req.query.limit ? Number(_req.query.limit) : undefined
  const search = (_req.query.search as string) ?? undefined

  const explorations = explorationStore.list({ sort, limit, search })
  res.json(explorations)
})

app.get('/api/explorations/:id', (req, res) => {
  const exploration = explorationStore.getById(req.params.id)
  if (!exploration) {
    res.status(404).json({ error: 'Exploration not found' })
    return
  }
  res.json(exploration)
})

app.post('/api/explorations/:id/verify', (req, res) => {
  const verified = (req.body as { verified?: boolean }).verified
  if (typeof verified !== 'boolean') {
    res.status(400).json({ error: 'verified must be a boolean' })
    return
  }

  const updated = explorationStore.verify(req.params.id, verified)
  if (!updated) {
    res.status(404).json({ error: 'Exploration not found' })
    return
  }
  res.json(updated)
})

app.post('/api/explorations/:id/vote', (req, res) => {
  const delta = (req.body as { delta?: number }).delta
  if (delta !== 1 && delta !== -1) {
    res.status(400).json({ error: 'delta must be 1 or -1' })
    return
  }

  const updated = explorationStore.vote(req.params.id, delta)
  if (!updated) {
    res.status(404).json({ error: 'Exploration not found' })
    return
  }
  res.json(updated)
})

const PORT = Number(process.env.PORT ?? 3201)
app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Explorer API listening on http://localhost:${PORT}`)
})
