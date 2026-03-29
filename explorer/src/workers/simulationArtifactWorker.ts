/// <reference lib="webworker" />

import { parseSimulationArtifactToBlocks, type SimulationRenderableArtifact } from '../lib/simulation-artifact-blocks'
import type { Block } from '../types/blocks'

interface ParseRequest {
  readonly id: number
  readonly artifact: SimulationRenderableArtifact
  readonly rawText: string
}

interface ParseSuccess {
  readonly id: number
  readonly ok: true
  readonly blocks: readonly Block[]
}

interface ParseFailure {
  readonly id: number
  readonly ok: false
  readonly error: string
}

self.onmessage = (event: MessageEvent<ParseRequest>) => {
  const { id, artifact, rawText } = event.data

  try {
    const response: ParseSuccess = {
      id,
      ok: true,
      blocks: parseSimulationArtifactToBlocks(artifact, rawText),
    }
    self.postMessage(response)
  } catch (error) {
    const response: ParseFailure = {
      id,
      ok: false,
      error: error instanceof Error ? error.message : 'Failed to parse simulation artifact.',
    }
    self.postMessage(response)
  }
}

export {}
