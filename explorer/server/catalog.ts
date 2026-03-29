/**
 * Auto-generates Claude tool_use JSON schema from the Zod block schemas.
 * Single source of truth: types/blocks.ts defines the schemas, this file
 * converts them to tool_use format. No manual duplication.
 */

import type Anthropic from '@anthropic-ai/sdk'
import { zodToJsonSchema } from 'zod-to-json-schema'
import { z } from 'zod/v4'
import {
  statBlockSchema,
  insightBlockSchema,
  chartBlockSchema,
  comparisonBlockSchema,
  tableBlockSchema,
  caveatBlockSchema,
  sourceBlockSchema,
  mapBlockSchema,
  timeSeriesBlockSchema,
} from '../src/types/blocks.ts'
import { simulationViewSpecSchema } from '../src/types/simulation-view.ts'

/** Convert a Zod schema to a clean JSON Schema object (no $schema, no $ref) */
function toToolSchema(schema: z.ZodType): Record<string, unknown> {
  const full = zodToJsonSchema(schema, { target: 'openAi' })
  // zodToJsonSchema wraps in { $schema, ...rest } — strip the meta keys
  const { $schema: _s, default: _d, ...clean } = full as Record<string, unknown>
  return clean
}

/** All tool definitions for Claude tool_use */
export function buildTools(): Anthropic.Messages.Tool[] {
  const blockSchemas = [
    statBlockSchema,
    insightBlockSchema,
    chartBlockSchema,
    comparisonBlockSchema,
    tableBlockSchema,
    caveatBlockSchema,
    sourceBlockSchema,
    mapBlockSchema,
    timeSeriesBlockSchema,
  ].map(s => toToolSchema(s))
  const simulationViewSchema = toToolSchema(simulationViewSpecSchema)

  return [
    {
      name: 'search_topic_cards',
      description:
        'Search the curated findings library for editorial topic cards extracted from the paper. ' +
        'Use this first when the user is asking about known paper findings, comparisons, experiments, or metrics.',
      input_schema: {
        type: 'object' as const,
        properties: {
          query: {
            type: 'string',
            description: 'Free-text query to match against curated topic titles, descriptions, and example prompts.',
          },
          limit: {
            type: 'integer',
            description: 'Maximum number of curated topic matches to return (default: 5).',
          },
        },
        required: [],
      },
    },
    {
      name: 'get_topic_card',
      description:
        'Retrieve a single curated topic card by ID, including its editorial blocks and suggested prompts. ' +
        'Use this after search_topic_cards when a curated card appears to directly answer the question.',
      input_schema: {
        type: 'object' as const,
        properties: {
          id: {
            type: 'string',
            description: 'The curated topic card ID.',
          },
        },
        required: ['id'],
      },
    },
    {
      name: 'search_explorations',
      description:
        'Search the community exploration pool for prior questions and answers about the study. ' +
        'Use this to check if a similar question has been asked before, or to find related explorations. ' +
        'Returns a list of exploration summaries with IDs, vote counts, and tags.',
      input_schema: {
        type: 'object' as const,
        properties: {
          query: {
            type: 'string',
            description: 'Free-text search query to match against exploration titles and summaries',
          },
          paradigm: {
            type: 'string',
            enum: ['SSP', 'MSP'],
            description: 'Filter to explorations about a specific paradigm',
          },
          experiment: {
            type: 'string',
            enum: ['SE1', 'SE2', 'SE3', 'SE4'],
            description: 'Filter to explorations about a specific sensitivity experiment',
          },
          verified_only: {
            type: 'boolean',
            description: 'Only return researcher-verified explorations',
          },
          sort: {
            type: 'string',
            enum: ['recent', 'top'],
            description: 'Sort order (default: recent)',
          },
          limit: {
            type: 'integer',
            description: 'Maximum number of results (default: 10)',
          },
        },
        required: [],
      },
    },
    {
      name: 'get_exploration',
      description:
        'Retrieve a single exploration by its ID. Returns the full exploration including all blocks, ' +
        'follow-up questions, votes, and verification status.',
      input_schema: {
        type: 'object' as const,
        properties: {
          id: {
            type: 'string',
            description: 'The UUID of the exploration to retrieve',
          },
        },
        required: ['id'],
      },
    },
    {
      name: 'suggest_underexplored_topics',
      description:
        'Suggest promising follow-up topics that are lightly covered in the public exploration history. ' +
        'Use this when the user wants ideas, next questions, or underexplored angles to investigate.',
      input_schema: {
        type: 'object' as const,
        properties: {
          query: {
            type: 'string',
            description: 'Optional current user question to bias suggestions toward related but underexplored areas.',
          },
          limit: {
            type: 'integer',
            description: 'Maximum number of suggested topics to return (default: 3).',
          },
        },
        required: [],
      },
    },
    {
      name: 'build_simulation_config',
      description:
        'Compose a bounded exact-mode simulation configuration for the Simulation Lab without running it. ' +
        'Use this when the user asks what experiment to run, wants a paper-aligned preset, or asks how to encode a scenario.',
      input_schema: {
        type: 'object' as const,
        properties: {
          preset: {
            type: 'string',
            enum: [
              'baseline-ssp',
              'baseline-msp',
              'latency-aligned',
              'latency-misaligned',
              'heterogeneous-start',
              'eip-7782',
            ],
            description: 'Optional paper-aligned preset to start from before applying overrides.',
          },
          paradigm: {
            type: 'string',
            enum: ['SSP', 'MSP'],
            description: 'Block-building paradigm.',
          },
          validators: {
            type: 'integer',
            description: 'Validator count between 25 and 1000.',
          },
          slots: {
            type: 'integer',
            description: 'Number of slots between 50 and 10000.',
          },
          distribution: {
            type: 'string',
            enum: ['homogeneous', 'homogeneous-gcp', 'heterogeneous', 'random'],
            description: 'Initial validator distribution. Use homogeneous to match the upstream baseline default.',
          },
          sourcePlacement: {
            type: 'string',
            enum: ['homogeneous', 'latency-aligned', 'latency-misaligned'],
            description: 'Information-source or relay placement pattern.',
          },
          migrationCost: {
            type: 'number',
            description: 'Migration cost in ETH between 0 and 0.02.',
          },
          attestationThreshold: {
            type: 'number',
            description: 'Attestation threshold gamma as a fraction between 0 and 1.',
          },
          slotTime: {
            type: 'integer',
            enum: [6, 8, 12],
            description: 'Slot time in seconds.',
          },
          seed: {
            type: 'integer',
            description: 'Deterministic RNG seed.',
          },
        },
        required: [],
      },
    },
    {
      name: 'render_simulation_view_spec',
      description:
        'Compose a simulation-specific view specification without inventing UI code or raw chart data. ' +
        'Use this as the FINAL step for Simulation Lab questions. ' +
        'Reference only supported metrics and known artifact names from the exact simulation manifest. ' +
        'If the question is outside bounds, return guidance and suggested prompts instead of fabricating a run.',
      input_schema: simulationViewSchema,
    },
    {
      name: 'render_blocks',
      description:
        'Compose visual blocks to answer the user\'s question about the geo-decentralization study. ' +
        'Use a mix of block types: stat for key numbers, insight for explanations, chart for data, ' +
        'comparison for SSP vs MSP, table for structured data, map for geography, timeseries for trends, ' +
        'caveat for limitations, source for references. Maximum 6 blocks. ' +
        'Use this as the final presentation step after searching curated cards, prior explorations, or building a simulation config.',
      input_schema: {
        type: 'object' as const,
        properties: {
          summary: {
            type: 'string',
            description: 'One-sentence answer shown as heading above the blocks',
          },
          blocks: {
            type: 'array',
            description: 'Array of visual blocks to render',
            items: {
              anyOf: blockSchemas,
            },
          },
          follow_ups: {
            type: 'array',
            description: '2-3 follow-up questions the user might want to explore next',
            items: { type: 'string' },
          },
        },
        required: ['summary', 'blocks'],
      },
    },
    {
      name: 'verify_exploration',
      description:
        'Mark an exploration as researcher-verified (accurate and confirmed) or remove verification. ' +
        'Only use this when the exploration content has been checked for factual accuracy against the paper.',
      input_schema: {
        type: 'object' as const,
        properties: {
          id: {
            type: 'string',
            description: 'The UUID of the exploration to verify',
          },
          verified: {
            type: 'boolean',
            description: 'Whether to mark as verified (true) or unverified (false)',
          },
        },
        required: ['id', 'verified'],
      },
    },
  ]
}
