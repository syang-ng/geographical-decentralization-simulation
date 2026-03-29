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

/** Convert a Zod schema to a clean JSON Schema object (no $schema, no $ref) */
function toToolSchema(schema: z.ZodType): Record<string, unknown> {
  const full = zodToJsonSchema(schema, { target: 'openAi' })
  // zodToJsonSchema wraps in { $schema, ...rest } — strip the meta keys
  const { $schema: _s, default: _d, ...clean } = full as Record<string, unknown>
  return clean
}

/** The render_blocks tool definition for Claude tool_use */
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

  return [
    {
      name: 'render_blocks',
      description:
        'Compose visual blocks to answer the user\'s question about the geo-decentralization study. ' +
        'Use a mix of block types: stat for key numbers, insight for explanations, chart for data, ' +
        'comparison for SSP vs MSP, table for structured data, map for geography, timeseries for trends, ' +
        'caveat for limitations, source for references. Maximum 6 blocks.',
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
  ]
}
