import { z } from 'zod/v4'

// --- Individual block schemas ---

export const statBlockSchema = z.object({
  type: z.literal('stat'),
  value: z.string(),
  label: z.string(),
  sublabel: z.string().optional(),
  delta: z.string().optional(),
  sentiment: z.enum(['positive', 'negative', 'neutral']).optional(),
})

export const insightBlockSchema = z.object({
  type: z.literal('insight'),
  title: z.string().optional(),
  text: z.string(),
  emphasis: z.enum(['normal', 'key-finding', 'surprising']).optional(),
})

export const chartBlockSchema = z.object({
  type: z.literal('chart'),
  title: z.string(),
  data: z.array(z.object({
    label: z.string(),
    value: z.number(),
    category: z.string().optional(),
  })),
  unit: z.string().optional(),
  chartType: z.enum(['bar', 'line']).optional(),
})

export const comparisonBlockSchema = z.object({
  type: z.literal('comparison'),
  title: z.string(),
  left: z.object({
    label: z.string(),
    items: z.array(z.object({ key: z.string(), value: z.string() })),
  }),
  right: z.object({
    label: z.string(),
    items: z.array(z.object({ key: z.string(), value: z.string() })),
  }),
  verdict: z.string().optional(),
})

export const tableBlockSchema = z.object({
  type: z.literal('table'),
  title: z.string(),
  headers: z.array(z.string()),
  rows: z.array(z.array(z.string())),
  highlight: z.array(z.number()).optional(),
})

export const caveatBlockSchema = z.object({
  type: z.literal('caveat'),
  text: z.string(),
})

export const sourceBlockSchema = z.object({
  type: z.literal('source'),
  refs: z.array(z.object({
    label: z.string(),
    section: z.string().optional(),
    url: z.string().optional(),
  })),
})

export const mapBlockSchema = z.object({
  type: z.literal('map'),
  title: z.string(),
  regions: z.array(z.object({
    name: z.string(),
    lat: z.number(),
    lon: z.number(),
    value: z.number(),
    label: z.string().optional(),
  })),
  colorScale: z.enum(['density', 'change', 'binary']).optional(),
})

export const timeSeriesBlockSchema = z.object({
  type: z.literal('timeseries'),
  title: z.string(),
  series: z.array(z.object({
    label: z.string(),
    data: z.array(z.object({ x: z.number(), y: z.number() })),
    color: z.string().optional(),
  })),
  xLabel: z.string().optional(),
  yLabel: z.string().optional(),
  annotations: z.array(z.object({
    x: z.number(),
    label: z.string(),
  })).optional(),
})

// --- Discriminated union ---

export const blockSchema = z.discriminatedUnion('type', [
  statBlockSchema,
  insightBlockSchema,
  chartBlockSchema,
  comparisonBlockSchema,
  tableBlockSchema,
  caveatBlockSchema,
  sourceBlockSchema,
  mapBlockSchema,
  timeSeriesBlockSchema,
])

// --- TypeScript types (inferred from schemas) ---

export type StatBlock = z.infer<typeof statBlockSchema>
export type InsightBlock = z.infer<typeof insightBlockSchema>
export type ChartBlock = z.infer<typeof chartBlockSchema>
export type ComparisonBlock = z.infer<typeof comparisonBlockSchema>
export type TableBlock = z.infer<typeof tableBlockSchema>
export type CaveatBlock = z.infer<typeof caveatBlockSchema>
export type SourceBlock = z.infer<typeof sourceBlockSchema>
export type MapBlock = z.infer<typeof mapBlockSchema>
export type TimeSeriesBlock = z.infer<typeof timeSeriesBlockSchema>
export type Block = z.infer<typeof blockSchema>

// --- Validation helper ---

export function parseBlocks(raw: unknown[]): Block[] {
  return raw.flatMap(item => {
    const result = blockSchema.safeParse(item)
    return result.success ? [result.data] : []
  })
}
