import { z } from 'zod/v4'

export const simulationDistributionValues = [
  'homogeneous',
  'homogeneous-gcp',
  'heterogeneous',
  'random',
 ] as const

export const simulationDistributionSchema = z.enum(simulationDistributionValues)

export const simulationSourcePlacementValues = [
  'homogeneous',
  'latency-aligned',
  'latency-misaligned',
 ] as const

export const simulationSourcePlacementSchema = z.enum(simulationSourcePlacementValues)

export const simulationConfigSchema = z.object({
  paradigm: z.enum(['SSP', 'MSP']),
  validators: z.number().int(),
  slots: z.number().int(),
  distribution: simulationDistributionSchema,
  sourcePlacement: simulationSourcePlacementSchema,
  migrationCost: z.number(),
  attestationThreshold: z.number(),
  slotTime: z.number().int(),
  seed: z.number().int(),
})

export const simulationRenderableArtifactNames = [
  'avg_mev.json',
  'supermajority_success.json',
  'failed_block_proposals.json',
  'utility_increase.json',
  'proposal_time_avg.json',
  'attestation_sum.json',
  'top_regions_final.json',
 ] as const

export const simulationRenderableArtifactNameSchema = z.enum(simulationRenderableArtifactNames)

export const simulationArtifactBundles = [
  'core-outcomes',
  'timing-and-attestation',
  'geography-overview',
 ] as const

export const simulationArtifactBundleSchema = z.enum(simulationArtifactBundles)

export const simulationMetricKeys = [
  'finalAverageMev',
  'finalSupermajoritySuccess',
  'finalFailedBlockProposals',
  'finalUtilityIncrease',
  'slotsRecorded',
  'attestationCutoffMs',
  'validators',
  'slots',
  'migrationCost',
  'attestationThreshold',
  'slotTime',
  'seed',
 ] as const

export const simulationMetricKeySchema = z.enum(simulationMetricKeys)

export const simulationChartMetricKeys = [
  'finalAverageMev',
  'finalSupermajoritySuccess',
  'finalFailedBlockProposals',
  'finalUtilityIncrease',
  'validators',
  'slots',
  'migrationCost',
  'attestationThreshold',
  'slotTime',
  'attestationCutoffMs',
 ] as const

export const simulationChartMetricKeySchema = z.enum(simulationChartMetricKeys)

export const simulationViewModes = ['answer', 'guidance', 'proposed-run'] as const
export const simulationSectionKinds = [
  'metric',
  'artifact',
  'artifact-bundle',
  'summary-chart',
  'insight',
  'caveat',
  'source',
] as const
export const simulationMetricSentiments = ['positive', 'negative', 'neutral'] as const
export const simulationInsightEmphases = ['normal', 'key-finding', 'surprising'] as const

export const simulationViewSectionSchema = z.discriminatedUnion('kind', [
  z.object({
    kind: z.literal('metric'),
    metric: simulationMetricKeySchema,
    label: z.string().optional(),
    sublabel: z.string().optional(),
    sentiment: z.enum(['positive', 'negative', 'neutral']).optional(),
  }),
  z.object({
    kind: z.literal('artifact'),
    artifactName: simulationRenderableArtifactNameSchema,
    title: z.string().optional(),
    note: z.string().optional(),
  }),
  z.object({
    kind: z.literal('artifact-bundle'),
    bundle: simulationArtifactBundleSchema,
    title: z.string().optional(),
    note: z.string().optional(),
  }),
  z.object({
    kind: z.literal('summary-chart'),
    title: z.string(),
    metrics: z.array(simulationChartMetricKeySchema).min(2).max(6),
    unit: z.string().optional(),
    note: z.string().optional(),
  }),
  z.object({
    kind: z.literal('insight'),
    title: z.string().optional(),
    text: z.string(),
    emphasis: z.enum(['normal', 'key-finding', 'surprising']).optional(),
  }),
  z.object({
    kind: z.literal('caveat'),
    text: z.string(),
  }),
  z.object({
    kind: z.literal('source'),
    refs: z.array(z.object({
      label: z.string(),
      section: z.string().optional(),
      url: z.string().optional(),
    })),
  }),
])

export const simulationViewSpecSchema = z.object({
  mode: z.enum(simulationViewModes),
  summary: z.string(),
  guidance: z.string().optional(),
  truthBoundary: z.string().optional(),
  suggestedPrompts: z.array(z.string()).max(4).optional(),
  proposedConfig: simulationConfigSchema.optional(),
  sections: z.array(simulationViewSectionSchema).max(10),
})

export type SimulationConfigDraft = z.infer<typeof simulationConfigSchema>
export type SimulationRenderableArtifactName = z.infer<typeof simulationRenderableArtifactNameSchema>
export type SimulationArtifactBundle = z.infer<typeof simulationArtifactBundleSchema>
export type SimulationMetricKey = z.infer<typeof simulationMetricKeySchema>
export type SimulationChartMetricKey = z.infer<typeof simulationChartMetricKeySchema>
export type SimulationViewSection = z.infer<typeof simulationViewSectionSchema>
export type SimulationViewSpec = z.infer<typeof simulationViewSpecSchema>
