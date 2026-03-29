import { z } from 'zod/v4'

export const simulationDistributionSchema = z.enum([
  'homogeneous',
  'homogeneous-gcp',
  'heterogeneous',
  'random',
])

export const simulationSourcePlacementSchema = z.enum([
  'homogeneous',
  'latency-aligned',
  'latency-misaligned',
])

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

export const simulationRenderableArtifactNameSchema = z.enum([
  'avg_mev.json',
  'supermajority_success.json',
  'failed_block_proposals.json',
  'utility_increase.json',
  'proposal_time_avg.json',
  'attestation_sum.json',
  'top_regions_final.json',
])

export const simulationMetricKeySchema = z.enum([
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
])

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
  mode: z.enum(['answer', 'guidance', 'proposed-run']),
  summary: z.string(),
  guidance: z.string().optional(),
  suggestedPrompts: z.array(z.string()).max(4).optional(),
  proposedConfig: simulationConfigSchema.optional(),
  sections: z.array(simulationViewSectionSchema).max(10),
})

export type SimulationConfigDraft = z.infer<typeof simulationConfigSchema>
export type SimulationRenderableArtifactName = z.infer<typeof simulationRenderableArtifactNameSchema>
export type SimulationMetricKey = z.infer<typeof simulationMetricKeySchema>
export type SimulationViewSection = z.infer<typeof simulationViewSectionSchema>
export type SimulationViewSpec = z.infer<typeof simulationViewSpecSchema>
