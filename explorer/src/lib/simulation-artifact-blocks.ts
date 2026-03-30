import { GCP_REGIONS } from '../data/gcp-regions'
import { parseBlocks, type Block } from '../types/blocks'
import type {
  SimulationArtifactBundle,
  SimulationChartMetricKey,
} from '../types/simulation-view'

export interface SimulationRenderableArtifact {
  readonly name: string
  readonly label: string
  readonly kind: 'timeseries' | 'map' | 'table' | 'raw'
}

const regionIndex = new Map(GCP_REGIONS.map(region => [region.id, region]))

function toTimeSeries(title: string, label: string, values: readonly number[], yLabel: string): Block {
  return {
    type: 'timeseries',
    title,
    xLabel: 'Slot',
    yLabel,
    series: [
      {
        label,
        data: values.map((value, index) => ({
          x: index + 1,
          y: Number.isFinite(value) ? value : 0,
        })),
      },
    ],
  }
}

function parseNumericSeries(rawText: string): number[] {
  const values = JSON.parse(rawText) as unknown
  if (!Array.isArray(values)) {
    throw new Error('Expected a numeric series array.')
  }
  return values.map(value => (typeof value === 'number' && Number.isFinite(value) ? value : 0))
}

function parseTopRegions(rawText: string) {
  return JSON.parse(rawText) as Array<[string, number]>
}

export function parseSimulationBlockBundle(rawText: string): readonly Block[] {
  const values = JSON.parse(rawText) as unknown
  return Array.isArray(values) ? parseBlocks(values) : []
}

export function parseSimulationArtifactToBlocks(
  artifact: SimulationRenderableArtifact,
  rawText: string,
): readonly Block[] {
  switch (artifact.name) {
    case 'avg_mev.json': {
      const values = parseNumericSeries(rawText)
      return [toTimeSeries('Average MEV Earned', artifact.label, values, 'ETH')]
    }
    case 'supermajority_success.json': {
      const values = parseNumericSeries(rawText)
      return [toTimeSeries('Supermajority Success', artifact.label, values, 'Success Rate (%)')]
    }
    case 'failed_block_proposals.json': {
      const values = parseNumericSeries(rawText)
      return [toTimeSeries('Failed Block Proposals', artifact.label, values, 'Count')]
    }
    case 'utility_increase.json': {
      const values = parseNumericSeries(rawText)
      return [toTimeSeries('Utility Increase', artifact.label, values, 'ETH')]
    }
    case 'proposal_time_avg.json': {
      const values = parseNumericSeries(rawText)
      return [toTimeSeries('Average Proposal Time', artifact.label, values, 'Milliseconds')]
    }
    case 'attestation_sum.json': {
      const values = parseNumericSeries(rawText)
      return [toTimeSeries('Aggregate Attestations', artifact.label, values, 'Aggregate Attestations')]
    }
    case 'top_regions_final.json': {
      const rows = parseTopRegions(rawText)
      const regions = rows
        .map(([name, value]) => {
          const region = regionIndex.get(name)
          if (!region) return null
          return {
            name,
            lat: region.lat,
            lon: region.lon,
            value,
            label: `${name} - ${region.city}`,
          }
        })
        .filter((region): region is NonNullable<typeof region> => region !== null)

      return [
        {
          type: 'map',
          title: 'Final Validator Geography',
          colorScale: 'density',
          regions,
        },
        {
          type: 'table',
          title: 'Top Final Regions',
          headers: ['Region', 'Validators'],
          rows: rows.slice(0, 12).map(([name, value]) => [name, String(value)]),
          highlight: [0, 1, 2],
        },
      ]
    }
    default:
      return [
        {
          type: 'caveat',
          text: `${artifact.label} is available for download, but this artifact does not have an in-browser renderer yet.`,
        },
      ]
  }
}

export function buildSimulationSummaryChart(
  title: string,
  metricValues: ReadonlyArray<{
    metric: SimulationChartMetricKey
    label: string
    value: number
  }>,
  unit?: string,
): Block {
  return {
    type: 'chart',
    title,
    chartType: 'bar',
    unit,
    data: metricValues.map(entry => ({
      label: entry.label,
      value: entry.value,
    })),
  }
}

export function buildSimulationArtifactBundle(
  bundle: SimulationArtifactBundle,
  artifacts: Partial<Record<SimulationRenderableArtifact['name'], string>>,
): readonly Block[] {
  if (bundle === 'core-outcomes') {
    const blocks: Block[] = []
    if (artifacts['avg_mev.json']) {
      blocks.push(...parseSimulationArtifactToBlocks({
        name: 'avg_mev.json',
        label: 'Average MEV',
        kind: 'timeseries',
      }, artifacts['avg_mev.json']))
    }
    if (artifacts['supermajority_success.json']) {
      blocks.push(...parseSimulationArtifactToBlocks({
        name: 'supermajority_success.json',
        label: 'Supermajority Success',
        kind: 'timeseries',
      }, artifacts['supermajority_success.json']))
    }
    if (artifacts['failed_block_proposals.json']) {
      blocks.push(...parseSimulationArtifactToBlocks({
        name: 'failed_block_proposals.json',
        label: 'Failed Block Proposals',
        kind: 'timeseries',
      }, artifacts['failed_block_proposals.json']))
    }
    return blocks
  }

  if (bundle === 'timing-and-attestation') {
    const blocks: Block[] = []
    if (artifacts['proposal_time_avg.json']) {
      blocks.push(...parseSimulationArtifactToBlocks({
        name: 'proposal_time_avg.json',
        label: 'Average Proposal Time',
        kind: 'timeseries',
      }, artifacts['proposal_time_avg.json']))
    }
    if (artifacts['attestation_sum.json']) {
      blocks.push(...parseSimulationArtifactToBlocks({
        name: 'attestation_sum.json',
        label: 'Attestation Sum',
        kind: 'timeseries',
      }, artifacts['attestation_sum.json']))
    }
    return blocks
  }

  if (bundle === 'geography-overview') {
    if (!artifacts['top_regions_final.json']) return []
    return parseSimulationArtifactToBlocks({
      name: 'top_regions_final.json',
      label: 'Final Top Regions',
      kind: 'map',
    }, artifacts['top_regions_final.json'])
  }

  return []
}
