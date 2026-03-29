import { GCP_REGIONS } from '../data/gcp-regions'
import type { Block } from '../types/blocks'

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

export function parseSimulationArtifactToBlocks(
  artifact: SimulationRenderableArtifact,
  rawText: string,
): readonly Block[] {
  switch (artifact.name) {
    case 'avg_mev.json': {
      const values = JSON.parse(rawText) as number[]
      return [toTimeSeries('Average MEV Earned', artifact.label, values, 'ETH')]
    }
    case 'supermajority_success.json': {
      const values = JSON.parse(rawText) as number[]
      return [toTimeSeries('Supermajority Success', artifact.label, values, 'Success Rate (%)')]
    }
    case 'failed_block_proposals.json': {
      const values = JSON.parse(rawText) as number[]
      return [toTimeSeries('Failed Block Proposals', artifact.label, values, 'Count')]
    }
    case 'utility_increase.json': {
      const values = JSON.parse(rawText) as number[]
      return [toTimeSeries('Utility Increase', artifact.label, values, 'ETH')]
    }
    case 'proposal_time_avg.json': {
      const values = JSON.parse(rawText) as number[]
      return [toTimeSeries('Average Proposal Time', artifact.label, values, 'Milliseconds')]
    }
    case 'attestation_sum.json': {
      const values = JSON.parse(rawText) as number[]
      return [toTimeSeries('Aggregate Attestations', artifact.label, values, 'Aggregate Attestations')]
    }
    case 'top_regions_final.json': {
      const rows = JSON.parse(rawText) as Array<[string, number]>
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
