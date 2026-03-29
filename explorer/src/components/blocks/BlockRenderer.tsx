import type { Block } from '../../types/blocks'
import { StatBlock } from './StatBlock'
import { InsightBlock } from './InsightBlock'
import { ChartBlock } from './ChartBlock'
import { ComparisonBlock } from './ComparisonBlock'
import { TableBlock } from './TableBlock'
import { CaveatBlock } from './CaveatBlock'
import { SourceBlock } from './SourceBlock'
import { MapBlock } from './MapBlock'
import { TimeSeriesBlock } from './TimeSeriesBlock'

interface BlockRendererProps {
  block: Block
}

export function BlockRenderer({ block }: BlockRendererProps) {
  switch (block.type) {
    case 'stat':
      return <StatBlock block={block} />
    case 'insight':
      return <InsightBlock block={block} />
    case 'chart':
      return <ChartBlock block={block} />
    case 'comparison':
      return <ComparisonBlock block={block} />
    case 'table':
      return <TableBlock block={block} />
    case 'caveat':
      return <CaveatBlock block={block} />
    case 'source':
      return <SourceBlock block={block} />
    case 'map':
      return <MapBlock block={block} />
    case 'timeseries':
      return <TimeSeriesBlock block={block} />
  }
}
