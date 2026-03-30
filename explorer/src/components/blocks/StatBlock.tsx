import { cn } from '../../lib/cn'
import type { StatBlock as StatBlockType } from '../../types/blocks'

interface StatBlockProps {
  block: StatBlockType
}

export function StatBlock({ block }: StatBlockProps) {
  return (
    <div className="bg-white border border-border-subtle rounded-lg p-4 sm:p-5">
      <div className="text-3xl font-semibold tabular-nums text-text-primary">
        {block.value}
      </div>
      {block.sentiment && <span className="sr-only">({block.sentiment})</span>}
      <div className="text-sm font-medium text-text-primary mt-1">
        {block.label}
      </div>
      {block.sublabel && (
        <div className="text-xs text-muted mt-0.5">
          {block.sublabel}
        </div>
      )}
      {block.delta && (
        <div className={cn(
          'inline-flex items-center gap-1 mt-2 text-xs',
          block.sentiment === 'positive' && 'text-success',
          block.sentiment === 'negative' && 'text-danger',
          (!block.sentiment || block.sentiment === 'neutral') && 'text-muted',
        )}>
          {block.delta}
        </div>
      )}
    </div>
  )
}
