import { cn } from '../../lib/cn'
import type { StatBlock as StatBlockType } from '../../types/blocks'

interface StatBlockProps {
  block: StatBlockType
}

const sentimentColors = {
  positive: 'text-success',
  negative: 'text-danger',
  neutral: 'text-accent',
} as const

export function StatBlock({ block }: StatBlockProps) {
  const valueColor = block.sentiment
    ? sentimentColors[block.sentiment]
    : 'text-accent'

  return (
    <div className="bg-surface border border-border-subtle rounded-xl p-4 sm:p-5">
      <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-4">
        <div className={cn('text-4xl font-bold tabular-nums shrink-0', valueColor)}>
          {block.value}
        </div>
        {block.sentiment && <span className="sr-only">({block.sentiment})</span>}
        <div className="min-w-0">
          <div className="text-sm font-medium text-text-primary leading-snug">
            {block.label}
          </div>
          {block.sublabel && (
            <div className="text-xs text-muted mt-0.5">
              {block.sublabel}
            </div>
          )}
          {block.delta && (
            <div className={cn(
              'inline-flex items-center gap-1 mt-1.5 px-2 py-0.5 text-xs rounded-full',
              block.sentiment === 'positive' && 'bg-success/10 text-success',
              block.sentiment === 'negative' && 'bg-danger/10 text-danger',
              (!block.sentiment || block.sentiment === 'neutral') && 'bg-white/5 text-muted',
            )}>
              {block.delta}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
