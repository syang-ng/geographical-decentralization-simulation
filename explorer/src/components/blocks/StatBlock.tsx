import { motion } from 'framer-motion'
import { cn } from '../../lib/cn'
import { HOVER_LIFT } from '../../lib/theme'
import type { StatBlock as StatBlockType } from '../../types/blocks'

interface StatBlockProps {
  block: StatBlockType
}

export function StatBlock({ block }: StatBlockProps) {
  return (
    <motion.div
      {...HOVER_LIFT}
      className="bg-white border border-border-subtle rounded-lg p-5 topo-bg relative overflow-hidden group geo-accent-bar"
    >
      {/* Faint coordinate corner — reveals on hover */}
      <span className="coord-label absolute top-2 right-3 opacity-0 group-hover:opacity-100 transition-opacity">
        §
      </span>

      <div className="text-4xl font-bold tabular-nums tracking-tight text-text-primary leading-none">
        {block.value}
      </div>
      {block.sentiment && <span className="sr-only">({block.sentiment})</span>}
      <div className="text-sm font-medium text-text-primary mt-2">
        {block.label}
      </div>
      {block.sublabel && (
        <div className="text-xs text-muted mt-1">
          {block.sublabel}
        </div>
      )}
      {block.delta && (
        <div className={cn(
          'inline-flex items-center gap-1.5 mt-3 text-xs font-medium',
          block.sentiment === 'positive' && 'text-success',
          block.sentiment === 'negative' && 'text-danger',
          (!block.sentiment || block.sentiment === 'neutral') && 'text-muted',
        )}>
          <span className={cn(
            'w-1.5 h-1.5 rounded-full',
            block.sentiment === 'positive' && 'bg-success',
            block.sentiment === 'negative' && 'bg-danger',
            (!block.sentiment || block.sentiment === 'neutral') && 'bg-muted',
          )} />
          {block.delta}
        </div>
      )}
    </motion.div>
  )
}
