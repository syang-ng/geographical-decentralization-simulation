import { useState } from 'react'
import { motion } from 'framer-motion'
import { cn } from '../../lib/cn'
import { HOVER_LIFT } from '../../lib/theme'
import type { ComparisonBlock as ComparisonBlockType } from '../../types/blocks'

interface ComparisonBlockProps {
  block: ComparisonBlockType
}

export function ComparisonBlock({ block }: ComparisonBlockProps) {
  const [hoveredSide, setHoveredSide] = useState<'left' | 'right' | null>(null)

  return (
    <motion.div
      {...HOVER_LIFT}
      className="bg-white border border-border-subtle rounded-lg p-5 topo-bg"
    >
      <h3 className="text-base font-semibold text-text-primary mb-4 font-serif">
        {block.title}
      </h3>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-0 sm:divide-x sm:divide-border-subtle">
        {/* Left — SSP / ocean blue dot */}
        <div
          onMouseEnter={() => setHoveredSide('left')}
          onMouseLeave={() => setHoveredSide(null)}
          className={cn(
            'pb-4 sm:pb-0 sm:pr-5 border-b sm:border-b-0 border-border-subtle rounded-md transition-colors duration-150',
            hoveredSide === 'left' && 'bg-accent/[0.03]',
            hoveredSide === 'right' && 'opacity-60',
          )}
        >
          <div className="flex items-center gap-2 text-sm font-medium text-text-primary mb-3">
            <span className={cn(
              'w-2.5 h-2.5 rounded-full bg-accent transition-shadow duration-150',
              hoveredSide === 'left' && 'shadow-[0_0_6px_rgba(37,99,235,0.3)]',
            )} />
            {block.left.label}
          </div>
          <div className="space-y-2.5">
            {block.left.items.map(item => (
              <div key={item.key} className="flex justify-between gap-2">
                <span className="text-xs text-muted">{item.key}</span>
                <span className="text-sm text-text-primary font-semibold tabular-nums">{item.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right — MSP / terracotta dot */}
        <div
          onMouseEnter={() => setHoveredSide('right')}
          onMouseLeave={() => setHoveredSide(null)}
          className={cn(
            'pt-4 sm:pt-0 sm:pl-5 rounded-md transition-colors duration-150',
            hoveredSide === 'right' && 'bg-accent-warm/[0.03]',
            hoveredSide === 'left' && 'opacity-60',
          )}
        >
          <div className="flex items-center gap-2 text-sm font-medium text-text-primary mb-3">
            <span className={cn(
              'w-2.5 h-2.5 rounded-full bg-accent-warm transition-shadow duration-150',
              hoveredSide === 'right' && 'shadow-[0_0_6px_rgba(194,85,58,0.3)]',
            )} />
            {block.right.label}
          </div>
          <div className="space-y-2.5">
            {block.right.items.map(item => (
              <div key={item.key} className="flex justify-between gap-2">
                <span className="text-xs text-muted">{item.key}</span>
                <span className="text-sm text-text-primary font-semibold tabular-nums">{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {block.verdict && (
        <p className="text-sm text-muted italic mt-5 pt-4 border-t border-rule font-serif">
          {block.verdict}
        </p>
      )}
    </motion.div>
  )
}
