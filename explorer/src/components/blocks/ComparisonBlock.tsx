import { motion } from 'framer-motion'
import { HOVER_LIFT } from '../../lib/theme'
import type { ComparisonBlock as ComparisonBlockType } from '../../types/blocks'

interface ComparisonBlockProps {
  block: ComparisonBlockType
}

export function ComparisonBlock({ block }: ComparisonBlockProps) {
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
        <div className="pb-4 sm:pb-0 sm:pr-5 border-b sm:border-b-0 border-border-subtle">
          <div className="flex items-center gap-2 text-sm font-medium text-text-primary mb-3">
            <span className="w-2.5 h-2.5 rounded-full bg-accent" />
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
        <div className="pt-4 sm:pt-0 sm:pl-5">
          <div className="flex items-center gap-2 text-sm font-medium text-text-primary mb-3">
            <span className="w-2.5 h-2.5 rounded-full bg-accent-warm" />
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
