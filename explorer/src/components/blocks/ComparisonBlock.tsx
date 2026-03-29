import type { ComparisonBlock as ComparisonBlockType } from '../../types/blocks'

interface ComparisonBlockProps {
  block: ComparisonBlockType
}

export function ComparisonBlock({ block }: ComparisonBlockProps) {
  return (
    <div className="bg-surface border border-border-subtle rounded-xl p-5">
      <h3 className="text-sm font-medium text-text-primary mb-4">
        {block.title}
      </h3>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* Left — SSP / accent blue */}
        <div className="bg-[#0a0a0a] border border-border-subtle rounded-lg p-4 border-t-2 border-t-accent">
          <div className="text-xs font-medium text-accent uppercase tracking-wider mb-3">
            {block.left.label}
          </div>
          <div className="space-y-2">
            {block.left.items.map(item => (
              <div key={item.key} className="flex justify-between gap-2">
                <span className="text-xs text-muted">{item.key}</span>
                <span className="text-xs text-text-primary font-medium tabular-nums">{item.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Right — MSP / warm terracotta */}
        <div className="bg-[#0a0a0a] border border-border-subtle rounded-lg p-4 border-t-2 border-t-accent-warm">
          <div className="text-xs font-medium text-accent-warm uppercase tracking-wider mb-3">
            {block.right.label}
          </div>
          <div className="space-y-2">
            {block.right.items.map(item => (
              <div key={item.key} className="flex justify-between gap-2">
                <span className="text-xs text-muted">{item.key}</span>
                <span className="text-xs text-text-primary font-medium tabular-nums">{item.value}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {block.verdict && (
        <p className="text-xs text-muted italic mt-3 pt-3 border-t border-border-subtle">
          {block.verdict}
        </p>
      )}
    </div>
  )
}
