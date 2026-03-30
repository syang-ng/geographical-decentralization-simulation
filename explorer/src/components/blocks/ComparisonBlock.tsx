import type { ComparisonBlock as ComparisonBlockType } from '../../types/blocks'

interface ComparisonBlockProps {
  block: ComparisonBlockType
}

export function ComparisonBlock({ block }: ComparisonBlockProps) {
  return (
    <div className="bg-white border border-border-subtle rounded-lg p-5">
      <h3 className="text-sm font-medium text-text-primary mb-4">
        {block.title}
      </h3>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-0 sm:divide-x sm:divide-border-subtle">
        {/* Left — SSP / blue dot */}
        <div className="pb-4 sm:pb-0 sm:pr-4 border-b sm:border-b-0 border-border-subtle">
          <div className="flex items-center gap-2 text-sm font-medium text-text-primary mb-3">
            <span className="w-2 h-2 rounded-full bg-accent" />
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

        {/* Right — MSP / warm dot */}
        <div className="pt-4 sm:pt-0 sm:pl-4">
          <div className="flex items-center gap-2 text-sm font-medium text-text-primary mb-3">
            <span className="w-2 h-2 rounded-full bg-accent-warm" />
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
        <p className="text-xs text-muted italic mt-4 pt-3 border-t border-rule">
          {block.verdict}
        </p>
      )}
    </div>
  )
}
