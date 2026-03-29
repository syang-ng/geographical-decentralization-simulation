import { TriangleAlert } from 'lucide-react'
import type { CaveatBlock as CaveatBlockType } from '../../types/blocks'

interface CaveatBlockProps {
  block: CaveatBlockType
}

export function CaveatBlock({ block }: CaveatBlockProps) {
  return (
    <div className="bg-surface border border-border-subtle rounded-xl p-5 border-l-[3px] border-l-warning">
      <div className="flex gap-3">
        <TriangleAlert className="w-4 h-4 text-warning shrink-0 mt-0.5" />
        <p className="text-sm text-muted leading-relaxed">
          {block.text}
        </p>
      </div>
    </div>
  )
}
