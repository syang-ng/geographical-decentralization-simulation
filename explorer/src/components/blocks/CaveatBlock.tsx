import type { CaveatBlock as CaveatBlockType } from '../../types/blocks'

interface CaveatBlockProps {
  block: CaveatBlockType
}

export function CaveatBlock({ block }: CaveatBlockProps) {
  return (
    <div className="border-l-2 border-warning pl-5 py-1">
      <div className="flex gap-2.5">
        <span className="w-2 h-2 rounded-full bg-warning shrink-0 mt-1.5" />
        <p className="text-sm text-text-body leading-relaxed font-serif">
          {block.text}
        </p>
      </div>
    </div>
  )
}
