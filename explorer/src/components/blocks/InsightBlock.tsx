import { cn } from '../../lib/cn'
import type { InsightBlock as InsightBlockType } from '../../types/blocks'

interface InsightBlockProps {
  block: InsightBlockType
}

const emphasisStyles = {
  normal: 'border-accent',
  'key-finding': 'border-accent-warm',
  surprising: 'border-danger',
} as const

export function InsightBlock({ block }: InsightBlockProps) {
  const emphasis = block.emphasis ?? 'normal'

  return (
    <div className={cn(
      'bg-surface border border-border-subtle rounded-xl p-5 border-l-[3px]',
      emphasisStyles[emphasis],
    )}>
      {block.title && (
        <h3 className="text-lg font-semibold text-text-primary mb-2">
          {block.title}
        </h3>
      )}
      <p className="text-sm leading-relaxed text-muted font-serif">
        <InlineMarkdown text={block.text} />
      </p>
    </div>
  )
}

function InlineMarkdown({ text }: { text: string }) {
  const parts = text.split(/(\*\*.*?\*\*|\*.*?\*)/g)
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={i} className="text-text-primary font-semibold">{part.slice(2, -2)}</strong>
        }
        if (part.startsWith('*') && part.endsWith('*')) {
          return <em key={i}>{part.slice(1, -1)}</em>
        }
        return <span key={i}>{part}</span>
      })}
    </>
  )
}
