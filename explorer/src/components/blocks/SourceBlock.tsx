import { FileText, ExternalLink } from 'lucide-react'
import type { SourceBlock as SourceBlockType } from '../../types/blocks'

interface SourceBlockProps {
  block: SourceBlockType
}

export function SourceBlock({ block }: SourceBlockProps) {
  return (
    <div className="bg-surface border border-border-subtle rounded-xl p-4">
      <div className="space-y-1.5">
        {block.refs.map((ref, i) => {
          const isExternal = Boolean(ref.url)
          const Icon = isExternal ? ExternalLink : FileText

          return isExternal ? (
            <a
              key={i}
              href={ref.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2.5 text-xs text-accent hover:text-accent/80 transition-colors py-1"
            >
              <Icon className="w-3.5 h-3.5 shrink-0" />
              <span>{ref.label}</span>
              {ref.section && (
                <span className="text-muted/60">· {ref.section}</span>
              )}
            </a>
          ) : (
            <div key={i} className="flex items-center gap-2.5 text-xs text-muted py-1">
              <Icon className="w-3.5 h-3.5 shrink-0" />
              <span>{ref.label}</span>
              {ref.section && (
                <span className="text-muted/60">· {ref.section}</span>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
