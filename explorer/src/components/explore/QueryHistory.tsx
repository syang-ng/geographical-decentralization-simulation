import { motion, AnimatePresence } from 'framer-motion'
import { ChevronRight } from 'lucide-react'
import { cn } from '../../lib/cn'
import { SPRING } from '../../lib/theme'
import type { ExploreResponse } from '../../lib/api'

export interface HistoryEntry {
  readonly query: string
  readonly summary: string
  readonly timestamp: number
  readonly response: ExploreResponse
}

interface QueryHistoryProps {
  entries: readonly HistoryEntry[]
  onSelect: (entry: HistoryEntry) => void
  activeQuery?: string
}

const dotColor: Record<string, string> = {
  curated: 'bg-success',
  history: 'bg-warning',
  generated: 'bg-accent',
}

function formatHistoryTime(timestamp: number): string {
  const deltaSeconds = Math.floor((Date.now() - timestamp) / 1000)
  if (deltaSeconds < 60) return 'just now'
  const minutes = Math.floor(deltaSeconds / 60)
  if (minutes < 60) return `${minutes}m`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h`
  const days = Math.floor(hours / 24)
  return `${days}d`
}

export function QueryHistory({ entries, onSelect, activeQuery }: QueryHistoryProps) {
  if (entries.length === 0) return null

  return (
    <div className="mb-6">
      <div className="mb-2 flex items-center gap-1.5">
        <span className="text-xs text-muted">
          This session
        </span>
      </div>

      <div className="hide-scrollbar -mx-1 overflow-x-auto px-1 pb-1">
        <AnimatePresence mode="popLayout">
          <div className="flex min-w-full gap-2">
            {entries.map((entry, i) => {
            const isActive = activeQuery === entry.query
            const source = entry.response.provenance.source
            return (
              <motion.button
                key={entry.timestamp}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ ...SPRING, delay: i * 0.03 }}
                onClick={() => onSelect(entry)}
                title={`${entry.summary} · ${entry.response.provenance.label}`}
                whileHover={{ y: -2 }}
                className={cn(
                  'min-w-[220px] max-w-[260px] shrink-0 rounded-lg border px-3 py-2 text-left transition-all',
                  isActive
                    ? 'border-accent bg-white'
                    : 'border-border-subtle bg-white hover:border-border-hover',
                )}
              >
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="inline-flex items-center gap-1.5 text-xs text-muted">
                    <span className={cn('w-1.5 h-1.5 rounded-full', dotColor[source] ?? 'bg-accent')} />
                    {source}
                  </span>
                  <span className="text-xs text-text-faint">
                    {formatHistoryTime(entry.timestamp)}
                  </span>
                </div>

                <div className="flex items-start gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-xs font-medium text-text-primary">
                      {entry.query}
                    </div>
                    <div className="mt-1 line-clamp-2 text-xs text-muted">
                      {entry.summary}
                    </div>
                  </div>
                  <ChevronRight className="mt-0.5 h-3 w-3 shrink-0 text-text-faint" />
                </div>
              </motion.button>
            )
            })}
          </div>
        </AnimatePresence>
      </div>
    </div>
  )
}
