import { motion, AnimatePresence } from 'framer-motion'
import { Clock, ChevronRight } from 'lucide-react'
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

function provenanceTone(source: ExploreResponse['provenance']['source']): string {
  if (source === 'curated') return 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300'
  if (source === 'history') return 'border-amber-500/20 bg-amber-500/10 text-amber-300'
  return 'border-accent/20 bg-accent/10 text-accent'
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
        <Clock className="w-3 h-3 text-muted/60" />
        <span className="text-[10px] text-muted uppercase tracking-wider font-medium">
          This session
        </span>
      </div>

      <div className="hide-scrollbar -mx-1 overflow-x-auto px-1 pb-1">
        <AnimatePresence mode="popLayout">
          <div className="flex min-w-full gap-2">
            {entries.map((entry, i) => {
            const isActive = activeQuery === entry.query
            return (
              <motion.button
                key={entry.timestamp}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ ...SPRING, delay: i * 0.03 }}
                onClick={() => onSelect(entry)}
                title={`${entry.summary} · ${entry.response.provenance.label}`}
                whileHover={{ y: -1 }}
                className={cn(
                  'min-w-[220px] max-w-[260px] shrink-0 rounded-xl border px-3 py-2 text-left transition-all',
                  'bg-surface/70 backdrop-blur-sm',
                  isActive
                    ? 'border-accent/40 bg-accent/10 text-text-primary shadow-[0_10px_30px_rgba(59,130,246,0.10)]'
                    : 'border-border-subtle text-muted hover:border-white/10 hover:text-text-primary',
                )}
              >
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span
                    className={cn(
                      'inline-flex items-center rounded-full border px-2 py-0.5 text-[9px] font-medium uppercase tracking-[0.18em]',
                      provenanceTone(entry.response.provenance.source),
                    )}
                  >
                    {entry.response.provenance.source}
                  </span>
                  <span className="text-[10px] text-muted/70">
                    {formatHistoryTime(entry.timestamp)}
                  </span>
                </div>

                <div className="flex items-start gap-2">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[11px] font-medium text-text-primary">
                      {entry.query}
                    </div>
                    <div className="mt-1 line-clamp-2 text-[10px] text-muted">
                      {entry.summary}
                    </div>
                  </div>
                  <ChevronRight className="mt-0.5 h-3 w-3 shrink-0 opacity-40" />
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
