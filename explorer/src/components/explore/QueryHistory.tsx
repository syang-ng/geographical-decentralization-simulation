import { motion, AnimatePresence } from 'framer-motion'
import { Clock, ChevronRight } from 'lucide-react'
import { cn } from '../../lib/cn'
import { SPRING } from '../../lib/theme'

export interface HistoryEntry {
  readonly query: string
  readonly summary: string
  readonly timestamp: number
}

interface QueryHistoryProps {
  entries: readonly HistoryEntry[]
  onSelect: (entry: HistoryEntry) => void
  activeQuery?: string
}

export function QueryHistory({ entries, onSelect, activeQuery }: QueryHistoryProps) {
  if (entries.length === 0) return null

  return (
    <div className="mb-6">
      <div className="flex items-center gap-1.5 mb-2">
        <Clock className="w-3 h-3 text-muted/60" />
        <span className="text-[10px] text-muted uppercase tracking-wider font-medium">
          This session
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        <AnimatePresence mode="popLayout">
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
                className={cn(
                  'flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] transition-all',
                  'border max-w-[200px] truncate',
                  isActive
                    ? 'border-accent/40 bg-accent/10 text-accent'
                    : 'border-border-subtle bg-surface/50 text-muted hover:text-text-primary hover:border-white/10',
                )}
              >
                <span className="truncate">{entry.query}</span>
                <ChevronRight className="w-2.5 h-2.5 shrink-0 opacity-40" />
              </motion.button>
            )
          })}
        </AnimatePresence>
      </div>
    </div>
  )
}
