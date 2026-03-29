import { useState } from 'react'
import { Search, Sparkles, Loader2 } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '../../lib/cn'
import { SPRING } from '../../lib/theme'

const EXAMPLE_CHIPS = [
  'How does SSP compare to MSP?',
  'Why does attestation threshold have opposite effects?',
  'What happens with shorter slots?',
  'Where do validators concentrate geographically?',
] as const

interface QueryBarProps {
  onSubmit?: (query: string) => void
  disabled?: boolean
  loading?: boolean
}

export function QueryBar({ onSubmit, disabled, loading }: QueryBarProps) {
  const [query, setQuery] = useState('')
  const isEnabled = !disabled && !loading

  const handleSubmit = () => {
    const trimmed = query.trim()
    if (!trimmed || !isEnabled) return
    onSubmit?.(trimmed)
    setQuery('')
  }

  const handleChip = (text: string) => {
    if (!isEnabled) return
    onSubmit?.(text)
  }

  return (
    <div>
      <div className={cn(
        'glass-1 rounded-xl transition-all',
        isEnabled && 'focus-within:border-accent/30 focus-within:shadow-[0_0_0_2px_rgba(59,130,246,0.15)]',
      )}>
        <div className="flex items-center gap-3 px-4 py-3">
          {loading ? (
            <Loader2 className="w-4 h-4 text-accent shrink-0 animate-spin" />
          ) : (
            <Sparkles className="w-4 h-4 text-accent shrink-0" />
          )}
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSubmit()}
            placeholder={disabled ? 'AI search coming in Phase 2 — explore the findings below' : 'Ask anything about the paper...'}
            disabled={!isEnabled}
            aria-label="Search the paper"
            className="flex-1 bg-transparent text-sm text-text-primary placeholder:text-muted/60 outline-none disabled:opacity-50"
          />
          {query.trim() && isEnabled ? (
            <button
              onClick={handleSubmit}
              className="p-1.5 rounded-lg bg-accent text-white hover:bg-accent/80 transition-colors"
            >
              <Search className="w-3.5 h-3.5" />
            </button>
          ) : (
            <Search className="w-4 h-4 text-muted/40" />
          )}
        </div>
      </div>

      {/* Example chips — only when enabled and no query active */}
      {!disabled && !loading && (
        <AnimatePresence>
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={SPRING}
            className="flex flex-wrap gap-1.5 mt-2 justify-center"
          >
            {EXAMPLE_CHIPS.map((chip, i) => (
              <motion.button
                key={chip}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ ...SPRING, delay: i * 0.04 }}
                onClick={() => handleChip(chip)}
                className="px-2.5 py-1 rounded-full text-[10px] text-muted/70 border border-border-subtle hover:border-accent/30 hover:text-accent transition-all"
              >
                {chip}
              </motion.button>
            ))}
          </motion.div>
        </AnimatePresence>
      )}

      {disabled && (
        <p className="text-[10px] text-muted/40 text-center mt-1.5">
          AI search coming in Phase 2 — explore the findings below
        </p>
      )}
    </div>
  )
}
