import { motion } from 'framer-motion'
import { ArrowRight } from 'lucide-react'
import { SPRING } from '../../lib/theme'

interface WayfinderLink {
  readonly label: string
  readonly hint: string
  readonly onClick: () => void
}

interface WayfinderProps {
  readonly links: readonly WayfinderLink[]
}

export function Wayfinder({ links }: WayfinderProps) {
  if (links.length === 0) return null

  return (
    <div className="mt-10 pt-6 border-t border-rule">
      <span className="text-[10px] uppercase tracking-[0.12em] text-text-faint mb-3 block">
        Continue exploring
      </span>
      <div className="flex flex-wrap gap-3">
        {links.map(link => (
          <motion.button
            key={link.label}
            onClick={link.onClick}
            whileHover={{ y: -1 }}
            transition={SPRING}
            className="group flex items-center gap-2 rounded-lg border border-border-subtle bg-white px-4 py-2.5 text-left transition-colors hover:border-border-hover"
          >
            <div className="min-w-0">
              <span className="text-xs font-medium text-text-primary">
                {link.label}
              </span>
              <span className="block text-[11px] text-muted mt-0.5">
                {link.hint}
              </span>
            </div>
            <ArrowRight className="w-3.5 h-3.5 shrink-0 text-text-faint transition-colors group-hover:text-accent" />
          </motion.button>
        ))}
      </div>
    </div>
  )
}
