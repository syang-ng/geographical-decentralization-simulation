import * as React from 'react'
import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { cn } from '../../lib/cn'
import { SPRING } from '../../lib/theme'

export type TabId = 'findings' | 'history' | 'paper' | 'deep-dive' | 'simulation'

interface TabNavProps {
  activeTab: TabId
  onTabChange: (tab: TabId) => void
}

const tabs: { id: TabId; label: string; shortLabel: string; hint: string }[] = [
  { id: 'findings', label: 'Findings', shortLabel: 'Findings', hint: 'Curated entry points & AI exploration' },
  { id: 'history', label: 'Explore History', shortLabel: 'History', hint: 'Past explorations & community votes' },
  { id: 'paper', label: 'Paper', shortLabel: 'Paper', hint: 'Full paper with editorial reading guide' },
  { id: 'deep-dive', label: 'Deep Dive', shortLabel: 'Dive', hint: 'Section-by-section argument & blocks' },
  { id: 'simulation', label: 'Simulation Lab', shortLabel: 'Sim', hint: 'Run your own parameter experiments' },
]

export function TabNav({ activeTab, onTabChange }: TabNavProps) {
  const tabRefs = React.useRef<Map<TabId, HTMLButtonElement>>(new Map())
  const [hoveredTab, setHoveredTab] = useState<TabId | null>(null)

  React.useEffect(() => {
    tabRefs.current.get(activeTab)?.focus()
  }, [activeTab])

  return (
    <div className="sticky top-0 z-20 border-b border-border-subtle bg-white/95 backdrop-blur-sm">
      <div className="max-w-5xl mx-auto px-4 sm:px-6">
        <nav className="flex gap-1" role="tablist" aria-label="Explorer sections">
          {tabs.map(tab => {
            const isActive = activeTab === tab.id
            return (
              <div key={tab.id} className="relative">
                <button
                  ref={el => { if (el) tabRefs.current.set(tab.id, el) }}
                  role="tab"
                  onClick={() => onTabChange(tab.id)}
                  onMouseEnter={() => setHoveredTab(tab.id)}
                  onMouseLeave={() => setHoveredTab(null)}
                  onKeyDown={e => {
                    const currentIndex = tabs.findIndex(t => t.id === tab.id)
                    if (e.key === 'ArrowRight') {
                      e.preventDefault()
                      const next = tabs[(currentIndex + 1) % tabs.length]
                      onTabChange(next.id)
                    } else if (e.key === 'ArrowLeft') {
                      e.preventDefault()
                      const prev = tabs[(currentIndex - 1 + tabs.length) % tabs.length]
                      onTabChange(prev.id)
                    }
                  }}
                  aria-label={tab.label}
                  aria-selected={isActive}
                  tabIndex={isActive ? 0 : -1}
                  className={cn(
                    'relative flex items-center gap-1.5 px-3 py-3 text-sm transition-colors',
                    isActive
                      ? 'text-text-primary'
                      : 'text-muted hover:text-text-primary',
                  )}
                >
                  {isActive && (
                    <span className="w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
                  )}
                  <span className="text-xs sm:text-sm">{tab.shortLabel}</span>
                  {isActive && (
                    <motion.div
                      layoutId="tab-indicator"
                      className="absolute bottom-0 left-0 right-0 h-0.5 bg-accent rounded-full"
                      transition={SPRING}
                    />
                  )}
                </button>

                {/* Subtle tooltip on hover */}
                <AnimatePresence>
                  {hoveredTab === tab.id && !isActive && (
                    <motion.div
                      initial={{ opacity: 0, y: 2 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: 2 }}
                      transition={{ duration: 0.15 }}
                      className="absolute top-full left-1/2 -translate-x-1/2 mt-1.5 z-30 whitespace-nowrap pointer-events-none"
                    >
                      <div className="rounded-md bg-text-primary px-2.5 py-1.5 text-[11px] text-white shadow-sm">
                        {tab.hint}
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            )
          })}
        </nav>
      </div>
    </div>
  )
}
