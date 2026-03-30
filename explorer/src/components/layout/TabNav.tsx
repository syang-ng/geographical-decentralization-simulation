import { motion } from 'framer-motion'
import { cn } from '../../lib/cn'
import { SPRING } from '../../lib/theme'

export type TabId = 'findings' | 'history' | 'paper' | 'deep-dive' | 'simulation'

interface TabNavProps {
  activeTab: TabId
  onTabChange: (tab: TabId) => void
}

const tabs: { id: TabId; label: string; shortLabel: string }[] = [
  { id: 'findings', label: 'Findings', shortLabel: 'Findings' },
  { id: 'history', label: 'Explore History', shortLabel: 'History' },
  { id: 'paper', label: 'Paper', shortLabel: 'Paper' },
  { id: 'deep-dive', label: 'Deep Dive', shortLabel: 'Dive' },
  { id: 'simulation', label: 'Simulation Lab', shortLabel: 'Sim' },
]

export function TabNav({ activeTab, onTabChange }: TabNavProps) {
  return (
    <div className="sticky top-0 z-20 border-b border-border-subtle bg-white">
      <div className="max-w-5xl mx-auto px-4 sm:px-6">
        <nav className="flex gap-1" aria-label="Explorer sections">
          {tabs.map(tab => {
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                aria-label={tab.label}
                aria-current={isActive ? 'page' : undefined}
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
            )
          })}
        </nav>
      </div>
    </div>
  )
}
