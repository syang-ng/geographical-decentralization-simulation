import { motion } from 'framer-motion'
import { FlaskConical, Clock, BookOpen, Cpu, FileText } from 'lucide-react'
import { cn } from '../../lib/cn'
import { SPRING } from '../../lib/theme'

export type TabId = 'findings' | 'history' | 'paper' | 'deep-dive' | 'simulation'

interface TabNavProps {
  activeTab: TabId
  onTabChange: (tab: TabId) => void
}

const tabs: { id: TabId; label: string; shortLabel: string; icon: typeof FlaskConical }[] = [
  { id: 'findings', label: 'Findings', shortLabel: 'Findings', icon: FlaskConical },
  { id: 'history', label: 'Explore History', shortLabel: 'History', icon: Clock },
  { id: 'paper', label: 'Paper', shortLabel: 'Paper', icon: FileText },
  { id: 'deep-dive', label: 'Deep Dive', shortLabel: 'Dive', icon: BookOpen },
  { id: 'simulation', label: 'Simulation Lab', shortLabel: 'Sim', icon: Cpu },
]

export function TabNav({ activeTab, onTabChange }: TabNavProps) {
  return (
    <div className="sticky top-0 z-20 border-b border-border-subtle bg-canvas/80 backdrop-blur-sm">
      <div className="max-w-5xl mx-auto px-4 sm:px-6">
        <nav className="flex gap-0.5 sm:gap-1" aria-label="Explorer sections">
          {tabs.map(tab => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                aria-label={tab.label}
                aria-current={isActive ? 'page' : undefined}
                className={cn(
                  'relative flex items-center gap-1.5 px-2.5 py-3 text-sm transition-colors sm:gap-2 sm:px-4',
                  isActive
                    ? 'text-text-primary'
                    : 'text-muted hover:text-text-primary',
                )}
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span className="text-[10px] sm:text-sm">{tab.shortLabel}</span>
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
