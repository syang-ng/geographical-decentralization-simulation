import { motion } from 'framer-motion'
import { FlaskConical, Clock, BookOpen, Cpu, FileText } from 'lucide-react'
import { cn } from '../../lib/cn'
import { SPRING } from '../../lib/theme'

export type TabId = 'findings' | 'history' | 'paper' | 'deep-dive' | 'simulation'

interface TabNavProps {
  activeTab: TabId
  onTabChange: (tab: TabId) => void
}

const tabs: { id: TabId; label: string; icon: typeof FlaskConical }[] = [
  { id: 'findings', label: 'Findings', icon: FlaskConical },
  { id: 'history', label: 'Explore History', icon: Clock },
  { id: 'paper', label: 'Paper', icon: FileText },
  { id: 'deep-dive', label: 'Deep Dive', icon: BookOpen },
  { id: 'simulation', label: 'Simulation Lab', icon: Cpu },
]

export function TabNav({ activeTab, onTabChange }: TabNavProps) {
  return (
    <div className="sticky top-0 z-20 border-b border-border-subtle bg-canvas/80 backdrop-blur-sm">
      <div className="max-w-5xl mx-auto px-4 sm:px-6">
        <nav className="flex gap-1">
          {tabs.map(tab => {
            const Icon = tab.icon
            const isActive = activeTab === tab.id
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={cn(
                  'relative flex items-center gap-2 px-4 py-3 text-sm transition-colors',
                  isActive
                    ? 'text-text-primary'
                    : 'text-muted hover:text-text-primary',
                )}
              >
                <Icon className="w-4 h-4" />
                <span className="hidden sm:inline">{tab.label}</span>
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
