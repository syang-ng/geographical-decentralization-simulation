import { Suspense, lazy, useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Header } from './components/layout/Header'
import { TabNav, type TabId } from './components/layout/TabNav'
import { Footer } from './components/layout/Footer'
import { FindingsPage } from './pages/FindingsPage'
import { ExploreHistoryPage } from './pages/ExploreHistoryPage'
import { cn } from './lib/cn'
import { SPRING_SOFT } from './lib/theme'

const DeepDivePage = lazy(async () => {
  const module = await import('./pages/DeepDivePage')
  return { default: module.DeepDivePage }
})

const PaperReaderPage = lazy(async () => {
  const module = await import('./pages/PaperReaderPage')
  return { default: module.PaperReaderPage }
})

const SimulationLabPage = lazy(async () => {
  const module = await import('./pages/SimulationLabPage')
  return { default: module.SimulationLabPage }
})

const VALID_TABS: readonly TabId[] = ['findings', 'history', 'paper', 'deep-dive', 'simulation']

function getInitialTab(): TabId {
  const params = new URLSearchParams(window.location.search)
  const tab = params.get('tab') as TabId | null
  return tab && VALID_TABS.includes(tab) ? tab : 'findings'
}

function getInitialQuery(): string | null {
  return new URLSearchParams(window.location.search).get('q')
}

function App() {
  const [activeTab, setActiveTab] = useState<TabId>(getInitialTab)
  const [sharedQuery] = useState<string | null>(getInitialQuery)

  const handleTabChange = useCallback((tab: TabId) => {
    setActiveTab(tab)
    const url = new URL(window.location.href)
    if (tab === 'findings') {
      url.searchParams.delete('tab')
    } else {
      url.searchParams.set('tab', tab)
    }
    url.searchParams.delete('q')
    window.history.replaceState({}, '', url.toString())
  }, [])

  return (
    <div className="min-h-screen bg-canvas">
      <Header />
      <TabNav activeTab={activeTab} onTabChange={handleTabChange} />

      <main
        className={cn(
          'mx-auto px-4 py-8 sm:px-6',
          activeTab === 'paper' ? 'max-w-7xl' : 'max-w-5xl',
        )}
      >
        <AnimatePresence mode="wait">
          {activeTab === 'findings' && (
            <motion.div
              key="findings"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={SPRING_SOFT}
            >
              <FindingsPage initialQuery={sharedQuery} />
            </motion.div>
          )}

          {activeTab === 'history' && (
            <motion.div
              key="history"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={SPRING_SOFT}
            >
              <ExploreHistoryPage />
            </motion.div>
          )}

          {activeTab === 'paper' && (
            <motion.div
              key="paper"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={SPRING_SOFT}
            >
              <Suspense fallback={<TabLoading title="Loading Paper Reader" description="Preparing the editorial reading view of the paper." />}>
                <PaperReaderPage />
              </Suspense>
            </motion.div>
          )}

          {activeTab === 'deep-dive' && (
            <motion.div
              key="deep-dive"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={SPRING_SOFT}
            >
              <Suspense fallback={<TabLoading title="Loading Deep Dive" description="Preparing the paper deep-dive blocks." />}>
                <DeepDivePage />
              </Suspense>
            </motion.div>
          )}

          {activeTab === 'simulation' && (
            <motion.div
              key="simulation"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={SPRING_SOFT}
            >
              <Suspense fallback={<TabLoading title="Loading Simulation Lab" description="Preparing the exact-mode simulation controls." />}>
                <SimulationLabPage />
              </Suspense>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      <Footer />
    </div>
  )
}

function TabLoading({ title, description }: { title: string; description: string }) {
  return (
    <div className="overflow-hidden rounded-[1.75rem] border border-border-subtle bg-surface/70 shadow-[0_18px_50px_rgba(0,0,0,0.14)]">
      <div className="border-b border-border-subtle bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.12),transparent_38%),linear-gradient(180deg,rgba(255,255,255,0.03),transparent)] px-5 py-5 sm:px-6">
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.22em] text-accent/80">
          <div className="h-2 w-2 rounded-full bg-accent" />
          Loading view
        </div>
        <h2 className="mt-3 text-xl font-medium text-text-primary">{title}</h2>
        <p className="mt-2 max-w-xl text-sm text-muted">{description}</p>
      </div>

      <div className="grid gap-4 px-5 py-5 sm:grid-cols-[minmax(0,1fr)_280px] sm:px-6">
        <div className="space-y-4">
          <div className="animate-pulse rounded-2xl border border-white/6 bg-black/10 p-5">
            <div className="h-3 w-24 rounded bg-white/8" />
            <div className="mt-4 h-8 w-3/4 rounded bg-white/10" />
            <div className="mt-3 h-4 w-full rounded bg-white/8" />
            <div className="mt-2 h-4 w-5/6 rounded bg-white/8" />
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {Array.from({ length: 4 }).map((_, index) => (
              <div
                key={index}
                className="animate-pulse rounded-2xl border border-white/6 bg-black/10 p-4"
              >
                <div className="h-3 w-16 rounded bg-white/8" />
                <div className="mt-4 h-5 w-1/2 rounded bg-white/10" />
                <div className="mt-3 h-3 w-4/5 rounded bg-white/8" />
                <div className="mt-2 h-3 w-3/5 rounded bg-white/8" />
              </div>
            ))}
          </div>
        </div>

        <div className="animate-pulse rounded-2xl border border-white/6 bg-black/10 p-5">
          <div className="h-3 w-20 rounded bg-white/8" />
          <div className="mt-4 space-y-3">
            {Array.from({ length: 6 }).map((_, index) => (
              <div key={index} className="rounded-xl border border-white/6 bg-white/[0.03] p-3">
                <div className="h-3 w-12 rounded bg-white/8" />
                <div className="mt-3 h-4 w-full rounded bg-white/10" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
