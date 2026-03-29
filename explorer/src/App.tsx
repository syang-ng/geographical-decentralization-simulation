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
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <h2 className="text-lg font-medium text-text-primary mb-2">{title}</h2>
      <p className="text-sm text-muted max-w-md">{description}</p>
    </div>
  )
}

export default App
