import { Suspense, lazy, useState, useCallback, useEffect } from 'react'
import { Header } from './components/layout/Header'
import { TabNav, type TabId } from './components/layout/TabNav'
import { Footer } from './components/layout/Footer'
import { FindingsPage } from './pages/FindingsPage'
import { ExploreHistoryPage } from './pages/ExploreHistoryPage'
import { cn } from './lib/cn'

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

interface ExplorerRouteState {
  readonly tab: TabId
  readonly query: string | null
  readonly explorationId: string | null
}

function getInitialTab(): TabId {
  const params = new URLSearchParams(window.location.search)
  const tab = params.get('tab') as TabId | null
  return tab && VALID_TABS.includes(tab) ? tab : 'findings'
}

function getInitialQuery(): string | null {
  return new URLSearchParams(window.location.search).get('q')
}

function getInitialExplorationId(): string | null {
  return new URLSearchParams(window.location.search).get('eid')
}

function readRouteState(): ExplorerRouteState {
  return {
    tab: getInitialTab(),
    query: getInitialQuery(),
    explorationId: getInitialExplorationId(),
  }
}

function writeRouteState(next: ExplorerRouteState, replace = false) {
  const url = new URL(window.location.href)

  if (next.tab === 'findings') {
    url.searchParams.delete('tab')
  } else {
    url.searchParams.set('tab', next.tab)
  }

  if (next.query) {
    url.searchParams.set('q', next.query)
  } else {
    url.searchParams.delete('q')
  }

  if (next.explorationId) {
    url.searchParams.set('eid', next.explorationId)
  } else {
    url.searchParams.delete('eid')
  }

  if (replace) {
    window.history.replaceState({}, '', url.toString())
  } else {
    window.history.pushState({}, '', url.toString())
  }
}

function addVisitedTab(previous: readonly TabId[], tab: TabId): readonly TabId[] {
  return previous.includes(tab) ? previous : [...previous, tab]
}

function App() {
  const initialRoute = readRouteState()
  const [activeTab, setActiveTab] = useState<TabId>(initialRoute.tab)
  const [sharedQuery, setSharedQuery] = useState<string | null>(initialRoute.query)
  const [sharedExplorationId, setSharedExplorationId] = useState<string | null>(initialRoute.explorationId)
  const [visitedTabs, setVisitedTabs] = useState<readonly TabId[]>([initialRoute.tab])

  const applyRouteState = useCallback((next: ExplorerRouteState, replace = false) => {
    setActiveTab(next.tab)
    setSharedQuery(next.query)
    setSharedExplorationId(next.explorationId)
    setVisitedTabs(previous => addVisitedTab(previous, next.tab))
    writeRouteState(next, replace)
  }, [])

  const syncFromLocation = useCallback(() => {
    const next = readRouteState()
    setActiveTab(next.tab)
    setSharedQuery(next.query)
    setSharedExplorationId(next.explorationId)
    setVisitedTabs(previous => addVisitedTab(previous, next.tab))
  }, [])

  useEffect(() => {
    const handlePopState = () => {
      syncFromLocation()
    }

    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [syncFromLocation])

  const handleTabChange = useCallback((tab: TabId) => {
    applyRouteState({ tab, query: sharedQuery, explorationId: null }, false)
  }, [applyRouteState, sharedQuery])

  const handleFindingsQueryChange = useCallback((query: string | null) => {
    applyRouteState({ tab: 'findings', query, explorationId: null }, false)
  }, [applyRouteState])

  const handleExplorationIdChange = useCallback((explorationId: string | null) => {
    applyRouteState({ tab: 'findings', query: null, explorationId }, false)
  }, [applyRouteState])

  const shouldRenderLazyTab = useCallback((tab: TabId) => {
    return visitedTabs.includes(tab) || activeTab === tab
  }, [activeTab, visitedTabs])

  return (
    <div className="min-h-screen bg-canvas">
      <a href="#main-content" className="skip-to-content">
        Skip to content
      </a>
      <Header />
      <TabNav activeTab={activeTab} onTabChange={handleTabChange} />

      <main
        id="main-content"
        className={cn(
          'mx-auto px-4 py-8 sm:px-6',
          activeTab === 'paper' ? 'max-w-7xl' : 'max-w-5xl',
        )}
      >
        <div hidden={activeTab !== 'findings'} aria-hidden={activeTab !== 'findings'}>
          <FindingsPage
            initialQuery={sharedQuery}
            initialExplorationId={sharedExplorationId}
            isActive={activeTab === 'findings'}
            onQueryChange={handleFindingsQueryChange}
            onExplorationIdChange={handleExplorationIdChange}
          />
        </div>

        <div hidden={activeTab !== 'history'} aria-hidden={activeTab !== 'history'}>
          <ExploreHistoryPage
            onGoToFindings={() => handleTabChange('findings')}
            onOpenQuery={handleFindingsQueryChange}
          />
        </div>

        {shouldRenderLazyTab('paper') && (
          <div hidden={activeTab !== 'paper'} aria-hidden={activeTab !== 'paper'}>
            <Suspense fallback={<TabLoading title="Loading Paper Reader" description="Preparing the editorial reading view of the paper." />}>
              <PaperReaderPage />
            </Suspense>
          </div>
        )}

        {shouldRenderLazyTab('deep-dive') && (
          <div hidden={activeTab !== 'deep-dive'} aria-hidden={activeTab !== 'deep-dive'}>
            <Suspense fallback={<TabLoading title="Loading Deep Dive" description="Preparing the paper deep-dive blocks." />}>
              <DeepDivePage />
            </Suspense>
          </div>
        )}

        {shouldRenderLazyTab('simulation') && (
          <div hidden={activeTab !== 'simulation'} aria-hidden={activeTab !== 'simulation'}>
            <Suspense fallback={<TabLoading title="Loading Simulation Lab" description="Preparing the exact-mode simulation controls." />}>
              <SimulationLabPage />
            </Suspense>
          </div>
        )}
      </main>

      <Footer />
    </div>
  )
}

function TabLoading({ title, description }: { title: string; description: string }) {
  return (
    <div className="overflow-hidden rounded-lg border border-border-subtle bg-white">
      <div className="border-b border-border-subtle px-5 py-5 sm:px-6">
        <div className="flex items-center gap-2 text-xs text-muted">
          <span className="h-2 w-2 rounded-full bg-accent" />
          Loading view
        </div>
        <h2 className="mt-3 text-xl font-medium text-text-primary">{title}</h2>
        <p className="mt-2 max-w-xl text-sm text-muted">{description}</p>
      </div>

      <div className="grid gap-4 px-5 py-5 sm:grid-cols-[minmax(0,1fr)_280px] sm:px-6">
        <div className="space-y-4">
          <div className="animate-pulse rounded-lg border border-border-subtle p-5">
            <div className="h-3 w-24 rounded bg-[#F0F0EE]" />
            <div className="mt-4 h-8 w-3/4 rounded bg-[#F0F0EE]" />
            <div className="mt-3 h-4 w-full rounded bg-[#F0F0EE]" />
            <div className="mt-2 h-4 w-5/6 rounded bg-[#F0F0EE]" />
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            {Array.from({ length: 4 }).map((_, index) => (
              <div
                key={index}
                className="animate-pulse rounded-lg border border-border-subtle p-4"
              >
                <div className="h-3 w-16 rounded bg-[#F0F0EE]" />
                <div className="mt-4 h-5 w-1/2 rounded bg-[#F0F0EE]" />
                <div className="mt-3 h-3 w-4/5 rounded bg-[#F0F0EE]" />
                <div className="mt-2 h-3 w-3/5 rounded bg-[#F0F0EE]" />
              </div>
            ))}
          </div>
        </div>

        <div className="animate-pulse rounded-lg border border-border-subtle p-5">
          <div className="h-3 w-20 rounded bg-[#F0F0EE]" />
          <div className="mt-4 space-y-3">
            {Array.from({ length: 6 }).map((_, index) => (
              <div key={index} className="rounded-lg border border-border-subtle p-3">
                <div className="h-3 w-12 rounded bg-[#F0F0EE]" />
                <div className="mt-3 h-4 w-full rounded bg-[#F0F0EE]" />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
