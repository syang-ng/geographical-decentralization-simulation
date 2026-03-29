import { useState, useCallback, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowRight, ArrowLeft } from 'lucide-react'
import { cn } from '../lib/cn'
import { DEFAULT_BLOCKS, TOPIC_CARDS, type TopicCard } from '../data/default-blocks'
import { BlockCanvas } from '../components/explore/BlockCanvas'
import { QueryBar } from '../components/explore/QueryBar'
import { QueryHistory, type HistoryEntry } from '../components/explore/QueryHistory'
import { ShimmerLoading } from '../components/explore/ShimmerBlock'
import { ErrorDisplay } from '../components/explore/ErrorDisplay'
import { explore, type ExploreError, type ExploreProvenance, type ExploreResponse } from '../lib/api'
import { SPRING } from '../lib/theme'

function upsertHistory(previous: HistoryEntry[], next: HistoryEntry): HistoryEntry[] {
  return [
    next,
    ...previous.filter(entry => entry.query !== next.query),
  ].slice(0, 8)
}

function provenanceClasses(source: ExploreProvenance['source'], canonical: boolean): string {
  if (source === 'curated') {
    return canonical
      ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
      : 'border-emerald-500/20 bg-emerald-500/5 text-emerald-300'
  }
  if (source === 'history') {
    return 'border-amber-500/30 bg-amber-500/10 text-amber-300'
  }
  return 'border-accent/30 bg-accent/10 text-accent'
}

function fallbackCuratedProvenance(label: string, detail: string): ExploreProvenance {
  return {
    source: 'curated',
    label,
    detail,
    canonical: true,
  }
}

export function FindingsPage({ initialQuery = null }: { initialQuery?: string | null }) {
  const [activeTopic, setActiveTopic] = useState<TopicCard | null>(null)

  const [aiResponse, setAiResponse] = useState<ExploreResponse | null>(null)
  const [activeQuery, setActiveQuery] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<ExploreError | null>(null)
  const [history, setHistory] = useState<HistoryEntry[]>([])
  const initialQueryHandledRef = useRef(false)

  const handleTopicClick = (card: TopicCard) => {
    setAiResponse(null)
    setActiveQuery(null)
    setError(null)
    setLoading(false)
    setActiveTopic(previous => (previous?.id === card.id ? null : card))
  }

  const handleBackToOverview = () => {
    setActiveTopic(null)
    setAiResponse(null)
    setActiveQuery(null)
    setError(null)
    setLoading(false)
  }

  const handleQuery = useCallback(async (query: string) => {
    setActiveTopic(null)
    setActiveQuery(query)
    setAiResponse(null)
    setError(null)
    setLoading(true)

    const result = await explore(
      query,
      history.map(entry => ({ query: entry.query, summary: entry.summary })),
    )

    setLoading(false)

    if (result.ok) {
      setAiResponse(result.data)
      setHistory(previous => upsertHistory(previous, {
        query,
        summary: result.data.summary,
        response: result.data,
        timestamp: Date.now(),
      }))
    } else {
      setError(result.error)
    }
  }, [history])

  const handleHistorySelect = useCallback((entry: HistoryEntry) => {
    setActiveTopic(null)
    setActiveQuery(entry.query)
    setAiResponse(entry.response)
    setError(null)
    setLoading(false)
  }, [])

  const handleRetry = useCallback(() => {
    if (activeQuery) handleQuery(activeQuery)
  }, [activeQuery, handleQuery])

  useEffect(() => {
    if (!initialQuery || initialQueryHandledRef.current) return
    initialQueryHandledRef.current = true
    void handleQuery(initialQuery)
  }, [handleQuery, initialQuery])

  const showAi = aiResponse !== null || loading || error !== null
  const showTopic = activeTopic !== null && !showAi

  const heading = aiResponse
    ? aiResponse.summary
    : showTopic && activeTopic
      ? activeTopic.title
      : 'Key findings at a glance'

  const displayProvenance = aiResponse?.provenance
    ?? (showTopic && activeTopic
      ? fallbackCuratedProvenance('Curated topic card', 'Editorial paper finding selected from the curated findings library.')
      : fallbackCuratedProvenance('Curated overview', 'Editorial overview assembled from the paper’s main findings and caveats.'))

  return (
    <div>
      <div className="mb-6">
        <QueryBar onSubmit={handleQuery} loading={loading} />
      </div>

      <QueryHistory
        entries={history}
        onSelect={handleHistorySelect}
        activeQuery={activeQuery ?? undefined}
      />

      <div className="mb-8">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs text-muted uppercase tracking-wider font-medium">
            Explore a finding
          </span>
          {(showTopic || showAi) && (
            <button
              onClick={handleBackToOverview}
              className="flex items-center gap-1 text-xs text-muted hover:text-text-primary transition-colors"
            >
              <ArrowLeft className="w-3 h-3" />
              Back to overview
            </button>
          )}
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {TOPIC_CARDS.map(card => {
            const isActive = activeTopic?.id === card.id && !showAi
            const isDimmed = (activeTopic !== null || showAi) && !isActive

            return (
              <motion.button
                key={card.id}
                onClick={() => handleTopicClick(card)}
                layout
                transition={SPRING}
                aria-label={card.title}
                aria-pressed={isActive}
                className={cn(
                  'text-left rounded-lg p-3 border transition-all duration-200',
                  'bg-surface hover:border-white/10',
                  isActive
                    ? 'border-accent/40 bg-accent/5'
                    : isDimmed
                      ? 'border-border-subtle opacity-40'
                      : 'border-border-subtle',
                )}
              >
                <h4 className="text-xs font-medium text-text-primary leading-snug mb-1 line-clamp-2">
                  {card.title}
                </h4>
                <p className="text-[10px] text-muted leading-relaxed line-clamp-2 mb-2">
                  {card.description}
                </p>
                <span className={cn(
                  'flex items-center gap-1 text-[10px] font-medium',
                  isActive ? 'text-accent' : 'text-muted/60',
                )}>
                  {isActive ? 'Viewing' : 'Explore'}
                  {!isActive && <ArrowRight className="w-2.5 h-2.5" />}
                </span>
              </motion.button>
            )
          })}
        </div>
      </div>

      <div className="border-t border-dashed border-border-subtle mb-6" />

      <div className="flex flex-col gap-2 mb-4 sm:flex-row sm:items-start sm:justify-between">
        <span className="text-xs text-muted uppercase tracking-wider font-medium">
          {heading}
        </span>
        <div className="flex flex-col items-start sm:items-end gap-1">
          <span
            className={cn(
              'inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-[10px] font-medium uppercase tracking-wider',
              provenanceClasses(displayProvenance.source, displayProvenance.canonical),
            )}
          >
            {displayProvenance.label}
          </span>
          <span className="text-[11px] text-muted max-w-xl text-left sm:text-right">
            {displayProvenance.detail}
          </span>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {loading ? (
          <motion.div
            key="loading"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={SPRING}
          >
            <ShimmerLoading />
          </motion.div>
        ) : error ? (
          <motion.div
            key="error"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={SPRING}
          >
            <ErrorDisplay error={error} onRetry={handleRetry} />
          </motion.div>
        ) : aiResponse ? (
          <motion.div
            key={`ai-${activeQuery}`}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={SPRING}
          >
            <BlockCanvas blocks={aiResponse.blocks} />
            {aiResponse.followUps.length > 0 && (
              <div className="mt-6 pt-4 border-t border-dashed border-border-subtle">
                <span className="text-[10px] text-muted uppercase tracking-wider font-medium mb-2 block">
                  Keep exploring
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {aiResponse.followUps.map((query, index) => (
                    <button
                      key={`${query}-${index}`}
                      onClick={() => handleQuery(query)}
                      className="px-2.5 py-1 rounded-full text-[10px] text-muted/70 border border-border-subtle hover:border-accent/30 hover:text-accent transition-all"
                    >
                      {query}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        ) : showTopic && activeTopic ? (
          <motion.div
            key={activeTopic.id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={SPRING}
          >
            <BlockCanvas blocks={activeTopic.blocks} />
          </motion.div>
        ) : (
          <motion.div
            key="default"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={SPRING}
          >
            <BlockCanvas blocks={DEFAULT_BLOCKS} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
