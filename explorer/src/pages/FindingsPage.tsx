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

const dotColor: Record<string, string> = {
  curated: 'bg-success',
  history: 'bg-warning',
  generated: 'bg-accent',
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
      : fallbackCuratedProvenance('Curated overview', "Editorial overview assembled from the paper's main findings and caveats."))

  return (
    <div>
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-xl font-semibold text-text-primary">
          Explore the paper without losing provenance.
        </h1>
        <p className="mt-2 text-sm text-muted max-w-2xl">
          Start from curated findings, revisit session history instantly, or ask a bounded question and inspect the resulting blocks.
        </p>

        <div className="flex flex-wrap items-center gap-3 mt-3 text-xs text-muted">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-success" />
            Curated cards first
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-warning" />
            Session history reuse
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-accent" />
            Fresh generation last
          </span>
        </div>

        <div className="mt-4">
          <QueryBar onSubmit={handleQuery} loading={loading} />
        </div>
      </div>

      <QueryHistory
        entries={history}
        onSelect={handleHistorySelect}
        activeQuery={activeQuery ?? undefined}
      />

      {/* Topic cards */}
      <div className="mb-8">
        <div className="mb-3 flex items-center justify-between">
          <span className="text-xs text-muted">
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

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {TOPIC_CARDS.map(card => {
            const isActive = activeTopic?.id === card.id && !showAi
            const isDimmed = (activeTopic !== null || showAi) && !isActive

            return (
              <motion.button
                key={card.id}
                onClick={() => handleTopicClick(card)}
                layout
                whileHover={{ y: -2 }}
                whileTap={{ scale: 0.98 }}
                transition={SPRING}
                aria-label={card.title}
                aria-pressed={isActive}
                className={cn(
                  'text-left rounded-lg border p-3 transition-all',
                  isActive
                    ? 'border-accent bg-white'
                    : isDimmed
                      ? 'border-border-subtle bg-white opacity-40'
                      : 'border-border-subtle bg-white hover:border-border-hover',
                )}
              >
                <h4 className="text-xs font-medium text-text-primary leading-snug mb-1 line-clamp-2">
                  {card.title}
                </h4>
                <p className="text-xs text-muted leading-relaxed line-clamp-2 mb-2">
                  {card.description}
                </p>
                <span className={cn(
                  'flex items-center gap-1 text-xs',
                  isActive ? 'text-accent' : 'text-text-faint',
                )}>
                  {isActive ? 'Viewing' : 'Explore'}
                  {!isActive && <ArrowRight className="w-2.5 h-2.5" />}
                </span>
              </motion.button>
            )
          })}
        </div>
      </div>

      <hr className="border-rule mb-6" />

      {/* Active lens */}
      <div className="mb-6">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between mb-4">
          <div>
            <div className="text-base font-medium text-text-primary">
              {heading}
            </div>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-muted shrink-0">
            <span className={cn('w-1.5 h-1.5 rounded-full', dotColor[displayProvenance.source] ?? 'bg-accent')} />
            {displayProvenance.label}
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-3 text-xs text-muted border-t border-border-subtle pt-4">
          <div>
            <div className="text-xs text-muted mb-1">Mode</div>
            <div className="text-sm text-text-primary">
              {showAi ? 'Question-driven exploration' : showTopic ? 'Curated topic card' : 'Editorial overview'}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted mb-1">Current query</div>
            <div className="text-sm text-text-primary line-clamp-2">
              {activeQuery ?? activeTopic?.prompts[0] ?? 'What are the main findings?'}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted mb-1">Follow-ups</div>
            <div className="text-sm text-text-primary">
              {aiResponse?.followUps.length ?? 0} suggested next questions
            </div>
          </div>
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
              <div className="mt-6 pt-4 border-t border-rule">
                <span className="text-xs text-muted mb-2 block">
                  Keep exploring
                </span>
                <div className="flex flex-wrap gap-2">
                  {aiResponse.followUps.map((query, index) => (
                    <button
                      key={`${query}-${index}`}
                      onClick={() => handleQuery(query)}
                      className="text-xs text-muted hover:text-accent transition-colors"
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
