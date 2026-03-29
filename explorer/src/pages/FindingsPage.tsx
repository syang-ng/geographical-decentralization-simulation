import { useState, useCallback, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowRight, ArrowLeft } from 'lucide-react'
import { cn } from '../lib/cn'
import { DEFAULT_BLOCKS, TOPIC_CARDS, type TopicCard } from '../data/default-blocks'
import { BlockCanvas } from '../components/explore/BlockCanvas'
import { QueryBar } from '../components/explore/QueryBar'
import { QueryHistory, type HistoryEntry } from '../components/explore/QueryHistory'
import { ShimmerLoading } from '../components/explore/ShimmerBlock'
import { ErrorDisplay } from '../components/explore/ErrorDisplay'
import { explore, type ExploreError } from '../lib/api'
import { SPRING } from '../lib/theme'
import type { Block } from '../types/blocks'

interface AiResponse {
  readonly summary: string
  readonly blocks: readonly Block[]
  readonly followUps: readonly string[]
}

interface FindingsPageProps {
  readonly initialQuery?: string | null
}

export function FindingsPage({ initialQuery }: FindingsPageProps) {
  const [activeTopic, setActiveTopic] = useState<TopicCard | null>(null)

  // AI exploration state
  const [aiResponse, setAiResponse] = useState<AiResponse | null>(null)
  const [activeQuery, setActiveQuery] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<ExploreError | null>(null)
  const [history, setHistory] = useState<HistoryEntry[]>([])

  // Auto-execute shared query from URL ?q= param
  useEffect(() => {
    if (initialQuery?.trim()) {
      handleQuery(initialQuery.trim())
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Run once on mount only

  const handleTopicClick = (card: TopicCard) => {
    // Clear AI state when browsing topic cards
    setAiResponse(null)
    setActiveQuery(null)
    setError(null)
    setActiveTopic(prev => prev?.id === card.id ? null : card)
  }

  const handleBackToOverview = () => {
    setActiveTopic(null)
    setAiResponse(null)
    setActiveQuery(null)
    setError(null)
  }

  const handleQuery = useCallback(async (query: string) => {
    setActiveTopic(null)
    setActiveQuery(query)
    setAiResponse(null)
    setError(null)
    setLoading(true)

    const result = await explore(
      query,
      history.map(h => ({ query: h.query, summary: h.summary })),
    )

    setLoading(false)

    if (result.ok) {
      setAiResponse({
        summary: result.data.summary,
        blocks: result.data.blocks,
        followUps: result.data.followUps,
      })
      setHistory(prev => [
        ...prev,
        { query, summary: result.data.summary, timestamp: Date.now() },
      ])
    } else {
      setError(result.error)
    }
  }, [history])

  const handleHistorySelect = useCallback((entry: HistoryEntry) => {
    handleQuery(entry.query)
  }, [handleQuery])

  const handleRetry = useCallback(() => {
    if (activeQuery) handleQuery(activeQuery)
  }, [activeQuery, handleQuery])

  // Determine what to show in the block canvas area
  const showAi = aiResponse !== null || loading || error !== null
  const showTopic = activeTopic !== null && !showAi

  return (
    <div>
      {/* Query bar */}
      <div className="mb-6">
        <QueryBar onSubmit={handleQuery} loading={loading} />
      </div>

      {/* Query history */}
      <QueryHistory
        entries={history}
        onSelect={handleHistorySelect}
        activeQuery={activeQuery ?? undefined}
      />

      {/* Topic cards section */}
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

      {/* Divider */}
      <div className="border-t border-dashed border-border-subtle mb-6" />

      {/* Block canvas area */}
      <div className="mb-3">
        <span className="text-xs text-muted uppercase tracking-wider font-medium">
          {aiResponse
            ? aiResponse.summary
            : showTopic && activeTopic
              ? activeTopic.title
              : 'Key findings at a glance'}
        </span>
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
            {/* Follow-up suggestions */}
            {aiResponse.followUps.length > 0 && (
              <div className="mt-6 pt-4 border-t border-dashed border-border-subtle">
                <span className="text-[10px] text-muted uppercase tracking-wider font-medium mb-2 block">
                  Keep exploring
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {aiResponse.followUps.map((q, i) => (
                    <button
                      key={i}
                      onClick={() => handleQuery(q)}
                      className="px-2.5 py-1 rounded-full text-[10px] text-muted/70 border border-border-subtle hover:border-accent/30 hover:text-accent transition-all"
                    >
                      {q}
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
