import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { Search, ArrowUpDown, ThumbsUp, ThumbsDown, Tag, ChevronDown, ChevronUp } from 'lucide-react'
import { listExplorations, voteExploration, type Exploration } from '../lib/api'
import { BlockCanvas } from '../components/explore/BlockCanvas'
import { cn } from '../lib/cn'
import { SPRING, SPRING_SOFT } from '../lib/theme'

type SortMode = 'recent' | 'top'

export function ExploreHistoryPage({ onGoToFindings }: { readonly onGoToFindings?: () => void } = {}) {
  const [sort, setSort] = useState<SortMode>('recent')
  const [search, setSearch] = useState('')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const queryClient = useQueryClient()

  const { data: explorations = [], isLoading } = useQuery({
    queryKey: ['explorations', sort, search],
    queryFn: () => listExplorations({ sort, search: search || undefined }),
  })

  const voteMutation = useMutation({
    mutationFn: ({ id, delta }: { id: string; delta: 1 | -1 }) =>
      voteExploration(id, delta),
    onMutate: async ({ id, delta }) => {
      await queryClient.cancelQueries({ queryKey: ['explorations'] })
      const previous = queryClient.getQueryData<Exploration[]>(['explorations', sort, search])

      queryClient.setQueryData<Exploration[]>(
        ['explorations', sort, search],
        old => old?.map(e => (e.id === id ? { ...e, votes: e.votes + delta } : e)) ?? [],
      )

      return { previous }
    },
    onError: (_err, _vars, context) => {
      if (context?.previous) {
        queryClient.setQueryData(['explorations', sort, search], context.previous)
      }
    },
    onSettled: () => {
      void queryClient.invalidateQueries({ queryKey: ['explorations'] })
    },
  })

  const toggleSort = () => setSort(prev => (prev === 'recent' ? 'top' : 'recent'))
  const toggleExpand = (id: string) => setExpandedId(prev => (prev === id ? null : id))

  if (isLoading) {
    return <LoadingSkeleton />
  }

  if (explorations.length === 0 && !search) {
    return <EmptyState onGoToFindings={onGoToFindings} />
  }

  return (
    <div className="space-y-6">
      <HistoryHeader
        search={search}
        sort={sort}
        onSearchChange={setSearch}
        onToggleSort={toggleSort}
      />

      {explorations.length === 0 && search ? (
        <NoResults search={search} />
      ) : (
        <motion.div
          initial="hidden"
          animate="visible"
          variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.04 } } }}
          className="grid gap-4"
        >
          <AnimatePresence mode="popLayout">
            {explorations.map(exploration => (
              <ExplorationCard
                key={exploration.id}
                exploration={exploration}
                isExpanded={expandedId === exploration.id}
                onToggleExpand={() => toggleExpand(exploration.id)}
                onVote={delta => voteMutation.mutate({ id: exploration.id, delta })}
              />
            ))}
          </AnimatePresence>
        </motion.div>
      )}
    </div>
  )
}

// --- Sub-components ---

function HistoryHeader({
  search,
  sort,
  onSearchChange,
  onToggleSort,
}: {
  readonly search: string
  readonly sort: SortMode
  readonly onSearchChange: (value: string) => void
  readonly onToggleSort: () => void
}) {
  return (
    <div className="flex flex-col sm:flex-row gap-3">
      <div className="relative flex-1">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted" />
        <input
          type="text"
          value={search}
          onChange={e => onSearchChange(e.target.value)}
          placeholder="Search explorations..."
          className={cn(
            'w-full pl-10 pr-4 py-2.5 rounded-lg text-sm',
            'bg-white border border-border-subtle',
            'text-text-primary placeholder:text-muted',
            'focus:outline-none focus:ring-1 focus:ring-accent',
          )}
        />
      </div>
      <button
        onClick={onToggleSort}
        className={cn(
          'flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm',
          'bg-white border border-border-subtle',
          'text-text-primary hover:bg-surface-hover transition-colors',
        )}
      >
        <ArrowUpDown className="w-4 h-4" />
        <span>{sort === 'recent' ? 'Recent' : 'Top Voted'}</span>
      </button>
    </div>
  )
}

function ExplorationCard({
  exploration,
  isExpanded,
  onToggleExpand,
  onVote,
}: {
  readonly exploration: Exploration
  readonly isExpanded: boolean
  readonly onToggleExpand: () => void
  readonly onVote: (delta: 1 | -1) => void
}) {
  const timeAgo = formatTimeAgo(exploration.createdAt)
  const allTags = [...exploration.paradigmTags, ...exploration.experimentTags]

  return (
    <motion.div
      layout
      variants={{
        hidden: { opacity: 0, y: 12 },
        visible: { opacity: 1, y: 0, transition: SPRING },
      }}
      className={cn(
        'bg-white rounded-lg border border-border-subtle overflow-hidden',
        'transition-colors hover:border-border-hover',
      )}
    >
      {/* Card header — always visible */}
      <div className="flex gap-3 p-4">
        <VoteControls votes={exploration.votes} onVote={onVote} />

        <button
          onClick={onToggleExpand}
          className="flex-1 text-left min-w-0"
        >
          <p className="text-sm font-medium text-text-primary truncate">
            {exploration.query}
          </p>
          <p className="text-xs text-muted mt-1 line-clamp-2">
            {exploration.summary}
          </p>

          <div className="flex flex-wrap items-center gap-3 mt-3">
            <span className="inline-flex items-center gap-1.5 text-xs text-muted">
              <span className="w-1.5 h-1.5 rounded-full bg-accent" />
              Generated
            </span>

            {exploration.verified && (
              <span className="inline-flex items-center gap-1.5 text-xs text-muted">
                <span className="w-1.5 h-1.5 rounded-full bg-success" />
                Verified
              </span>
            )}
            {allTags.map(tag => (
              <span
                key={tag}
                className="inline-flex items-center gap-1.5 text-xs text-muted"
              >
                <span className={cn(
                  'w-1.5 h-1.5 rounded-full',
                  tag === 'SSP' && 'bg-accent',
                  tag === 'MSP' && 'bg-accent-warm',
                  tag.startsWith('SE') && 'bg-success',
                )} />
                {tag}
              </span>
            ))}

            <span className="text-xs text-text-faint ml-auto">
              {timeAgo}
            </span>
          </div>

          <div className="flex flex-wrap gap-2 mt-2 text-xs text-text-faint">
            {exploration.model && <span>{exploration.model}</span>}
            <span>{exploration.cached ? 'prompt cache hit' : 'new generation'}</span>
          </div>
        </button>

        <button
          onClick={onToggleExpand}
          className="self-start p-1 text-muted hover:text-text-primary transition-colors"
        >
          {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
      </div>

      {/* Expanded block canvas */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={SPRING_SOFT}
            className="overflow-hidden"
          >
            <div className="border-t border-border-subtle p-4">
              <BlockCanvas blocks={exploration.blocks} />
              {exploration.followUps.length > 0 && (
                <FollowUpList followUps={exploration.followUps} />
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

function VoteControls({
  votes,
  onVote,
}: {
  readonly votes: number
  readonly onVote: (delta: 1 | -1) => void
}) {
  return (
    <div className="flex flex-col items-center gap-1 shrink-0">
      <button
        onClick={e => { e.stopPropagation(); onVote(1) }}
        className="p-1 text-muted hover:text-accent transition-colors"
        aria-label="Upvote"
      >
        <ThumbsUp className="w-3.5 h-3.5" />
      </button>
      <span
        className={cn(
          'text-xs font-medium tabular-nums',
          votes > 0 && 'text-accent',
          votes < 0 && 'text-rose-400',
          votes === 0 && 'text-muted',
        )}
      >
        {votes}
      </span>
      <button
        onClick={e => { e.stopPropagation(); onVote(-1) }}
        className="p-1 text-muted hover:text-rose-400 transition-colors"
        aria-label="Downvote"
      >
        <ThumbsDown className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}

function FollowUpList({ followUps }: { readonly followUps: readonly string[] }) {
  return (
    <div className="mt-4 pt-3 border-t border-border-subtle">
      <span className="text-xs text-muted mb-2 block">
        Follow-up questions
      </span>
      <div className="flex flex-wrap gap-2">
        {followUps.map(q => (
          <span
            key={q}
            className="text-xs text-muted hover:text-text-primary transition-colors"
          >
            {q}
          </span>
        ))}
      </div>
    </div>
  )
}

function EmptyState({ onGoToFindings }: { readonly onGoToFindings?: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <Tag className="w-8 h-8 text-text-faint mb-4" />
      <h2 className="text-lg font-medium text-text-primary mb-2">No explorations yet</h2>
      <p className="text-sm text-muted max-w-md mb-5">
        Ask a question on the Findings tab to get started. Every Sonnet response is
        automatically saved here for the community to browse and vote on.
      </p>
      {onGoToFindings && (
        <button
          onClick={onGoToFindings}
          className="flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium bg-accent text-white hover:bg-accent/90 transition-colors"
        >
          <Search className="w-4 h-4" />
          Start exploring
        </button>
      )}
    </div>
  )
}

function NoResults({ search }: { readonly search: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-16 text-center">
      <Search className="w-6 h-6 text-text-faint mb-3" />
      <p className="text-sm text-muted">
        No results for &ldquo;{search}&rdquo;
      </p>
    </div>
  )
}

function LoadingSkeleton() {
  return (
    <div className="space-y-4 pt-6">
      {Array.from({ length: 3 }).map((_, i) => (
        <div key={i} className="bg-white rounded-lg border border-border-subtle p-4 animate-pulse">
          <div className="h-4 bg-[#F0F0EE] rounded w-3/4 mb-3" />
          <div className="h-3 bg-[#F0F0EE] rounded w-1/2" />
        </div>
      ))}
    </div>
  )
}

// --- Utilities ---

function formatTimeAgo(isoDate: string): string {
  const seconds = Math.floor((Date.now() - new Date(isoDate).getTime()) / 1000)

  if (seconds < 60) return 'just now'
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 30) return `${days}d ago`
  return new Date(isoDate).toLocaleDateString()
}
