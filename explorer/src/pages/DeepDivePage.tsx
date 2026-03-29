import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, BookOpen } from 'lucide-react'
import { BlockCanvas } from '../components/explore/BlockCanvas'
import { SPRING } from '../lib/theme'
import { PAPER_SECTIONS, type PaperSection } from '../data/paper-sections'

function summarizeSection(section: PaperSection): string[] {
  const tags: string[] = []
  const blockTypes = new Set(section.blocks.map(block => block.type))
  if (blockTypes.has('chart') || blockTypes.has('timeseries')) tags.push('charts')
  if (blockTypes.has('table')) tags.push('tables')
  if (blockTypes.has('comparison')) tags.push('comparisons')
  if (section.blocks.some(block => block.type === 'insight' && block.emphasis === 'surprising')) {
    tags.push('surprising result')
  }
  if (section.blocks.some(block => block.type === 'caveat')) tags.push('caveat')
  return tags.slice(0, 3)
}

export function DeepDivePage() {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())

  const toggle = (id: string) => {
    setExpandedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const expandAll = () => {
    setExpandedIds(new Set(PAPER_SECTIONS.map(section => section.id)))
  }

  const collapseAll = () => {
    setExpandedIds(new Set())
  }

  return (
    <div>
      <div className="mb-6 overflow-hidden rounded-2xl border border-border-subtle bg-surface/80 shadow-[0_24px_80px_rgba(0,0,0,0.18)]">
        <div className="border-b border-border-subtle bg-[radial-gradient(circle_at_top_left,rgba(59,130,246,0.10),transparent_42%),radial-gradient(circle_at_top_right,rgba(217,119,87,0.08),transparent_35%)] px-4 py-4 sm:px-5">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-2xl">
              <div className="flex items-center gap-2">
                <BookOpen className="h-4 w-4 text-accent" />
                <span className="text-xs font-medium uppercase tracking-wider text-muted">
                  Paper deep dive
                </span>
              </div>
              <h1 className="mt-2 text-xl font-medium text-text-primary sm:text-2xl">
                Walk the paper section by section.
              </h1>
              <p className="mt-2 text-sm leading-relaxed text-muted">
                Each accordion mirrors a section of the study and preserves the same blocks, metrics, and caveats used elsewhere in the explorer.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={expandAll}
                className="rounded-full border border-border-subtle bg-surface/70 px-3 py-1.5 text-[10px] uppercase tracking-[0.18em] text-muted transition-colors hover:border-white/10 hover:text-text-primary"
              >
                Expand all
              </button>
              <button
                onClick={collapseAll}
                className="rounded-full border border-border-subtle bg-surface/70 px-3 py-1.5 text-[10px] uppercase tracking-[0.18em] text-muted transition-colors hover:border-white/10 hover:text-text-primary"
              >
                Collapse all
              </button>
            </div>
          </div>
        </div>

        <div className="grid gap-px bg-border-subtle sm:grid-cols-3">
          <div className="bg-surface px-4 py-3 sm:px-5">
            <div className="text-[10px] uppercase tracking-[0.18em] text-muted">Sections</div>
            <div className="mt-1 text-sm text-text-primary">{PAPER_SECTIONS.length} paper checkpoints</div>
          </div>
          <div className="bg-surface px-4 py-3 sm:px-5">
            <div className="text-[10px] uppercase tracking-[0.18em] text-muted">Expanded</div>
            <div className="mt-1 text-sm text-text-primary">{expandedIds.size} currently open</div>
          </div>
          <div className="bg-surface px-4 py-3 sm:px-5">
            <div className="text-[10px] uppercase tracking-[0.18em] text-muted">Coverage</div>
            <div className="mt-1 text-sm text-text-primary">Model, experiments, results, and caveats</div>
          </div>
        </div>
      </div>

      <div className="space-y-2">
        {PAPER_SECTIONS.map(section => {
          const isExpanded = expandedIds.has(section.id)
          const summaryTags = summarizeSection(section)
          return (
            <motion.div
              key={section.id}
              layout
              whileHover={{ y: -2 }}
              className="overflow-hidden rounded-xl border border-border-subtle bg-surface/70 shadow-[0_14px_40px_rgba(0,0,0,0.12)]"
            >
              <button
                onClick={() => toggle(section.id)}
                className="w-full px-4 py-4 text-left transition-colors hover:bg-white/[0.02]"
              >
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 w-8 shrink-0 text-[10px] font-mono text-accent">
                    {section.number}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="min-w-0">
                        <h3 className="truncate text-sm font-medium text-text-primary">
                          {section.title}
                        </h3>
                        <p className="mt-1 text-[11px] text-muted">
                          {section.description}
                        </p>
                      </div>

                      <div className="flex items-center gap-2">
                        <span className="rounded-full border border-border-subtle bg-black/10 px-2 py-1 text-[10px] uppercase tracking-[0.18em] text-muted">
                          {section.blocks.length} blocks
                        </span>
                        <motion.div
                          animate={{ rotate: isExpanded ? 180 : 0 }}
                          transition={SPRING}
                        >
                          <ChevronDown className="h-4 w-4 shrink-0 text-muted/50" />
                        </motion.div>
                      </div>
                    </div>

                    {summaryTags.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {summaryTags.map(tag => (
                          <span
                            key={`${section.id}-${tag}`}
                            className="rounded-full border border-white/8 bg-white/[0.03] px-2 py-0.5 text-[10px] text-muted"
                          >
                            {tag}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </button>

              <AnimatePresence>
                {isExpanded && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={SPRING}
                    className="overflow-hidden"
                  >
                    <div className="border-t border-border-subtle/50 px-4 pb-4 pt-3">
                      <div className="mb-4 rounded-xl border border-white/6 bg-black/10 px-3 py-3 text-[11px] text-muted">
                        <span className="font-medium text-text-primary">Section focus:</span> {section.description}
                      </div>
                      <BlockCanvas blocks={section.blocks} />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}
