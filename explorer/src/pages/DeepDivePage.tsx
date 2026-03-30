import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown } from 'lucide-react'
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
  const [expandedIds, setExpandedIds] = useState<Set<string>>(
    () => new Set(PAPER_SECTIONS.length > 0 ? [PAPER_SECTIONS[0].id] : []),
  )

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
      {/* Page header */}
      <div className="mb-8">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
          <div className="max-w-2xl">
            <div className="flex items-center gap-2 mb-2">
              <span className="w-2 h-2 rounded-full bg-accent" />
              <span className="text-xs text-muted">Paper deep dive</span>
            </div>
            <h1 className="text-xl font-semibold text-text-primary">
              Walk the paper section by section.
            </h1>
            <p className="mt-2 text-sm text-muted">
              Each accordion mirrors a section of the study and preserves the same blocks, metrics, and caveats used elsewhere in the explorer.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={expandAll}
              className="rounded-md border border-border-subtle px-3 py-1.5 text-xs text-muted transition-colors hover:border-border-hover hover:text-text-primary"
            >
              Expand all
            </button>
            <button
              onClick={collapseAll}
              className="rounded-md border border-border-subtle px-3 py-1.5 text-xs text-muted transition-colors hover:border-border-hover hover:text-text-primary"
            >
              Collapse all
            </button>
          </div>
        </div>

        <div className="grid gap-4 sm:grid-cols-3 border-t border-border-subtle mt-4 pt-4 text-xs text-muted">
          <div>
            <div className="mb-1">Sections</div>
            <div className="text-sm text-text-primary">{PAPER_SECTIONS.length} paper checkpoints</div>
          </div>
          <div>
            <div className="mb-1">Expanded</div>
            <div className="text-sm text-text-primary">{expandedIds.size} currently open</div>
          </div>
          <div>
            <div className="mb-1">Coverage</div>
            <div className="text-sm text-text-primary">Model, experiments, results, and caveats</div>
          </div>
        </div>
      </div>

      {/* Accordion sections */}
      <div className="space-y-2">
        {PAPER_SECTIONS.map(section => {
          const isExpanded = expandedIds.has(section.id)
          const summaryTags = summarizeSection(section)
          return (
            <motion.div
              key={section.id}
              layout
              whileHover={{ y: -1 }}
              className="overflow-hidden rounded-lg border border-border-subtle bg-white"
            >
              <button
                onClick={() => toggle(section.id)}
                className="w-full px-4 py-4 text-left transition-colors hover:bg-surface-active"
              >
                <div className="flex items-start gap-3">
                  <span className="mt-0.5 w-8 shrink-0 text-xs font-mono text-accent">
                    {section.number}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="min-w-0">
                        <h3 className="truncate text-sm font-medium text-text-primary">
                          {section.title}
                        </h3>
                        <p className="mt-1 text-xs text-muted">
                          {section.description}
                        </p>
                      </div>

                      <div className="flex items-center gap-2">
                        <span className="text-xs text-muted">
                          {section.blocks.length} blocks
                        </span>
                        <motion.div
                          animate={{ rotate: isExpanded ? 180 : 0 }}
                          transition={SPRING}
                        >
                          <ChevronDown className="h-4 w-4 shrink-0 text-text-faint" />
                        </motion.div>
                      </div>
                    </div>

                    {summaryTags.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-2">
                        {summaryTags.map(tag => (
                          <span
                            key={`${section.id}-${tag}`}
                            className="text-xs text-muted"
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
                    <div className="border-t border-border-subtle px-4 pb-4 pt-3">
                      <div className="mb-4 rounded-md border border-border-subtle bg-[#FAFAF8] px-3 py-3 text-xs text-muted">
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
