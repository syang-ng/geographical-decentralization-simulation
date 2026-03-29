import { useCallback } from 'react'
import { motion } from 'framer-motion'
import { Download } from 'lucide-react'
import type { Block } from '../../types/blocks'
import { BlockRenderer } from '../blocks/BlockRenderer'
import { SPRING } from '../../lib/theme'
import { cn } from '../../lib/cn'

interface BlockCanvasProps {
  readonly blocks: readonly Block[]
  readonly showExport?: boolean
}

export function BlockCanvas({ blocks, showExport = true }: BlockCanvasProps) {
  const handleExport = useCallback(() => {
    const json = JSON.stringify(blocks, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `exploration-blocks-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }, [blocks])
  if (blocks.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-muted">
        No blocks to display
      </div>
    )
  }

  // First 3 blocks are stats — render in a 3-up grid if they're all stat type
  const leadingStats = blocks.slice(0, 3).every(b => b.type === 'stat')

  return (
    <div className="space-y-3">
      {showExport && blocks.length > 0 && (
        <div className="flex justify-end">
          <motion.button
            onClick={handleExport}
            whileHover={{ y: -1, scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[10px] font-medium',
              'text-muted hover:text-text-primary',
              'bg-surface border border-border-subtle hover:border-accent/30',
              'transition-colors',
            )}
          >
            <Download className="w-3 h-3" />
            Export JSON
          </motion.button>
        </div>
      )}

      <motion.div
        initial="hidden"
        animate="visible"
        variants={{
          hidden: {},
          visible: { transition: { staggerChildren: 0.08, delayChildren: 0.04 } },
        }}
        className="space-y-3"
      >
      {leadingStats ? (
        <>
          {/* 3-up stat grid */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {blocks.slice(0, 3).map((block, i) => (
              <motion.div
                key={i}
                variants={{
                  hidden: { opacity: 0, y: 16, filter: 'blur(6px)' },
                  visible: { opacity: 1, y: 0, filter: 'blur(0px)', transition: SPRING },
                }}
              >
                <BlockRenderer block={block} />
              </motion.div>
            ))}
          </div>
          {/* Remaining blocks */}
          {blocks.slice(3).map((block, i) => (
            <motion.div
              key={i + 3}
              variants={{
                hidden: { opacity: 0, y: 16, filter: 'blur(6px)' },
                visible: { opacity: 1, y: 0, filter: 'blur(0px)', transition: SPRING },
              }}
            >
              <BlockRenderer block={block} />
            </motion.div>
          ))}
        </>
      ) : (
        blocks.map((block, i) => (
          <motion.div
            key={i}
            variants={{
              hidden: { opacity: 0, y: 16, filter: 'blur(6px)' },
              visible: { opacity: 1, y: 0, filter: 'blur(0px)', transition: SPRING },
            }}
          >
            <BlockRenderer block={block} />
          </motion.div>
        ))
      )}
      </motion.div>
    </div>
  )
}
