import { useCallback } from 'react'
import { motion } from 'framer-motion'
import { Download } from 'lucide-react'
import type { Block } from '../../types/blocks'
import { BlockRenderer } from '../blocks/BlockRenderer'
import { STAGGER_CONTAINER, STAGGER_ITEM } from '../../lib/theme'

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
          <button
            onClick={handleExport}
            className="flex items-center gap-1 px-2 py-1 rounded text-[9px] text-muted/50 hover:text-muted transition-colors"
            title="Export block data as JSON"
          >
            <Download className="w-2.5 h-2.5" />
            JSON
          </button>
        </div>
      )}

      <motion.div
        initial="hidden"
        animate="visible"
        variants={STAGGER_CONTAINER}
        className="space-y-3"
      >
      {leadingStats ? (
        <>
          {/* 3-up stat grid */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {blocks.slice(0, 3).map((block, i) => (
              <motion.div
                key={i}
                variants={STAGGER_ITEM}
              >
                <BlockRenderer block={block} />
              </motion.div>
            ))}
          </div>
          {/* Remaining blocks */}
          {blocks.slice(3).map((block, i) => (
            <motion.div
              key={i + 3}
              variants={STAGGER_ITEM}
            >
              <BlockRenderer block={block} />
            </motion.div>
          ))}
        </>
      ) : (
        blocks.map((block, i) => (
          <motion.div
            key={i}
            variants={STAGGER_ITEM}
          >
            <BlockRenderer block={block} />
          </motion.div>
        ))
      )}
      </motion.div>
    </div>
  )
}
