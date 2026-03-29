import { motion } from 'framer-motion'
import type { Block } from '../../types/blocks'
import { BlockRenderer } from '../blocks/BlockRenderer'
import { SPRING } from '../../lib/theme'

interface BlockCanvasProps {
  blocks: readonly Block[]
}

export function BlockCanvas({ blocks }: BlockCanvasProps) {
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
    <motion.div
      initial="hidden"
      animate="visible"
      variants={{
        hidden: {},
        visible: { transition: { staggerChildren: 0.06 } },
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
                  hidden: { opacity: 0, y: 12 },
                  visible: { opacity: 1, y: 0, transition: SPRING },
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
                hidden: { opacity: 0, y: 12 },
                visible: { opacity: 1, y: 0, transition: SPRING },
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
              hidden: { opacity: 0, y: 12 },
              visible: { opacity: 1, y: 0, transition: SPRING },
            }}
          >
            <BlockRenderer block={block} />
          </motion.div>
        ))
      )}
    </motion.div>
  )
}
