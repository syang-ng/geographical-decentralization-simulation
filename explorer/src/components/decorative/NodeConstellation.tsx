import { motion } from 'framer-motion'
import { SPRING_SOFT } from '../../lib/theme'

interface Node {
  readonly x: number
  readonly y: number
  readonly r: number
  readonly color: string
  readonly opacity: number
}

const NODES: readonly Node[] = [
  { x: 15, y: 20, r: 2, color: 'var(--color-accent)', opacity: 0.4 },
  { x: 45, y: 12, r: 2.5, color: 'var(--color-accent)', opacity: 0.5 },
  { x: 75, y: 28, r: 1.5, color: 'var(--color-accent-warm)', opacity: 0.35 },
  { x: 30, y: 55, r: 2, color: 'var(--color-success)', opacity: 0.3 },
  { x: 60, y: 48, r: 1.5, color: 'var(--color-accent)', opacity: 0.25 },
  { x: 85, y: 60, r: 2, color: 'var(--color-accent-warm)', opacity: 0.4 },
  { x: 20, y: 80, r: 1.5, color: 'var(--color-accent)', opacity: 0.3 },
  { x: 55, y: 78, r: 2, color: 'var(--color-accent)', opacity: 0.35 },
  { x: 90, y: 85, r: 1.5, color: 'var(--color-success)', opacity: 0.25 },
]

const EDGES: readonly [number, number][] = [
  [0, 1], [1, 2], [1, 4], [0, 3], [3, 4], [4, 5], [3, 6], [6, 7], [7, 8], [4, 7], [5, 8],
]

/** Faint dot-and-line network — evokes validators spread across geography */
export function NodeConstellation({ className = '' }: { readonly className?: string }) {
  return (
    <motion.svg
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ ...SPRING_SOFT, delay: 0.5 }}
      className={className}
      viewBox="0 0 100 100"
      fill="none"
      aria-hidden="true"
      preserveAspectRatio="none"
    >
      {EDGES.map(([a, b]) => (
        <line
          key={`${a}-${b}`}
          x1={NODES[a].x}
          y1={NODES[a].y}
          x2={NODES[b].x}
          y2={NODES[b].y}
          stroke="var(--color-meridian)"
          strokeWidth="0.4"
          opacity="0.3"
        />
      ))}
      {NODES.map((node, i) => (
        <circle
          key={i}
          cx={node.x}
          cy={node.y}
          r={node.r}
          fill={node.color}
          opacity={node.opacity}
        />
      ))}
    </motion.svg>
  )
}
