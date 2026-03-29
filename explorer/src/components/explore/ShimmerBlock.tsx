import { motion } from 'framer-motion'
import { SPRING } from '../../lib/theme'

/** Skeleton placeholder blocks shown while Claude generates a response */
export function ShimmerLoading() {
  return (
    <div className="space-y-3">
      {/* Stat row skeleton */}
      <div className="grid grid-cols-3 gap-3">
        {[0, 1, 2].map(i => (
          <motion.div
            key={`stat-${i}`}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ ...SPRING, delay: i * 0.06 }}
            className="glass-1 rounded-lg p-4"
          >
            <div className="shimmer h-8 w-16 rounded mb-2" />
            <div className="shimmer h-3 w-24 rounded mb-1" />
            <div className="shimmer h-2 w-20 rounded" />
          </motion.div>
        ))}
      </div>

      {/* Insight skeleton */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...SPRING, delay: 0.18 }}
        className="glass-1 rounded-lg p-4 border-l-2 border-accent/30"
      >
        <div className="shimmer h-4 w-48 rounded mb-3" />
        <div className="shimmer h-3 w-full rounded mb-2" />
        <div className="shimmer h-3 w-4/5 rounded mb-2" />
        <div className="shimmer h-3 w-3/5 rounded" />
      </motion.div>

      {/* Chart/comparison skeleton */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ ...SPRING, delay: 0.24 }}
        className="glass-1 rounded-lg p-4"
      >
        <div className="shimmer h-4 w-40 rounded mb-4" />
        <div className="flex gap-4">
          <div className="flex-1 space-y-2">
            {[0, 1, 2].map(i => (
              <div key={i} className="shimmer h-6 rounded" style={{ width: `${80 - i * 15}%` }} />
            ))}
          </div>
          <div className="flex-1 space-y-2">
            {[0, 1, 2].map(i => (
              <div key={i} className="shimmer h-6 rounded" style={{ width: `${70 - i * 10}%` }} />
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  )
}
