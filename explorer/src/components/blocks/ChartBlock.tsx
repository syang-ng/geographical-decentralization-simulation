import { motion } from 'framer-motion'
import { cn } from '../../lib/cn'
import { SPRING } from '../../lib/theme'
import type { ChartBlock as ChartBlockType } from '../../types/blocks'

interface ChartBlockProps {
  block: ChartBlockType
}

export function ChartBlock({ block }: ChartBlockProps) {
  const maxValue = Math.max(...block.data.map(d => Math.abs(d.value)), 1)

  return (
    <div className="bg-surface border border-border-subtle rounded-xl p-5">
      <h3 className="text-sm font-medium text-text-primary mb-4">
        {block.title}
      </h3>

      {block.chartType === 'line' ? (
        <LineChart data={block.data} unit={block.unit} />
      ) : (
        <div className="space-y-2.5">
          {block.data.map((d, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="text-xs text-muted w-28 text-right shrink-0 truncate">
                {d.label}
              </span>
              <div className="flex-1 h-6 bg-white/5 rounded-sm overflow-hidden">
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${(Math.abs(d.value) / maxValue) * 100}%` }}
                  transition={{ ...SPRING, delay: i * 0.05 }}
                  className={cn(
                    'h-full rounded-sm',
                    d.value >= 0 ? 'bg-accent/40' : 'bg-danger/40',
                  )}
                />
              </div>
              <span className="text-xs text-text-primary tabular-nums w-16 text-right shrink-0">
                {d.value}{block.unit ?? ''}
              </span>
            </div>
          ))}
        </div>
      )}

      {block.unit && block.chartType !== 'line' && (
        <div className="text-[10px] text-muted/60 text-right mt-2">
          unit: {block.unit}
        </div>
      )}
    </div>
  )
}

function LineChart({ data, unit }: { data: ChartBlockType['data']; unit?: string }) {
  const padding = { top: 10, right: 40, bottom: 30, left: 10 }
  const width = 500
  const height = 160
  const chartW = width - padding.left - padding.right
  const chartH = height - padding.top - padding.bottom

  const maxY = Math.max(...data.map(d => d.value), 1)
  const minY = Math.min(...data.map(d => d.value), 0)
  const rangeY = maxY - minY || 1

  const points = data.map((d, i) => ({
    x: padding.left + (i / Math.max(data.length - 1, 1)) * chartW,
    y: padding.top + chartH - ((d.value - minY) / rangeY) * chartH,
    label: d.label,
    value: d.value,
  }))

  const pathD = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`)
    .join(' ')

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" preserveAspectRatio="xMidYMid meet">
      {[0, 0.25, 0.5, 0.75, 1].map(frac => {
        const y = padding.top + chartH * (1 - frac)
        return (
          <line key={frac} x1={padding.left} y1={y} x2={width - padding.right} y2={y}
            stroke="#222222" strokeWidth={0.5} />
        )
      })}

      <motion.path
        d={pathD}
        fill="none"
        stroke="#3B82F6"
        strokeWidth={2}
        initial={{ pathLength: 0 }}
        animate={{ pathLength: 1 }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
      />

      {points.map((p, i) => (
        <g key={i}>
          <circle cx={p.x} cy={p.y} r={3} fill="#3B82F6" />
          <text x={p.x} y={height - 5} textAnchor="middle" className="fill-muted text-[9px]">
            {p.label}
          </text>
          <text x={p.x + 6} y={p.y - 6} className="fill-text-primary text-[9px]">
            {p.value}{unit ?? ''}
          </text>
        </g>
      ))}
    </svg>
  )
}
