import { useId } from 'react'
import { motion } from 'framer-motion'
import { cn } from '../../lib/cn'
import { BLOCK_COLORS, SPRING } from '../../lib/theme'
import type { ChartBlock as ChartBlockType } from '../../types/blocks'

interface ChartBlockProps {
  block: ChartBlockType
}

export function ChartBlock({ block }: ChartBlockProps) {
  const maxValue = Math.max(1, ...block.data.map(d => Math.abs(d.value)))
  const categoryColors = new Map<string, string>()
  let colorIndex = 0

  for (const datum of block.data) {
    if (!datum.category || categoryColors.has(datum.category)) continue
    categoryColors.set(datum.category, BLOCK_COLORS[colorIndex % BLOCK_COLORS.length])
    colorIndex += 1
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-border-subtle bg-surface/95 shadow-[0_20px_60px_rgba(0,0,0,0.24)]">
      <div className="border-b border-border-subtle bg-white/[0.02] px-5 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium text-text-primary">
              {block.title}
            </h3>
            <p className="mt-1 text-[11px] text-muted">
              {block.chartType === 'line'
                ? 'Trend view across the selected values.'
                : 'Exact values rendered as proportional bars.'}
            </p>
          </div>

          {categoryColors.size > 0 && (
            <div className="flex flex-wrap items-center gap-1.5">
              {[...categoryColors.entries()].map(([category, color]) => (
                <span
                  key={category}
                  className="inline-flex items-center gap-1.5 rounded-full border border-white/8 bg-black/15 px-2 py-1 text-[10px] text-muted"
                >
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
                  {category}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="px-5 py-5">
        {block.chartType === 'line' ? (
          <LineChart data={block.data} unit={block.unit} />
        ) : (
          <div className="space-y-3">
            {block.data.map((d, i) => {
              const barColor = d.category
                ? (categoryColors.get(d.category) ?? BLOCK_COLORS[i % BLOCK_COLORS.length])
                : (d.value >= 0 ? 'rgba(59,130,246,0.75)' : 'rgba(244,63,94,0.75)')

              return (
                <div key={`${d.label}-${i}`} className="rounded-xl border border-white/6 bg-black/10 px-3 py-2.5">
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate text-xs font-medium text-text-primary">
                        {d.label}
                      </div>
                      {d.category && (
                        <div className="mt-0.5 text-[10px] uppercase tracking-[0.18em] text-muted/80">
                          {d.category}
                        </div>
                      )}
                    </div>
                    <span className="shrink-0 text-xs text-text-primary tabular-nums">
                      {d.value}{block.unit ?? ''}
                    </span>
                  </div>

                  <div className="relative h-8 overflow-hidden rounded-lg border border-white/5 bg-white/[0.03]">
                    <div className="absolute inset-0 bg-[linear-gradient(90deg,rgba(255,255,255,0.04)_0,rgba(255,255,255,0.04)_1px,transparent_1px,transparent_20%)] opacity-60" />
                    <motion.div
                      initial={{ width: 0, opacity: 0.65 }}
                      animate={{ width: `${(Math.abs(d.value) / maxValue) * 100}%`, opacity: 1 }}
                      transition={{ ...SPRING, delay: i * 0.06 }}
                      className="absolute inset-y-0 left-0 rounded-r-lg"
                      style={{
                        background: `linear-gradient(90deg, ${barColor}, rgba(255,255,255,0.12))`,
                        boxShadow: `0 0 28px ${barColor}33`,
                      }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {block.unit && block.chartType !== 'line' && (
          <div className="mt-3 text-right text-[10px] text-muted/60">
            unit: {block.unit}
          </div>
        )}
      </div>
    </div>
  )
}

function LineChart({ data, unit }: { data: ChartBlockType['data']; unit?: string }) {
  const gradientId = useId()
  const padding = { top: 10, right: 40, bottom: 30, left: 10 }
  const width = 500
  const height = 160
  const chartW = width - padding.left - padding.right
  const chartH = height - padding.top - padding.bottom

  const maxY = Math.max(1, ...data.map(d => d.value))
  const minY = Math.min(0, ...data.map(d => d.value))
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
  const baselineY = padding.top + chartH - ((Math.min(Math.max(0, minY), maxY) - minY) / rangeY) * chartH
  const areaD = points.length > 0
    ? `${pathD} L ${points[points.length - 1].x} ${baselineY} L ${points[0].x} ${baselineY} Z`
    : ''

  return (
    <div className="rounded-xl border border-white/6 bg-black/10 p-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id={gradientId} x1="0%" x2="0%" y1="0%" y2="100%">
            <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#3B82F6" stopOpacity="0.02" />
          </linearGradient>
        </defs>

        {[0, 0.25, 0.5, 0.75, 1].map(frac => {
          const y = padding.top + chartH * (1 - frac)
          return (
            <line
              key={frac}
              x1={padding.left}
              y1={y}
              x2={width - padding.right}
              y2={y}
              stroke="#222222"
              strokeWidth={0.5}
            />
          )
        })}

        <line
          x1={padding.left}
          y1={baselineY}
          x2={width - padding.right}
          y2={baselineY}
          stroke="#ffffff22"
          strokeWidth={0.75}
        />

        {areaD && (
          <motion.path
            d={areaD}
            fill={`url(#${gradientId})`}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.45, ease: 'easeOut', delay: 0.2 }}
          />
        )}

        <motion.path
          d={pathD}
          fill="none"
          stroke="#3B82F6"
          strokeWidth={2.5}
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          style={{ filter: 'drop-shadow(0 0 10px rgba(59,130,246,0.32))' }}
        />

        {points.map((p, i) => (
          <g key={`${p.label}-${i}`}>
            <circle cx={p.x} cy={p.y} r={5} fill="#0b1020" />
            <circle cx={p.x} cy={p.y} r={3} fill="#3B82F6" />
            <text x={p.x} y={height - 5} textAnchor="middle" className="fill-muted text-[9px]">
              {p.label}
            </text>
          </g>
        ))}
      </svg>

      <div className="mt-3 grid gap-2 sm:grid-cols-2">
        {points.map((point, index) => (
          <div
            key={`${point.label}-label-${index}`}
            className={cn(
              'rounded-lg border border-white/6 bg-white/[0.02] px-2.5 py-2 text-[11px]',
              index === points.length - 1 && 'border-accent/20 bg-accent/[0.06]',
            )}
          >
            <div className="text-muted">{point.label}</div>
            <div className="mt-1 font-medium tabular-nums text-text-primary">
              {point.value}{unit ?? ''}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
