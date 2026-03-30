import { useId, useState } from 'react'
import { motion } from 'framer-motion'
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
    <div className="overflow-hidden rounded-xl border border-border-subtle bg-white">
      <div className="border-b border-border-subtle px-5 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-accent" />
            <h3 className="text-sm font-medium text-text-primary">{block.title}</h3>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-xs text-muted">
            <span>{block.data.length} points</span>
            {block.unit && <span className="font-mono text-[10px]">{block.unit}</span>}
            {categoryColors.size > 0 && [...categoryColors.entries()].map(([category, color]) => (
              <span key={category} className="inline-flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
                {category}
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="px-5 py-4">
        {block.chartType === 'line' ? (
          <LineChart data={block.data} unit={block.unit} />
        ) : (
          <BarChart data={block.data} maxValue={maxValue} unit={block.unit} categoryColors={categoryColors} />
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
  const latestPoint = points[points.length - 1]
  const highestPoint = [...points].sort((left, right) => right.value - left.value)[0]
  const labelIndices = data.length <= 6
    ? new Set(data.map((_, index) => index))
    : new Set([0, Math.floor((data.length - 1) / 2), data.length - 1])

  return (
    <div>
      <div className="rounded-lg border border-border-subtle bg-[#FAFAF8] p-3">
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full" preserveAspectRatio="xMidYMid meet">
          <defs>
            <linearGradient id={gradientId} x1="0%" x2="0%" y1="0%" y2="100%">
              <stop offset="0%" stopColor="#2563EB" stopOpacity="0.12" />
              <stop offset="100%" stopColor="#2563EB" stopOpacity="0.01" />
            </linearGradient>
          </defs>

          {[0, 0.25, 0.5, 0.75, 1].map(frac => {
            const y = padding.top + chartH * (1 - frac)
            return (
              <line key={frac} x1={padding.left} y1={y} x2={width - padding.right} y2={y}
                stroke="#E8E8E6" strokeWidth={0.5} />
            )
          })}

          <line x1={padding.left} y1={baselineY} x2={width - padding.right} y2={baselineY}
            stroke="#CBD5E1" strokeWidth={0.8} />

          {areaD && (
            <motion.path d={areaD} fill={`url(#${gradientId})`}
              initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              transition={{ duration: 0.4, ease: 'easeOut', delay: 0.16 }} />
          )}

          <motion.path d={pathD} fill="none" stroke="#2563EB" strokeWidth={2}
            strokeLinecap="round" strokeLinejoin="round"
            initial={{ pathLength: 0 }} animate={{ pathLength: 1 }}
            transition={{ duration: 0.72, ease: 'easeOut' }} />

          {points.map((p, i) => (
            <g key={`${p.label}-${i}`}>
              <circle cx={p.x} cy={p.y} r={i === points.length - 1 ? 3.5 : 2}
                fill="white" stroke="#2563EB" strokeWidth={1.5} />
              {labelIndices.has(i) && (
                <text x={p.x} y={height - 5} textAnchor="middle"
                  className="fill-muted text-[9px]" fontFamily="var(--font-mono)">
                  {p.label}
                </text>
              )}
            </g>
          ))}
        </svg>
      </div>

      <div className="mt-2 flex gap-4 text-xs text-muted">
        {highestPoint && (
          <span>Peak: <span className="text-text-primary font-medium tabular-nums">{highestPoint.label} {highestPoint.value}{unit ?? ''}</span></span>
        )}
        {latestPoint && (
          <span>Latest: <span className="text-text-primary font-medium tabular-nums">{latestPoint.label} {latestPoint.value}{unit ?? ''}</span></span>
        )}
      </div>
    </div>
  )
}

function BarChart({
  data,
  maxValue,
  unit,
  categoryColors,
}: {
  data: ChartBlockType['data']
  maxValue: number
  unit?: string
  categoryColors: Map<string, string>
}) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)

  return (
    <div className="space-y-2">
      {data.map((d, i) => {
        const barColor = d.category
          ? (categoryColors.get(d.category) ?? BLOCK_COLORS[i % BLOCK_COLORS.length])
          : (d.value >= 0 ? '#2563EB' : '#DC2626')
        const isHovered = hoveredIndex === i
        const isDimmed = hoveredIndex !== null && !isHovered

        return (
          <div
            key={`${d.label}-${i}`}
            onMouseEnter={() => setHoveredIndex(i)}
            onMouseLeave={() => setHoveredIndex(null)}
            className="transition-opacity duration-150"
            style={{ opacity: isDimmed ? 0.4 : 1 }}
          >
            <div className="flex items-baseline justify-between gap-3 mb-1">
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-xs text-text-primary truncate">{d.label}</span>
                {d.category && <span className="text-[10px] text-muted">{d.category}</span>}
              </div>
              <span className="text-xs text-text-primary tabular-nums font-medium shrink-0">
                {d.value}{unit ?? ''}
              </span>
            </div>
            <div className="relative h-2 overflow-hidden rounded-full bg-surface-active">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${(Math.abs(d.value) / maxValue) * 100}%` }}
                transition={{ ...SPRING, delay: i * 0.04 }}
                className="absolute inset-y-0 left-0 rounded-full transition-shadow duration-150"
                style={{
                  backgroundColor: barColor,
                  boxShadow: isHovered ? `0 0 8px ${barColor}40` : 'none',
                }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}
