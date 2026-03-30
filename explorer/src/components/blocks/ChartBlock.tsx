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
  const minValue = Math.min(...block.data.map(d => d.value))
  const topDatum = [...block.data].sort((left, right) => Math.abs(right.value) - Math.abs(left.value))[0]
  const categoryColors = new Map<string, string>()
  let colorIndex = 0

  for (const datum of block.data) {
    if (!datum.category || categoryColors.has(datum.category)) continue
    categoryColors.set(datum.category, BLOCK_COLORS[colorIndex % BLOCK_COLORS.length])
    colorIndex += 1
  }

  return (
    <div className="lab-panel overflow-hidden rounded-xl">
      <div className="border-b border-border-subtle px-5 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium text-text-primary">
              {block.title}
            </h3>
            <p className="mt-1 text-xs text-muted">
              {block.chartType === 'line'
                ? 'Trend view across the selected values.'
                : 'Exact values rendered as proportional bars.'}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {topDatum && (
              <span className="lab-chip">
                peak {topDatum.label}: {topDatum.value}{block.unit ?? ''}
              </span>
            )}
            <span className="lab-chip">
              range {minValue} to {maxValue}{block.unit ?? ''}
            </span>
          </div>

          {categoryColors.size > 0 && (
            <div className="flex flex-wrap items-center gap-2">
              {[...categoryColors.entries()].map(([category, color]) => (
                <span
                  key={category}
                  className="inline-flex items-center gap-1.5 text-xs text-muted"
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
                : (d.value >= 0 ? '#3B82F6' : '#EF4444')

              return (
                <div key={`${d.label}-${i}`} className="rounded-xl border border-border-subtle bg-[linear-gradient(180deg,rgba(255,255,255,0.94),rgba(245,245,243,0.8))] p-3">
                  <div className="mb-2 flex items-center justify-between gap-3">
                    <div className="min-w-0">
                      <div className="truncate text-[11px] font-medium uppercase tracking-[0.16em] text-text-faint">
                        Measurement
                      </div>
                      <div className="mt-1 truncate text-xs font-medium text-text-primary">
                        {d.label}
                      </div>
                      {d.category && (
                        <div className="mt-0.5 text-xs text-muted">
                          {d.category}
                        </div>
                      )}
                    </div>
                    <span className="shrink-0 text-xs text-text-primary tabular-nums font-medium">
                      {d.value}{block.unit ?? ''}
                    </span>
                  </div>

                  <div className="relative h-6 overflow-hidden rounded-full border border-border-subtle bg-[#F5F5F3]">
                    <div className="absolute inset-y-0 left-0 w-full bg-[linear-gradient(90deg,rgba(26,26,26,0.03)_1px,transparent_1px)] bg-[length:14%_100%]" />
                    <motion.div
                      initial={{ width: 0, opacity: 0.7 }}
                      animate={{ width: `${(Math.abs(d.value) / maxValue) * 100}%`, opacity: 1 }}
                      transition={{ ...SPRING, delay: i * 0.06 }}
                      className="absolute inset-y-0 left-0 rounded-full shadow-[inset_0_1px_0_rgba(255,255,255,0.35)]"
                      style={{ backgroundColor: barColor }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {block.unit && block.chartType !== 'line' && (
          <div className="mt-3 text-right text-xs text-muted">
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
  const latestPoint = points[points.length - 1]
  const highestPoint = [...points].sort((left, right) => right.value - left.value)[0]

  return (
    <div>
      <div className="rounded-xl border border-border-subtle bg-[radial-gradient(circle_at_top,rgba(59,130,246,0.08),transparent_60%),linear-gradient(180deg,rgba(255,255,255,0.96),rgba(245,245,243,0.82))] p-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id={gradientId} x1="0%" x2="0%" y1="0%" y2="100%">
            <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#3B82F6" stopOpacity="0.02" />
          </linearGradient>
        </defs>

        {/* Thin gray gridlines */}
        {[0, 0.25, 0.5, 0.75, 1].map(frac => {
          const y = padding.top + chartH * (1 - frac)
          return (
            <line
              key={frac}
              x1={padding.left}
              y1={y}
              x2={width - padding.right}
              y2={y}
              stroke="#E8E8E6"
              strokeWidth={0.5}
            />
          )
        })}

        <line
          x1={padding.left}
          y1={baselineY}
          x2={width - padding.right}
          y2={baselineY}
          stroke="#D4D4D2"
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
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />

        {points.map((p, i) => (
          <g key={`${p.label}-${i}`}>
            <circle cx={p.x} cy={p.y} r={3.5} fill="white" stroke="#3B82F6" strokeWidth={1.5} />
            <text x={p.x} y={height - 5} textAnchor="middle" className="fill-[#6B7280] text-[9px]">
              {p.label}
            </text>
          </g>
        ))}
      </svg>
      </div>

      <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {highestPoint && (
          <div className="rounded-xl border border-border-subtle bg-[linear-gradient(180deg,rgba(255,255,255,0.94),rgba(245,245,243,0.8))] px-2.5 py-2 text-xs">
            <div className="text-text-faint">Highest point</div>
            <div className="mt-1 font-medium tabular-nums text-text-primary">
              {highestPoint.label}: {highestPoint.value}{unit ?? ''}
            </div>
          </div>
        )}
        {latestPoint && (
          <div className="rounded-xl border border-accent/20 bg-[#F8FAFF] px-2.5 py-2 text-xs">
            <div className="text-text-faint">Latest value</div>
            <div className="mt-1 font-medium tabular-nums text-text-primary">
              {latestPoint.label}: {latestPoint.value}{unit ?? ''}
            </div>
          </div>
        )}
        {points.map((point, index) => (
          <div
            key={`${point.label}-label-${index}`}
            className={cn(
              'rounded-xl border border-border-subtle bg-[linear-gradient(180deg,rgba(255,255,255,0.94),rgba(245,245,243,0.8))] px-2.5 py-2 text-xs',
              index === points.length - 1 && 'border-accent bg-[#F8FAFF]',
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
