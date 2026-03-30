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
  const observationCount = block.data.length
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
                ? 'Exact trend view with restrained chrome and instrument-style reference framing.'
                : 'Exact values rendered as proportional bars with the dominant measurement surfaced first.'}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="lab-chip">{observationCount} observations</span>
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
                <div key={`${d.label}-${i}`} className="rounded-2xl border border-border-subtle bg-[linear-gradient(180deg,rgba(255,255,255,0.97),rgba(246,245,241,0.86))] p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.75)]">
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
  const labelIndices = data.length <= 6
    ? new Set(data.map((_, index) => index))
    : new Set([0, Math.floor((data.length - 1) / 2), data.length - 1])

  return (
    <div>
      <div className="rounded-2xl border border-border-subtle bg-[radial-gradient(circle_at_15%_0%,rgba(59,130,246,0.1),transparent_32%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(243,242,238,0.86))] p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.82)]">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="text-[11px] uppercase tracking-[0.18em] text-text-faint">
            Measurement rail
          </div>
          <div className="text-[11px] text-muted">
            {data.length} points{unit ? ` · ${unit}` : ''}
          </div>
        </div>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full" preserveAspectRatio="xMidYMid meet">
          <defs>
            <linearGradient id={gradientId} x1="0%" x2="0%" y1="0%" y2="100%">
              <stop offset="0%" stopColor="#3B82F6" stopOpacity="0.16" />
              <stop offset="100%" stopColor="#3B82F6" stopOpacity="0.015" />
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
            stroke="#BFC5CC"
            strokeWidth={0.8}
          />

          {areaD && (
            <motion.path
              d={areaD}
              fill={`url(#${gradientId})`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4, ease: 'easeOut', delay: 0.16 }}
            />
          )}

          <motion.path
            d={pathD}
            fill="none"
            stroke="#2563EB"
            strokeWidth={2.2}
            strokeLinecap="round"
            strokeLinejoin="round"
            initial={{ pathLength: 0 }}
            animate={{ pathLength: 1 }}
            transition={{ duration: 0.72, ease: 'easeOut' }}
          />

          {points.map((p, i) => (
            <g key={`${p.label}-${i}`}>
              <circle cx={p.x} cy={p.y} r={i === points.length - 1 ? 4 : 2.5} fill="white" stroke="#2563EB" strokeWidth={1.5} />
              {labelIndices.has(i) && (
                <text x={p.x} y={height - 5} textAnchor="middle" className="fill-[#6B7280] text-[9px]">
                  {p.label}
                </text>
              )}
            </g>
          ))}
        </svg>
      </div>

      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        {highestPoint && (
          <div className="rounded-2xl border border-border-subtle bg-[linear-gradient(180deg,rgba(255,255,255,0.97),rgba(246,245,241,0.86))] px-3 py-2.5 text-xs shadow-[inset_0_1px_0_rgba(255,255,255,0.75)]">
            <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Highest point</div>
            <div className="mt-1 font-medium tabular-nums text-text-primary">
              {highestPoint.label}: {highestPoint.value}{unit ?? ''}
            </div>
          </div>
        )}
        {latestPoint && (
          <div className="rounded-2xl border border-accent/20 bg-[#F8FAFF] px-3 py-2.5 text-xs shadow-[inset_0_1px_0_rgba(255,255,255,0.75)]">
            <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Latest value</div>
            <div className="mt-1 font-medium tabular-nums text-text-primary">
              {latestPoint.label}: {latestPoint.value}{unit ?? ''}
            </div>
          </div>
        )}
        <div className={cn(
          'rounded-2xl border border-border-subtle bg-[linear-gradient(180deg,rgba(255,255,255,0.97),rgba(246,245,241,0.86))] px-3 py-2.5 text-xs shadow-[inset_0_1px_0_rgba(255,255,255,0.75)]',
        )}>
          <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Observed range</div>
          <div className="mt-1 font-medium tabular-nums text-text-primary">
            {minY}{unit ?? ''} to {maxY}{unit ?? ''}
          </div>
        </div>
      </div>
    </div>
  )
}
