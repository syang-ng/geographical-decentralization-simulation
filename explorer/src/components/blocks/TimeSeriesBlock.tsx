import { useId, useState } from 'react'
import { motion } from 'framer-motion'
import { BLOCK_COLORS } from '../../lib/theme'
import type { TimeSeriesBlock as TimeSeriesBlockType } from '../../types/blocks'

interface TimeSeriesBlockProps {
  block: TimeSeriesBlockType
}

function formatSeriesNumber(value: number): string {
  if (!Number.isFinite(value)) return '0'
  if (Math.abs(value) >= 100) return value.toFixed(0)
  if (Math.abs(value) >= 10) return value.toFixed(1)
  return value.toFixed(2)
}

export function TimeSeriesBlock({ block }: TimeSeriesBlockProps) {
  const [hover, setHover] = useState<{ x: number; svgX: number } | null>(null)
  const gradientBaseId = useId().replace(/:/g, '')

  const padding = { top: 20, right: 60, bottom: 35, left: 45 }
  const svgW = 600
  const svgH = 240
  const chartW = svgW - padding.left - padding.right
  const chartH = svgH - padding.top - padding.bottom

  const allPoints = block.series.flatMap(series => series.data)
  if (allPoints.length === 0) {
    return (
      <div className="lab-panel rounded-xl p-5">
        <h3 className="text-sm font-medium text-text-primary">{block.title}</h3>
        <p className="mt-3 text-sm text-muted">No time series data to display.</p>
      </div>
    )
  }

  const minX = Math.min(...allPoints.map(point => point.x))
  const maxX = Math.max(...allPoints.map(point => point.x))
  const minY = Math.min(...allPoints.map(point => point.y))
  const maxY = Math.max(...allPoints.map(point => point.y))
  const rangeX = maxX - minX || 1
  const rangeY = maxY - minY || 1
  const latestValues = block.series.flatMap(series => {
    const latest = series.data[series.data.length - 1]
    return latest ? [{ label: series.label, value: latest.y }] : []
  })
  const highestValue = Math.max(...allPoints.map(point => point.y))
  const pointStep = Math.max(1, Math.floor(Math.max(...block.series.map(series => series.data.length)) / 18))

  function toSvg(x: number, y: number) {
    return {
      sx: padding.left + ((x - minX) / rangeX) * chartW,
      sy: padding.top + chartH - ((y - minY) / rangeY) * chartH,
    }
  }

  const yTicks = Array.from({ length: 5 }, (_, index) => minY + (rangeY * index) / 4)
  const xTicks = Array.from({ length: 5 }, (_, index) => Math.round(minX + (rangeX * index) / 4))

  return (
    <div className="lab-panel overflow-hidden rounded-xl">
      <div className="border-b border-border-subtle px-5 py-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium text-text-primary">
              {block.title}
            </h3>
            <p className="mt-1 text-xs text-muted">
              Exact series values, rendered with the same slot ordering as the simulation output and framed like a measurement surface.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="lab-chip">
              range {formatSeriesNumber(minY)} to {formatSeriesNumber(maxY)}
            </span>
            <span className="lab-chip">
              peak {formatSeriesNumber(highestValue)}
            </span>
            {block.series.map((series, index) => {
              const latest = series.data[series.data.length - 1]
              const color = series.color ?? BLOCK_COLORS[index % BLOCK_COLORS.length]
              return (
                <span
                  key={series.label}
                  className="lab-chip"
                >
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  {series.label}
                  {latest && (
                    <span className="font-medium tabular-nums text-text-primary">
                      {formatSeriesNumber(latest.y)}
                    </span>
                  )}
                </span>
              )
            })}
          </div>
        </div>
      </div>

      <div className="px-5 py-5">
        <div className="mb-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
          {latestValues.map((entry, index) => (
            <div
              key={`${entry.label}-${index}`}
              className="rounded-2xl border border-border-subtle bg-[linear-gradient(180deg,rgba(255,255,255,0.97),rgba(246,245,241,0.86))] px-3 py-2.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.78)]"
            >
              <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Latest</div>
              <div className="mt-1 text-xs font-medium text-text-primary">{entry.label}</div>
              <div className="mt-1 text-sm font-semibold tabular-nums text-text-primary">
                {formatSeriesNumber(entry.value)}
              </div>
            </div>
          ))}
        </div>

        <div className="rounded-2xl border border-border-subtle bg-[radial-gradient(circle_at_15%_0%,rgba(59,130,246,0.1),transparent_28%),radial-gradient(circle_at_85%_0%,rgba(194,85,58,0.08),transparent_24%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(243,242,238,0.84))] px-3 py-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.84)]">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div className="text-[11px] uppercase tracking-[0.18em] text-text-faint">
              Measurement deck
            </div>
            <div className="text-[11px] text-muted">
              {block.series.length} series · x {formatSeriesNumber(minX)} to {formatSeriesNumber(maxX)}
            </div>
          </div>
          <svg
            viewBox={`0 0 ${svgW} ${svgH}`}
            className="w-full"
            preserveAspectRatio="xMidYMid meet"
            onMouseMove={event => {
              const rect = event.currentTarget.getBoundingClientRect()
              const relX = ((event.clientX - rect.left) / rect.width) * svgW
              if (relX >= padding.left && relX <= svgW - padding.right) {
                setHover({ x: minX + ((relX - padding.left) / chartW) * rangeX, svgX: relX })
              }
            }}
            onMouseLeave={() => setHover(null)}
          >
            <defs>
              {block.series.map((series, index) => {
                const color = series.color ?? BLOCK_COLORS[index % BLOCK_COLORS.length]
                return (
                  <linearGradient
                    key={series.label}
                    id={`${gradientBaseId}-${index}`}
                    x1="0%"
                    x2="0%"
                    y1="0%"
                    y2="100%"
                  >
                    <stop offset="0%" stopColor={color} stopOpacity="0.15" />
                    <stop offset="100%" stopColor={color} stopOpacity="0.02" />
                  </linearGradient>
                )
              })}
            </defs>

            {yTicks.map(tick => {
              const { sy } = toSvg(0, tick)
              return (
                <g key={tick}>
                  <line
                    x1={padding.left}
                    y1={sy}
                    x2={svgW - padding.right}
                    y2={sy}
                    stroke="#E8E8E6"
                    strokeWidth={0.5}
                  />
                  <text
                    x={padding.left - 6}
                    y={sy + 3}
                    textAnchor="end"
                    className="fill-[#6B7280] text-[9px]"
                  >
                    {formatSeriesNumber(tick)}
                  </text>
                </g>
              )
            })}

            {xTicks.map(tick => {
              const { sx } = toSvg(tick, minY)
              return (
                <text
                  key={tick}
                  x={sx}
                  y={svgH - 5}
                  textAnchor="middle"
                  className="fill-[#6B7280] text-[9px]"
                >
                  {tick}
                </text>
              )
            })}

            {block.yLabel && (
              <text
                x={12}
                y={padding.top + chartH / 2}
                textAnchor="middle"
                className="fill-[#6B7280] text-[9px]"
                transform={`rotate(-90, 12, ${padding.top + chartH / 2})`}
              >
                {block.yLabel}
              </text>
            )}

            {block.xLabel && (
              <text
                x={padding.left + chartW / 2}
                y={svgH - 2}
                textAnchor="middle"
                className="fill-[#6B7280] text-[9px]"
              >
                {block.xLabel}
              </text>
            )}

            {block.series.map((series, index) => {
              const color = series.color ?? BLOCK_COLORS[index % BLOCK_COLORS.length]
              const coordinates = series.data.map(point => toSvg(point.x, point.y))
              const pathD = coordinates
                .map((point, pointIndex) => `${pointIndex === 0 ? 'M' : 'L'} ${point.sx} ${point.sy}`)
                .join(' ')
              const baselineY = padding.top + chartH
              const areaD = coordinates.length > 0
                ? `${pathD} L ${coordinates[coordinates.length - 1].sx} ${baselineY} L ${coordinates[0].sx} ${baselineY} Z`
                : ''
              const latest = coordinates[coordinates.length - 1]

              return (
                <g key={series.label}>
                  {areaD && (
                    <motion.path
                      d={areaD}
                      fill={`url(#${gradientBaseId}-${index})`}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ duration: 0.4, ease: 'easeOut', delay: index * 0.05 }}
                    />
                  )}
                  <motion.path
                    d={pathD}
                    fill="none"
                    stroke={color}
                    strokeWidth={2.15}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: 1 }}
                    transition={{ duration: 0.72, ease: 'easeOut', delay: index * 0.08 }}
                  />
                  {coordinates.map((point, pointIndex) => (
                    (pointIndex % pointStep === 0 || pointIndex === coordinates.length - 1) ? (
                    <circle
                      key={`${series.label}-${pointIndex}`}
                      cx={point.sx}
                      cy={point.sy}
                      r={pointIndex === coordinates.length - 1 ? 3.75 : 1.8}
                      fill={color}
                      opacity={pointIndex === coordinates.length - 1 ? 0.95 : 0.65}
                    />
                    ) : null
                  ))}
                  {latest && (
                    <circle cx={latest.sx} cy={latest.sy} r={4} fill="white" stroke={color} strokeWidth={1.5} />
                  )}
                </g>
              )
            })}

            {block.annotations?.map(annotation => {
              const { sx } = toSvg(annotation.x, minY)
              return (
                <g key={annotation.x}>
                  <line
                    x1={sx}
                    y1={padding.top}
                    x2={sx}
                    y2={padding.top + chartH}
                    stroke="#C2553A"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                  />
                  <text
                    x={sx}
                    y={padding.top - 5}
                    textAnchor="middle"
                    className="fill-[#8B5E4D] text-[8px]"
                  >
                    {annotation.label}
                  </text>
                </g>
              )
            })}

            {hover && (
              <line
                x1={hover.svgX}
                y1={padding.top}
                x2={hover.svgX}
                y2={padding.top + chartH}
                stroke="#9CA3AF"
                strokeWidth={0.5}
                strokeDasharray="2 2"
              />
            )}
          </svg>
        </div>

        {block.annotations && block.annotations.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {block.annotations.map(annotation => (
              <span
                key={`${annotation.x}-${annotation.label}`}
                className="lab-chip"
              >
                x={annotation.x}: {annotation.label}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
