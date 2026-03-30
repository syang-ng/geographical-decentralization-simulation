import { useId, useState } from 'react'
import { motion } from 'framer-motion'
import { BLOCK_COLORS } from '../../lib/theme'
import type { TimeSeriesBlock as TimeSeriesBlockType } from '../../types/blocks'

interface TimeSeriesBlockProps {
  block: TimeSeriesBlockType
}

export function TimeSeriesBlock({ block }: TimeSeriesBlockProps) {
  const [hover, setHover] = useState<{ x: number; svgX: number } | null>(null)
  const gradientBaseId = useId().replace(/:/g, '')

  const padding = { top: 20, right: 60, bottom: 35, left: 45 }
  const svgW = 600
  const svgH = 240
  const chartW = svgW - padding.left - padding.right
  const chartH = svgH - padding.top - padding.bottom

  const allPoints = block.series.flatMap(s => s.data)
  if (allPoints.length === 0) {
    return (
      <div className="bg-white border border-border-subtle rounded-lg p-5">
        <h3 className="text-sm font-medium text-text-primary">{block.title}</h3>
        <p className="mt-3 text-sm text-muted">No time series data to display.</p>
      </div>
    )
  }

  const minX = Math.min(...allPoints.map(p => p.x))
  const maxX = Math.max(...allPoints.map(p => p.x))
  const minY = Math.min(...allPoints.map(p => p.y))
  const maxY = Math.max(...allPoints.map(p => p.y))
  const rangeX = maxX - minX || 1
  const rangeY = maxY - minY || 1

  function toSvg(x: number, y: number) {
    return {
      sx: padding.left + ((x - minX) / rangeX) * chartW,
      sy: padding.top + chartH - ((y - minY) / rangeY) * chartH,
    }
  }

  const yTicks = Array.from({ length: 5 }, (_, i) => minY + (rangeY * i) / 4)
  const xTicks = Array.from({ length: 5 }, (_, i) => Math.round(minX + (rangeX * i) / 4))

  return (
    <div className="bg-white border border-border-subtle rounded-lg overflow-hidden">
      <div className="border-b border-border-subtle px-5 py-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium text-text-primary">
              {block.title}
            </h3>
            <p className="mt-1 text-xs text-muted">
              Exact series values, with annotations preserved from the generated block.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {block.series.map((s, i) => {
              const latest = s.data[s.data.length - 1]
              const color = s.color ?? BLOCK_COLORS[i % BLOCK_COLORS.length]
              return (
                <span
                  key={s.label}
                  className="inline-flex items-center gap-1.5 text-xs text-muted"
                >
                  <span
                    className="h-2 w-2 rounded-full"
                    style={{ backgroundColor: color }}
                  />
                  {s.label}
                  {latest && (
                    <span className="font-medium tabular-nums text-text-primary">
                      {latest.y.toFixed(2)}
                    </span>
                  )}
                </span>
              )
            })}
          </div>
        </div>
      </div>

      <div className="px-5 py-5">
        <svg
          viewBox={`0 0 ${svgW} ${svgH}`}
          className="w-full"
          preserveAspectRatio="xMidYMid meet"
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect()
            const relX = ((e.clientX - rect.left) / rect.width) * svgW
            if (relX >= padding.left && relX <= svgW - padding.right) {
              setHover({ x: minX + ((relX - padding.left) / chartW) * rangeX, svgX: relX })
            }
          }}
          onMouseLeave={() => setHover(null)}
        >
          <defs>
            {block.series.map((s, i) => {
              const color = s.color ?? BLOCK_COLORS[i % BLOCK_COLORS.length]
              return (
                <linearGradient
                  key={s.label}
                  id={`${gradientBaseId}-${i}`}
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

          {/* Thin gray gridlines */}
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
                  {tick.toFixed(2)}
                </text>
              </g>
            )
          })}

          {/* X-axis labels */}
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

          {/* Axis labels */}
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

          {/* Series fills + lines */}
          {block.series.map((s, i) => {
            const color = s.color ?? BLOCK_COLORS[i % BLOCK_COLORS.length]
            const coordinates = s.data.map(point => toSvg(point.x, point.y))
            const pathD = coordinates
              .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.sx} ${point.sy}`)
              .join(' ')
            const baselineY = padding.top + chartH
            const areaD = coordinates.length > 0
              ? `${pathD} L ${coordinates[coordinates.length - 1].sx} ${baselineY} L ${coordinates[0].sx} ${baselineY} Z`
              : ''
            const latest = coordinates[coordinates.length - 1]

            return (
              <g key={s.label}>
                {areaD && (
                  <motion.path
                    d={areaD}
                    fill={`url(#${gradientBaseId}-${i})`}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.45, ease: 'easeOut', delay: i * 0.08 }}
                  />
                )}
                <motion.path
                  d={pathD}
                  fill="none"
                  stroke={color}
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.8, ease: 'easeOut', delay: i * 0.12 }}
                />
                {coordinates.map((point, index) => (
                  <circle
                    key={`${s.label}-${index}`}
                    cx={point.sx}
                    cy={point.sy}
                    r={2}
                    fill={color}
                    opacity={0.85}
                  />
                ))}
                {latest && (
                  <>
                    <circle cx={latest.sx} cy={latest.sy} r={4} fill="white" stroke={color} strokeWidth={1.5} />
                  </>
                )}
              </g>
            )
          })}

          {/* Annotations */}
          {block.annotations?.map(ann => {
            const { sx } = toSvg(ann.x, minY)
            return (
              <g key={ann.x}>
                <line
                  x1={sx}
                  y1={padding.top}
                  x2={sx}
                  y2={padding.top + chartH}
                  stroke="#9CA3AF"
                  strokeWidth={1}
                  strokeDasharray="3 3"
                />
                <text
                  x={sx}
                  y={padding.top - 5}
                  textAnchor="middle"
                  className="fill-[#6B7280] text-[8px]"
                >
                  {ann.label}
                </text>
              </g>
            )
          })}

          {/* Hover crosshair */}
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

        {block.annotations && block.annotations.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {block.annotations.map(annotation => (
              <span
                key={`${annotation.x}-${annotation.label}`}
                className="text-xs text-muted"
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
