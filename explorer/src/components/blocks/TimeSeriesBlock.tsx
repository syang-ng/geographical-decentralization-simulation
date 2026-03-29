import { useState } from 'react'
import { motion } from 'framer-motion'
import { BLOCK_COLORS } from '../../lib/theme'
import type { TimeSeriesBlock as TimeSeriesBlockType } from '../../types/blocks'

interface TimeSeriesBlockProps {
  block: TimeSeriesBlockType
}

export function TimeSeriesBlock({ block }: TimeSeriesBlockProps) {
  const [hover, setHover] = useState<{ x: number; svgX: number } | null>(null)

  const padding = { top: 20, right: 60, bottom: 35, left: 45 }
  const svgW = 600
  const svgH = 240
  const chartW = svgW - padding.left - padding.right
  const chartH = svgH - padding.top - padding.bottom

  // Compute axis bounds across all series
  const allPoints = block.series.flatMap(s => s.data)
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

  // Y-axis ticks
  const yTicks = Array.from({ length: 5 }, (_, i) => minY + (rangeY * i) / 4)

  // X-axis ticks
  const xTicks = Array.from({ length: 5 }, (_, i) => Math.round(minX + (rangeX * i) / 4))

  return (
    <div className="bg-surface border border-border-subtle rounded-xl p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-text-primary">
          {block.title}
        </h3>
        {/* Legend */}
        <div className="flex items-center gap-3">
          {block.series.map((s, i) => (
            <span key={s.label} className="flex items-center gap-1.5 text-[10px] text-muted">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: s.color ?? BLOCK_COLORS[i % BLOCK_COLORS.length] }}
              />
              {s.label}
            </span>
          ))}
        </div>
      </div>

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
        {/* Grid lines */}
        {yTicks.map(tick => {
          const { sy } = toSvg(0, tick)
          return (
            <g key={tick}>
              <line x1={padding.left} y1={sy} x2={svgW - padding.right} y2={sy}
                stroke="#222222" strokeWidth={0.5} />
              <text x={padding.left - 6} y={sy + 3} textAnchor="end"
                className="fill-muted text-[9px]">
                {tick.toFixed(2)}
              </text>
            </g>
          )
        })}

        {/* X-axis labels */}
        {xTicks.map(tick => {
          const { sx } = toSvg(tick, 0)
          return (
            <text key={tick} x={sx} y={svgH - 5} textAnchor="middle"
              className="fill-muted text-[9px]">
              {tick}
            </text>
          )
        })}

        {/* Axis labels */}
        {block.yLabel && (
          <text x={12} y={padding.top + chartH / 2} textAnchor="middle"
            className="fill-muted text-[9px]"
            transform={`rotate(-90, 12, ${padding.top + chartH / 2})`}>
            {block.yLabel}
          </text>
        )}
        {block.xLabel && (
          <text x={padding.left + chartW / 2} y={svgH - 2} textAnchor="middle"
            className="fill-muted text-[9px]">
            {block.xLabel}
          </text>
        )}

        {/* Series lines */}
        {block.series.map((s, i) => {
          const color = s.color ?? BLOCK_COLORS[i % BLOCK_COLORS.length]
          const pathD = s.data
            .map((p, j) => {
              const { sx, sy } = toSvg(p.x, p.y)
              return `${j === 0 ? 'M' : 'L'} ${sx} ${sy}`
            })
            .join(' ')

          return (
            <motion.path
              key={s.label}
              d={pathD}
              fill="none"
              stroke={color}
              strokeWidth={2}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.8, ease: 'easeOut', delay: i * 0.15 }}
            />
          )
        })}

        {/* Annotations */}
        {block.annotations?.map(ann => {
          const { sx } = toSvg(ann.x, 0)
          return (
            <g key={ann.x}>
              <line x1={sx} y1={padding.top} x2={sx} y2={padding.top + chartH}
                stroke="#87867f" strokeWidth={1} strokeDasharray="3 3" />
              <text x={sx} y={padding.top - 5} textAnchor="middle"
                className="fill-muted text-[8px]">
                {ann.label}
              </text>
            </g>
          )
        })}

        {/* Hover crosshair */}
        {hover && (
          <line x1={hover.svgX} y1={padding.top} x2={hover.svgX} y2={padding.top + chartH}
            stroke="#87867f" strokeWidth={0.5} strokeDasharray="2 2" />
        )}
      </svg>
    </div>
  )
}
