import { useState } from 'react'
import { motion } from 'framer-motion'
import { SPRING } from '../../lib/theme'
import type { MapBlock as MapBlockType } from '../../types/blocks'

interface MapBlockProps {
  block: MapBlockType
}

function latLonToMercator(lat: number, lon: number, width: number, height: number) {
  const x = ((lon + 180) / 360) * width
  const latRad = (lat * Math.PI) / 180
  const mercN = Math.log(Math.tan(Math.PI / 4 + latRad / 2))
  const y = height / 2 - (mercN / Math.PI) * (height / 2)
  return { x, y }
}

function getDotRadius(value: number, maxValue: number): number {
  const normalized = Math.max(value / Math.max(maxValue, 1), 0.1)
  return 3 + normalized * 9 // 3px min, 12px max
}

function getDotColor(value: number, maxValue: number, colorScale?: string): string {
  if (colorScale === 'binary') return value > 0 ? '#2dd4bf' : '#222222'
  if (colorScale === 'change') {
    if (value > 0) return '#2dd4bf'
    if (value < 0) return '#f43f5e'
    return '#87867f'
  }
  // density: green → yellow → red
  const t = Math.min(value / Math.max(maxValue, 1), 1)
  if (t < 0.33) return '#2dd4bf'
  if (t < 0.66) return '#fbbf24'
  return '#f43f5e'
}

export function MapBlock({ block }: MapBlockProps) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string; value: number } | null>(null)
  const maxValue = Math.max(...block.regions.map(r => r.value), 1)

  const svgW = 800
  const svgH = 420

  return (
    <div className="bg-surface border border-border-subtle rounded-xl p-5">
      <h3 className="text-sm font-medium text-text-primary mb-3">
        {block.title}
      </h3>

      <div className="relative overflow-hidden rounded-lg bg-[#0a0a0a]" style={{ minHeight: 280 }}>
        <svg
          viewBox={`0 0 ${svgW} ${svgH}`}
          className="w-full"
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={block.title}
        >
          {/* Simplified world outline — just a bounding frame + equator */}
          <rect x={0} y={0} width={svgW} height={svgH} fill="none" stroke="#1a1a1a" strokeWidth={1} />
          <line x1={0} y1={svgH / 2} x2={svgW} y2={svgH / 2} stroke="#1a1a1a" strokeWidth={0.5} strokeDasharray="4 4" />
          <line x1={svgW / 2} y1={0} x2={svgW / 2} y2={svgH} stroke="#1a1a1a" strokeWidth={0.5} strokeDasharray="4 4" />

          {/* Region dots */}
          {block.regions.map((region, i) => {
            const { x, y } = latLonToMercator(region.lat, region.lon, svgW, svgH)
            const r = getDotRadius(region.value, maxValue)
            const color = getDotColor(region.value, maxValue, block.colorScale)

            return (
              <motion.circle
                key={region.name}
                cx={x}
                cy={y}
                r={r}
                fill={color}
                fillOpacity={0.7}
                stroke={color}
                strokeWidth={1}
                strokeOpacity={0.3}
                initial={{ scale: 0, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ ...SPRING, delay: i * 0.02 }}
                style={{ cursor: 'pointer' }}
                aria-label={`${region.label ?? region.name}: ${region.value} validators`}
                onMouseEnter={() => setTooltip({
                  x, y,
                  label: region.label ?? region.name,
                  value: region.value,
                })}
                onMouseLeave={() => setTooltip(null)}
                whileHover={{ scale: 1.3 }}
              />
            )
          })}
        </svg>

        {/* Tooltip */}
        {tooltip && (
          <div
            role="tooltip"
            className="absolute pointer-events-none z-10 glass-2 rounded-md px-2.5 py-1.5 text-xs"
            style={{
              left: `${(tooltip.x / svgW) * 100}%`,
              top: `${(tooltip.y / svgH) * 100}%`,
              transform: 'translate(-50%, -120%)',
            }}
          >
            <div className="text-text-primary font-medium">{tooltip.label}</div>
            <div className="text-muted tabular-nums">{tooltip.value} validators</div>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 text-[10px] text-muted">
        <span className="flex items-center gap-1">
          <span className="w-2 h-2 rounded-full bg-success" /> {'<10'}
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 rounded-full bg-warning" /> 10–50
        </span>
        <span className="flex items-center gap-1">
          <span className="w-4 h-4 rounded-full bg-danger" /> 50+
        </span>
        <span className="ml-auto">{block.regions.length} GCP regions</span>
      </div>
    </div>
  )
}
