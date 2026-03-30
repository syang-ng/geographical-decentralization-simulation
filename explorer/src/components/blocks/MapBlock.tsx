import { useState } from 'react'
import { motion } from 'framer-motion'
import { ExternalLink } from 'lucide-react'
import { SPRING } from '../../lib/theme'
import { cn } from '../../lib/cn'
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
  return 3 + normalized * 9
}

function getDotColor(value: number, maxValue: number, colorScale?: string): string {
  if (colorScale === 'binary') return value > 0 ? '#22C55E' : '#D4D4D2'
  if (colorScale === 'change') {
    if (value > 0) return '#22C55E'
    if (value < 0) return '#EF4444'
    return '#9CA3AF'
  }
  const t = Math.min(value / Math.max(maxValue, 1), 1)
  if (t < 0.33) return '#22C55E'
  if (t < 0.66) return '#F59E0B'
  return '#EF4444'
}

export function MapBlock({ block }: MapBlockProps) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string; value: number } | null>(null)
  const maxValue = Math.max(...block.regions.map(r => r.value), 1)
  const topRegions = [...block.regions]
    .toSorted((left, right) => right.value - left.value)
    .slice(0, 3)

  const svgW = 800
  const svgH = 420

  return (
    <div className="bg-white border border-border-subtle rounded-lg overflow-hidden">
      <div className="border-b border-border-subtle px-5 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium text-text-primary">
              {block.title}
            </h3>
            <p className="mt-1 text-xs text-muted">
              Geographic distribution rendered directly from the block&apos;s region values.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {topRegions.map(region => (
              <span
                key={region.name}
                className="text-xs text-muted"
              >
                {region.label ?? region.name}
                <span className="ml-1 font-medium tabular-nums text-text-primary">
                  {region.value}
                </span>
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="px-5 py-5">
        <div className="relative overflow-hidden rounded-lg bg-[#FAFAF8]" style={{ minHeight: 280 }}>
        <svg
          viewBox={`0 0 ${svgW} ${svgH}`}
          className="w-full"
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={block.title}
        >
          {/* Light grid */}
          <rect x={0} y={0} width={svgW} height={svgH} fill="none" stroke="#E8E8E6" strokeWidth={1} />
          <line x1={0} y1={svgH / 2} x2={svgW} y2={svgH / 2} stroke="#E8E8E6" strokeWidth={0.5} strokeDasharray="4 4" />
          <line x1={svgW / 2} y1={0} x2={svgW / 2} y2={svgH} stroke="#E8E8E6" strokeWidth={0.5} strokeDasharray="4 4" />
          {[0.25, 0.75].map(frac => (
            <line
              key={`lat-${frac}`}
              x1={0}
              y1={svgH * frac}
              x2={svgW}
              y2={svgH * frac}
              stroke="#E8E8E6"
              strokeWidth={0.5}
              strokeDasharray="3 6"
            />
          ))}
          {[0.25, 0.75].map(frac => (
            <line
              key={`lon-${frac}`}
              x1={svgW * frac}
              y1={0}
              x2={svgW * frac}
              y2={svgH}
              stroke="#E8E8E6"
              strokeWidth={0.5}
              strokeDasharray="3 6"
            />
          ))}

          {/* Region dots */}
          {block.regions.map((region, i) => {
            const { x, y } = latLonToMercator(region.lat, region.lon, svgW, svgH)
            const r = getDotRadius(region.value, maxValue)
            const color = getDotColor(region.value, maxValue, block.colorScale)

            return (
              <g key={region.name}>
                <motion.circle
                  cx={x}
                  cy={y}
                  r={r * 1.6}
                  fill={color}
                  fillOpacity={0.12}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ ...SPRING, delay: i * 0.02 }}
                />
                <motion.circle
                  cx={x}
                  cy={y}
                  r={r}
                  fill={color}
                  fillOpacity={0.85}
                  stroke="white"
                  strokeWidth={1.5}
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
                  whileHover={{ scale: 1.28 }}
                />
              </g>
            )
          })}
        </svg>

        {/* Tooltip */}
        {tooltip && (
          <div
            role="tooltip"
            className="absolute pointer-events-none z-10 bg-white border border-border-subtle rounded-md px-2.5 py-1.5 text-xs shadow-sm"
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
      </div>

      {/* Legend + 3D viewer link */}
      <div className="flex flex-wrap items-center gap-4 border-t border-border-subtle px-5 py-3 text-xs text-muted">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-success" /> {'<10'}
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-full bg-warning" /> 10–50
        </span>
        <span className="flex items-center gap-1">
          <span className="h-3 w-3 rounded-full bg-danger" /> 50+
        </span>
        <span className="ml-auto flex items-center gap-3">
          {block.regions.length} GCP regions
          <a
            href="https://geo-decentralization.github.io/"
            target="_blank"
            rel="noopener noreferrer"
            className={cn(
              'flex items-center gap-1 text-accent transition-colors hover:text-accent/80',
            )}
          >
            <ExternalLink className="h-3 w-3" />
            Open in 3D Viewer
          </a>
        </span>
      </div>
    </div>
  )
}
