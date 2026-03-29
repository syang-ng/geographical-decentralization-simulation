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
  const topRegions = [...block.regions]
    .toSorted((left, right) => right.value - left.value)
    .slice(0, 3)

  const svgW = 800
  const svgH = 420

  return (
    <div className="overflow-hidden rounded-2xl border border-border-subtle bg-surface/95 shadow-[0_20px_60px_rgba(0,0,0,0.24)]">
      <div className="border-b border-border-subtle bg-white/[0.02] px-5 py-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium text-text-primary">
              {block.title}
            </h3>
            <p className="mt-1 text-[11px] text-muted">
              Geographic distribution rendered directly from the block&apos;s region values.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {topRegions.map(region => (
              <span
                key={region.name}
                className="rounded-full border border-white/8 bg-black/15 px-2.5 py-1 text-[10px] text-muted"
              >
                {region.label ?? region.name}
                <span className="ml-1.5 font-medium tabular-nums text-text-primary">
                  {region.value}
                </span>
              </span>
            ))}
          </div>
        </div>
      </div>

      <div className="px-5 py-5">
        <div className="relative overflow-hidden rounded-xl border border-white/6 bg-[#0a0a0a]" style={{ minHeight: 280 }}>
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(59,130,246,0.12),transparent_42%),radial-gradient(circle_at_bottom_right,rgba(45,212,191,0.08),transparent_35%)]" />
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.02),transparent_25%,transparent_75%,rgba(255,255,255,0.03))]" />
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
          {[0.25, 0.75].map(frac => (
            <line
              key={`lat-${frac}`}
              x1={0}
              y1={svgH * frac}
              x2={svgW}
              y2={svgH * frac}
              stroke="#151515"
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
              stroke="#151515"
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
                  r={r * 1.9}
                  fill={color}
                  fillOpacity={0.08}
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ ...SPRING, delay: i * 0.02 }}
                />
                <motion.circle
                  cx={x}
                  cy={y}
                  r={r}
                  fill={color}
                  fillOpacity={0.78}
                  stroke={color}
                  strokeWidth={1}
                  strokeOpacity={0.35}
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
      </div>

      {/* Legend + 3D viewer link */}
      <div className="flex flex-wrap items-center gap-4 border-t border-border-subtle px-5 py-3 text-[10px] text-muted">
        <span className="flex items-center gap-1">
          <span className="h-2 w-2 rounded-full bg-success" /> {'<10'}
        </span>
        <span className="flex items-center gap-1">
          <span className="h-3 w-3 rounded-full bg-warning" /> 10–50
        </span>
        <span className="flex items-center gap-1">
          <span className="h-4 w-4 rounded-full bg-danger" /> 50+
        </span>
        <span className="ml-auto flex items-center gap-3">
          {block.regions.length} GCP regions
          <a
            href="/dash"
            target="_blank"
            rel="noopener noreferrer"
            className={cn(
              'flex items-center gap-1 rounded border border-accent/20 px-2 py-0.5',
              'text-accent transition-colors hover:border-accent/40 hover:text-accent/80',
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
