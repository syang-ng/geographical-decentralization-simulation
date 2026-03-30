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
  const normalized = Math.max(value / Math.max(maxValue, 1), 0.08)
  return 4 + normalized * 8
}

function getDotColor(value: number, maxValue: number, colorScale?: string): string {
  if (colorScale === 'binary') return value > 0 ? '#22C55E' : '#D4D4D2'
  if (colorScale === 'change') {
    if (value > 0) return '#22C55E'
    if (value < 0) return '#EF4444'
    return '#9CA3AF'
  }
  const t = Math.min(value / Math.max(maxValue, 1), 1)
  if (t < 0.33) return '#5B8DEF'
  if (t < 0.66) return '#D97757'
  return '#B45309'
}

export function MapBlock({ block }: MapBlockProps) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; label: string; value: number } | null>(null)
  const maxValue = Math.max(...block.regions.map(region => region.value), 1)
  const topRegions = [...block.regions]
    .toSorted((left, right) => right.value - left.value)
    .slice(0, 4)

  const svgW = 800
  const svgH = 420

  return (
    <div className="lab-panel overflow-hidden rounded-xl">
      <div className="border-b border-border-subtle px-5 py-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h3 className="text-sm font-medium text-text-primary">
              {block.title}
            </h3>
            <p className="mt-1 text-xs text-muted">
              Geographic distribution rendered directly from exact region counts, with emphasis on dominant corridors and spatial reading rather than decorative 3D distortion.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="lab-chip">{block.regions.length} measured regions</span>
            <span className="lab-chip">max {maxValue} validators</span>
          </div>
        </div>
      </div>

      <div className="grid gap-4 px-5 py-5 lg:grid-cols-[1.45fr_minmax(0,0.7fr)]">
        <div className="relative overflow-hidden rounded-2xl border border-border-subtle bg-[radial-gradient(circle_at_20%_15%,rgba(59,130,246,0.12),transparent_26%),radial-gradient(circle_at_80%_18%,rgba(217,119,87,0.12),transparent_28%),linear-gradient(180deg,#fcfcfa,#f0efeb)] shadow-[inset_0_1px_0_rgba(255,255,255,0.84)]" style={{ minHeight: 320 }}>
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(transparent_31px,rgba(26,26,26,0.03)_32px),linear-gradient(90deg,transparent_31px,rgba(26,26,26,0.03)_32px)] bg-[size:32px_32px]" />
          <div className="pointer-events-none absolute inset-x-[14%] top-4 h-20 rounded-full bg-[radial-gradient(circle,rgba(255,255,255,0.92),transparent_72%)] blur-2xl" />
          <div className="pointer-events-none absolute inset-x-[12%] bottom-[-22%] h-40 rounded-[50%] bg-[radial-gradient(circle,rgba(26,26,26,0.08),transparent_72%)] blur-2xl" />
          <div className="globe-grid pointer-events-none absolute right-[-12px] top-[-18px] h-48 w-48 opacity-60" />

          <svg
            viewBox={`0 0 ${svgW} ${svgH}`}
            className="relative z-[1] w-full"
            preserveAspectRatio="xMidYMid meet"
            role="img"
            aria-label={block.title}
          >
            <rect x={0} y={0} width={svgW} height={svgH} fill="none" stroke="#E8E8E6" strokeWidth={1} />
            <ellipse cx={svgW / 2} cy={svgH / 2} rx={svgW * 0.28} ry={svgH * 0.42} fill="none" stroke="#E6EBF0" strokeWidth={1} />
            <ellipse cx={svgW / 2} cy={svgH / 2} rx={svgW * 0.17} ry={svgH * 0.42} fill="none" stroke="#EDF1F5" strokeWidth={1} />
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

            {block.regions.map((region, index) => {
              const { x, y } = latLonToMercator(region.lat, region.lon, svgW, svgH)
              const radius = getDotRadius(region.value, maxValue)
              const color = getDotColor(region.value, maxValue, block.colorScale)
              const isTopRegion = topRegions.some(topRegion => topRegion.name === region.name)

              return (
                <g key={region.name}>
                  {isTopRegion && (
                    <motion.circle
                      cx={x}
                      cy={y}
                      r={radius * 2.1}
                      fill={color}
                      fillOpacity={0.08}
                      initial={{ scale: 0.82, opacity: 0 }}
                      animate={{ scale: [1, 1.04, 1], opacity: [0.08, 0.14, 0.08] }}
                      transition={{ duration: 5.2, repeat: Infinity, ease: 'easeInOut', delay: index * 0.1 }}
                    />
                  )}
                  <motion.circle
                    cx={x}
                    cy={y}
                    r={radius * 1.55}
                    fill={color}
                    fillOpacity={0.12}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ ...SPRING, delay: index * 0.018 }}
                  />
                  <motion.circle
                    cx={x}
                    cy={y}
                    r={radius}
                    fill={color}
                    fillOpacity={0.88}
                    stroke="white"
                    strokeWidth={1.5}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ ...SPRING, delay: index * 0.018 }}
                    style={{ cursor: 'pointer' }}
                    aria-label={`${region.label ?? region.name}: ${region.value} validators`}
                    onMouseEnter={() => setTooltip({
                      x,
                      y,
                      label: region.label ?? region.name,
                      value: region.value,
                    })}
                    onMouseLeave={() => setTooltip(null)}
                    whileHover={{ scale: 1.14 }}
                  />
                </g>
              )
            })}
          </svg>

          {tooltip && (
            <div
              role="tooltip"
              className="absolute z-10 rounded-md border border-border-subtle bg-white/95 px-2.5 py-1.5 text-xs shadow-sm backdrop-blur-sm"
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

        <div className="space-y-3">
          <div className="rounded-2xl border border-border-subtle bg-[linear-gradient(180deg,rgba(255,255,255,0.97),rgba(246,245,241,0.86))] p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.78)]">
            <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Leading regions</div>
            <div className="mt-3 space-y-2">
              {topRegions.map((region, index) => (
                <div
                  key={region.name}
                  className="flex items-center justify-between gap-3 rounded-lg border border-border-subtle bg-white/80 px-3 py-2"
                >
                  <div className="min-w-0">
                    <div className="text-xs font-medium text-text-primary">
                      {index + 1}. {region.label ?? region.name}
                    </div>
                    <div className="mt-0.5 text-xs text-muted">{region.name}</div>
                  </div>
                  <div className="text-sm font-semibold tabular-nums text-text-primary">
                    {region.value}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-2xl border border-border-subtle bg-[linear-gradient(180deg,rgba(255,255,255,0.97),rgba(246,245,241,0.86))] p-3 text-xs text-muted shadow-[inset_0_1px_0_rgba(255,255,255,0.78)]">
            <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Legend</div>
            <div className="mt-3 space-y-2">
              <div className="flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-[#5B8DEF]" />
                Lower concentration
              </div>
              <div className="flex items-center gap-2">
                <span className="h-2.5 w-2.5 rounded-full bg-[#D97757]" />
                Medium concentration
              </div>
              <div className="flex items-center gap-2">
                <span className="h-3 w-3 rounded-full bg-[#B45309]" />
                Dominant concentration
              </div>
            </div>
          </div>

          <a
            href="https://geo-decentralization.github.io/"
            target="_blank"
            rel="noopener noreferrer"
            className={cn(
              'inline-flex items-center gap-1.5 rounded-xl border border-border-subtle bg-white/92 px-3 py-2 text-xs text-accent transition-colors hover:border-border-hover hover:text-accent/80',
            )}
          >
            <ExternalLink className="h-3 w-3" />
            Open in 3D Viewer
          </a>
        </div>
      </div>
    </div>
  )
}
