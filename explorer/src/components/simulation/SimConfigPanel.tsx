import { motion } from 'framer-motion'
import { Ban, Play, RotateCcw, Sparkles } from 'lucide-react'
import { cn } from '../../lib/cn'
import { SPRING } from '../../lib/theme'
import type { SimulationConfig } from '../../lib/simulation-api'

const THRESHOLD_OPTIONS = [
  { label: '1/3', value: 1 / 3 },
  { label: '1/2', value: 1 / 2 },
  { label: '2/3', value: 2 / 3 },
  { label: '4/5', value: 4 / 5 },
] as const

const SLOT_OPTIONS = [
  { label: '6s', value: 6 },
  { label: '8s', value: 8 },
  { label: '12s', value: 12 },
] as const

function attestationCutoffMs(slotTime: number): number {
  if (slotTime === 6) return 3000
  if (slotTime === 8) return 4000
  return 4000
}

interface SimConfigPanelProps {
  readonly config: SimulationConfig
  readonly onConfigChange: <K extends keyof SimulationConfig>(key: K, value: SimulationConfig[K]) => void
  readonly onSubmit: () => void
  readonly onReset: () => void
  readonly isSubmitting: boolean
  readonly canCancel: boolean
  readonly onCancel: () => void
  readonly paperScenarioLabels: readonly string[]
}

export function SimConfigPanel({
  config,
  onConfigChange,
  onSubmit,
  onReset,
  isSubmitting,
  canCancel,
  onCancel,
  paperScenarioLabels,
}: SimConfigPanelProps) {
  return (
    <div className="lab-stage p-5 mb-6">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="text-xs text-muted mb-1.5 block">
            Paradigm
          </label>
          <div className="flex gap-1">
            {(['SSP', 'MSP'] as const).map(paradigm => (
              <button
                key={paradigm}
                onClick={() => onConfigChange('paradigm', paradigm)}
                className={cn(
                  'flex-1 py-1.5 rounded-lg text-xs font-medium transition-all',
                  config.paradigm === paradigm
                    ? paradigm === 'SSP'
                      ? 'bg-white text-accent border border-accent'
                      : 'bg-white text-accent-warm border border-accent-warm'
                    : 'bg-white text-muted border border-border-subtle hover:border-border-hover',
                )}
              >
                {paradigm}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="text-xs text-muted mb-1.5 block">
            Validator Distribution
          </label>
          <select
            value={config.distribution}
            onChange={event => onConfigChange('distribution', event.target.value as SimulationConfig['distribution'])}
            className="w-full bg-white border border-border-subtle rounded-lg px-3 py-1.5 text-xs text-text-primary outline-none focus:ring-1 focus:ring-accent"
          >
            <option value="homogeneous">Homogeneous (upstream baseline default)</option>
            <option value="homogeneous-gcp">Homogeneous per GCP region</option>
            <option value="heterogeneous">Heterogeneous (real ETH data)</option>
            <option value="random">Random</option>
          </select>
        </div>

        <div>
          <label className="text-xs text-muted mb-1.5 block">
            Information Source Placement
          </label>
          <select
            value={config.sourcePlacement}
            onChange={event => onConfigChange('sourcePlacement', event.target.value as SimulationConfig['sourcePlacement'])}
            className="w-full bg-white border border-border-subtle rounded-lg px-3 py-1.5 text-xs text-text-primary outline-none focus:ring-1 focus:ring-accent"
          >
            <option value="homogeneous">Homogeneous</option>
            <option value="latency-aligned">Latency-aligned</option>
            <option value="latency-misaligned">Latency-misaligned</option>
          </select>
        </div>

        <div>
          <label className="text-xs text-muted mb-1.5 block">
            Seed
          </label>
          <input
            type="number"
            value={config.seed}
            min={0}
            max={2147483647}
            onChange={event => onConfigChange('seed', Number(event.target.value))}
            className="w-full bg-white border border-border-subtle rounded-lg px-3 py-1.5 text-xs text-text-primary outline-none focus:ring-1 focus:ring-accent"
          />
        </div>

        <div>
          <label className="text-xs text-muted mb-1.5 block">
            Validators
          </label>
          <input
            type="number"
            min={1}
            max={1000}
            step={1}
            value={config.validators}
            onChange={event => onConfigChange('validators', Number(event.target.value))}
            className="w-full bg-white border border-border-subtle rounded-lg px-3 py-1.5 text-xs text-text-primary outline-none focus:ring-1 focus:ring-accent"
          />
          <div className="mt-1 text-xs text-muted">Upstream defaults and paper baselines use 1,000 validators.</div>
        </div>

        <div>
          <label className="text-xs text-muted mb-1.5 block">
            Slots
          </label>
          <input
            type="number"
            min={1}
            max={10000}
            step={1}
            value={config.slots}
            onChange={event => onConfigChange('slots', Number(event.target.value))}
            className="w-full bg-white border border-border-subtle rounded-lg px-3 py-1.5 text-xs text-text-primary outline-none focus:ring-1 focus:ring-accent"
          />
          <div className="mt-1 text-xs text-muted">The upstream presets run up to 10,000 slots; shorter runs remain exact but are noisier.</div>
        </div>

        <div>
          <label className="text-xs text-muted mb-1.5 block">
            Migration Cost: {config.migrationCost.toFixed(4)} ETH
          </label>
          <input
            type="range"
            min={0}
            max={0.005}
            step={0.0001}
            value={config.migrationCost}
            onChange={event => onConfigChange('migrationCost', Number(event.target.value))}
            className="w-full accent-accent"
          />
          <div className="flex justify-between text-xs text-text-faint">
            <span>0</span>
            <span>0.005</span>
          </div>
        </div>

        <div>
          <label className="text-xs text-muted mb-1.5 block">
            Attestation Threshold (γ)
          </label>
          <div className="flex gap-1">
            {THRESHOLD_OPTIONS.map(option => (
              <button
                key={option.label}
                onClick={() => onConfigChange('attestationThreshold', option.value)}
                className={cn(
                  'flex-1 py-1.5 rounded-lg text-xs font-medium transition-all',
                  Math.abs(config.attestationThreshold - option.value) < 0.01
                    ? 'bg-white text-accent border border-accent'
                    : 'bg-white text-muted border border-border-subtle hover:border-border-hover',
                )}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="text-xs text-muted mb-1.5 block">
            Slot Time (Δ)
          </label>
          <div className="flex gap-1">
            {SLOT_OPTIONS.map(option => (
              <button
                key={option.label}
                onClick={() => onConfigChange('slotTime', option.value)}
                className={cn(
                  'flex-1 py-1.5 rounded-lg text-xs font-medium transition-all',
                  config.slotTime === option.value
                    ? 'bg-white text-accent border border-accent'
                    : 'bg-white text-muted border border-border-subtle hover:border-border-hover',
                )}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex items-center gap-3 mt-5 pt-4 border-t border-border-subtle/50">
        <motion.button
          onClick={onSubmit}
          whileTap={{ scale: 0.98 }}
          transition={SPRING}
          disabled={isSubmitting}
          className={cn(
            'flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-medium transition-all',
            'bg-accent text-white hover:bg-accent/80 disabled:opacity-60 disabled:cursor-not-allowed',
          )}
        >
          <Play className="w-3 h-3" />
          {isSubmitting ? 'Submitting…' : 'Run Exact Simulation'}
        </motion.button>

        <button
          onClick={onReset}
          className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs text-muted hover:text-text-primary transition-colors"
        >
          <RotateCcw className="w-3 h-3" />
          Reset
        </button>

        {canCancel && (
          <button
            onClick={onCancel}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs text-danger hover:text-danger transition-colors"
          >
            <Ban className="w-3 h-3" />
            Cancel
          </button>
        )}

        <div className="flex-1" />

        <div className="text-xs text-text-faint text-right">
          <div className="flex items-center gap-1 justify-end">
            <Sparkles className="w-3 h-3" />
            Exact mode only
          </div>
          <div>Slot cutoff: {attestationCutoffMs(config.slotTime)} ms</div>
          <div>{paperScenarioLabels.join(' · ')}</div>
        </div>
      </div>
    </div>
  )
}
