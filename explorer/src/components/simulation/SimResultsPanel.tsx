import { startTransition } from 'react'
import { Check, Copy } from 'lucide-react'
import { BlockCanvas } from '../explore/BlockCanvas'
import { cn } from '../../lib/cn'
import type {
  SimulationArtifact,
  SimulationConfig,
  SimulationManifest,
  SimulationOverviewBundle,
} from '../../lib/simulation-api'
import type { SimulationArtifactBundle } from '../../types/simulation-view'
import type { Block } from '../../types/blocks'

function formatNumber(value: number, digits = 2): string {
  return value.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  })
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${formatNumber(bytes / 1024, 1)} KB`
  return `${formatNumber(bytes / (1024 * 1024), 1)} MB`
}

function paperScenarioLabels(config: SimulationConfig): string[] {
  const labels: string[] = []

  if (config.distribution === 'heterogeneous' && config.sourcePlacement !== 'homogeneous') {
    labels.push('SE3 joint heterogeneity')
  } else if (config.distribution === 'heterogeneous') {
    labels.push('SE2 heterogeneous validators')
  } else if (config.distribution === 'homogeneous-gcp') {
    labels.push('Equal per-GCP validator start')
  } else if (config.sourcePlacement === 'latency-aligned') {
    labels.push('SE1 latency-aligned sources')
  } else if (config.sourcePlacement === 'latency-misaligned') {
    labels.push('SE1 latency-misaligned sources')
  } else {
    labels.push('Baseline geography/source setup')
  }

  if (config.slotTime === 6) {
    labels.push('SE4b shorter slots')
  } else if (Math.abs(config.attestationThreshold - 2 / 3) > 0.01) {
    labels.push('SE4a gamma variation')
  }

  labels.push(config.paradigm === 'SSP' ? 'SSP exact mode' : 'MSP exact mode')
  return labels
}

function buildRunSummary(manifest: SimulationManifest): string {
  return [
    `Exact simulation run`,
    `Paradigm: ${manifest.config.paradigm}`,
    `Scenario: ${paperScenarioLabels(manifest.config).join(' | ')}`,
    `Seed: ${manifest.config.seed}`,
    `Validators: ${manifest.config.validators}`,
    `Slots: ${manifest.config.slots}`,
    `Runtime: ${formatNumber(manifest.runtimeSeconds, 2)}s`,
    `Final average MEV: ${formatNumber(manifest.summary.finalAverageMev, 4)} ETH`,
    `Final supermajority success: ${formatNumber(manifest.summary.finalSupermajoritySuccess, 2)}%`,
    `Execution: ${manifest.cacheHit ? 'Exact cache hit' : 'Fresh exact run'}`,
    `Cache key: ${manifest.cacheKey}`,
  ].join('\n')
}

function isManifestOverviewBundle(
  bundle: OverviewBundleOption | SimulationOverviewBundle | null,
): bundle is SimulationOverviewBundle {
  return Boolean(bundle && 'bytes' in bundle)
}

type OverviewBundleOption = {
  readonly bundle: SimulationArtifactBundle
  readonly label: string
  readonly description: string
}

interface SimResultsPanelProps {
  readonly manifest: SimulationManifest
  readonly overviewBundleOptions: ReadonlyArray<OverviewBundleOption | SimulationOverviewBundle>
  readonly selectedBundle: SimulationArtifactBundle
  readonly onSelectBundle: (bundle: SimulationArtifactBundle) => void
  readonly selectedOverviewBundleMetrics: SimulationOverviewBundle | null
  readonly overviewBlocks: readonly Block[]
  readonly isOverviewLoading: boolean
  readonly selectedArtifact: SimulationArtifact | null
  readonly selectedArtifactName: string | null
  readonly onSelectArtifact: (name: string) => void
  readonly isArtifactFetching: boolean
  readonly isParsing: boolean
  readonly parseError: string | null
  readonly parsedBlocks: readonly Block[]
  readonly copyState: 'config' | 'run' | null
  readonly onCopy: (text: string, kind: 'config' | 'run') => void
}

export function SimResultsPanel({
  manifest,
  overviewBundleOptions,
  selectedBundle,
  onSelectBundle,
  selectedOverviewBundleMetrics,
  overviewBlocks,
  isOverviewLoading,
  selectedArtifact,
  selectedArtifactName,
  onSelectArtifact,
  isArtifactFetching,
  isParsing,
  parseError,
  parsedBlocks,
  copyState,
  onCopy,
}: SimResultsPanelProps) {
  return (
    <>
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3 mb-6">
        <div className="bg-white border border-border-subtle rounded-lg p-4">
          <div className="text-xs text-muted mb-2">Avg MEV</div>
          <div className="text-2xl font-semibold text-text-primary">
            {formatNumber(manifest.summary.finalAverageMev, 4)}
          </div>
          <div className="text-xs text-muted mt-1">ETH</div>
        </div>

        <div className="bg-white border border-border-subtle rounded-lg p-4">
          <div className="text-xs text-muted mb-2">Supermajority</div>
          <div className="text-2xl font-semibold text-text-primary">
            {formatNumber(manifest.summary.finalSupermajoritySuccess, 2)}%
          </div>
        </div>

        <div className="bg-white border border-border-subtle rounded-lg p-4">
          <div className="text-xs text-muted mb-2">Runtime</div>
          <div className="text-2xl font-semibold text-text-primary">
            {formatNumber(manifest.runtimeSeconds, 2)}s
          </div>
        </div>

        <div className="bg-white border border-border-subtle rounded-lg p-4">
          <div className="text-xs text-muted mb-2">Slots</div>
          <div className="text-2xl font-semibold text-text-primary">
            {manifest.summary.slotsRecorded.toLocaleString()}
          </div>
        </div>
      </div>

      <div className="lab-stage p-5 mb-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div>
            <div className="text-xs text-muted mb-1">
              Run provenance
            </div>
            <div className="text-sm text-text-primary">
              {manifest.cacheHit ? 'Exact cache hit' : 'Fresh exact execution'}
            </div>
            <div className="text-xs text-muted mt-1 max-w-2xl">
              {manifest.cacheHit
                ? 'Reused an identical exact run from the shared exact cache. Outputs are unchanged for the same inputs.'
                : 'Executed the canonical exact simulator with the current configuration and seed.'}
            </div>
            <div className="flex flex-wrap gap-2 mt-3">
              {paperScenarioLabels(manifest.config).map(label => (
                <span
                  key={label}
                  className="inline-flex items-center gap-1.5 text-xs text-muted"
                >
                  <span className="w-1.5 h-1.5 rounded-full bg-accent" />
                  {label}
                </span>
              ))}
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => onCopy(JSON.stringify(manifest.config, null, 2), 'config')}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border-subtle bg-white px-3 py-2 text-xs text-text-primary hover:border-border-hover transition-colors"
            >
              {copyState === 'config' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
              {copyState === 'config' ? 'Copied config' : 'Copy config JSON'}
            </button>
            <button
              onClick={() => onCopy(buildRunSummary(manifest), 'run')}
              className="inline-flex items-center gap-1.5 rounded-lg border border-border-subtle bg-white px-3 py-2 text-xs text-text-primary hover:border-border-hover transition-colors"
            >
              {copyState === 'run' ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
              {copyState === 'run' ? 'Copied summary' : 'Copy run summary'}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4 text-xs text-muted">
          <div>
            <span className="block text-xs text-text-faint">Seed</span>
            {manifest.config.seed}
          </div>
          <div>
            <span className="block text-xs text-text-faint">Validators</span>
            {manifest.config.validators.toLocaleString()}
          </div>
          <div>
            <span className="block text-xs text-text-faint">Slots</span>
            {manifest.config.slots.toLocaleString()}
          </div>
          <div>
            <span className="block text-xs text-text-faint">Cache key</span>
            {manifest.cacheKey.slice(0, 12)}
          </div>
        </div>
      </div>

      <div className="lab-stage p-5 mb-6">
        <div className="flex items-center justify-between gap-3 mb-4">
          <div>
            <div className="text-xs text-muted mb-1">
              Exact overview
            </div>
            <div className="text-sm text-text-primary">
              Prebuilt exact overview bundles for the current run.
            </div>
          </div>
          <div className="text-xs text-muted">
            {selectedOverviewBundleMetrics
              ? [
                  formatBytes(selectedOverviewBundleMetrics.bytes),
                  selectedOverviewBundleMetrics.brotliBytes != null
                    ? `br ${formatBytes(selectedOverviewBundleMetrics.brotliBytes)}`
                    : null,
                  selectedOverviewBundleMetrics.gzipBytes != null
                    ? `gzip ${formatBytes(selectedOverviewBundleMetrics.gzipBytes)}`
                    : null,
                ].filter(Boolean).join(' · ')
              : 'Manifest-ready sidecars'}
          </div>
        </div>

        <div className="flex flex-wrap gap-2 mb-4">
          {overviewBundleOptions.map(option => (
            <button
              key={option.bundle}
              onClick={() => startTransition(() => onSelectBundle(option.bundle))}
              className={cn(
                'rounded-lg border px-3 py-2 text-left transition-colors',
                selectedBundle === option.bundle
                  ? 'border-accent bg-white'
                  : 'border-border-subtle bg-white hover:border-border-hover',
              )}
            >
              <div className="text-xs font-medium text-text-primary">{option.label}</div>
              <div className="text-xs text-muted">{option.description}</div>
              {isManifestOverviewBundle(option) && (
                <div className="mt-1 text-[11px] text-text-faint">
                  {[
                    formatBytes(option.bytes),
                    option.brotliBytes != null ? `br ${formatBytes(option.brotliBytes)}` : null,
                    option.gzipBytes != null ? `gzip ${formatBytes(option.gzipBytes)}` : null,
                  ].filter(Boolean).join(' · ')}
                </div>
              )}
            </button>
          ))}
        </div>

        {isOverviewLoading && overviewBlocks.length === 0 && (
          <div className="py-12 text-sm text-muted text-center">
            Preparing exact overview charts…
          </div>
        )}

        {!isOverviewLoading && overviewBlocks.length > 0 && (
          <BlockCanvas blocks={overviewBlocks} />
        )}

        {!isOverviewLoading && overviewBlocks.length === 0 && (
          <div className="py-12 text-sm text-muted text-center">
            This exact run does not have a ready overview sidecar for the selected bundle yet.
          </div>
        )}
      </div>

      <div className="lab-stage p-5 mb-6">
        <div className="flex items-center justify-between gap-3 mb-4">
          <div>
            <div className="text-xs text-muted mb-1">
              Artifact manifest
            </div>
            <div className="text-sm text-text-primary">
              Summary data is already loaded. Pick a derived artifact or a raw export with compression ready.
            </div>
          </div>
          <div className="text-xs text-muted text-right">
            {manifest.cacheHit ? 'Served from exact cache' : 'Fresh exact run'}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          {manifest.artifacts.map(artifact => (
            <button
              key={artifact.name}
              onClick={() => onSelectArtifact(artifact.name)}
              disabled={!artifact.renderable}
              className={cn(
                'text-left rounded-lg border px-4 py-3 transition-all',
                selectedArtifactName === artifact.name
                  ? 'border-accent bg-white'
                  : 'border-border-subtle bg-white hover:border-border-hover',
                !artifact.renderable && 'opacity-60 cursor-not-allowed',
              )}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-sm font-medium text-text-primary">{artifact.label}</div>
                  <div className="text-xs text-muted mt-1">{artifact.description}</div>
                </div>
                <div className="text-xs text-muted whitespace-nowrap">
                  {artifact.lazy ? 'lazy' : 'ready'}
                </div>
              </div>

              <div className="flex items-center gap-3 mt-3 text-xs text-muted">
                <span>{formatBytes(artifact.bytes)}</span>
                {artifact.brotliBytes != null && <span>br {formatBytes(artifact.brotliBytes)}</span>}
                {artifact.gzipBytes != null && <span>gzip {formatBytes(artifact.gzipBytes)}</span>}
                <span>{artifact.kind}</span>
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="lab-stage p-5">
        <div className="flex items-center justify-between gap-3 mb-4">
          <div>
            <div className="text-xs text-muted mb-1">
              Rendered artifact
            </div>
            <div className="text-sm text-text-primary">
              {selectedArtifact?.label ?? 'Select an artifact to render'}
            </div>
          </div>
          {selectedArtifact && (
            <div className="text-xs text-muted">
              {selectedArtifact.kind} · {selectedArtifact.lazy ? 'lazy-loaded' : 'manifest-ready'}
            </div>
          )}
        </div>

        {((isArtifactFetching && !parsedBlocks.length) || isParsing) && (
          <div className="py-12 text-sm text-muted text-center">
            Parsing artifact in a browser worker…
          </div>
        )}

        {parseError && (
          <div className="rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
            {parseError}
          </div>
        )}

        {!isArtifactFetching && !isParsing && !parseError && parsedBlocks.length > 0 && (
          <BlockCanvas blocks={parsedBlocks} />
        )}

        {!isArtifactFetching && !isParsing && !parseError && parsedBlocks.length === 0 && (
          <div className="py-12 text-sm text-muted text-center">
            Pick a renderable artifact to inspect the exact run.
          </div>
        )}
      </div>
    </>
  )
}
