import { useEffect, useMemo, useState } from 'react'
import { ArrowUpRight } from 'lucide-react'
import { cn } from '../../lib/cn'
import { formatNumber } from './simulation-constants'

interface ResearchMetadata {
  readonly v?: number
  readonly cost?: number
  readonly delta?: number
  readonly cutoff?: number
  readonly gamma?: number
  readonly description?: string
}

interface ResearchDatasetEntry {
  readonly evaluation: string
  readonly paradigm: string
  readonly result: string
  readonly path: string
  readonly sourceRole?: string
  readonly metadata?: ResearchMetadata
}

interface ResearchCatalog {
  readonly introBlurb: string
  readonly defaultSelection: {
    readonly evaluation: string
    readonly paradigm: string
    readonly result: string
    readonly path: string
  } | null
  readonly datasets: readonly ResearchDatasetEntry[]
}

declare global {
  interface Window {
    RESEARCH_CATALOG?: ResearchCatalog
  }
}

interface ResearchDemoSurfaceProps {
  readonly catalogScriptUrl: string
  readonly viewerBaseUrl: string
}

function readResearchCatalog(): ResearchCatalog | null {
  return typeof window !== 'undefined' ? window.RESEARCH_CATALOG ?? null : null
}

function uniqueOrdered(values: readonly string[]): string[] {
  return [...new Set(values)]
}

function formatEth(value: number | undefined): string {
  if (typeof value !== 'number') return 'N/A'
  return `${formatNumber(value, 4)} ETH`
}

function formatMilliseconds(value: number | undefined): string {
  if (typeof value !== 'number') return 'N/A'
  return `${formatNumber(value, 0)} ms`
}

export function ResearchDemoSurface({
  catalogScriptUrl,
  viewerBaseUrl,
}: ResearchDemoSurfaceProps) {
  const [catalog, setCatalog] = useState<ResearchCatalog | null>(() => readResearchCatalog())
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const [selectedEvaluation, setSelectedEvaluation] = useState('')
  const [selectedParadigm, setSelectedParadigm] = useState('')
  const [selectedResult, setSelectedResult] = useState('')
  const [theme, setTheme] = useState<'auto' | 'light' | 'dark'>('auto')
  const [step, setStep] = useState<1 | 10 | 50>(1)
  const [autoplay, setAutoplay] = useState(true)

  useEffect(() => {
    const existing = readResearchCatalog()
    if (existing) {
      setCatalog(existing)
      setCatalogError(null)
      return
    }

    const scriptId = 'research-demo-catalog-script'
    let script = document.getElementById(scriptId) as HTMLScriptElement | null

    const handleLoad = () => {
      const loadedCatalog = readResearchCatalog()
      if (loadedCatalog) {
        setCatalog(loadedCatalog)
        setCatalogError(null)
        return
      }
      setCatalogError('The frozen research catalog loaded, but no datasets were exposed.')
    }

    const handleError = () => {
      setCatalogError('The frozen research catalog could not be loaded.')
    }

    if (!script) {
      script = document.createElement('script')
      script.id = scriptId
      script.src = catalogScriptUrl
      script.async = true
      document.head.appendChild(script)
    }

    script.addEventListener('load', handleLoad)
    script.addEventListener('error', handleError)

    return () => {
      script?.removeEventListener('load', handleLoad)
      script?.removeEventListener('error', handleError)
    }
  }, [catalogScriptUrl])

  useEffect(() => {
    if (!catalog) return

    const fallback = catalog.defaultSelection ?? catalog.datasets[0] ?? null
    if (!fallback) return

    setSelectedEvaluation(previous => previous || fallback.evaluation)
    setSelectedParadigm(previous => previous || fallback.paradigm)
    setSelectedResult(previous => previous || fallback.result)
  }, [catalog])

  const evaluationOptions = useMemo(
    () => uniqueOrdered((catalog?.datasets ?? []).map(entry => entry.evaluation)),
    [catalog],
  )

  const paradigmOptions = useMemo(
    () => uniqueOrdered(
      (catalog?.datasets ?? [])
        .filter(entry => entry.evaluation === selectedEvaluation)
        .map(entry => entry.paradigm),
    ),
    [catalog, selectedEvaluation],
  )

  const resultOptions = useMemo(
    () => uniqueOrdered(
      (catalog?.datasets ?? [])
        .filter(entry => entry.evaluation === selectedEvaluation && entry.paradigm === selectedParadigm)
        .map(entry => entry.result),
    ),
    [catalog, selectedEvaluation, selectedParadigm],
  )

  useEffect(() => {
    if (!catalog || evaluationOptions.length === 0) return

    const defaultSelection = catalog.defaultSelection
    const nextEvaluation = evaluationOptions.includes(selectedEvaluation)
      ? selectedEvaluation
      : defaultSelection?.evaluation && evaluationOptions.includes(defaultSelection.evaluation)
        ? defaultSelection.evaluation
        : evaluationOptions[0]!

    if (nextEvaluation !== selectedEvaluation) {
      setSelectedEvaluation(nextEvaluation)
      return
    }

    const nextParadigmOptions = uniqueOrdered(
      catalog.datasets
        .filter(entry => entry.evaluation === nextEvaluation)
        .map(entry => entry.paradigm),
    )
    if (nextParadigmOptions.length === 0) return

    const nextParadigm = nextParadigmOptions.includes(selectedParadigm)
      ? selectedParadigm
      : defaultSelection?.evaluation === nextEvaluation && defaultSelection.paradigm && nextParadigmOptions.includes(defaultSelection.paradigm)
        ? defaultSelection.paradigm
        : nextParadigmOptions[0]!

    if (nextParadigm !== selectedParadigm) {
      setSelectedParadigm(nextParadigm)
      return
    }

    const nextResultOptions = uniqueOrdered(
      catalog.datasets
        .filter(entry => entry.evaluation === nextEvaluation && entry.paradigm === nextParadigm)
        .map(entry => entry.result),
    )
    if (nextResultOptions.length === 0) return

    const nextResult = nextResultOptions.includes(selectedResult)
      ? selectedResult
      : defaultSelection?.evaluation === nextEvaluation
        && defaultSelection.paradigm === nextParadigm
        && defaultSelection.result
        && nextResultOptions.includes(defaultSelection.result)
          ? defaultSelection.result
          : nextResultOptions[0]!

    if (nextResult !== selectedResult) {
      setSelectedResult(nextResult)
    }
  }, [catalog, evaluationOptions, selectedEvaluation, selectedParadigm, selectedResult])

  const selectedDataset = useMemo(
    () => (catalog?.datasets ?? []).find(entry =>
      entry.evaluation === selectedEvaluation
      && entry.paradigm === selectedParadigm
      && entry.result === selectedResult,
    ) ?? null,
    [catalog, selectedEvaluation, selectedParadigm, selectedResult],
  )

  const introParagraphs = useMemo(
    () => (catalog?.introBlurb ?? '')
      .split(/\n\s*\n/)
      .map(paragraph => paragraph.trim())
      .filter(Boolean),
    [catalog],
  )

  const handleLaunchViewer = () => {
    if (!selectedDataset) return

    const settings = {
      dataset: selectedDataset.path,
      theme,
      step,
      autoplay,
    }

    try {
      window.localStorage.setItem('app_settings', JSON.stringify(settings))
    } catch {
      // Ignore storage failures and rely on query params.
    }

    const params = new URLSearchParams({
      dataset: selectedDataset.path,
      theme,
      step: String(step),
      autoplay: String(autoplay),
    })
    const viewerUrl = `${viewerBaseUrl}/viewer.html?${params.toString()}`
    const popup = window.open(viewerUrl, '_blank', 'noopener,noreferrer')
    if (!popup) {
      window.location.assign(viewerUrl)
    }
  }

  const handleFillDemoValues = () => {
    const demoEntry = (catalog?.datasets ?? []).find(entry =>
      entry.evaluation === 'Test' && entry.paradigm === 'External' && entry.result === 'data',
    )

    if (demoEntry) {
      setSelectedEvaluation(demoEntry.evaluation)
      setSelectedParadigm(demoEntry.paradigm)
      setSelectedResult(demoEntry.result)
    }

    setTheme('dark')
    setStep(10)
    setAutoplay(true)
  }

  if (catalogError) {
    return (
      <div className="lab-stage p-5">
        <div className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-3 text-sm text-danger">
          {catalogError}
        </div>
      </div>
    )
  }

  if (!catalog) {
    return (
      <div className="lab-stage p-5">
        <div className="py-12 text-sm text-muted text-center">
          Loading frozen research launcher…
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="lab-stage p-6">
        <div className="flex flex-col gap-6 lg:grid lg:grid-cols-[minmax(0,1.3fr)_minmax(320px,0.95fr)]">
          <div>
            <div className="inline-flex items-center gap-2 rounded-full border border-border-subtle bg-white px-3 py-1 text-[11px] font-medium uppercase tracking-[0.16em] text-text-primary">
              Published Research Demo
            </div>
            <h2 className="mt-4 text-2xl font-semibold text-text-primary">
              Geographical Decentralization Simulation
            </h2>
            <div className="mt-3 space-y-3 text-sm text-muted">
              {introParagraphs.map(paragraph => (
                <p key={paragraph}>{paragraph}</p>
              ))}
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-1">
            <div className="rounded-xl border border-border-subtle bg-white px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Surface</div>
              <div className="mt-1 text-sm font-medium text-text-primary">Static published results</div>
              <div className="mt-1 text-xs text-muted">This side swaps among checked-in published datasets. It does not run new simulations.</div>
            </div>
            <div className="rounded-xl border border-border-subtle bg-white px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Controls</div>
              <div className="mt-1 text-sm font-medium text-text-primary">Viewer settings only</div>
              <div className="mt-1 text-xs text-muted">Theme, step size, and autoplay affect presentation. They do not change the model.</div>
            </div>
            <div className="rounded-xl border border-border-subtle bg-white px-4 py-3">
              <div className="text-[10px] uppercase tracking-[0.16em] text-text-faint">Parity</div>
              <div className="mt-1 text-sm font-medium text-text-primary">Canonical dataset selector</div>
              <div className="mt-1 text-xs text-muted">Scenario, `Local`/`External`, and result choices mirror the frozen researcher catalog.</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(320px,0.85fr)]">
        <div className="lab-stage p-5">
          <div className="flex items-center justify-between gap-3 mb-4">
            <div>
              <div className="text-xs text-muted mb-1">Dataset selection</div>
              <div className="text-sm text-text-primary">
                Match the paper-facing launcher with our app styling.
              </div>
            </div>
            <button
              onClick={handleFillDemoValues}
              className="text-xs text-muted hover:text-text-primary transition-colors"
            >
              Fill demo values
            </button>
          </div>

          <div className="grid gap-4 sm:grid-cols-[minmax(0,1.15fr)_minmax(0,0.95fr)_minmax(0,1fr)]">
            <div>
              <label className="text-xs text-muted mb-1.5 block">Dataset</label>
              <select
                value={selectedEvaluation}
                onChange={event => setSelectedEvaluation(event.target.value)}
                className="w-full rounded-lg border border-border-subtle bg-white px-3 py-2 text-sm text-text-primary outline-none focus:ring-1 focus:ring-accent"
              >
                {evaluationOptions.map(option => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-xs text-muted mb-1.5 block">Block building</label>
              <div className="grid grid-cols-2 gap-2">
                {paradigmOptions.map(option => (
                  <button
                    key={option}
                    onClick={() => setSelectedParadigm(option)}
                    className={cn(
                      'rounded-lg border px-3 py-2 text-sm font-medium transition-colors',
                      selectedParadigm === option
                        ? 'border-accent bg-white text-accent'
                        : 'border-border-subtle bg-white text-text-primary hover:border-border-hover',
                    )}
                  >
                    {option}
                  </button>
                ))}
              </div>
            </div>

            <div>
              <label className="text-xs text-muted mb-1.5 block">Result</label>
              <select
                value={selectedResult}
                onChange={event => setSelectedResult(event.target.value)}
                className="w-full rounded-lg border border-border-subtle bg-white px-3 py-2 text-sm text-text-primary outline-none focus:ring-1 focus:ring-accent"
              >
                {resultOptions.map(option => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="mt-4 text-xs text-muted">
            The `Local` / `External` switch mirrors the researcher launcher. It changes which frozen dataset path is opened, not the simulation engine.
          </div>
        </div>

        <div className="lab-stage p-5">
          <div className="text-xs text-muted mb-1">Viewer options</div>
          <div className="text-sm text-text-primary mb-4">
            These map directly onto the frozen viewer contract.
          </div>

          <div className="grid gap-4">
            <div>
              <label className="text-xs text-muted mb-1.5 block">Theme</label>
              <select
                value={theme}
                onChange={event => setTheme(event.target.value as 'auto' | 'light' | 'dark')}
                className="w-full rounded-lg border border-border-subtle bg-white px-3 py-2 text-sm text-text-primary outline-none focus:ring-1 focus:ring-accent"
              >
                <option value="auto">Auto</option>
                <option value="light">Light</option>
                <option value="dark">Dark</option>
              </select>
            </div>

            <div>
              <label className="text-xs text-muted mb-1.5 block">Step size</label>
              <select
                value={step}
                onChange={event => setStep(Number(event.target.value) as 1 | 10 | 50)}
                className="w-full rounded-lg border border-border-subtle bg-white px-3 py-2 text-sm text-text-primary outline-none focus:ring-1 focus:ring-accent"
              >
                <option value={1}>1</option>
                <option value={10}>10</option>
                <option value={50}>50</option>
              </select>
            </div>

            <div>
              <label className="text-xs text-muted mb-1.5 block">Autoplay</label>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: 'On', value: true },
                  { label: 'Off', value: false },
                ].map(option => (
                  <button
                    key={option.label}
                    onClick={() => setAutoplay(option.value)}
                    className={cn(
                      'rounded-lg border px-3 py-2 text-sm font-medium transition-colors',
                      autoplay === option.value
                        ? 'border-accent bg-white text-accent'
                        : 'border-border-subtle bg-white text-text-primary hover:border-border-hover',
                    )}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="mt-4 rounded-xl border border-border-subtle bg-white px-4 py-3">
            <div className="text-xs text-text-faint">Current viewer payload</div>
            <div className="mt-1 text-sm font-medium text-text-primary break-all">
              {selectedDataset?.path ?? 'Choose a dataset'}
            </div>
          </div>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(320px,0.85fr)]">
        <div className="lab-stage p-5">
          <div className="text-xs text-muted mb-1">Scenario summary</div>
          <div className="text-sm text-text-primary">
            {selectedDataset?.metadata?.description ?? 'Select a dataset to see the published scenario description.'}
          </div>

          <div className="grid grid-cols-2 gap-3 mt-4 text-xs text-muted sm:grid-cols-3">
            <div className="rounded-lg border border-border-subtle bg-white px-3 py-3">
              <div className="text-text-faint">Validators</div>
              <div className="mt-1 text-sm font-medium text-text-primary">
                {selectedDataset?.metadata?.v?.toLocaleString() ?? 'N/A'}
              </div>
            </div>
            <div className="rounded-lg border border-border-subtle bg-white px-3 py-3">
              <div className="text-text-faint">Migration cost</div>
              <div className="mt-1 text-sm font-medium text-text-primary">
                {formatEth(selectedDataset?.metadata?.cost)}
              </div>
            </div>
            <div className="rounded-lg border border-border-subtle bg-white px-3 py-3">
              <div className="text-text-faint">Delta</div>
              <div className="mt-1 text-sm font-medium text-text-primary">
                {formatMilliseconds(selectedDataset?.metadata?.delta)}
              </div>
            </div>
            <div className="rounded-lg border border-border-subtle bg-white px-3 py-3">
              <div className="text-text-faint">Cutoff</div>
              <div className="mt-1 text-sm font-medium text-text-primary">
                {formatMilliseconds(selectedDataset?.metadata?.cutoff)}
              </div>
            </div>
            <div className="rounded-lg border border-border-subtle bg-white px-3 py-3">
              <div className="text-text-faint">Gamma</div>
              <div className="mt-1 text-sm font-medium text-text-primary">
                {typeof selectedDataset?.metadata?.gamma === 'number'
                  ? formatNumber(selectedDataset.metadata.gamma, 4)
                  : 'N/A'}
              </div>
            </div>
            <div className="rounded-lg border border-border-subtle bg-white px-3 py-3">
              <div className="text-text-faint">Source role</div>
              <div className="mt-1 text-sm font-medium capitalize text-text-primary">
                {selectedDataset?.sourceRole ?? 'N/A'}
              </div>
            </div>
          </div>
        </div>

        <div className="lab-stage p-5">
          <div className="text-xs text-muted mb-1">Launch</div>
          <div className="text-sm text-text-primary">
            Open the frozen viewer with the current researcher-style selection.
          </div>
          <div className="mt-2 text-xs text-muted">
            This opens the viewer against the same static dataset contract as the baseline launcher, while keeping this page in the app.
          </div>

          <div className="flex flex-col gap-3 mt-5 sm:flex-row">
            <button
              onClick={handleLaunchViewer}
              disabled={!selectedDataset}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent/85 disabled:cursor-not-allowed disabled:opacity-60"
            >
              Launch Viewer
              <ArrowUpRight className="h-4 w-4" />
            </button>
            <a
              href={`${viewerBaseUrl}/`}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center justify-center rounded-lg border border-border-subtle bg-white px-4 py-2 text-sm font-medium text-text-primary transition-colors hover:border-border-hover"
            >
              Open Baseline Page
            </a>
          </div>
        </div>
      </div>
    </div>
  )
}
