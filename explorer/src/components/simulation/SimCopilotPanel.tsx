import { useState } from 'react'
import { Sparkles } from 'lucide-react'
import { BlockCanvas } from '../explore/BlockCanvas'
import { cn } from '../../lib/cn'
import type { SimulationCopilotResponse, SimulationConfig } from '../../lib/simulation-api'

interface SimCopilotPanelProps {
  readonly copilotQuestion: string
  readonly onQuestionChange: (value: string) => void
  readonly onAsk: (question: string) => void
  readonly onApplyConfig: (config: SimulationConfig) => void
  readonly copilotResponse: SimulationCopilotResponse | null
  readonly copilotAvailable: boolean
  readonly isHealthLoading: boolean
  readonly isMutating: boolean
  readonly mutationError: Error | null
  readonly hasManifest: boolean
  readonly promptSuggestions: readonly string[]
}

export function SimCopilotPanel({
  copilotQuestion,
  onQuestionChange,
  onAsk,
  onApplyConfig,
  copilotResponse,
  copilotAvailable,
  isHealthLoading,
  isMutating,
  mutationError,
  hasManifest,
  promptSuggestions,
}: SimCopilotPanelProps) {
  const [assistantOpen, setAssistantOpen] = useState(false)
  const copilotDisabled = isHealthLoading || !copilotAvailable || isMutating
  const showAssistant = assistantOpen || Boolean(copilotResponse) || Boolean(mutationError) || isMutating

  return (
    <div className="lab-stage p-5 mb-6">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="text-xs text-muted mb-1">
            Assistant framing (optional)
          </div>
          <div className="text-sm text-text-primary">
            The exact run surface is primary. The assistant is an opt-in layer for bounded help.
          </div>
        </div>
        <div className="max-w-xl text-xs text-muted">
          It can reorganize supported charts, suggest bounded configs, and add clearly labeled framing.
          It cannot invent metrics, replace the default result surface, or change the exact engine.
        </div>
      </div>

      <div className="mt-3 text-xs text-muted">
        {isHealthLoading
          ? 'Checking assistant availability...'
          : copilotAvailable
            ? 'Assistant framing is available for bounded exact-mode questions. Use it when you want help with comparison setup or chart ordering, not as the default source of truth.'
            : 'Assistant framing is offline. Add ANTHROPIC_API_KEY to explorer/.env to enable bounded simulation guidance.'}
      </div>

      {!showAssistant && (
        <div className="mt-4 flex flex-col gap-3 rounded-2xl border border-border-subtle bg-white/88 px-4 py-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.75)] lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="text-xs text-text-faint uppercase tracking-[0.16em]">Default mode</div>
            <div className="mt-1 text-sm text-text-primary">
              Read the exact metadata, artifact labels, and charts first.
            </div>
            <div className="mt-1 text-xs text-muted">
              Open the assistant only if you want bounded help with phrasing a run, comparing scenarios, or reordering supported visuals.
            </div>
          </div>
          <button
            onClick={() => setAssistantOpen(true)}
            disabled={!copilotAvailable && !isHealthLoading}
            className={cn(
              'flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-xs font-medium transition-all',
              'bg-accent text-white hover:bg-accent/80 disabled:opacity-60 disabled:cursor-not-allowed',
            )}
          >
            <Sparkles className="w-3 h-3" />
            Open assistant
          </button>
        </div>
      )}

      {showAssistant && (
        <>
          <div className="mt-4 flex items-center justify-between gap-3">
            <div className="text-xs text-muted">
              {hasManifest
                ? 'Optional layer over the current exact run.'
                : 'Optional help for shaping a bounded exact run.'}
            </div>
            <button
              onClick={() => setAssistantOpen(false)}
              className="text-xs text-muted transition-colors hover:text-text-primary"
            >
              Hide assistant
            </button>
          </div>

          <div className="mt-3 flex flex-col gap-3 lg:flex-row">
            <div className="flex-1">
              <div className="mb-2 text-xs text-muted">
                {hasManifest
                  ? 'Ask about this exact run, or request the next bounded experiment.'
                  : 'Ask for a bounded exact run setup that stays within the paper and simulator surface.'}
              </div>
              <textarea
                value={copilotQuestion}
                onChange={event => onQuestionChange(event.target.value)}
                rows={3}
                placeholder={hasManifest
                  ? 'Example: Show avg_mev, then supermajority_success, then explain the top regions.'
                  : 'Example: Set up the paper baseline SSP run, then tell me what to inspect first.'}
                disabled={isHealthLoading || !copilotAvailable}
                className="min-h-[92px] w-full resize-y bg-white border border-border-subtle rounded-lg px-3 py-2 text-sm text-text-primary outline-none focus:ring-1 focus:ring-accent"
              />
            </div>

            <div className="flex flex-col gap-2 lg:w-48">
              <button
                onClick={() => onAsk(copilotQuestion.trim())}
                disabled={!copilotQuestion.trim() || copilotDisabled}
                className={cn(
                  'flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-xs font-medium transition-all',
                  'bg-accent text-white hover:bg-accent/80 disabled:opacity-60 disabled:cursor-not-allowed',
                )}
              >
                <Sparkles className="w-3 h-3" />
                {isHealthLoading
                  ? 'Checking...'
                  : isMutating
                    ? 'Thinking...'
                    : 'Ask assistant'}
              </button>

              {copilotResponse?.proposedConfig && (
                <button
                  onClick={() => onApplyConfig({ ...copilotResponse.proposedConfig! })}
                  className="rounded-lg border border-border-subtle bg-white px-3 py-2 text-xs text-text-primary hover:border-border-hover transition-colors"
                >
                  Apply proposed config
                </button>
              )}
            </div>
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            {promptSuggestions.map(prompt => (
              <button
                key={prompt}
                onClick={() => {
                  if (!copilotAvailable) return
                  onQuestionChange(prompt)
                  onAsk(prompt)
                }}
                disabled={!copilotAvailable}
                className="text-xs text-muted hover:text-text-primary transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {prompt}
              </button>
            ))}
          </div>

          {mutationError && (
            <div className="mt-4 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
              {mutationError.message}
            </div>
          )}

          {copilotResponse && (
            <div className="mt-5 space-y-4">
              <div className="border-l-2 border-warning pl-4 py-2">
                <div className="flex items-center gap-1.5 text-xs font-medium text-text-primary mb-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-warning" />
                  {copilotResponse.truthBoundary.label}
                </div>
                <div className="text-xs text-muted">
                  {copilotResponse.truthBoundary.detail}
                </div>
              </div>

              <div className="rounded-xl border border-border-subtle bg-white/90 px-4 py-3">
                <div className="text-xs text-muted mb-1">
                  Assistant framing
                </div>
                <div className="text-sm text-text-primary">{copilotResponse.summary}</div>
                {copilotResponse.guidance && (
                  <div className="mt-2">
                    <div className="text-[11px] uppercase tracking-[0.14em] text-text-faint">Guide interpretation</div>
                    <div className="mt-1 text-xs text-muted">{copilotResponse.guidance}</div>
                  </div>
                )}
                <div className="mt-2 text-xs text-muted">
                  {copilotResponse.mode === 'proposed-run'
                    ? 'Proposed bounded run'
                    : copilotResponse.mode === 'guidance'
                      ? 'Guidance only'
                      : 'Current exact result with assistant framing'}
                  {copilotResponse.cached
                    ? ' · assistant reused cached study context'
                    : ' · fresh assistant response over the bounded simulation surface'}
                </div>
              </div>

              {copilotResponse.proposedConfig && (
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs text-muted">
                  <div>
                    <span className="block text-xs text-text-faint">Paradigm</span>
                    {copilotResponse.proposedConfig.paradigm}
                  </div>
                  <div>
                    <span className="block text-xs text-text-faint">Distribution</span>
                    {copilotResponse.proposedConfig.distribution}
                  </div>
                  <div>
                    <span className="block text-xs text-text-faint">Validators</span>
                    {copilotResponse.proposedConfig.validators.toLocaleString()}
                  </div>
                  <div>
                    <span className="block text-xs text-text-faint">Slots</span>
                    {copilotResponse.proposedConfig.slots.toLocaleString()}
                  </div>
                </div>
              )}

              <BlockCanvas blocks={copilotResponse.blocks} />
            </div>
          )}
        </>
      )}
    </div>
  )
}
