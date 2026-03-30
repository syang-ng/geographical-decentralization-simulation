import { AlertCircle, RefreshCw, KeyRound } from 'lucide-react'
import type { ExploreError } from '../../lib/api'

interface ErrorDisplayProps {
  error: ExploreError
  onRetry: () => void
}

export function ErrorDisplay({ error, onRetry }: ErrorDisplayProps) {
  const isRateLimit = error.status === 429
  const isAuth = error.status === 401
  const isNetwork = error.status === 0

  return (
    <div className="bg-white border border-border-subtle rounded-lg p-6 text-center">
      <div className="flex justify-center mb-3">
        {isAuth ? (
          <KeyRound className="w-8 h-8 text-muted" />
        ) : (
          <AlertCircle className="w-8 h-8 text-muted" />
        )}
      </div>

      <h3 className="text-sm font-medium text-text-primary mb-1">
        {isRateLimit
          ? 'Rate limit reached'
          : isAuth
            ? 'API key not configured'
            : isNetwork
              ? 'Cannot reach API server'
              : 'Something went wrong'}
      </h3>

      <p className="text-xs text-muted mb-4 max-w-sm mx-auto">
        {isRateLimit
          ? 'Too many requests. Wait a moment and try again.'
          : isAuth
            ? 'Set ANTHROPIC_API_KEY in the server environment to enable AI exploration.'
            : isNetwork
              ? 'Make sure the API server is running (npx tsx server/index.ts).'
              : error.error}
      </p>

      {!isAuth && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border border-border-subtle text-text-primary hover:bg-surface-active transition-colors"
        >
          <RefreshCw className="w-3 h-3" />
          Try again
        </button>
      )}
    </div>
  )
}
