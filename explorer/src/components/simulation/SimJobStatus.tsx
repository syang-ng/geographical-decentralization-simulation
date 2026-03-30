import { motion } from 'framer-motion'
import { SPRING } from '../../lib/theme'
import type { SimulationJob } from '../../lib/simulation-api'

type JobStatus = 'idle' | 'submitting' | 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

interface SimJobStatusProps {
  readonly status: JobStatus
  readonly jobData: SimulationJob | null
  readonly submitError: Error | null
  readonly cancelError: Error | null
}

export function SimJobStatus({
  status,
  jobData,
  submitError,
  cancelError,
}: SimJobStatusProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={SPRING}
      className="lab-panel rounded-xl p-5 mb-6"
    >
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <div className="text-xs text-muted mb-1">
            Job status
          </div>
          <div className="text-sm font-medium text-text-primary">
            {status === 'idle' && 'Ready'}
            {status === 'submitting' && 'Submitting configuration'}
            {status === 'queued' && 'Queued for exact execution'}
            {status === 'running' && 'Running exact simulation'}
            {status === 'completed' && 'Completed'}
            {status === 'failed' && 'Failed'}
            {status === 'cancelled' && 'Cancelled'}
          </div>
        </div>

        {jobData && (
          <div className="grid grid-cols-2 gap-3 text-xs text-muted min-w-[220px]">
            <div>
              <span className="block text-xs text-text-faint">Queue</span>
              {jobData.queuePosition ?? 'live'}
            </div>
            <div>
              <span className="block text-xs text-text-faint">Cache</span>
              {jobData.cacheHit ? 'hit' : 'fresh'}
            </div>
          </div>
        )}
      </div>

      {(submitError || jobData?.error || cancelError) && (
        <div className="mt-4 rounded-lg border border-danger/30 bg-danger/10 px-3 py-2 text-xs text-danger">
          {submitError?.message
            ?? cancelError?.message
            ?? jobData?.error}
        </div>
      )}
    </motion.div>
  )
}
