import { useState, useEffect } from 'react'
import { getRuns } from '../utils/api'

function formatDuration(seconds) {
  if (!seconds) return null
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function formatDate(iso) {
  const diff = Date.now() - new Date(iso)
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  if (days < 7) return `${days}d ago`
  return new Date(iso).toLocaleDateString()
}

const TYPE_CHIP = {
  video:  'text-blue-400 bg-blue-400/10',
  audio:  'text-green-400 bg-green-400/10',
  text:   'text-purple-400 bg-purple-400/10',
}

export default function HistoryDrawer({ onClose, onSelect }) {
  const [runs, setRuns] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    getRuns()
      .then(data => setRuns(data.runs))
      .catch(e => setError(e.message))
  }, [])

  return (
    <>
      <div className="fixed inset-0 z-40 bg-black/40" onClick={onClose} />

      <div className="fixed right-0 top-0 h-full w-80 z-50 bg-gray-950 border-l border-gray-800 flex flex-col">
        <div className="h-12 flex items-center justify-between px-4 border-b border-gray-800 shrink-0">
          <span className="text-sm font-semibold text-gray-200">History</span>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-200 transition text-xl leading-none"
          >
            ×
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {error && (
            <p className="text-red-400 text-sm p-4">{error}</p>
          )}

          {!runs && !error && (
            <div className="flex items-center justify-center h-32 text-gray-600 text-sm">
              Loading...
            </div>
          )}

          {runs && runs.length === 0 && (
            <div className="flex items-center justify-center h-32 text-gray-600 text-sm">
              No analyses yet
            </div>
          )}

          {runs && runs.map(run => (
            <div
              key={run.job_id}
              onClick={() => onSelect?.(run.job_id)}
              className="flex gap-3 px-4 py-3 border-b border-gray-800/50 hover:bg-gray-900 transition cursor-pointer"
            >
              <div className="w-16 h-10 shrink-0 rounded bg-gray-800 overflow-hidden">
                {run.thumbnail_url ? (
                  <img src={run.thumbnail_url} alt="" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-[10px] text-gray-600 font-mono">
                    {run.input_type === 'text' ? 'TXT' : 'AUD'}
                  </div>
                )}
              </div>

              <div className="flex-1 min-w-0">
                <p className="text-xs text-gray-200 truncate leading-tight">{run.filename}</p>
                <div className="flex items-center gap-1.5 mt-1">
                  {run.input_type && (
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${TYPE_CHIP[run.input_type] ?? 'text-gray-400 bg-gray-400/10'}`}>
                      {run.input_type}
                    </span>
                  )}
                  {run.duration_seconds && (
                    <span className="text-[10px] text-gray-500">
                      {formatDuration(run.duration_seconds)}
                    </span>
                  )}
                </div>
                <p className="text-[10px] text-gray-600 mt-0.5">{formatDate(run.timestamp)}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  )
}
