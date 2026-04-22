import { useState } from 'react'
import UploadModal from './UploadModal'
import HistoryDrawer from './HistoryDrawer'

export default function TopBar({ onLoadJob }) {
  const [modalOpen, setModalOpen] = useState(false)
  const [modalMode, setModalMode] = useState('file')
  const [historyOpen, setHistoryOpen] = useState(false)

  return (
    <>
      <div className="h-12 bg-gray-950 border-b border-gray-800 flex items-center justify-between px-4">
        <span className="text-lg font-bold text-red-500">neuroLoop</span>
        <div className="flex gap-2">
          <button
            onClick={() => setHistoryOpen(true)}
            className="px-3 py-1.5 text-sm bg-gray-800 hover:bg-gray-700 rounded-md transition"
            title="History"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
          </button>
          <button
            onClick={() => { setModalMode('file'); setModalOpen(true) }}
            className="px-3 py-1.5 text-sm bg-gray-800 hover:bg-gray-700 rounded-md transition"
          >
            Upload Video
          </button>
          <button
            onClick={() => { setModalMode('text'); setModalOpen(true) }}
            className="px-3 py-1.5 text-sm bg-gray-800 hover:bg-gray-700 rounded-md transition"
          >
            Paste Text
          </button>
        </div>
      </div>
      {modalOpen && (
        <UploadModal mode={modalMode} onClose={() => setModalOpen(false)} />
      )}
      {historyOpen && (
        <HistoryDrawer
          onClose={() => setHistoryOpen(false)}
          onSelect={(jobId) => { onLoadJob?.(jobId); setHistoryOpen(false) }}
        />
      )}
    </>
  )
}
