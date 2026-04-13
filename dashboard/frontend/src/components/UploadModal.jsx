import { useState, useRef } from 'react'
import { getUploadUrl, startPredict } from '../utils/api'
import useStore from '../stores/useStore'

export default function UploadModal({ mode, onClose }) {
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [text, setText] = useState('')
  const fileRef = useRef()
  const setJob = useStore((s) => s.setJob)
  const setInput = useStore((s) => s.setInput)
  const reset = useStore((s) => s.reset)

  async function handleFileUpload() {
    const file = fileRef.current?.files?.[0]
    if (!file) return
    setUploading(true)
    setError(null)
    try {
      reset()
      const { upload_url, s3_key } = await getUploadUrl(file.name, file.type)
      await fetch(upload_url, { method: 'PUT', body: file, headers: { 'Content-Type': file.type } })
      const inputType = file.type.startsWith('audio') ? 'audio' : 'video'
      setInput(inputType, URL.createObjectURL(file), null)
      const { job_id } = await startPredict(s3_key, inputType)
      setJob({ jobId: job_id, jobStatus: 'processing', jobProgress: 0 })
      onClose()
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  async function handleTextSubmit() {
    if (!text.trim()) return
    setUploading(true)
    setError(null)
    try {
      reset()
      const blob = new Blob([text], { type: 'text/plain' })
      const filename = 'input.txt'
      const { upload_url, s3_key } = await getUploadUrl(filename, 'text/plain')
      await fetch(upload_url, { method: 'PUT', body: blob, headers: { 'Content-Type': 'text/plain' } })
      setInput('text', null, text)
      const { job_id } = await startPredict(s3_key, 'text')
      setJob({ jobId: job_id, jobStatus: 'processing', jobProgress: 0 })
      onClose()
    } catch (e) {
      setError(e.message)
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-gray-900 rounded-xl p-6 w-full max-w-md border border-gray-700" onClick={(e) => e.stopPropagation()}>
        <h2 className="text-lg font-semibold mb-4">
          {mode === 'file' ? 'Upload Video or Audio' : 'Paste Text'}
        </h2>

        {mode === 'file' ? (
          <input
            ref={fileRef}
            type="file"
            accept="video/*,audio/*"
            className="w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:bg-gray-700 file:text-white hover:file:bg-gray-600"
          />
        ) : (
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to analyze..."
            rows={6}
            className="w-full bg-gray-800 rounded-lg p-3 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-red-500"
          />
        )}

        {error && <p className="text-red-400 text-sm mt-2">{error}</p>}

        <div className="flex justify-end gap-2 mt-4">
          <button onClick={onClose} className="px-4 py-2 text-sm text-gray-400 hover:text-white transition">
            Cancel
          </button>
          <button
            onClick={mode === 'file' ? handleFileUpload : handleTextSubmit}
            disabled={uploading}
            className="px-4 py-2 text-sm bg-red-600 hover:bg-red-500 rounded-md transition disabled:opacity-50"
          >
            {uploading ? 'Processing...' : 'Analyze'}
          </button>
        </div>
      </div>
    </div>
  )
}
