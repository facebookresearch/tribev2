import useStore from '../stores/useStore'

export default function TextDisplay() {
  const textContent = useStore((s) => s.textContent)

  if (!textContent) {
    return (
      <div className="w-full h-full bg-gray-900 rounded-lg flex items-center justify-center text-gray-600 text-sm">
        No text loaded
      </div>
    )
  }

  return (
    <div className="w-full h-full bg-gray-900 rounded-lg p-4 overflow-y-auto">
      <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">{textContent}</p>
    </div>
  )
}
