import { useRef, useEffect } from 'react'
import useStore from '../stores/useStore'

export default function VideoPlayer() {
  const videoRef = useRef()
  const mediaUrl = useStore((s) => s.mediaUrl)
  const isPlaying = useStore((s) => s.isPlaying)
  const currentTime = useStore((s) => s.currentTime)
  const setDuration = useStore((s) => s.setDuration)
  const setCurrentTime = useStore((s) => s.setCurrentTime)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    if (isPlaying) {
      video.play().catch(() => {})
    } else {
      video.pause()
    }
  }, [isPlaying])

  useEffect(() => {
    const video = videoRef.current
    if (!video || isPlaying) return
    if (Math.abs(video.currentTime - currentTime) > 0.3) {
      video.currentTime = currentTime
    }
  }, [currentTime, isPlaying])

  function handleTimeUpdate() {
    const video = videoRef.current
    if (!video || !isPlaying) return
    setCurrentTime(video.currentTime)
  }

  function handleLoadedMetadata() {
    const video = videoRef.current
    if (video) setDuration(video.duration)
  }

  if (!mediaUrl) {
    return (
      <div className="w-full h-full bg-gray-900 rounded-lg flex items-center justify-center text-gray-600 text-sm">
        No media loaded
      </div>
    )
  }

  return (
    <div className="w-full h-full bg-black rounded-lg overflow-hidden flex items-center justify-center">
      <video
        ref={videoRef}
        src={mediaUrl}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        className="max-w-full max-h-full"
      />
    </div>
  )
}
