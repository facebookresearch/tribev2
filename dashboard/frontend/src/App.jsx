import { useEffect } from 'react'
import TopBar from './components/TopBar'
import VideoPlayer from './components/VideoPlayer'
import TextDisplay from './components/TextDisplay'
import BrainViewer from './components/BrainViewer'
import Timeline from './components/Timeline'
import RegionPanel from './components/RegionPanel'
import useStore from './stores/useStore'
import { getMesh, getResults } from './utils/api'

export default function App() {
  const mesh = useStore((s) => s.mesh)
  const setMesh = useStore((s) => s.setMesh)
  const inputType = useStore((s) => s.inputType)
  const jobId = useStore((s) => s.jobId)
  const jobStatus = useStore((s) => s.jobStatus)
  const jobProgress = useStore((s) => s.jobProgress)
  const setJob = useStore((s) => s.setJob)
  const setPredictions = useStore((s) => s.setPredictions)
  const setDuration = useStore((s) => s.setDuration)
  const isPlaying = useStore((s) => s.isPlaying)

  // Load mesh on mount
  useEffect(() => {
    if (mesh) return
    getMesh().then((data) => {
      setMesh({
        vertices: new Float32Array(data.vertices.flat()),
        faces: new Uint32Array(data.faces.flat()),
        nVertices: data.n_vertices,
      })
    }).catch((err) => console.error('Failed to load mesh:', err))
  }, [mesh, setMesh])

  // Poll job status
  useEffect(() => {
    if (!jobId || jobStatus !== 'processing') return
    const interval = setInterval(async () => {
      try {
        const res = await getResults(jobId)
        if (res.status === 'done') {
          // Fetch prediction data + full meta + regions in parallel
          const [predsResp, metaResp, regionsResp] = await Promise.all([
            fetch(res.preds_url).then((r) => r.arrayBuffer()),
            fetch(res.meta_url).then((r) => r.json()),
            fetch(res.regions_url).then((r) => r.json()),
          ])

          // Parse raw float32 binary (no numpy header)
          const nTimesteps = metaResp.n_timesteps
          const nVerts = metaResp.n_vertices
          const predsRaw = new Float32Array(predsResp)
          const preds = []
          for (let t = 0; t < nTimesteps; t++) {
            preds.push(predsRaw.subarray(t * nVerts, (t + 1) * nVerts))
          }

          setPredictions({
            preds,
            regions: regionsResp.regions,
            fineGroups: regionsResp.fine_groups,
            coarseGroups: regionsResp.coarse_groups,
            globalVmin: metaResp.global_vmin,
            globalVmax: metaResp.global_vmax,
            segmentTimes: metaResp.segment_times,
            hemodynamicLag: metaResp.hemodynamic_lag,
          })

          setDuration(metaResp.duration_seconds)
          setJob({ jobStatus: 'done', jobProgress: 1 })
          clearInterval(interval)
        } else if (res.status === 'error') {
          setJob({ jobStatus: 'error' })
          clearInterval(interval)
        } else {
          setJob({ jobProgress: res.progress })
        }
      } catch (err) {
        console.error('Poll error:', err)
      }
    }, 2000)
    return () => clearInterval(interval)
  }, [jobId, jobStatus, setJob, setPredictions, setDuration])

  // Play loop for text-only mode (video element drives time for video/audio)
  useEffect(() => {
    if (!isPlaying) return
    if (inputType === 'video' || inputType === 'audio') return
    const interval = setInterval(() => {
      const { currentTime, duration, isPlaying } = useStore.getState()
      if (!isPlaying) return
      if (currentTime >= duration) {
        useStore.getState().setPlaying(false)
        return
      }
      useStore.getState().setCurrentTime(currentTime + 0.1)
    }, 100)
    return () => clearInterval(interval)
  }, [inputType, isPlaying])

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-white">
      <TopBar />

      {jobStatus === 'processing' && (
        <div className="h-1 bg-gray-800">
          <div
            className="h-full transition-all duration-500"
            style={{
              width: `${jobProgress * 100}%`,
              background: 'linear-gradient(90deg, #e94560, #533483)',
            }}
          />
        </div>
      )}

      <div className="flex-1 flex min-h-0">
        <div className="w-1/2 border-r border-gray-800">
          {inputType === 'text' ? <TextDisplay /> : <VideoPlayer />}
        </div>
        <div className="w-1/2">
          <BrainViewer />
        </div>
      </div>

      <Timeline />

      <div className="h-48 border-t border-gray-800">
        <RegionPanel />
      </div>
    </div>
  )
}
