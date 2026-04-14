import { create } from 'zustand'

const useStore = create((set, get) => ({
  // Timeline
  currentTime: 0,        // seconds (float) — wall-clock time in the stimulus
  duration: 0,           // total duration in seconds
  isPlaying: false,
  timestep: 0,           // integer index into predictions array

  // Segment-based time mapping
  segmentTimes: null,    // array of { start, duration } from backend
  hemodynamicLag: 5.0,   // seconds — brain response lags stimulus

  setCurrentTime: (t) => {
    const { duration, segmentTimes, hemodynamicLag } = get()
    const clamped = Math.max(0, Math.min(t, duration))
    // Map wall-clock time to the correct prediction timestep
    // Account for hemodynamic lag: brain at time T reflects stimulus at T - lag
    const brainTime = clamped + hemodynamicLag
    let step = 0
    if (segmentTimes) {
      // Find the segment whose start time is closest to brainTime
      for (let i = 0; i < segmentTimes.length; i++) {
        if (segmentTimes[i].start <= brainTime) {
          step = i
        } else {
          break
        }
      }
    } else {
      // Fallback: linear mapping
      const { preds } = get()
      step = preds ? Math.min(Math.floor(clamped), preds.length - 1) : 0
    }
    set({ currentTime: clamped, timestep: Math.max(0, step) })
  },
  setDuration: (d) => set({ duration: d }),
  setPlaying: (p) => set((s) => ({
    isPlaying: p,
    selectedRegion: p ? null : s.selectedRegion,
  })),
  togglePlaying: () => set((s) => ({
    isPlaying: !s.isPlaying,
    selectedRegion: !s.isPlaying ? null : s.selectedRegion,
  })),

  // Mesh (loaded once)
  mesh: null,
  setMesh: (m) => set({ mesh: m }),

  // Predictions
  preds: null,           // Array of Float32Array, one per timestep
  regions: null,         // { regionName: [value_per_timestep] }
  fineGroups: null,      // { regionName: "Fine Group Name" }
  coarseGroups: null,    // { regionName: "Coarse Group Name" }
  regionVertices: null,  // { regionName: [vertex_idx, ...] }
  globalVmin: 0,         // global min across all timesteps (1st percentile)
  globalVmax: 1,         // global max across all timesteps (99th percentile)
  setPredictions: ({ preds, regions, fineGroups, coarseGroups, regionVertices, globalVmin, globalVmax, segmentTimes, hemodynamicLag }) =>
    set({
      preds, regions, fineGroups, coarseGroups,
      regionVertices: regionVertices ?? null,
      globalVmin: globalVmin ?? 0,
      globalVmax: globalVmax ?? 1,
      segmentTimes: segmentTimes ?? null,
      hemodynamicLag: hemodynamicLag ?? 5.0,
    }),

  // Region selection (for highlighting + camera focus)
  selectedRegion: null,  // region name or null
  setSelectedRegion: (name) => set({ selectedRegion: name }),

  // Job tracking
  jobId: null,
  jobStatus: null,       // "processing" | "done" | "error"
  jobProgress: 0,
  setJob: (j) => set(j),

  // Input
  inputType: null,       // "video" | "audio" | "text"
  mediaUrl: null,        // blob URL for video/audio, or null for text
  textContent: null,     // raw text for text input
  setInput: (inputType, mediaUrl, textContent) =>
    set({ inputType, mediaUrl: mediaUrl ?? null, textContent: textContent ?? null }),

  // Reset for new prediction
  reset: () => set({
    currentTime: 0, duration: 0, isPlaying: false, timestep: 0,
    preds: null, regions: null, fineGroups: null, coarseGroups: null,
    regionVertices: null, selectedRegion: null,
    globalVmin: 0, globalVmax: 1, segmentTimes: null, hemodynamicLag: 5.0,
    jobId: null, jobStatus: null, jobProgress: 0,
    inputType: null, mediaUrl: null, textContent: null,
  }),
}))

export default useStore
