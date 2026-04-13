import { create } from 'zustand'

const useStore = create((set, get) => ({
  currentTime: 0,
  duration: 0,
  isPlaying: false,
  timestep: 0,

  setCurrentTime: (t) => {
    const { duration, preds } = get()
    const clamped = Math.max(0, Math.min(t, duration))
    const step = preds ? Math.min(Math.floor(clamped), preds.length - 1) : 0
    set({ currentTime: clamped, timestep: Math.max(0, step) })
  },
  setDuration: (d) => set({ duration: d }),
  setPlaying: (p) => set({ isPlaying: p }),
  togglePlaying: () => set((s) => ({ isPlaying: !s.isPlaying })),

  mesh: null,
  setMesh: (m) => set({ mesh: m }),

  preds: null,
  regions: null,
  fineGroups: null,
  coarseGroups: null,
  setPredictions: ({ preds, regions, fineGroups, coarseGroups }) =>
    set({ preds, regions, fineGroups, coarseGroups }),

  jobId: null,
  jobStatus: null,
  jobProgress: 0,
  setJob: (j) => set(j),

  inputType: null,
  mediaUrl: null,
  setInput: (inputType, mediaUrl) => set({ inputType, mediaUrl }),

  reset: () => set({
    currentTime: 0, duration: 0, isPlaying: false, timestep: 0,
    preds: null, regions: null, fineGroups: null, coarseGroups: null,
    jobId: null, jobStatus: null, jobProgress: 0,
    inputType: null, mediaUrl: null,
  }),
}))

export default useStore
