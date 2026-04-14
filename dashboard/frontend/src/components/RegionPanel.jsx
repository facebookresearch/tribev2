import useStore from '../stores/useStore'

const COARSE_COLORS = {
  'Visual': '#e94560',
  'Somatomotor': '#3b82f6',
  'Dorsal Attention': '#10b981',
  'Ventral Attention': '#f59e0b',
  'Limbic': '#8b5cf6',
  'Frontoparietal': '#ec4899',
  'Default': '#6b7280',
}

// Human-readable descriptions for functional network groups
const GROUP_DESCRIPTIONS = {
  'Visual': 'Processing what you see',
  'Somatomotor': 'Movement and physical sensation',
  'Dorsal Attention': 'Focusing on something specific',
  'Ventral Attention': 'Noticing something unexpected',
  'Limbic': 'Emotion and memory',
  'Frontoparietal': 'Thinking and decision making',
  'Default': 'Mind wandering and self-reflection',
}

// Human-readable descriptions for fine-grained networks
const FINE_DESCRIPTIONS = {
  'Primary Visual': 'Basic image processing',
  'Early Visual': 'Detecting edges and shapes',
  'Dorsal Stream Visual': 'Tracking where things are',
  'Ventral Stream Visual': 'Recognizing what things are',
  'MT+ Complex and Neighboring Areas': 'Perceiving motion',
  'Somatosensory and Motor': 'Touch and movement control',
  'Paracentral Lobular and Mid-Cingulate': 'Leg and trunk movement',
  'Premotor': 'Planning movements and eye gaze',
  'Posterior Opercular': 'Secondary touch and balance',
  'Early Auditory': 'Basic sound processing',
  'Auditory Association': 'Understanding speech and complex sounds',
  'Insular and Frontal Opercular': 'Gut feelings and emotional awareness',
  'Medial Temporal': 'Forming new memories',
  'Lateral Temporal': 'Understanding meaning and language',
  'Temporo-Parieto-Occipital Junction': 'Multisensory integration and social awareness',
  'Superior Parietal': 'Guiding reach, grasp, and gaze',
  'Inferior Parietal': 'Language, attention, and meaning',
  'Posterior Cingulate': 'Memory retrieval and self-reflection',
  'Anterior Cingulate and Medial Prefrontal': 'Social thinking and error monitoring',
  'Orbital and Polar Frontal': 'Judging value and reward',
  'Inferior Frontal': 'Speech, grammar, and word meaning',
  'Dorsolateral Prefrontal': 'Working memory and planning',
}

export default function RegionPanel() {
  const regions = useStore((s) => s.regions)
  const fineGroups = useStore((s) => s.fineGroups)
  const coarseGroups = useStore((s) => s.coarseGroups)
  const timestep = useStore((s) => s.timestep)
  const globalVmin = useStore((s) => s.globalVmin)
  const globalVmax = useStore((s) => s.globalVmax)
  const isPlaying = useStore((s) => s.isPlaying)
  const selectedRegion = useStore((s) => s.selectedRegion)
  const setSelectedRegion = useStore((s) => s.setSelectedRegion)

  if (!regions) {
    return (
      <div className="h-full bg-gray-950 p-4 flex items-center justify-center text-gray-600 text-sm">
        Run a prediction to see region scores
      </div>
    )
  }

  // Use global scale so bars reflect absolute activation, not relative to top region
  const scaleMax = globalVmax || 1

  const entries = Object.entries(regions)
    .map(([name, values]) => ({ name, value: values[timestep] ?? 0 }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 10)

  const clickable = !isPlaying

  return (
    <div className="h-full bg-gray-950 p-4 overflow-y-auto">
      <div className="text-xs text-gray-500 font-semibold mb-3 flex items-center justify-between">
        <span>BRAIN ACTIVITY <span className="text-gray-600 font-normal">@ t={timestep}s</span></span>
        {clickable && (
          <span className="text-[10px] text-gray-600 font-normal">
            {selectedRegion ? `Focused: ${selectedRegion}` : 'Click a region to focus'}
          </span>
        )}
      </div>
      <div className="flex flex-col gap-2">
        {entries.map(({ name, value }) => {
          const coarse = coarseGroups?.[name] || ''
          const fine = fineGroups?.[name] || ''
          const color = COARSE_COLORS[coarse] || '#6b7280'
          const pct = Math.max(0, Math.min(100, (value / scaleMax) * 100))
          const fineDesc = FINE_DESCRIPTIONS[fine] || fine
          const coarseDesc = GROUP_DESCRIPTIONS[coarse] || coarse
          const isSelected = selectedRegion === name

          return (
            <div
              key={name}
              onClick={() => {
                if (!clickable) return
                setSelectedRegion(isSelected ? null : name)
              }}
              className={`group rounded-md transition ${
                clickable ? 'cursor-pointer hover:bg-gray-900 px-2 -mx-2 py-1 -my-1' : ''
              } ${isSelected ? 'bg-gray-900 ring-1 ring-red-500/50' : ''}`}
            >
              <div className="flex items-center gap-3">
                <div className="w-36 shrink-0">
                  <div className="text-xs font-medium text-gray-200">{fineDesc}</div>
                  <div className="text-[10px] text-gray-500">{name} — {coarseDesc}</div>
                </div>
                <div className="flex-1 h-2 bg-gray-800 rounded-full">
                  <div
                    className="h-full rounded-full transition-all duration-200"
                    style={{ width: `${pct}%`, backgroundColor: color }}
                  />
                </div>
                <span className="text-[10px] text-gray-400 w-10 text-right font-mono">
                  {(pct).toFixed(0)}%
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
