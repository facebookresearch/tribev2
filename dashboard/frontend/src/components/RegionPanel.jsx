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

export default function RegionPanel() {
  const regions = useStore((s) => s.regions)
  const fineGroups = useStore((s) => s.fineGroups)
  const coarseGroups = useStore((s) => s.coarseGroups)
  const timestep = useStore((s) => s.timestep)

  if (!regions) {
    return (
      <div className="h-full bg-gray-950 p-4 flex items-center justify-center text-gray-600 text-sm">
        Run a prediction to see region scores
      </div>
    )
  }

  const entries = Object.entries(regions)
    .map(([name, values]) => ({ name, value: values[timestep] ?? 0 }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 10)

  const maxVal = entries[0]?.value || 1

  return (
    <div className="h-full bg-gray-950 p-4 overflow-y-auto">
      <div className="text-xs text-gray-500 font-semibold mb-2">
        TOP ACTIVATED REGIONS <span className="text-gray-600 font-normal">@ t={timestep}s</span>
      </div>
      <div className="flex flex-col gap-1.5">
        {entries.map(({ name, value }) => {
          const coarse = coarseGroups?.[name] || ''
          const fine = fineGroups?.[name] || ''
          const color = COARSE_COLORS[coarse] || '#6b7280'
          const pct = maxVal > 0 ? (value / maxVal) * 100 : 0

          return (
            <div key={name} className="flex items-center gap-2">
              <span className="text-xs text-gray-300 w-28 truncate" title={`${name} — ${fine}`}>
                {name}
                <span className="text-gray-600 ml-1 text-[10px]">({fine})</span>
              </span>
              <div className="flex-1 h-1.5 bg-gray-800 rounded">
                <div
                  className="h-full rounded"
                  style={{ width: `${pct}%`, backgroundColor: color }}
                />
              </div>
              <span className="text-[10px] text-gray-500 w-8 text-right">{value.toFixed(2)}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
