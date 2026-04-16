import { useState, useEffect } from 'react'
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

const REGION_DESCRIPTIONS = {
  // ── Primary Visual ──
  "V1": "First stage of vision — edges, contrast, orientation",
  // ── Early Visual ──
  "V2": "Second visual area — contours and simple patterns",
  "V3": "Third visual area — form and dynamic shape",
  "V4": "Color and shape processing in mid-level vision",
  // ── Dorsal Stream Visual ──
  "V3A": "Motion and spatial depth processing",
  "V3B": "Motion boundaries and moving edges",
  "V6": "Wide-field motion for self-movement awareness",
  "V6A": "Visual guidance of reaching movements",
  "V7": "Spatial integration near the center of gaze",
  "IPS1": "Top-down spatial attention and object tracking",
  // ── Ventral Stream Visual ──
  "V8": "Color perception in the lower visual pathway",
  "VVC": "Color, contrast, and texture integration",
  "PIT": "Object color properties — hue and brightness",
  "FFC": "Face and body recognition",
  "VMV1": "Scene processing — visual features",
  "VMV2": "Scene processing — spatial layout",
  "VMV3": "Scene processing — context and meaning",
  // ── MT+ Complex ──
  "V3CD": "Integrates object detail for recognition",
  "LO1": "Object orientation processing",
  "LO2": "Object shape discrimination",
  "LO3": "Object form processing",
  "V4t": "Object velocity near the center of gaze",
  "FST": "Complex motion and optic flow analysis",
  "MT": "Core motion perception — speed and direction",
  "MST": "Heading direction and optic flow",
  "PH": "High-level motion processing near parietal cortex",
  // ── Somatosensory and Motor ──
  "4": "Primary motor cortex — fine movement control",
  "3a": "Sense of body position and muscle stretch",
  "3b": "Primary touch — first stage of tactile processing",
  "1": "Texture and surface feel by touch",
  "2": "Deep pressure and joint position sensing",
  // ── Paracentral Lobular and Mid-Cingulate ──
  "24dd": "Mid-cingulate motor — movement coordination",
  "24dv": "Mid-cingulate motor — choosing which action to take",
  "6mp": "Supplementary motor — movement sequencing",
  "6ma": "Supplementary motor — visually triggered actions",
  "SCEF": "Supplementary eye field — goal-directed gaze",
  "5m": "Sensory integration for reaching and pointing",
  "5L": "Hand movement guidance without vision",
  "5mv": "Body sensation and movement imitation",
  // ── Premotor ──
  "55b": "Language-related area between eye fields",
  "6d": "Dorsal premotor — upper limb planning",
  "6a": "Dorsal premotor — visually guided hand movement",
  "FEF": "Frontal eye field — voluntary eye movements",
  "6v": "Ventral premotor — grasping and object handling",
  "6r": "Rostral premotor — speech-related motor area",
  "PEF": "Premotor eye field — reflexive eye movements",
  // ── Posterior Opercular ──
  "43": "Swallowing, taste, and mouth/tongue movement",
  "FOP1": "Frontal opercular — pain and motor imagery",
  "OP4": "Sensory-motor integration for object handling",
  "OP1": "Secondary touch — pain and tactile memory",
  "OP2-3": "Balance and body-position processing",
  "PFcm": "Language vocabulary and speech articulation",
  // ── Early Auditory ──
  "A1": "Primary auditory cortex — basic sound analysis",
  "LBelt": "Lateral belt — spectral sound features",
  "MBelt": "Medial belt — frequency-selective processing",
  "PBelt": "Posterior belt — complex sound integration",
  "RI": "Retroinsular — links hearing and touch",
  // ── Auditory Association ──
  "A4": "Higher auditory processing — sound patterns",
  "A5": "Auditory association — speech sound streams",
  "STSdp": "Posterior dorsal STS — links sight and sound",
  "STSda": "Anterior dorsal STS — speech processing",
  "STSvp": "Posterior ventral STS — language comprehension",
  "STSva": "Anterior ventral STS — speech and language",
  "STGa": "Anterior superior temporal — voice identity",
  "TA2": "Temporal association — auditory processing",
  // ── Insular and Frontal Opercular ──
  "52": "Auditory and arithmetic processing",
  "PI": "Posterior insula — bridges hearing and touch",
  "Ig": "Granular insula — touch and taste processing",
  "PoI1": "Posterior insula — multi-sense input processing",
  "PoI2": "Posterior insula — movement and sensation",
  "FOP2": "Frontal opercular — sensory-motor relay",
  "FOP3": "Frontal opercular — taste and motor processing",
  "MI": "Middle insula — body state and internal awareness",
  "AVI": "Anterior insula — gut feelings and autonomic state",
  "AAIC": "Anterior insula — self-awareness and salience",
  "Pir": "Piriform cortex — smell and odor discrimination",
  "FOP4": "Frontal opercular — language and motor control",
  "FOP5": "Frontal opercular — speech initiation",
  // ── Medial Temporal ──
  "H": "Hippocampus — memory formation and navigation",
  "PreS": "Presubiculum — head direction and orientation",
  "EC": "Entorhinal cortex — memory encoding gateway",
  "PeEc": "Perirhinal cortex — object memory and familiarity",
  "PHA1": "Parahippocampal — scene and place recognition",
  "PHA2": "Parahippocampal — visual scene processing",
  "PHA3": "Parahippocampal — scene detail and tool knowledge",
  // ── Lateral Temporal ──
  "PHT": "Posterior middle temporal — concept retrieval",
  "TE1p": "Posterior middle temporal — face and visual memory",
  "TE1m": "Middle temporal — visual working memory",
  "TE1a": "Anterior middle temporal — meaning and language",
  "TE2p": "Posterior inferior temporal — tool and object recognition",
  "TE2a": "Anterior inferior temporal — meaning and visual processing",
  "TGv": "Ventral temporal pole — social and emotional meaning",
  "TGd": "Dorsal temporal pole — social cognition and faces",
  "TF": "Fusiform — face, object, and word recognition",
  // ── Temporo-Parieto-Occipital Junction ──
  "TPOJ1": "TPO junction — multi-sense integration hub",
  "TPOJ2": "TPO junction — body perception and action",
  "TPOJ3": "TPO junction — theory of mind and semantics",
  "STV": "Superior temporal visual — speech and face reading",
  "PSL": "Perisylvian language — speech monitoring",
  // ── Superior Parietal ──
  "LIPv": "Lateral intraparietal — reaching and pointing",
  "LIPd": "Lateral intraparietal — eye movement control",
  "VIP": "Ventral intraparietal — motion tracking, optic flow",
  "AIP": "Anterior intraparietal — grasping and object shape",
  "MIP": "Medial intraparietal — reaching and arm control",
  "7PC": "Postcentral — vision, motion, and movement execution",
  "7AL": "Anterior-lateral SPL — spatial working memory",
  "7Am": "Anterior-medial SPL — mental imagery, attention",
  "7PL": "Posterior-lateral SPL — episodic memory, eye movement",
  "7Pm": "Posterior-medial SPL — memory and spatial cognition",
  // ── Inferior Parietal ──
  "PGp": "Posterior angular gyrus — spatial sense and memory",
  "PGs": "Superior angular gyrus — biological motion perception",
  "PGi": "Inferior angular gyrus — narrative and face processing",
  "PFm": "Mid-supramarginal — attention and decision-making",
  "PF": "Supramarginal — action observation, mirror system",
  "PFt": "Anterior supramarginal — imitation and tool use",
  "PFop": "Opercular supramarginal — motor planning",
  "IP0": "Posterior intraparietal — number and spatial maps",
  "IP1": "Middle intraparietal — arithmetic and face processing",
  "IP2": "Anterior intraparietal — mental arithmetic",
  // ── Posterior Cingulate ──
  "DVT": "Dorsal visual transition — scene processing",
  "ProS": "Prostriate — rapid scene-to-spatial conversion",
  "POS1": "Parieto-occipital — early visual scene input",
  "POS2": "Parieto-occipital — visual scene relay",
  "RSC": "Retrosplenial cortex — landmarks and navigation",
  "v23ab": "Ventral posterior cingulate — episodic memory",
  "d23ab": "Dorsal posterior cingulate — episodic memory",
  "31pv": "Posterior cingulate — memory and reward",
  "31pd": "Posterior cingulate — memory encoding",
  "31a": "Anterior area 31 — spatial navigation and planning",
  "23d": "Dorsal area 23 — reward and goal navigation",
  "23c": "Cingulate motor — learning from action outcomes",
  "PCV": "Precuneus — spatial 'where' processing",
  "7m": "Medial area 7 — linking objects to memory",
  // ── Anterior Cingulate and Medial Prefrontal ──
  "33pr": "Anterior cingulate — autonomic and body control",
  "p24pr": "Posterior cingulate motor — task selection",
  "a24pr": "Anterior cingulate motor — response selection",
  "p24": "Posterior ACC — focused attention",
  "a24": "Anterior ACC — emotion and motivation",
  "p32pr": "Posterior area 32 — attention and language tasks",
  "a32pr": "Anterior area 32 — motivation and error encoding",
  "d32": "Dorsal area 32 — reward value and social judgment",
  "p32": "Medial prefrontal — social cognition, error monitoring",
  "s32": "Subcallosal area 32 — emotion and reward expectation",
  "8BM": "Medial area 8B — spatial coordination of vision",
  "9m": "Medial area 9 — spatial information monitoring",
  "10v": "Ventromedial prefrontal — value-based decisions",
  "10r": "Rostral area 10 — attention and working memory",
  "25": "Subgenual cortex — emotion and mood regulation",
  // ── Orbital and Polar Frontal ──
  "47s": "Orbital area 47 — meaning retrieval",
  "47m": "Medial orbital — emotion-guided decisions",
  "a47r": "Anterior area 47 — meaning and reward processing",
  "11l": "Lateral area 11 — smell and food reward",
  "13l": "Lateral area 13 — taste, smell, and fullness",
  "a10p": "Anterior frontopolar — abstract thinking",
  "p10p": "Posterior frontopolar — multitask coordination",
  "10pp": "Polar area 10 — complex working memory",
  "10d": "Dorsal area 10 — default mode, self-reflection",
  "OFC": "Orbitofrontal cortex — reward and self-regulation",
  "pOFC": "Posterior OFC — evaluating sensory rewards",
  // ── Inferior Frontal ──
  "44": "Broca's area (pars opercularis) — speech production",
  "45": "Broca's area (pars triangularis) — language meaning",
  "IFJp": "Posterior inferior frontal junction — attention hub",
  "IFJa": "Anterior inferior frontal junction — cognitive control",
  "IFSp": "Posterior inferior frontal sulcus — working memory",
  "IFSa": "Anterior inferior frontal sulcus — following instructions",
  "47l": "Lateral pars orbitalis — language and meaning",
  "p47r": "Posterior area 47 — complex word processing",
  // ── Dorsolateral Prefrontal ──
  "8C": "Lateral area 8 — spatial working memory",
  "8Av": "Ventral area 8A — visual attention and working memory",
  "i6-8": "Inferior transitional 6-8 — spatial maintenance",
  "s6-8": "Superior transitional 6-8 — motor-cognitive bridge",
  "SFL": "Superior frontal language — speech planning",
  "8BL": "Lateral area 8B — coordinating sight and sound in space",
  "9p": "Posterior area 9 — executive control and monitoring",
  "9a": "Anterior area 9 — working memory maintenance",
  "8Ad": "Dorsal area 8A — spatial hearing and vision",
  "p9-46v": "Posterior ventral 9-46 — goal-directed cognition",
  "a9-46v": "Anterior ventral 9-46 — prefrontal control",
  "46": "Dorsolateral PFC — planning and cognitive control",
  "9-46d": "Dorsal 9-46 — conscious behavioral control",
}

export default function RegionPanel() {
  const regions = useStore((s) => s.regions)
  const coarseGroups = useStore((s) => s.coarseGroups)
  const timestep = useStore((s) => s.timestep)
  const globalVmax = useStore((s) => s.globalVmax)
  const isPlaying = useStore((s) => s.isPlaying)
  const selectedRegion = useStore((s) => s.selectedRegion)
  const setSelectedRegion = useStore((s) => s.setSelectedRegion)

  const scaleMax = globalVmax || 1

  // Throttle the sort to ~4 updates/sec during playback
  const [entries, setEntries] = useState([])
  useEffect(() => {
    if (!regions) { setEntries([]); return }
    const id = setTimeout(() => {
      setEntries(
        Object.entries(regions)
          .map(([name, values]) => ({ name, value: values[timestep] ?? 0 }))
          .sort((a, b) => b.value - a.value)
      )
    }, isPlaying ? 250 : 0)
    return () => clearTimeout(id)
  }, [regions, timestep, isPlaying])

  if (!regions) {
    return (
      <div className="h-full bg-gray-950 p-4 flex items-center justify-center text-gray-600 text-sm">
        Run a prediction to see region scores
      </div>
    )
  }

  const clickable = !isPlaying

  return (
    <div className="h-full bg-gray-950 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="shrink-0 px-5 pt-3 pb-2 flex items-center justify-between border-b border-gray-800/50">
        <span className="text-[11px] text-gray-400 font-semibold tracking-wide uppercase">
          Region Activity
        </span>
        {clickable && (
          <span className="text-[10px] text-gray-600">
            {selectedRegion ? `Focused: ${selectedRegion}` : 'Click to focus'}
          </span>
        )}
      </div>

      {/* Two-column scrollable grid */}
      <div className="flex-1 overflow-y-auto px-3 py-2">
        <div className="grid grid-cols-2 gap-x-2 gap-y-0.5">
          {entries.map(({ name, value }, i) => {
            const coarse = coarseGroups?.[name] || ''
            const color = COARSE_COLORS[coarse] || '#6b7280'
            const pct = Math.max(0, Math.min(100, (value / scaleMax) * 100))
            const desc = REGION_DESCRIPTIONS[name]
            const isSelected = selectedRegion === name

            // Split description at " — " to show function part prominently
            let label, detail
            if (desc && desc.includes(' — ')) {
              const idx = desc.indexOf(' — ')
              label = desc.slice(idx + 3)   // function part (after —)
              detail = desc.slice(0, idx)    // anatomy part (before —)
            } else {
              label = desc || name
              detail = null
            }

            return (
              <div
                key={name}
                onClick={() => {
                  if (!clickable) return
                  setSelectedRegion(isSelected ? null : name)
                }}
                className={`flex items-center gap-1.5 px-2 py-1 rounded transition-colors ${
                  clickable ? 'cursor-pointer hover:bg-gray-800/60' : ''
                } ${isSelected ? 'bg-gray-800 ring-1 ring-red-500/40' : ''}`}
              >
                {/* Color dot */}
                <span
                  className="w-1.5 h-1.5 rounded-full shrink-0"
                  style={{ backgroundColor: color, opacity: Math.max(0.3, pct / 100) }}
                />

                {/* Labels: function description prominent, area code dimmed */}
                <div className="flex-1 min-w-0 flex items-baseline gap-1.5">
                  <span className="text-[11px] text-gray-200 truncate leading-tight">
                    {label}
                  </span>
                  <span className="text-[9px] text-gray-600 shrink-0 font-mono">
                    {name}
                  </span>
                </div>

                {/* Bar + percentage */}
                <div className="w-14 shrink-0 flex items-center gap-1">
                  <div className="flex-1 h-1 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${pct}%`, backgroundColor: color }}
                    />
                  </div>
                  <span className="text-[9px] text-gray-500 font-mono w-6 text-right">
                    {pct.toFixed(0)}%
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
