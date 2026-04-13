export function hotColor(t) {
  t = Math.max(0, Math.min(1, t))
  const r = Math.min(1, t * 2.5)
  const g = Math.max(0, Math.min(1, (t - 0.4) * 2.5))
  const b = Math.max(0, Math.min(1, (t - 0.7) * 3.33))
  return [r, g, b]
}

export function activationsToColors(activations, vmin, vmax) {
  const n = activations.length
  const colors = new Float32Array(n * 3)
  const range = vmax - vmin || 1
  for (let i = 0; i < n; i++) {
    const t = (activations[i] - vmin) / range
    const [r, g, b] = hotColor(t)
    colors[i * 3] = r
    colors[i * 3 + 1] = g
    colors[i * 3 + 2] = b
  }
  return colors
}
