const BASE = '/api'

export async function fetchJSON(path, opts = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export async function getUploadUrl(filename, contentType) {
  return fetchJSON('/upload', {
    method: 'POST',
    body: JSON.stringify({ filename, content_type: contentType }),
  })
}

export async function startPredict(s3Key, inputType) {
  return fetchJSON('/predict', {
    method: 'POST',
    body: JSON.stringify({ s3_key: s3Key, input_type: inputType }),
  })
}

export async function getResults(jobId) {
  return fetchJSON(`/results/${jobId}`)
}

export async function getMesh() {
  return fetchJSON('/mesh')
}

export async function getRuns() {
  return fetchJSON('/runs')
}
