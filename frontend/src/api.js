// Thin wrapper around the gateway's REST surface. Relative paths work in both dev
// (proxied by Vite, see vite.config.js) and prod (served by nginx alongside the SPA).

async function request(path, options) {
  const res = await fetch(path, options)
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`${path} -> ${res.status} ${body}`.slice(0, 300))
  }
  return res.json()
}

export const api = {
  health: () => request('/health'),
  stats: () => request('/stats'),
  recent: (limit = 25) => request(`/requests/recent?limit=${limit}`),
  chat: (query, { threshold, noCascade } = {}) =>
    request('/v1/chat', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ query, threshold, no_cascade: noCascade }),
    }),
  drift: () => request('/drift'),
  simulateDrift: () => request('/admin/simulate-drift', { method: 'POST' }),
  retrain: () => request('/admin/retrain', { method: 'POST' }),
}
