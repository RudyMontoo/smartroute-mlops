import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import { HeroStrip } from './components/HeroStrip'
import { RequestFeed } from './components/RequestFeed'
import {
  CostSavedChart,
  DriftChart,
  LatencyByModelChart,
  RoutingDistributionChart,
} from './components/Charts'
import { ControlPanel } from './components/ControlPanel'
import { useInterval } from './useInterval'
import './App.css'

const POLL_MS = 2000

export default function App() {
  const [stats, setStats] = useState(null)
  const [recent, setRecent] = useState([])
  const [health, setHealth] = useState(null)
  const [driftHistory, setDriftHistory] = useState([])
  const [error, setError] = useState(null)

  const refresh = useCallback(async () => {
    try {
      const [statsRes, recentRes, healthRes] = await Promise.all([
        api.stats(),
        api.recent(30),
        api.health(),
      ])
      setStats(statsRes)
      setRecent(recentRes)
      setHealth(healthRes)
      setError(null)
    } catch (err) {
      setError(String(err.message ?? err))
    }
  }, [])

  const refreshDrift = useCallback(async () => {
    try {
      const drift = await api.drift()
      setDriftHistory((prev) => [...prev, { psi: drift.max_psi ?? drift.psi ?? 0 }].slice(-40))
    } catch {
      // /drift ships in a later block; the chart just stays empty until then.
    }
  }, [])

  useEffect(() => {
    refresh()
    refreshDrift()
  }, [refresh, refreshDrift])

  useInterval(refresh, POLL_MS)
  useInterval(refreshDrift, POLL_MS * 2)

  return (
    <div className="app">
      <header>
        <h1>SmartRoute</h1>
        <span className="subtitle">LLM gateway with learned model routing</span>
        {error && <span className="error-banner">⚠ {error}</span>}
      </header>

      <HeroStrip stats={stats} modelInfo={health?.model} />

      <div className="grid">
        <RequestFeed rows={recent} />
        <ControlPanel onRequestServed={refresh} onRefreshDrift={refreshDrift} />
      </div>

      <div className="grid charts-grid">
        <CostSavedChart timeline={stats?.timeline ?? []} />
        <LatencyByModelChart byModel={stats?.by_model ?? []} />
        <RoutingDistributionChart byLabel={stats?.by_label ?? []} />
        <DriftChart history={driftHistory} />
      </div>

      <footer>
        provider: {health?.provider ?? '—'} · threshold: {health?.threshold ?? '—'} · model:{' '}
        {health?.model
          ? `${health.model.alias} v${health.model.version} (${health.model.source})`
          : '—'}
      </footer>
    </div>
  )
}
