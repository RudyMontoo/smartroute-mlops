import { useState } from 'react'
import { api } from '../api'

export function ControlPanel({ onRequestServed, onRefreshDrift }) {
  const [query, setQuery] = useState(
    'How many clips did Natalia sell altogether if she sold 48 in April and half as many in May?',
  )
  const [threshold, setThreshold] = useState(0.6)
  const [busy, setBusy] = useState(null)
  const [log, setLog] = useState([])

  function note(text, tone = 'info') {
    setLog((prev) => [{ text, tone, id: Date.now() }, ...prev].slice(0, 6))
  }

  async function sendQuery() {
    if (!query.trim()) return
    setBusy('chat')
    try {
      const result = await api.chat(query, { threshold })
      const { routing, usage } = result
      note(
        `routed to ${routing.model.includes('120b') ? 'expensive' : 'cheap'} ` +
          `(${routing.predicted_label}, conf ${routing.confidence.toFixed(2)})` +
          `${routing.escalated ? ' → escalated' : ''} — $${usage.cost_usd.toFixed(5)}`,
        'ok',
      )
      onRequestServed?.()
    } catch (err) {
      note(String(err.message ?? err), 'error')
    } finally {
      setBusy(null)
    }
  }

  async function simulateDrift() {
    setBusy('drift')
    try {
      const result = await api.simulateDrift()
      note(`drift simulation injected (${result.injected ?? '?'} queries)`, 'ok')
      onRefreshDrift?.()
    } catch (err) {
      note('simulate-drift not available yet', 'error')
    } finally {
      setBusy(null)
    }
  }

  async function retrain() {
    setBusy('retrain')
    try {
      const result = await api.retrain()
      note(
        result.promoted
          ? `retrained → v${result.version} promoted to @champion`
          : `retrained → v${result.version} did not beat champion`,
        'ok',
      )
    } catch (err) {
      note('retrain not available yet', 'error')
    } finally {
      setBusy(null)
    }
  }

  return (
    <div className="card control-panel">
      <h2>Control panel</h2>

      <label className="field-label" htmlFor="query">
        Test query
      </label>
      <textarea
        id="query"
        rows={2}
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask anything…"
      />

      <div className="row">
        <label className="field-label" htmlFor="threshold">
          Confidence threshold ({threshold.toFixed(2)})
        </label>
        <input
          id="threshold"
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={threshold}
          onChange={(e) => setThreshold(Number(e.target.value))}
        />
      </div>

      <div className="button-row">
        <button onClick={sendQuery} disabled={busy !== null}>
          {busy === 'chat' ? 'Sending…' : 'Send query'}
        </button>
        <button onClick={simulateDrift} disabled={busy !== null} className="secondary">
          {busy === 'drift' ? 'Injecting…' : 'Simulate drift'}
        </button>
        <button onClick={retrain} disabled={busy !== null} className="secondary">
          {busy === 'retrain' ? 'Retraining…' : 'Retrain router'}
        </button>
      </div>

      <div className="log">
        {log.map((entry) => (
          <div key={entry.id} className={`log-line log-${entry.tone}`}>
            {entry.text}
          </div>
        ))}
      </div>
    </div>
  )
}
