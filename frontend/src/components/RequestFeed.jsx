const MODEL_SHORT = {
  'openai/gpt-oss-20b': 'cheap',
  'openai/gpt-oss-120b': 'expensive',
}

function shortModel(model) {
  return MODEL_SHORT[model] ?? model
}

function timeAgo(iso) {
  const seconds = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 1000))
  if (seconds < 60) return `${seconds}s ago`
  return `${Math.round(seconds / 60)}m ago`
}

export function RequestFeed({ rows }) {
  return (
    <div className="card">
      <h2>Live request feed</h2>
      <div className="feed">
        <div className="feed-row feed-header">
          <span>query</span>
          <span>difficulty</span>
          <span>model</span>
          <span>flags</span>
          <span>latency</span>
          <span>cost</span>
          <span>when</span>
        </div>
        {rows.length === 0 && <div className="feed-empty">No requests yet — send one below.</div>}
        {rows.map((row) => (
          <div className="feed-row" key={row.id}>
            <span className="truncate" title={row.query}>
              {row.query}
            </span>
            <span className={`pill pill-${row.predicted_label}`}>{row.predicted_label}</span>
            <span className={`pill pill-model-${shortModel(row.model)}`}>
              {shortModel(row.model)}
            </span>
            <span className="flags">
              {!!row.escalated && <span title="escalated to the expensive model">⬆ esc</span>}
              {!!row.cached && <span title="served from cache">⚡ cache</span>}
              {!!row.redactions && <span title="PII redacted">🔒 {row.redactions}</span>}
              {!!row.flagged && <span title="prompt-injection flagged">⚠ flag</span>}
            </span>
            <span>{row.latency_ms.toFixed(0)}ms</span>
            <span>${row.cost_usd.toFixed(5)}</span>
            <span className="muted">{timeAgo(row.ts)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
