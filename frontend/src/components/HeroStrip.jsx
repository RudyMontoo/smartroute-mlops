const fmtUsd = (n) => `$${(n ?? 0).toFixed(4)}`
const fmtPct = (n) => `${(n ?? 0).toFixed(1)}%`

function Stat({ label, value, accent }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${accent ?? ''}`}>{value}</div>
    </div>
  )
}

export function HeroStrip({ stats, modelInfo }) {
  if (!stats) return <div className="card hero">loading…</div>

  return (
    <div className="card hero">
      <Stat label="Requests served" value={stats.requests ?? 0} />
      <Stat label="Cost saved" value={fmtUsd(stats.saved_usd)} accent="good" />
      <Stat label="Savings vs always-expensive" value={fmtPct(stats.savings_pct)} accent="good" />
      <Stat
        label="Escalation rate"
        value={fmtPct(stats.requests ? (100 * stats.escalations) / stats.requests : 0)}
      />
      <Stat label="Cache hit rate" value={fmtPct(stats.cache_hit_rate_pct)} />
      <Stat label="PII redactions" value={stats.redactions ?? 0} />
      <Stat
        label="Serving model"
        value={modelInfo ? `${modelInfo.alias} v${modelInfo.version}` : '—'}
        accent="muted"
      />
    </div>
  )
}
