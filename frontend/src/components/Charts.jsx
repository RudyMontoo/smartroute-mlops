import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const COLORS = ['#6366f1', '#f59e0b', '#10b981', '#ef4444']

function ChartCard({ title, children }) {
  return (
    <div className="card chart-card">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={220}>
        {children}
      </ResponsiveContainer>
    </div>
  )
}

export function CostSavedChart({ timeline }) {
  const data = timeline.map((point, i) => ({ i, saved: point.saved, spent: point.spent }))
  return (
    <ChartCard title="Cumulative cost saved">
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
        <XAxis dataKey="i" tick={false} />
        <YAxis tickFormatter={(v) => `$${v.toFixed(3)}`} width={60} />
        <Tooltip formatter={(v) => `$${v.toFixed(5)}`} />
        <Area type="monotone" dataKey="saved" stroke="#10b981" fill="#10b98133" />
        <Area type="monotone" dataKey="spent" stroke="#6366f1" fill="#6366f133" />
      </AreaChart>
    </ChartCard>
  )
}

export function LatencyByModelChart({ byModel }) {
  const data = byModel.map((m) => ({
    model: m.model.includes('120b') ? 'expensive' : 'cheap',
    ms: Math.round(m.avg_latency_ms),
  }))
  return (
    <ChartCard title="Avg latency per model">
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
        <XAxis dataKey="model" />
        <YAxis width={40} />
        <Tooltip formatter={(v) => `${v}ms`} />
        <Bar dataKey="ms" fill="#6366f1" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ChartCard>
  )
}

export function RoutingDistributionChart({ byLabel }) {
  const data = byLabel.map((l) => ({ name: l.predicted_label, value: l.n }))
  return (
    <ChartCard title="Routing distribution">
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" outerRadius={80} label>
          {data.map((entry, i) => (
            <Cell key={entry.name} fill={COLORS[i % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip />
      </PieChart>
    </ChartCard>
  )
}

export function DriftChart({ history, threshold = 0.2 }) {
  const data = history.map((point, i) => ({ i, psi: point.psi }))
  return (
    <ChartCard title="Drift score (PSI) vs threshold">
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2f3a" />
        <XAxis dataKey="i" tick={false} />
        <YAxis width={40} />
        <Tooltip />
        <Line type="monotone" dataKey="psi" stroke="#f59e0b" dot={false} />
        <Line
          type="monotone"
          dataKey={() => threshold}
          stroke="#ef4444"
          strokeDasharray="4 4"
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ChartCard>
  )
}
