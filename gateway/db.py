"""SQLite request log.

Every routing decision is persisted: what came in, what the classifier thought, which
model served it, what it cost, and what it *would* have cost on the expensive model.
That last column is what makes the savings figure defensible -- it is a measured
counterfactual on real traffic, not an estimate.

Writes go through `asyncio.to_thread` so a disk hiccup cannot block the event loop.
"""

from __future__ import annotations

import asyncio
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from gateway.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                 TEXT    NOT NULL,
    query              TEXT    NOT NULL,
    predicted_label    TEXT    NOT NULL,
    confidence         REAL    NOT NULL,
    model              TEXT    NOT NULL,
    reason             TEXT    NOT NULL,
    provider           TEXT    NOT NULL,
    input_tokens       INTEGER NOT NULL,
    output_tokens      INTEGER NOT NULL,
    cost_usd           REAL    NOT NULL,
    baseline_cost_usd  REAL    NOT NULL,
    latency_ms         REAL    NOT NULL,
    model_version      TEXT,
    escalated          INTEGER NOT NULL DEFAULT 0,
    cached             INTEGER NOT NULL DEFAULT 0,
    redactions         INTEGER NOT NULL DEFAULT 0,
    flagged            INTEGER NOT NULL DEFAULT 0,
    synthetic          INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_requests_ts ON requests(ts);
"""


@dataclass
class RequestRecord:
    query: str
    predicted_label: str
    confidence: float
    model: str
    reason: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    baseline_cost_usd: float
    latency_ms: float
    model_version: str | None = None
    escalated: bool = False
    cached: bool = False
    redactions: int = 0
    flagged: bool = False
    #: True for rows injected by /admin/simulate-drift, so the demo can be reset
    #: without wiping genuine traffic.
    synthetic: bool = False
    ts: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@contextmanager
def _connect(path: Path | None = None):
    target = Path(path or settings.db_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | None = None) -> None:
    with _connect(path) as conn:
        conn.executescript(SCHEMA)


def _insert(record: RequestRecord) -> None:
    data = asdict(record)
    for key in ("escalated", "cached", "flagged", "synthetic"):
        data[key] = int(data[key])
    columns = ", ".join(data)
    placeholders = ", ".join(f":{k}" for k in data)
    with _connect() as conn:
        conn.execute(f"INSERT INTO requests ({columns}) VALUES ({placeholders})", data)


async def log_request(record: RequestRecord) -> None:
    await asyncio.to_thread(_insert, record)


def recent(limit: int = 25) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM requests ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def stats() -> dict[str, Any]:
    """Aggregates for the dashboard hero strip."""
    with _connect() as conn:
        totals = conn.execute(
            """
            SELECT COUNT(*)                        AS requests,
                   COALESCE(SUM(cost_usd), 0)          AS spent,
                   COALESCE(SUM(baseline_cost_usd), 0) AS baseline,
                   COALESCE(SUM(escalated), 0)         AS escalations,
                   COALESCE(SUM(cached), 0)            AS cache_hits,
                   COALESCE(SUM(redactions), 0)        AS redactions,
                   COALESCE(SUM(flagged), 0)           AS flagged
            FROM requests
            """
        ).fetchone()

        by_model = conn.execute(
            """
            SELECT model,
                   COUNT(*)        AS n,
                   AVG(latency_ms) AS avg_latency_ms
            FROM requests GROUP BY model
            """
        ).fetchall()

        by_label = conn.execute(
            "SELECT predicted_label, COUNT(*) AS n FROM requests GROUP BY predicted_label"
        ).fetchall()

        latencies = [
            row["latency_ms"]
            for row in conn.execute("SELECT latency_ms FROM requests ORDER BY latency_ms")
        ]

    spent, baseline = totals["spent"], totals["baseline"]
    requests = totals["requests"]

    return {
        "requests": requests,
        "spent_usd": spent,
        "baseline_usd": baseline,
        "saved_usd": baseline - spent,
        "savings_pct": (100.0 * (1 - spent / baseline)) if baseline else 0.0,
        "escalations": totals["escalations"],
        "cache_hits": totals["cache_hits"],
        "cache_hit_rate_pct": (100.0 * totals["cache_hits"] / requests) if requests else 0.0,
        "redactions": totals["redactions"],
        "flagged": totals["flagged"],
        "by_model": [dict(r) for r in by_model],
        "by_label": [dict(r) for r in by_label],
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
    }


def _percentile(sorted_values: list[float], pct: int) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, int(round((pct / 100) * (len(sorted_values) - 1))))
    return float(sorted_values[index])


def recent_queries(limit: int = 300) -> list[str]:
    """Live query texts, newest first -- the current window for drift detection."""
    with _connect() as conn:
        rows = conn.execute("SELECT query FROM requests ORDER BY id DESC LIMIT ?", (limit,))
        return [row["query"] for row in rows]


def cost_timeline(limit: int = 200) -> list[dict[str, Any]]:
    """Cumulative spend vs counterfactual, oldest first, for the savings chart."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT ts, cost_usd, baseline_cost_usd FROM requests
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()

    spent = baseline = 0.0
    timeline = []
    for row in reversed(rows):
        spent += row["cost_usd"]
        baseline += row["baseline_cost_usd"]
        timeline.append({"ts": row["ts"], "spent": spent, "saved": baseline - spent})
    return timeline
