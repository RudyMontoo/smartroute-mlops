"""Prometheus collectors.

Scraped by the Prometheus container and rendered in Grafana. Model-level counters
(routing decisions, escalations, drift score) sit alongside the usual latency
histogram, because for an ML service the interesting failures are in the model's
behaviour, not just in the HTTP layer.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "smartroute_requests_total",
    "Requests served, by routed model and provider.",
    ["model", "provider"],
)

ROUTING_DECISIONS = Counter(
    "smartroute_routing_decisions_total",
    "Routing decisions, by the difficulty class the router predicted.",
    ["predicted_label", "model"],
)

ESCALATIONS = Counter(
    "smartroute_escalations_total",
    "Cheap answers rejected by the cascade verifier and retried on the large model.",
    ["trigger"],
)

CACHE_EVENTS = Counter(
    "smartroute_cache_events_total",
    "Response cache outcomes.",
    ["outcome"],  # hit | miss
)

PII_REDACTIONS = Counter(
    "smartroute_pii_redactions_total",
    "PII spans masked before the query left for a third-party provider.",
    ["kind"],
)

INJECTION_FLAGS = Counter(
    "smartroute_injection_flags_total",
    "Requests flagged by the prompt-injection heuristic.",
)

PROVIDER_FALLBACKS = Counter(
    "smartroute_provider_fallbacks_total",
    "Times the upstream provider failed and the mock served the request instead.",
)

LATENCY = Histogram(
    "smartroute_request_latency_seconds",
    "End-to-end request latency.",
    ["model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

COST = Counter(
    "smartroute_cost_usd_total",
    "Cumulative spend, and what the always-expensive baseline would have cost.",
    ["kind"],  # actual | baseline
)

DRIFT_SCORE = Gauge(
    "smartroute_drift_psi",
    "Population Stability Index of a monitored feature against its training reference.",
    ["feature"],
)

MODEL_INFO = Gauge(
    "smartroute_model_version_info",
    "Always 1; the labels carry the serving model's identity.",
    ["version", "alias", "source"],
)
