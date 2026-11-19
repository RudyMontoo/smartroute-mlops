"""End-to-end tests for the gateway HTTP surface."""

from __future__ import annotations

import pytest

from router import policy

SIMPLE = "Who wrote the novel Pride and Prejudice?"
COMPLEX = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether? Show your working step by step."
)


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"]["version"]
    assert body["provider"] == "mock"


def test_simple_query_is_routed_to_the_cheap_model(client):
    body = client.post("/v1/chat", json={"query": SIMPLE}).json()
    assert body["routing"]["predicted_label"] == "simple"
    assert body["routing"]["model"] == policy.CHEAP
    assert body["answer"]


def test_complex_query_is_routed_to_the_expensive_model(client):
    body = client.post("/v1/chat", json={"query": COMPLEX}).json()
    assert body["routing"]["predicted_label"] == "complex"
    assert body["routing"]["model"] == policy.EXPENSIVE


def test_cheap_routing_actually_costs_less_than_the_baseline(client):
    usage = client.post("/v1/chat", json={"query": SIMPLE}).json()["usage"]
    assert usage["cost_usd"] < usage["baseline_cost_usd"]
    assert usage["saved_usd"] == pytest.approx(
        usage["baseline_cost_usd"] - usage["cost_usd"], rel=1e-9
    )


def test_expensive_routing_saves_nothing_and_does_not_pretend_to(client):
    """A complex query goes to the big model, so the counterfactual is itself.

    Worth pinning down: an off-by-one in the cost accounting here would silently
    inflate the headline savings number the whole project is judged on.
    """
    usage = client.post("/v1/chat", json={"query": COMPLEX}).json()["usage"]
    assert usage["saved_usd"] == pytest.approx(0.0, abs=1e-12)


def test_threshold_override_pushes_ambiguous_queries_to_the_big_model(client):
    """At threshold=1.0 no confidence can clear the bar, so every `moderate` query
    must escalate. Guards the one branch of the policy that reads from the request."""
    query = "What are some good strategies for learning a musical instrument as an adult?"
    body = client.post("/v1/chat", json={"query": query, "threshold": 1.0}).json()
    if body["routing"]["predicted_label"] == "moderate":
        assert body["routing"]["model"] == policy.EXPENSIVE


def test_empty_query_is_rejected(client):
    assert client.post("/v1/chat", json={"query": ""}).status_code == 422


def test_stats_accumulate_across_requests(client):
    client.post("/v1/chat", json={"query": SIMPLE})
    stats = client.get("/stats").json()
    assert stats["requests"] > 0
    assert stats["baseline_usd"] >= stats["spent_usd"]
    assert stats["saved_usd"] >= 0


def test_recent_feed_returns_newest_first(client):
    client.post("/v1/chat", json={"query": COMPLEX})
    rows = client.get("/requests/recent", params={"limit": 5}).json()
    assert rows
    assert [row["id"] for row in rows] == sorted((row["id"] for row in rows), reverse=True)


def test_metrics_endpoint_exposes_prometheus_counters(client):
    text = client.get("/metrics").text
    assert "smartroute_requests_total" in text
    assert "smartroute_cost_usd_total" in text
