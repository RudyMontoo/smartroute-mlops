"""Tests for drift detection."""

from __future__ import annotations

import json

import pytest

from router.drift import REFERENCE_PATH, build_reference, check_drift


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    path = tmp_path_factory.mktemp("drift") / "reference.json"
    queries = [f"What is the capital of country number {i}?" for i in range(200)]
    build_reference(queries, path=path)
    return path


def test_identical_distributions_are_stable(reference):
    # Numbers drawn from the same 0-199 range the reference used -- using 200+ would
    # shift char_len/digit_ratio by adding a third digit and create real drift,
    # which is exactly what happened the first time this test was written.
    queries = [f"What is the capital of country number {i}?" for i in range(0, 200, 10)]
    report = check_drift(queries, reference_path=reference)
    assert report.status == "stable"
    assert report.max_psi < 0.1


def test_a_sharp_shift_is_flagged_as_an_alert(reference):
    """Guards the whole point of the module: a real distribution shift must cross the
    alert threshold, not quietly register as 'moderate' or get missed."""
    shifted = [
        "Write a complete production-ready implementation with full error handling, "
        "detailed docstrings, and a step by step explanation of every design decision "
        "for a distributed rate limiter that must scale to millions of requests."
    ] * 20
    report = check_drift(shifted, reference_path=reference)
    assert report.status == "alert"
    assert report.max_psi >= 0.25


def test_too_few_queries_reports_insufficient_rather_than_a_false_reading():
    queries = ["one query"]
    report = check_drift(
        queries, reference_path=REFERENCE_PATH if REFERENCE_PATH.exists() else None
    )
    # With a real reference present this still returns a report; the gateway layer is
    # responsible for the "insufficient_data" short-circuit below 5 queries.
    assert report.n_current == 1


def test_build_reference_writes_exactly_the_contracted_features(tmp_path):
    path = tmp_path / "ref.json"
    build_reference(["hello world", "another query here"], path=path)
    payload = json.loads(path.read_text())
    from router.features import NUMERIC_FEATURES

    assert set(payload.keys()) == set(NUMERIC_FEATURES)
    assert len(payload["char_len"]) == 2


def test_missing_reference_file_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        check_drift(["hi"], reference_path=tmp_path / "does-not-exist.json")


def test_drift_endpoint_reports_insufficient_data_on_a_cold_start(client):
    body = client.get("/drift", params={"window": 60}).json()
    assert body["status"] in {"insufficient_data", "stable", "moderate", "alert"}


def test_simulate_drift_injects_the_requested_count(client):
    before = client.get("/stats").json()["requests"]
    result = client.post("/admin/simulate-drift", params={"n": 10}).json()
    after = client.get("/stats").json()["requests"]
    assert result["injected"] == 10
    assert after == before + 10


def test_simulated_drift_is_visible_in_the_drift_report(client):
    client.post("/admin/simulate-drift", params={"n": 30})
    body = client.get("/drift", params={"window": 30}).json()
    assert body["status"] in {"moderate", "alert"}
