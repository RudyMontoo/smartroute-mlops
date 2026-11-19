"""Tests for the CI model-quality gate.

A gate that silently passes is worse than no gate, because it manufactures confidence.
These tests pin down that it actually fails on each metric it claims to guard.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GOOD = {"macro_f1": 0.93, "accuracy": 0.93, "cost_savings_pct": 31.4}


def _run(*args: str) -> subprocess.CompletedProcess:
    # cwd is pinned to the repo root so `-m scripts.quality_gate` resolves no matter
    # where pytest itself was invoked from.
    return subprocess.run(
        [sys.executable, "-m", "scripts.quality_gate", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def run_gate(tmp_path, metrics: dict, *extra: str) -> subprocess.CompletedProcess:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics))
    return _run("--metrics", str(path), *extra)


def test_a_healthy_model_passes(tmp_path):
    assert run_gate(tmp_path, GOOD).returncode == 0


def test_an_accuracy_regression_fails_the_build(tmp_path):
    result = run_gate(tmp_path, {**GOOD, "macro_f1": 0.40})
    assert result.returncode == 1
    assert "macro_f1" in result.stdout


def test_a_savings_regression_fails_even_when_accuracy_is_fine(tmp_path):
    """The regression this gate exists for: the classifier is still accurate, but the
    routing has stopped saving money. No ordinary test suite would notice."""
    result = run_gate(tmp_path, {**GOOD, "cost_savings_pct": 2.0})
    assert result.returncode == 1
    assert "cost_savings_pct" in result.stdout


def test_missing_metrics_fail_closed(tmp_path):
    """If training never ran, the gate must not wave the build through."""
    assert _run("--metrics", str(tmp_path / "nope.json")).returncode == 1


def test_floors_are_configurable(tmp_path):
    assert run_gate(tmp_path, GOOD, "--min-f1", "0.99").returncode == 1
