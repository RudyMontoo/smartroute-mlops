"""Fail the build when the model regresses.

    python -m scripts.quality_gate            # reads models/metrics.json
    python -m scripts.quality_gate --strict   # also fails on a savings regression

Ordinary CI asks "does the code still work". For an ML service that is not enough: a
change to the featurizer, the data, or the routing policy can leave every test green
while quietly making the model worse or more expensive. This gate is what turns the
pipeline from "tests pass" into "the model is still good enough to deploy".

The floors are set below the measured baseline with deliberate headroom -- tight enough
to catch a real regression, loose enough not to fail on run-to-run noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "models" / "metrics.json"

# Baseline at the time of writing: macro_f1 0.9285, cost savings 31.4%.
MIN_MACRO_F1 = 0.85
MIN_COST_SAVINGS_PCT = 25.0
MIN_ACCURACY = 0.85


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--min-f1", type=float, default=MIN_MACRO_F1)
    parser.add_argument("--min-savings", type=float, default=MIN_COST_SAVINGS_PCT)
    parser.add_argument("--min-accuracy", type=float, default=MIN_ACCURACY)
    args = parser.parse_args()

    if not args.metrics.exists():
        print(f"FAIL: no metrics at {args.metrics}. Run `python -m router.train` first.")
        return 1

    metrics = json.loads(args.metrics.read_text())

    checks = [
        ("macro_f1", metrics.get("macro_f1", 0.0), args.min_f1),
        ("accuracy", metrics.get("accuracy", 0.0), args.min_accuracy),
        ("cost_savings_pct", metrics.get("cost_savings_pct", 0.0), args.min_savings),
    ]

    failures = []
    print("Model quality gate")
    print("-" * 52)
    for name, actual, floor in checks:
        ok = actual >= floor
        print(f"  {'PASS' if ok else 'FAIL'}  {name:18s} {actual:8.4f}  (floor {floor})")
        if not ok:
            failures.append(f"{name}={actual:.4f} below floor {floor}")

    print("-" * 52)
    if failures:
        print("QUALITY GATE FAILED -- not deployable:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("QUALITY GATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
