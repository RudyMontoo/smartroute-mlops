"""Model and feature ablation, logged to MLflow.

    python -m router.sweep

Trains every (estimator, feature set) combination and logs each as its own MLflow
run -- including the ones that lose. The losing runs are the point. A single training
run tells you nothing about whether the choices were any good; six runs sitting side by
side in the experiment UI show *why* the shipped configuration was chosen, and the
TF-IDF-only arm answers the question of whether the hand-engineered numeric features
are earning their place at all.

Runs here are intentionally not registered. The registry should contain models that
were actually promoted to serve traffic, not every experiment that was ever tried.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow

from router import policy
from router.features import build_frame
from router.train import (
    DEFAULT_TRACKING_URI,
    ESTIMATORS,
    build_pipeline,
    evaluate_pipeline,
    load_split,
)

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "router-ablation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=policy.DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    args = parser.parse_args()

    X_train, X_val, y_train, y_val = load_split(seed=args.seed)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    results = []
    for estimator in sorted(ESTIMATORS):
        for use_numeric in (True, False):
            arm = "tfidf+numeric" if use_numeric else "tfidf-only"
            run_name = f"{estimator}/{arm}"

            pipe = build_pipeline(estimator, use_numeric=use_numeric)
            pipe.fit(build_frame(X_train), y_train)
            metrics = evaluate_pipeline(pipe, X_val, y_val, args.threshold)

            scalars = {
                k: v
                for k, v in metrics.items()
                if isinstance(v, int | float) and k not in {"confusion_matrix"}
            }

            with mlflow.start_run(run_name=run_name):
                mlflow.log_params(
                    {
                        "estimator": estimator,
                        "use_numeric_features": use_numeric,
                        "feature_set": arm,
                        "threshold": args.threshold,
                        "seed": args.seed,
                    }
                )
                mlflow.log_metrics(scalars)
                mlflow.log_text(metrics["report"], "classification_report.txt")

            results.append({"estimator": estimator, "feature_set": arm, **scalars})
            print(
                f"{run_name:24s} macro_f1={metrics['macro_f1']:.4f} "
                f"acc={metrics['accuracy']:.4f} savings={metrics['cost_savings_pct']:.1f}%"
            )

    results.sort(key=lambda r: r["macro_f1"], reverse=True)
    best = results[0]

    out = ROOT / "eval" / "results" / "ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"threshold": args.threshold, "runs": results}, indent=2))

    print("\n--- ranked by macro_f1 ---")
    for rank, row in enumerate(results, 1):
        print(
            f"{rank}. {row['estimator']:8s} {row['feature_set']:14s} "
            f"macro_f1={row['macro_f1']:.4f}  savings={row['cost_savings_pct']:.1f}%"
        )

    numeric = {r["estimator"]: r for r in results if r["feature_set"] == "tfidf+numeric"}
    tfidf = {r["estimator"]: r for r in results if r["feature_set"] == "tfidf-only"}
    print("\n--- do the hand-engineered features earn their place? ---")
    for estimator in sorted(numeric):
        delta = numeric[estimator]["macro_f1"] - tfidf[estimator]["macro_f1"]
        verdict = "helps" if delta > 0.005 else ("hurts" if delta < -0.005 else "no effect")
        print(f"{estimator:8s} {delta:+.4f} macro_f1 from numeric features -> {verdict}")

    print(f"\nbest: {best['estimator']} / {best['feature_set']} ({best['macro_f1']:.4f})")
    print(f"wrote {out.relative_to(ROOT)}  |  mlflow experiment '{EXPERIMENT}'")


if __name__ == "__main__":
    main()
