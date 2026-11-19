"""Train the query-difficulty router, log to MLflow, and promote it if it earns it.

Run directly for a single training run:

    python -m router.train

The model is intentionally small (TF-IDF + a linear classifier) and trains in a few
seconds. That is not a shortcut, it is what makes the rest of the project possible: a
model that takes 20 minutes to fit cannot be retrained inside a CI job, cannot be
promoted live during a demo, and cannot be hot-swapped without a restart.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from router import policy
from router.features import build_frame, build_vectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "queries.csv"
MODEL_DIR = ROOT / "models"
LABELS = ["simple", "moderate", "complex"]

REGISTERED_MODEL = "query-difficulty-router"
EXPERIMENT = "query-difficulty-router"
DEFAULT_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")

ESTIMATORS: dict[str, Any] = {
    "logreg": lambda: LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"),
    "nb": lambda: MultinomialNB(alpha=0.3),
    "linsvc": lambda: LinearSVC(C=0.5, class_weight="balanced"),
}


def build_pipeline(estimator: str = "logreg", use_numeric: bool = True) -> Pipeline:
    if estimator not in ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; expected one of {list(ESTIMATORS)}")
    return Pipeline(
        [
            ("features", build_vectorizer(use_numeric=use_numeric)),
            ("clf", ESTIMATORS[estimator]()),
        ]
    )


def predict_with_confidence(pipe: Pipeline, queries: list[str]) -> tuple[list[str], list[float]]:
    """Return predicted labels and a confidence in [0, 1] for each query.

    LinearSVC has no `predict_proba`. Rather than drop it from the comparison, its
    decision margin is squashed through a softmax so every candidate exposes the same
    interface -- the routing policy needs a confidence regardless of model family.
    """
    frame = build_frame(queries)
    labels = list(pipe.predict(frame))

    if hasattr(pipe, "predict_proba"):
        confidences = pipe.predict_proba(frame).max(axis=1)
    else:
        margins = pipe.decision_function(frame)
        exp = np.exp(margins - margins.max(axis=1, keepdims=True))
        confidences = (exp / exp.sum(axis=1, keepdims=True)).max(axis=1)

    return labels, [float(c) for c in confidences]


def evaluate_pipeline(
    pipe: Pipeline, queries: list[str], truth: list[str], threshold: float
) -> dict[str, Any]:
    """Score a fitted pipeline on both axes we care about: accuracy and money."""
    predicted, confidences = predict_with_confidence(pipe, queries)
    routed_cost, baseline_cost = policy.projected_costs(queries, predicted, confidences, threshold)

    return {
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
        "accuracy": float(accuracy_score(truth, predicted)),
        "cost_savings_pct": float(100.0 * (1.0 - routed_cost / baseline_cost)),
        "routed_cost_usd": routed_cost,
        "baseline_cost_usd": baseline_cost,
        "mean_confidence": float(np.mean(confidences)),
        "pct_routed_cheap": float(
            100.0
            * sum(
                policy.decide(lbl, c, threshold).model == policy.CHEAP
                for lbl, c in zip(predicted, confidences, strict=True)
            )
            / len(predicted)
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
        "report": classification_report(truth, predicted, labels=LABELS, zero_division=0),
        "predictions": predicted,
    }


def load_split(seed: int = 42, test_size: float = 0.2):
    df = pd.read_csv(DATA)
    return train_test_split(
        df["query"].tolist(),
        df["label"].tolist(),
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )


def _current_champion_f1(client, name: str) -> float | None:
    """macro_f1 of the model currently serving, or None if there isn't one yet."""
    try:
        champion = client.get_model_version_by_alias(name, "champion")
        run = client.get_run(champion.run_id)
        return float(run.data.metrics["macro_f1"])
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", default="logreg", choices=sorted(ESTIMATORS))
    parser.add_argument("--no-numeric", action="store_true", help="TF-IDF features only")
    parser.add_argument("--threshold", type=float, default=policy.DEFAULT_THRESHOLD)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--no-promote",
        action="store_true",
        help="register the version but never move the champion alias",
    )
    args = parser.parse_args()

    use_numeric = not args.no_numeric
    X_train, X_val, y_train, y_val = load_split(seed=args.seed)

    pipe = build_pipeline(args.estimator, use_numeric)
    pipe.fit(build_frame(X_train), y_train)
    metrics = evaluate_pipeline(pipe, X_val, y_val, args.threshold)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "router.joblib")
    serialisable = {k: v for k, v in metrics.items() if k not in {"report", "predictions"}}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(serialisable, indent=2))

    # mlflow.log_metrics only accepts scalars; the confusion matrix goes up as an artifact.
    scalar_metrics = {k: v for k, v in serialisable.items() if isinstance(v, int | float)}

    print(f"\nestimator={args.estimator}  numeric_features={use_numeric}  n_train={len(X_train)}")
    print(metrics["report"])
    print(f"macro_f1         {metrics['macro_f1']:.4f}")
    print(f"accuracy         {metrics['accuracy']:.4f}")
    print(f"cost_savings_pct {metrics['cost_savings_pct']:.2f}%")
    print(f"pct_routed_cheap {metrics['pct_routed_cheap']:.1f}%")

    if args.no_mlflow:
        print("\n[mlflow skipped]")
        return

    import mlflow
    from mlflow.models import infer_signature

    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name=f"{args.estimator}-{'full' if use_numeric else 'tfidf'}"):
        mlflow.log_params(
            {
                "estimator": args.estimator,
                "use_numeric_features": use_numeric,
                "threshold": args.threshold,
                "seed": args.seed,
                "n_train": len(X_train),
                "n_val": len(X_val),
            }
        )
        mlflow.log_metrics(scalar_metrics)
        mlflow.log_text(metrics["report"], "classification_report.txt")
        mlflow.log_dict(
            {"labels": LABELS, "matrix": metrics["confusion_matrix"]}, "confusion_matrix.json"
        )

        sample = build_frame(X_val[:5])
        info = mlflow.sklearn.log_model(
            pipe,
            artifact_path="model",
            signature=infer_signature(sample, pipe.predict(sample)),
            registered_model_name=REGISTERED_MODEL,
        )

    client = mlflow.MlflowClient()
    version = client.search_model_versions(
        f"name='{REGISTERED_MODEL}'", order_by=["version_number DESC"], max_results=1
    )[0].version

    incumbent = _current_champion_f1(client, REGISTERED_MODEL)
    print(f"\nregistered {REGISTERED_MODEL} v{version} ({info.model_uri})")

    if args.no_promote:
        client.set_registered_model_alias(REGISTERED_MODEL, "challenger", version)
        print(f"-> tagged @challenger (promotion disabled); champion macro_f1={incumbent}")
    elif incumbent is None or metrics["macro_f1"] >= incumbent:
        client.set_registered_model_alias(REGISTERED_MODEL, "champion", version)
        was = "no incumbent" if incumbent is None else f"beat {incumbent:.4f}"
        print(f"-> promoted to @champion ({was})")
    else:
        client.set_registered_model_alias(REGISTERED_MODEL, "challenger", version)
        print(
            f"-> NOT promoted: {metrics['macro_f1']:.4f} < champion {incumbent:.4f}; "
            "tagged @challenger instead"
        )


if __name__ == "__main__":
    main()
