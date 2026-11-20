"""Model-quality monitoring and retraining workflow.

Extends the existing query-difficulty router with the five-task workflow required by
the MLOps practical assessment: prepare new data, evaluate the current champion on
it, retrain a candidate on the combined data using the *same* preprocessing pipeline,
run a quality gate that only ever records a promotion decision (it never overwrites
the champion file itself), and leave behind every artifact needed to demonstrate the
run.

    python -m scripts.monitor_retrain

Every run writes a fresh, timestamped directory under `monitoring/runs/` so repeated
runs don't clobber each other's evidence.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from router import train as router_train
from router.features import build_frame

ROOT = router_train.ROOT
TRAIN_DATA = router_train.DATA  # data/queries.csv -- the existing training data
NEW_DATA = ROOT / "data" / "holdout.csv"  # stand-in for an incoming, labelled batch
MODEL_DIR = router_train.MODEL_DIR
CHAMPION_PATH = MODEL_DIR / "router.joblib"
CANDIDATE_PATH = MODEL_DIR / "candidate_router.joblib"
MONITOR_DIR = ROOT / "monitoring" / "runs"
LABELS = router_train.LABELS

REQUIRED_COLUMNS = {"query", "label"}

# Task D decision rule.
PROMOTION_METRIC = "f1_macro"
MIN_IMPROVEMENT = 0.0  # candidate must not regress vs. champion on the shared eval slice


def task_a_prepare_new_data(path: Path = NEW_DATA) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the incoming batch, validate it, and report its shape."""
    df = pd.read_csv(path)

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"new data batch is missing required columns: {missing_cols}")

    n_missing = int(df[["query", "label"]].isna().any(axis=1).sum())
    n_dupes = int(df.duplicated(subset=["query"]).sum())
    df = df.dropna(subset=["query", "label"]).drop_duplicates(subset=["query"]).reset_index(drop=True)

    report = {
        "source": str(path),
        "n_rows": int(len(df)),
        "n_rows_dropped_missing": n_missing,
        "n_rows_dropped_duplicate": n_dupes,
        "label_distribution": df["label"].value_counts().to_dict(),
    }

    print("Task A -- prepare new data")
    print(json.dumps(report, indent=2))
    return df, report


def _evaluate(pipe, queries: list[str], truth: list[str]) -> dict[str, Any]:
    predicted = list(pipe.predict(build_frame(queries)))
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision_macro": float(precision_score(truth, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(truth, predicted, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
        "report": classification_report(truth, predicted, labels=LABELS, zero_division=0),
    }


def task_b_evaluate_existing(new_data: pd.DataFrame) -> dict[str, Any]:
    """Score the deployed champion model on the new labelled batch."""
    pipe = joblib.load(CHAMPION_PATH)
    metrics = _evaluate(pipe, new_data["query"].tolist(), new_data["label"].tolist())

    print("\nTask B -- evaluate the existing (champion) model on the new data")
    print(metrics["report"])
    return metrics


def task_c_retrain(new_data: pd.DataFrame, seed: int = 42):
    """Combine existing training data with the new batch and fit a candidate model."""
    existing = pd.read_csv(TRAIN_DATA)
    combined = pd.concat(
        [existing[["query", "label"]], new_data[["query", "label"]]], ignore_index=True
    ).drop_duplicates(subset=["query"]).reset_index(drop=True)

    X_train, X_val, y_train, y_val = train_test_split(
        combined["query"].tolist(),
        combined["label"].tolist(),
        test_size=0.2,
        random_state=seed,
        stratify=combined["label"],
    )

    # Same preprocessing + estimator as router/train.py -- no train/serve skew between
    # the deployed champion and the candidate being compared against it.
    pipe = router_train.build_pipeline("logreg", use_numeric=True)
    pipe.fit(build_frame(X_train), y_train)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, CANDIDATE_PATH)
    candidate_metrics = _evaluate(pipe, X_val, y_val)

    mlflow_run_id = None
    try:
        import mlflow

        mlflow.set_tracking_uri(router_train.DEFAULT_TRACKING_URI)
        mlflow.set_experiment(router_train.EXPERIMENT)
        with mlflow.start_run(run_name="candidate-retrain") as run:
            mlflow.log_params(
                {
                    "n_combined": len(combined),
                    "n_train": len(X_train),
                    "n_val": len(X_val),
                    "seed": seed,
                    "source": "scripts.monitor_retrain",
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in candidate_metrics.items() if isinstance(v, int | float)}
            )
            mlflow.log_text(candidate_metrics["report"], "candidate_classification_report.txt")
            mlflow.sklearn.log_model(pipe, artifact_path="model")
            mlflow_run_id = run.info.run_id
    except Exception as exc:  # pragma: no cover -- MLflow is optional per the brief
        print(f"[MLflow tracking skipped: {exc}]")

    print("\nTask C -- retrain a candidate on existing + new data")
    print(f"combined_n={len(combined)}  train_n={len(X_train)}  val_n={len(X_val)}")
    print(f"candidate model saved separately -> {CANDIDATE_PATH}")
    print(candidate_metrics["report"])

    return pipe, X_val, y_val, candidate_metrics, mlflow_run_id


def task_d_quality_gate(
    champion_metrics_on_val: dict[str, Any], candidate_metrics: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Compare champion vs. candidate on the same held-out slice and record a decision.

    Metric: macro F1. The router is a 3-class classifier with class imbalance across
    its source datasets (SQuAD/Dolly/GSM8K+MBPP); macro F1 weighs every class equally
    so a candidate can't look better purely by improving the majority class. It's also
    the metric the project's existing CI quality gate (scripts/quality_gate.py) and the
    champion/challenger promotion rule in router/train.py already use, so this decision
    is consistent with the rest of the project rather than introducing a second, rival
    notion of "better".
    """
    champ = champion_metrics_on_val[PROMOTION_METRIC]
    cand = candidate_metrics[PROMOTION_METRIC]
    eligible = cand >= champ - MIN_IMPROVEMENT

    decision = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metric": PROMOTION_METRIC,
        "metric_rationale": (
            "macro_f1 on a shared held-out slice: equal weight per class, and the same "
            "metric already used by scripts/quality_gate.py and router/train.py's "
            "champion/challenger promotion rule."
        ),
        "champion_score": champ,
        "candidate_score": cand,
        "decision": "eligible_for_promotion" if eligible else "retain_champion",
        "champion_metrics": {k: v for k, v in champion_metrics_on_val.items() if k != "report"},
        "candidate_metrics": {k: v for k, v in candidate_metrics.items() if k != "report"},
    }
    return decision, eligible


def main() -> dict[str, Any]:
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = MONITOR_DIR / run_id
    run_dir.mkdir()

    # Task A
    new_data, data_report = task_a_prepare_new_data()
    (run_dir / "task_a_new_data_report.json").write_text(json.dumps(data_report, indent=2))

    # Task B
    eval_report = task_b_evaluate_existing(new_data)
    (run_dir / "task_b_evaluation_report.json").write_text(
        json.dumps({k: v for k, v in eval_report.items() if k != "report"}, indent=2)
    )
    (run_dir / "task_b_classification_report.txt").write_text(eval_report["report"])

    # Task C
    _pipe, X_val, y_val, candidate_metrics, mlflow_run_id = task_c_retrain(new_data)
    (run_dir / "task_c_candidate_metrics.json").write_text(
        json.dumps({k: v for k, v in candidate_metrics.items() if k != "report"}, indent=2)
    )
    (run_dir / "task_c_candidate_classification_report.txt").write_text(candidate_metrics["report"])

    # Task D -- evaluate the champion on the SAME val slice the candidate was scored on,
    # so the comparison is apples-to-apples rather than two different data slices.
    champion_pipe = joblib.load(CHAMPION_PATH)
    champion_metrics_on_val = _evaluate(champion_pipe, X_val, y_val)

    decision, eligible = task_d_quality_gate(champion_metrics_on_val, candidate_metrics)
    decision["mlflow_run_id"] = mlflow_run_id
    (run_dir / "task_d_quality_gate_decision.json").write_text(json.dumps(decision, indent=2))

    print("\nTask D -- quality gate decision")
    print(
        json.dumps(
            {k: v for k, v in decision.items() if k not in {"champion_metrics", "candidate_metrics"}},
            indent=2,
        )
    )

    if eligible:
        print(
            f"\nCandidate is ELIGIBLE for promotion (not applied automatically).\n"
            f"To promote by hand: cp {CANDIDATE_PATH} {CHAMPION_PATH}"
        )
    else:
        print("\nCandidate does NOT meet the quality bar -- champion retained, unchanged.")

    print(f"\nTask E -- run artifacts written to {run_dir}")
    print(f"  new-data report:        {run_dir / 'task_a_new_data_report.json'}")
    print(f"  champion evaluation:    {run_dir / 'task_b_evaluation_report.json'}")
    print(f"  candidate model:        {CANDIDATE_PATH}")
    print(f"  candidate metrics:      {run_dir / 'task_c_candidate_metrics.json'}")
    print(f"  quality-gate decision:  {run_dir / 'task_d_quality_gate_decision.json'}")

    return {"run_dir": run_dir, "decision": decision}


if __name__ == "__main__":
    main()
