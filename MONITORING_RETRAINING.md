# Model monitoring and retraining workflow

This extends the deployed query-difficulty router with the five-task monitoring and
retraining workflow required by the MLOps practical assessment. Entry point:

```bash
python -m scripts.monitor_retrain
```

Every run writes a fresh, timestamped evidence directory under `monitoring/runs/`
(JSON reports + classification reports) so repeated runs never overwrite each
other's evidence, and the script itself never overwrites the champion model file.

## Task A — Prepare new data

`data/holdout.csv` (198 rows, carved off the dataset at prep time and never used by
`router/train.py`) stands in for a newly arrived, labelled batch — realistic, because
it genuinely wasn't used to fit the deployed model. The script validates it before
using it: checks `query`/`label` columns are present, drops any row with a missing
value, drops duplicate queries, then reports size and target distribution.

```json
{
  "source": "data/holdout.csv",
  "n_rows": 198,
  "n_rows_dropped_missing": 0,
  "n_rows_dropped_duplicate": 0,
  "label_distribution": { "simple": 66, "complex": 66, "moderate": 66 }
}
```

## Task B — Evaluate the existing model

The currently deployed champion (`models/router.joblib`) is loaded and scored against
the new batch's true labels — accuracy, and macro precision/recall/F1 (macro, because
this is a 3-class problem and we don't want the majority class to hide a weak minority
class):

```
              precision    recall  f1-score   support
      simple       0.79      0.83      0.81        66
    moderate       0.84      0.74      0.79        66
     complex       0.94      1.00      0.97        66
    accuracy                           0.86       198
```

Saved as `monitoring/runs/<ts>/task_b_evaluation_report.json` (+ the full
`classification_report` text alongside it).

## Task C — Retrain the model

The new batch is combined with the existing training data (`data/queries.csv`), and a
candidate is fit with the **same** feature pipeline the champion uses
(`router/features.py`'s TF-IDF + hand-engineered numeric features, via
`router.train.build_pipeline`) — so there's no train/serve or train/retrain skew
between the two models being compared. The combined set (900 rows) is split 80/20
(stratified, seed 42) purely for the candidate's own training/validation; the
candidate is saved **separately** as `models/candidate_router.joblib`, next to but
never overwriting `models/router.joblib`.

If MLflow is reachable (it is here — the project already runs a local MLflow backed
by `mlflow.db`), the run is logged to the existing `query-difficulty-router`
experiment with its params, metrics, classification report, and model artifact, the
same way `router/train.py`'s regular runs are. MLflow is optional per the brief — if
it isn't available, the script prints a note and continues using the local metrics.

## Task D — Quality gate

**Metric: macro F1**, computed for both models on the *same* held-out validation
slice (the one carved out of the combined dataset while retraining the candidate —
comparing the two models on the same data is what makes the comparison meaningful).
Macro F1 is the metric already used by the project's two other quality mechanisms —
the CI gate (`scripts/quality_gate.py`) and the champion/challenger promotion rule in
`router/train.py` — so this decision stays consistent with the rest of the project
instead of introducing a second, competing notion of "better".

**Decision rule:** the candidate is `eligible_for_promotion` only if its macro F1 on
the shared validation slice is `>=` the champion's macro F1 on that same slice.
Otherwise the champion is retained. The decision — both scores, the metric used, the
rationale, and the MLflow run id — is always written to
`monitoring/runs/<ts>/task_d_quality_gate_decision.json`, whether or not the candidate
passes. **The script itself never copies the candidate over the champion file** —
promotion is a deliberate, separate, human step (`cp models/candidate_router.joblib
models/router.joblib`) taken only after reading the recorded decision.

### A real run

```json
{
  "metric": "f1_macro",
  "champion_score": 0.9553,
  "candidate_score": 0.8815,
  "decision": "retain_champion"
}
```

The champion scores higher on this particular validation slice than the candidate
does, so the gate correctly refuses to recommend promotion — a useful negative
result: it demonstrates the gate isn't a rubber stamp, and that retraining on more
data doesn't automatically mean a better model on every random split.

## Task E — Demonstration

```
Task A -- prepare new data            -> 198 rows, balanced 66/66/66, no missing/dupes
Task B -- champion on new data        -> accuracy 0.86, macro-F1 0.86
Task C -- candidate retrain           -> combined_n=900, val macro-F1 0.88, saved to
                                          models/candidate_router.joblib, MLflow run
                                          a52db52264834c54a55fc33450a6b9ba
Task D -- quality gate                -> champion 0.9553 vs candidate 0.8815 on the
                                          same slice -> retain_champion (recorded, not
                                          auto-applied)
Task E -- artifacts                   -> monitoring/runs/20261120-102303/*.json
```

Evidence from this run lives in `monitoring/runs/20261120-102303/`:

| File | Task |
|---|---|
| `task_a_new_data_report.json` | A — validation + distribution of the new batch |
| `task_b_evaluation_report.json`, `task_b_classification_report.txt` | B — champion scored on the new batch |
| `task_c_candidate_metrics.json`, `task_c_candidate_classification_report.txt` | C — candidate's own validation metrics |
| `task_d_quality_gate_decision.json` | D — the promotion decision, recorded regardless of outcome |

`models/candidate_router.joblib` is the Task C model artifact, kept separate from the
deployed `models/router.joblib` champion.
