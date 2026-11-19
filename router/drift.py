"""Data drift detection: is live traffic still shaped like the training set?

Monitors the same numeric features the router was trained on (`router/features.py`),
not the raw text -- PSI on a TF-IDF vocabulary is noisy and hard to threshold
meaningfully, whereas "average query length just doubled" is something a PSI score can
say cleanly and something a human can sanity-check at a glance.

Two complementary statistics, computed per feature:

  PSI (Population Stability Index)   the standard MLOps drift metric. Buckets both
                                      distributions the same way and compares bucket
                                      weights. < 0.1 stable, 0.1-0.25 moderate, > 0.25
                                      is the conventional "investigate" threshold.

  KS (two-sample Kolmogorov-Smirnov) a statistical test with a p-value, included
                                      because PSI has no notion of statistical
                                      significance on its own -- a small window can
                                      swing PSI around by chance. Reported alongside,
                                      not instead of, PSI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from router.features import NUMERIC_FEATURES, numeric_features

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = ROOT / "data" / "reference_features.json"

PSI_BINS = 10
PSI_WARN = 0.1
PSI_ALERT = 0.25
#: A tiny floor so an empty bucket contributes a large-but-finite penalty instead of
#: a divide-by-zero. The classic practitioner trick for this exact metric.
EPS = 1e-4


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    reference_mean: float
    current_mean: float
    status: str  # "stable" | "moderate" | "alert"


@dataclass
class DriftReport:
    n_current: int
    n_reference: int
    max_psi: float
    status: str
    features: list[FeatureDrift]


def build_reference(queries: list[str], path: Path = REFERENCE_PATH) -> None:
    """Snapshot the training set's feature distribution. Run once after `prepare_data`,
    before the model ever sees production traffic, so the reference is never
    contaminated by drift it is supposed to detect."""
    frame = pd.DataFrame([numeric_features(q) for q in queries])
    payload = {col: frame[col].tolist() for col in NUMERIC_FEATURES}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = PSI_BINS) -> float:
    """PSI with bin edges fixed from the reference distribution's quantiles, so a
    feature that is literally constant in the reference collapses to one bin instead
    of raising on degenerate `np.histogram` edges."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(len(reference), 1) + EPS
    cur_pct = cur_counts / max(len(current), 1) + EPS
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _status(psi: float) -> str:
    if psi >= PSI_ALERT:
        return "alert"
    if psi >= PSI_WARN:
        return "moderate"
    return "stable"


def check_drift(current_queries: list[str], reference_path: Path = REFERENCE_PATH) -> DriftReport:
    if not reference_path.exists():
        raise FileNotFoundError(
            f"no drift reference at {reference_path}; run build_reference() once "
            "against the training set after `python -m router.prepare_data`"
        )
    reference = pd.DataFrame(json.loads(reference_path.read_text()))

    if not current_queries:
        return DriftReport(0, len(reference), 0.0, "stable", [])

    current = pd.DataFrame([numeric_features(q) for q in current_queries])

    features = []
    for col in NUMERIC_FEATURES:
        ref_col, cur_col = reference[col].to_numpy(), current[col].to_numpy()
        psi = _psi(ref_col, cur_col)
        ks_stat, ks_p = stats.ks_2samp(ref_col, cur_col)
        features.append(
            FeatureDrift(
                feature=col,
                psi=round(psi, 4),
                ks_statistic=round(float(ks_stat), 4),
                ks_pvalue=round(float(ks_p), 4),
                reference_mean=round(float(ref_col.mean()), 4),
                current_mean=round(float(cur_col.mean()), 4),
                status=_status(psi),
            )
        )

    max_psi = max((f.psi for f in features), default=0.0)
    return DriftReport(
        n_current=len(current),
        n_reference=len(reference),
        max_psi=round(max_psi, 4),
        status=_status(max_psi),
        features=features,
    )


def report_to_dict(report: DriftReport) -> dict:
    return {**asdict(report), "features": [asdict(f) for f in report.features]}


if __name__ == "__main__":
    df = pd.read_csv(ROOT / "data" / "queries.csv")
    build_reference(df["query"].tolist())
    print(f"wrote drift reference from {len(df)} training queries -> {REFERENCE_PATH}")
