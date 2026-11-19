"""Routing policy and cost model.

Imported by both training and serving. Keeping the decision rule and the price table
in one place means the cost-savings number reported by `train.py`, the one measured by
`eval/`, and the one the live gateway shows on its dashboard are all computed by the
same code. If they disagree, it is a real disagreement and not a bookkeeping bug.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

CHEAP = os.getenv("MODEL_CHEAP", "openai/gpt-oss-20b")
EXPENSIVE = os.getenv("MODEL_EXPENSIVE", "openai/gpt-oss-120b")

# Per-million-token rates in USD, from the provider's published price list.
#
# These are the one set of numbers in the project that cannot be derived from the code
# or the data, so they are isolated here, overridable from the environment, and quoted
# with their source in RESULTS.md. Every cost figure the project reports is a function
# of exactly these six constants and measured token counts -- change them and every
# reported number moves consistently, because nothing else hard-codes a price.
PRICING: dict[str, dict[str, float]] = {
    CHEAP: {
        "input": float(os.getenv("PRICE_CHEAP_IN", "0.10")),
        "output": float(os.getenv("PRICE_CHEAP_OUT", "0.50")),
    },
    EXPENSIVE: {
        "input": float(os.getenv("PRICE_EXPENSIVE_IN", "0.15")),
        "output": float(os.getenv("PRICE_EXPENSIVE_OUT", "0.75")),
    },
}

#: Confidence below which a `moderate` query is escalated to the expensive model.
#: Tuned empirically -- see `eval/sweep_threshold.py` and the Pareto curve in RESULTS.md.
DEFAULT_THRESHOLD = float(os.getenv("ROUTER_THRESHOLD", "0.60"))

#: Mean output length per difficulty class, used only to *project* cost during training
#: and offline evaluation. The live gateway always bills real token counts.
ASSUMED_OUTPUT_TOKENS = {"simple": 60, "moderate": 220, "complex": 420}


@dataclass(frozen=True)
class RoutingDecision:
    model: str
    reason: str
    confidence: float


def estimate_tokens(text: str) -> int:
    """Rough token count: ~4 characters per token.

    Deliberately crude. Real token counts come back from the provider on every live
    request; this is only for pre-flight projections where no response exists yet.
    """
    return max(1, len(text or "") // 4)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price = PRICING[model]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def decide(label: str, confidence: float, threshold: float = DEFAULT_THRESHOLD) -> RoutingDecision:
    """Pick a model for a classified query.

    `simple` and `complex` are routed on the label alone. `moderate` is the genuinely
    ambiguous class, so it only goes to the cheap model when the classifier is
    confident; otherwise we pay for the big model rather than risk a bad answer. This
    asymmetry is why the threshold is worth tuning at all.
    """
    if label == "complex":
        return RoutingDecision(EXPENSIVE, "label=complex", confidence)
    if label == "simple":
        return RoutingDecision(CHEAP, "label=simple", confidence)
    if confidence >= threshold:
        return RoutingDecision(CHEAP, f"label=moderate, confident ({confidence:.2f})", confidence)
    return RoutingDecision(
        EXPENSIVE, f"label=moderate, low confidence ({confidence:.2f})", confidence
    )


def projected_costs(
    queries: list[str], labels: list[str], confidences: list[float], threshold: float
) -> tuple[float, float]:
    """Return (routed_cost, always_expensive_cost) over a set of queries.

    Used as a training-time metric so the CI quality gate can fail a model that keeps
    its accuracy but quietly stops saving money -- a regression that pure macro-F1
    would never catch.
    """
    routed = 0.0
    baseline = 0.0
    for query, label, conf in zip(queries, labels, confidences, strict=True):
        in_tok = estimate_tokens(query)
        out_tok = ASSUMED_OUTPUT_TOKENS.get(label, 200)
        routed += cost_usd(decide(label, conf, threshold).model, in_tok, out_tok)
        baseline += cost_usd(EXPENSIVE, in_tok, out_tok)
    return routed, baseline
