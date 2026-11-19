"""Cascade verification: catch a bad cheap answer and retry on the large model.

The router classifies a query *before* any answer exists, so it is guessing. The
cascade closes that loop: when the small model answers, a cheap set of heuristics
inspects the response, and a response that looks like a failure is retried on the large
model. Routing stops being a single irreversible bet.

The checks are deliberately cheap and syntactic -- no second model call, no embeddings.
Anything expensive here would eat the savings the router just made.

Escalation is not free and the accounting does not pretend otherwise: an escalated
request pays for *both* calls, so it costs more than always having used the large model.
That shows up in the measured savings, which is the point -- a verifier tuned too
aggressively would quietly destroy the economics, and the number would say so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Explicit refusals and capability disclaimers.
_REFUSAL = re.compile(
    r"\b(i (can'?t|cannot|am unable to|won'?t be able)|as an ai|i'?m (just )?an ai|"
    r"i do not have access|unable to (help|assist|answer))\b",
    re.IGNORECASE,
)

#: Low-confidence hedging that usually signals the model is out of its depth.
_HEDGE = re.compile(
    r"\b(i'?m not (sure|certain)|i don'?t know|it'?s (unclear|hard to say)|"
    r"without more (context|information)|cannot determine)\b",
    re.IGNORECASE,
)

#: Queries that demand a numeric result.
_NEEDS_NUMBER = re.compile(
    r"\b(how many|how much|calculate|compute|total|sum|average|percent|"
    r"what is \d|\d+\s*[\+\-\*/x]\s*\d+)\b",
    re.IGNORECASE,
)
_HAS_NUMBER = re.compile(r"\d")

#: Below this an answer is almost certainly a non-answer rather than a terse one.
MIN_ANSWER_CHARS = 25


@dataclass(frozen=True)
class Verdict:
    escalate: bool
    trigger: str

    @property
    def reason(self) -> str:
        return {
            "ok": "cheap answer accepted",
            "refusal": "cheap model refused",
            "hedge": "cheap model hedged",
            "too_short": "cheap answer implausibly short",
            "missing_number": "numeric question answered without a number",
        }.get(self.trigger, self.trigger)


ACCEPTED = Verdict(False, "ok")


def verify(query: str, answer: str) -> Verdict:
    """Decide whether a cheap model's answer should be retried on the large model."""
    text = (answer or "").strip()

    if len(text) < MIN_ANSWER_CHARS:
        return Verdict(True, "too_short")
    if _REFUSAL.search(text):
        return Verdict(True, "refusal")
    if _HEDGE.search(text):
        return Verdict(True, "hedge")

    # A question that asks "how many" and comes back with no digits anywhere did not
    # actually answer the question, whatever else it did.
    if _NEEDS_NUMBER.search(query or "") and not _HAS_NUMBER.search(text):
        return Verdict(True, "missing_number")

    return ACCEPTED
