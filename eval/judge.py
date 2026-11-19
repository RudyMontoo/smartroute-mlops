"""Scoring answers against ground truth, with an LLM judge for the open-ended ones.

Two scoring regimes, picked per query by `answer_type`:

  numeric (GSM8K) and span (SQuAD)   exact / containment match against the gold
                                     answer the dataset ships. Free, deterministic,
                                     and completely immune to judge bias.

  freeform (Dolly) and code (MBPP)   no mechanical gold answer exists, so the large
                                     model grades the candidate against the reference.

Using ground truth wherever it exists is the whole reason the dataset sources were
chosen the way they were -- roughly two thirds of the holdout is scored without an LLM
in the loop at all, which limits how much the judge's opinion can distort the result.

Known limitation, stated plainly in RESULTS.md: the judge is the same 70B model that
produces the `expensive` answers, so it may mildly prefer its own output. That biases
*against* the router (it inflates the baseline it is measured against), so the reported
quality-retained figure is, if anything, conservative.
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path

from gateway.providers import Completion
from router import policy

RESULTS_DIR = Path(__file__).resolve().parent / "results"
JUDGMENTS_CACHE = RESULTS_DIR / "judgments.json"

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = str.maketrans("", "", string.punctuation)
_NUMBER = re.compile(r"-?\$?\d[\d,]*\.?\d*")

JUDGE_PROMPT = """You are grading a candidate answer against a reference answer.

Question:
{question}

Reference answer:
{reference}

Candidate answer:
{candidate}

Reply with exactly one word:
GOOD     - the candidate is correct and addresses the question
PARTIAL  - partially correct, or correct but missing important content
BAD      - incorrect, irrelevant, or a refusal

Answer:"""

_VERDICTS = {"GOOD": 1.0, "PARTIAL": 0.5, "BAD": 0.0}


# ----------------------------------------------------------------- mechanical


def normalise(text: str) -> str:
    """SQuAD-style normalisation: casefold, drop articles and punctuation."""
    text = _ARTICLES.sub(" ", (text or "").lower())
    return " ".join(text.translate(_PUNCT).split())


def extract_final_number(text: str) -> float | None:
    """GSM8K answers end with the result, so take the last number in the response."""
    matches = _NUMBER.findall(text or "")
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", "").replace("$", ""))
    except ValueError:
        return None


def score_numeric(gold: str, candidate: str) -> float:
    try:
        expected = float(str(gold).replace(",", "").replace("$", ""))
    except (TypeError, ValueError):
        return 0.0
    actual = extract_final_number(candidate)
    if actual is None:
        return 0.0
    return 1.0 if abs(actual - expected) < 1e-6 else 0.0


def score_span(gold: str, candidate: str) -> float:
    gold_norm, candidate_norm = normalise(gold), normalise(candidate)
    if not gold_norm:
        return 0.0
    return 1.0 if gold_norm in candidate_norm else 0.0


def is_mechanical(answer_type: str) -> bool:
    return answer_type in {"numeric", "span"}


# ---------------------------------------------------------------- llm judge


class Judge:
    """Scores answers, caching every judgement so a rerun costs nothing."""

    def __init__(self, client) -> None:
        self._client = client
        self._cache: dict[str, float] = {}
        self.llm_calls = 0

        if JUDGMENTS_CACHE.exists():
            self._cache = json.loads(JUDGMENTS_CACHE.read_text())

    @staticmethod
    def cache_key(query: str, candidate: str) -> str:
        import hashlib

        digest = hashlib.sha256(f"{query}||{candidate}".encode()).hexdigest()[:24]
        return digest

    def has_judgement(self, query: str, candidate: str) -> bool:
        return self.cache_key(query, candidate) in self._cache

    async def score(self, query: str, gold: str, answer_type: str, candidate: str) -> float:
        if answer_type == "numeric":
            return score_numeric(gold, candidate)
        if answer_type == "span":
            return score_span(gold, candidate)

        key = self.cache_key(query, candidate)
        if key in self._cache:
            return self._cache[key]

        verdict = await self._ask_llm(query, gold, candidate)
        self._cache[key] = verdict
        return verdict

    async def _ask_llm(self, query: str, reference: str, candidate: str) -> float:
        # Inputs are trimmed hard: the judge needs enough to grade, and every extra
        # character comes out of an 8,000 tokens-per-minute budget.
        prompt = JUDGE_PROMPT.format(
            question=query[:700], reference=(reference or "")[:700], candidate=candidate[:1200]
        )
        self.llm_calls += 1
        completion: Completion = await self._client.complete(policy.EXPENSIVE, prompt)

        head = completion.text.strip().upper()
        for verdict, value in _VERDICTS.items():
            if head.startswith(verdict) or verdict in head[:40]:
                return value
        # An unparseable verdict is scored as partial rather than silently as zero;
        # treating judge failures as model failures would understate every strategy.
        return 0.5

    def flush(self) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        JUDGMENTS_CACHE.write_text(json.dumps(self._cache, indent=0, sort_keys=True))
