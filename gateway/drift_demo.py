"""Synthetic traffic for the /admin/simulate-drift demo button.

Deliberately shaped to shift the feature distribution the router was trained on:
much longer than the training mean, code-heavy, and reasoning-keyword-dense. The point
is not realism, it is to *reliably* move PSI past the alert threshold on demand so the
drift chart has something to show in a five-minute viva slot instead of waiting for it
to happen on its own.
"""

from __future__ import annotations

import random

_TEMPLATES = [
    (
        "Write a complete, production-ready Python function with full type hints, "
        "docstrings, and error handling that implements a {algo} for a list of up to "
        "{n} integers, then explain step by step why each part of the implementation "
        "is correct and analyze its time and space complexity in detail."
    ),
    (
        "Design and implement, in {n} clear steps, a {algo}-based solution for the "
        "following scenario: given {n} concurrent requests hitting a distributed "
        "system, derive the exact formula for expected latency, show every "
        "intermediate calculation, and justify each assumption you make along the way."
    ),
    (
        "Step by step, derive and prove why a {algo} algorithm applied to {n} items "
        "has the time complexity it does, including a full worked example with actual "
        "numbers, a discussion of edge cases, and a comparison against at least two "
        "alternative approaches with their own complexity derivations."
    ),
]

_ALGOS = ["merge sort", "binary search tree", "dynamic programming", "graph traversal", "hash map"]


def generate(n: int = 30, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    return [
        rng.choice(_TEMPLATES).format(algo=rng.choice(_ALGOS), n=rng.randint(50, 500))
        for _ in range(n)
    ]
