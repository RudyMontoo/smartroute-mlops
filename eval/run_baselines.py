"""Measure the router against four baselines on the held-out set.

    python -m eval.run_baselines            # use cached generations (free, offline)
    python -m eval.run_baselines --live     # call Groq for anything not cached

This is the experiment the whole project rests on. "We built a router" is a claim;
"the router keeps N% of the large model's answer quality for M% of the cost, and beats
a three-line length heuristic by K points" is a result.

The length heuristic is in here deliberately. It is the honest null hypothesis: if a
trained classifier cannot beat `len(query) > 80`, then the machine learning is
decoration, and that belongs in the write-up rather than hidden.

Efficiency note: every strategy is a *selection* between the same two cached answers
per query, so the holdout only ever needs 2 generations per query (not 5). That turns
~1000 API calls into ~400, which matters on a free tier.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import time
from pathlib import Path

import pandas as pd

from eval.client import RateLimitedClient
from eval.judge import Judge, is_mechanical
from gateway.config import settings
from gateway.providers import GroqProvider
from router import policy
from router.features import build_frame
from router.train import build_pipeline, load_split, predict_with_confidence

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "data" / "holdout.csv"
RESULTS_DIR = ROOT / "eval" / "results"
GENERATIONS_CACHE = RESULTS_DIR / "generations.json"

#: The naive rule a reviewer would reasonably ask "why not just do this?".
LENGTH_THRESHOLD_CHARS = 80


# --------------------------------------------------------------- generation


def _cache_key(model: str, query: str) -> str:
    import hashlib

    return f"{model}:{hashlib.sha256(query.encode()).hexdigest()[:24]}"


async def generate_all(queries: list[str], client, cache: dict, live: bool) -> dict:
    """Answer every query on both models, skipping anything already cached.

    Sequential by design -- see eval/client.py. The cache is flushed every few answers
    so an interrupted run (or an exhausted daily quota) never loses work.
    """
    todo = [
        (model, query)
        for model in (policy.CHEAP, policy.EXPENSIVE)
        for query in queries
        if _cache_key(model, query) not in cache
    ]

    if not todo:
        print(f"all {len(queries) * 2} generations already cached")
        return cache
    if not live:
        raise SystemExit(
            f"{len(todo)} generations missing from the cache. Re-run with --live "
            "(and a GROQ_API_KEY in .env) to produce them."
        )

    print(f"generating {len(todo)} answers sequentially (rate-limit aware) ...")
    started = time.perf_counter()

    for index, (model, query) in enumerate(todo, 1):
        completion = await client.complete(model, query)

        # Belt and braces: the strict client should never return a mock, but a mock
        # answer silently entering this cache would corrupt every quality number.
        if completion.provider != "groq":
            raise RuntimeError(f"refusing to cache a {completion.provider!r} generation")

        cache[_cache_key(model, query)] = {
            "text": completion.text,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "latency_ms": completion.latency_ms,
            "provider": completion.provider,
        }

        if index % 10 == 0 or index == len(todo):
            elapsed = time.perf_counter() - started
            eta = elapsed / index * (len(todo) - index)
            _save_generations(cache)
            print(f"  {index}/{len(todo)}  elapsed {elapsed / 60:.1f}m  eta {eta / 60:.1f}m")

    _save_generations(cache)
    return cache


def _save_generations(cache: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GENERATIONS_CACHE.write_text(json.dumps(cache, indent=0))


# --------------------------------------------------------------- strategies


def strategy_choices(
    df: pd.DataFrame, threshold: float, seed: int = 42
) -> tuple[dict[str, list[str]], list[str], list[float]]:
    """For each strategy, which model each holdout query would have been sent to."""
    queries = df["query"].tolist()
    rng = random.Random(seed)

    # Train on data/queries.csv only -- the holdout has never been seen by this model.
    X_train, _, y_train, _ = load_split(seed=seed, test_size=0.01)
    pipe = build_pipeline("logreg", use_numeric=True)
    pipe.fit(build_frame(X_train), y_train)
    labels, confidences = predict_with_confidence(pipe, queries)

    return (
        {
            "always-cheap": [policy.CHEAP] * len(queries),
            "always-expensive": [policy.EXPENSIVE] * len(queries),
            "random-50/50": [rng.choice([policy.CHEAP, policy.EXPENSIVE]) for _ in queries],
            "length-heuristic": [
                policy.EXPENSIVE if len(q) > LENGTH_THRESHOLD_CHARS else policy.CHEAP
                for q in queries
            ],
            "learned-router": [
                policy.decide(label, confidence, threshold).model
                for label, confidence in zip(labels, confidences, strict=True)
            ],
        },
        labels,
        confidences,
    )


# ----------------------------------------------------------------- scoring


async def score_strategies(
    df: pd.DataFrame, choices: dict[str, list[str]], cache: dict, judge: Judge, live: bool
) -> dict:
    rows = df.to_dict("records")
    needed = {
        (_cache_key(model, row["query"]), row["query"], row["answer"], row["answer_type"], model)
        for models in choices.values()
        for row, model in zip(rows, models, strict=True)
    }

    if not live:
        unscored = [
            item
            for item in needed
            if not is_mechanical(item[3])
            and not judge.has_judgement(item[1], cache[item[0]]["text"])
        ]
        if unscored:
            raise SystemExit(
                f"{len(unscored)} judgements missing from the cache. Re-run with --live."
            )

    scores: dict[str, float] = {}
    for key, query, gold, answer_type, _model in needed:
        scores[key] = await judge.score(query, gold, answer_type, cache[key]["text"])
    judge.flush()

    results = {}
    for name, models in choices.items():
        quality = []
        cost = 0.0
        cheap_count = 0
        latencies = []

        for row, model in zip(rows, models, strict=True):
            entry = cache[_cache_key(model, row["query"])]
            quality.append(scores[_cache_key(model, row["query"])])
            cost += policy.cost_usd(model, entry["input_tokens"], entry["output_tokens"])
            latencies.append(entry["latency_ms"])
            cheap_count += model == policy.CHEAP

        results[name] = {
            "mean_quality": sum(quality) / len(quality),
            "total_cost_usd": cost,
            "pct_routed_cheap": 100.0 * cheap_count / len(models),
            "mean_latency_ms": sum(latencies) / len(latencies),
            "n": len(models),
        }

    reference = results["always-expensive"]
    for row in results.values():
        row["quality_retained_pct"] = 100.0 * row["mean_quality"] / reference["mean_quality"]
        row["cost_vs_expensive_pct"] = 100.0 * row["total_cost_usd"] / reference["total_cost_usd"]
    return results


# --------------------------------------------------------------------- main


def render_table(results: dict) -> str:
    header = (
        f"{'strategy':<18}{'quality':>9}{'retained':>10}{'cost $':>11}"
        f"{'vs 70B':>9}{'cheap %':>9}"
    )
    lines = [header, "-" * len(header)]
    order = [
        "always-cheap",
        "always-expensive",
        "random-50/50",
        "length-heuristic",
        "learned-router",
    ]
    for name in order:
        r = results[name]
        lines.append(
            f"{name:<18}{r['mean_quality']:>9.3f}{r['quality_retained_pct']:>9.1f}%"
            f"{r['total_cost_usd']:>11.5f}{r['cost_vs_expensive_pct']:>8.1f}%"
            f"{r['pct_routed_cheap']:>8.1f}%"
        )
    return "\n".join(lines)


async def run(args) -> None:
    df = pd.read_csv(HOLDOUT)
    df["answer"] = df["answer"].fillna("")
    queries = df["query"].tolist()

    cache = json.loads(GENERATIONS_CACHE.read_text()) if GENERATIONS_CACHE.exists() else {}

    provider = None
    client = None
    if args.live:
        if not settings.groq_api_key:
            raise SystemExit(
                "--live needs a GROQ_API_KEY in .env. This harness will not fall back "
                "to the mock provider: mock answers would make every quality number "
                "meaningless rather than merely approximate."
            )
        provider = GroqProvider(
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            timeout=settings.request_timeout_s,
        )
        client = RateLimitedClient(provider, max_tokens=args.max_tokens)

    try:
        cache = await generate_all(queries, client, cache, live=args.live)
        choices, labels, confidences = strategy_choices(df, args.threshold)
        judge = Judge(client)
        results = await score_strategies(df, choices, cache, judge, live=args.live)
    finally:
        if provider:
            await provider.aclose()

    results_path = RESULTS_DIR / "baselines.json"
    payload = {
        "threshold": args.threshold,
        "n_holdout": len(df),
        "generated_by": "groq",
        "judge_llm_calls": judge.llm_calls,
        "strategies": results,
        "router_label_distribution": pd.Series(labels).value_counts().to_dict(),
        "mean_router_confidence": float(sum(confidences) / len(confidences)),
    }
    results_path.write_text(json.dumps(payload, indent=2))

    print()
    print(render_table(results))
    print()

    router = results["learned-router"]
    heuristic = results["length-heuristic"]
    delta = router["quality_retained_pct"] - heuristic["quality_retained_pct"]
    verdict = (
        f"the learned router beats the length heuristic by {delta:+.1f} quality points"
        if delta > 0
        else f"the length heuristic matches or beats the learned router ({delta:+.1f} points)"
    )
    print(
        f"headline: {router['quality_retained_pct']:.1f}% of the large model's quality "
        f"for {router['cost_vs_expensive_pct']:.1f}% of the cost"
    )
    print(f"baseline: {verdict}")
    print(f"\nwrote {results_path.relative_to(ROOT)}  ({judge.llm_calls} judge calls)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="allow API calls for cache misses")
    parser.add_argument("--threshold", type=float, default=policy.DEFAULT_THRESHOLD)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="output cap per generation; the free tier bills 8k tokens/minute",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
