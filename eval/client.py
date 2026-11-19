"""A strict, rate-limit-aware client for offline evaluation.

Why this exists instead of reusing `ProviderPool`: the gateway's pool falls back to the
mock provider when Groq fails, which is the right behaviour for a live service and
exactly the wrong behaviour for an experiment. A mock answer silently cached as if it
were a real generation would corrupt the measured quality of every strategy -- which is
precisely what happened on the first run here, and was only recoverable because every
cached entry records which provider produced it.

So this client fails loudly instead of degrading, and paces itself against the real
rate-limit headers: the free tier allows 8,000 tokens per minute, which is the binding
constraint on a ~280k-token evaluation.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

import httpx

from gateway.providers import Completion, GroqProvider

log = logging.getLogger(__name__)

#: Stay this far clear of the per-minute token ceiling before pausing.
TOKEN_HEADROOM = 1200


def parse_duration(value: str | None) -> float:
    """Groq expresses resets as '31m40.8s', '577ms', '2.5s'."""
    if not value:
        return 0.0
    total = 0.0
    for amount, unit in re.findall(r"([\d.]+)\s*(ms|m(?!s)|s|h)", value):
        scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}[unit]
        total += float(amount) * scale
    if not total:
        try:
            total = float(value)
        except ValueError:
            total = 0.0
    return total


class RateLimitedClient:
    """Serialises calls and sleeps when the token budget runs low.

    Concurrency is deliberately 1. With an 8k tokens-per-minute ceiling, parallelism
    buys nothing -- it just converts throughput into 429s.
    """

    def __init__(self, provider: GroqProvider, max_tokens: int = 256) -> None:
        self._provider = provider
        self._lock = asyncio.Lock()
        self.max_tokens = max_tokens
        self.calls = 0
        self.tokens_used = 0
        self.waits = 0

    async def complete(self, model: str, prompt: str) -> Completion:
        async with self._lock:
            return await self._complete_with_retry(model, prompt)

    async def _complete_with_retry(self, model: str, prompt: str) -> Completion:
        for attempt in range(6):
            await self._respect_budget()
            try:
                completion = await self._provider.complete(model, prompt, self.max_tokens)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429 or attempt == 5:
                    raise
                wait = (
                    parse_duration(exc.response.headers.get("retry-after"))
                    or parse_duration(exc.response.headers.get("x-ratelimit-reset-tokens"))
                    or 2 ** (attempt + 1)
                )
                self.waits += 1
                log.info("429; sleeping %.1fs", wait + 0.5)
                await asyncio.sleep(wait + 0.5)
                continue

            self.calls += 1
            self.tokens_used += completion.input_tokens + completion.output_tokens
            return completion

        raise RuntimeError("exhausted retries against the provider")

    async def _respect_budget(self) -> None:
        """Pause pre-emptively when the remaining per-minute token budget is thin."""
        headers = self._provider.last_rate_limit
        remaining = headers.get("x-ratelimit-remaining-tokens")
        if remaining is None:
            return
        try:
            if int(remaining) >= TOKEN_HEADROOM:
                return
        except ValueError:
            return

        wait = parse_duration(headers.get("x-ratelimit-reset-tokens")) + 1.0
        if wait > 0:
            self.waits += 1
            log.info("token budget low (%s left); sleeping %.1fs", remaining, wait)
            await asyncio.sleep(wait)

    def summary(self) -> str:
        return f"{self.calls} calls, ~{self.tokens_used:,} tokens, {self.waits} rate-limit pauses"


class Stopwatch:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started
