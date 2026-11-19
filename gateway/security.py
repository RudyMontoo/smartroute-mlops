"""Security building blocks: API-key auth, rate limiting, PII redaction, injection guard.

`gateway/main.py` is what sequences these -- `require_api_key` as a route dependency,
then `redact`/`check_injection` at the top of `_handle_chat` -- in this order:

    auth -> rate limit -> PII redact -> injection guard

Authenticate before anything else runs (no point rate-limiting an unauthenticated
caller's retries). Redact before the query ever leaves this process -- rehydration
happens on the way back out, so the user still gets a coherent answer and the
third-party provider never sees the raw PII at all, not even in the request we send.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from threading import Lock

# --------------------------------------------------------------------- auth

#: Demo keys. In a real deployment these would be hashed and stored externally;
#: hashing here is still worthwhile so a log/traceback leak doesn't hand out the
#: literal key.
_DEMO_KEYS = {"demo-key-smartroute", "viva-demo-key"}
_KEY_HASHES = {hashlib.sha256(k.encode()).hexdigest() for k in _DEMO_KEYS}


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def is_valid_key(raw_key: str | None) -> bool:
    return bool(raw_key) and hash_key(raw_key) in _KEY_HASHES


# -------------------------------------------------------------- rate limiting


class TokenBucket:
    """Classic token bucket, one per API key, refilled continuously."""

    def __init__(self, capacity: int, refill_per_second: float) -> None:
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self.tokens = float(capacity)
        self.updated = time.monotonic()

    def take(self, cost: float = 1.0) -> tuple[bool, float]:
        now = time.monotonic()
        elapsed = now - self.updated
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self.updated = now

        if self.tokens >= cost:
            self.tokens -= cost
            return True, 0.0

        missing = cost - self.tokens
        retry_after = missing / self.refill_per_second
        return False, retry_after


class RateLimiter:
    """Per-key token buckets behind a lock -- request volume here is low enough that
    a single lock is simpler and plenty fast, and simple is worth more than fast at
    this scale."""

    def __init__(self, capacity: int = 20, refill_per_second: float = 1.0) -> None:
        self._capacity = capacity
        self._refill = refill_per_second
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = Lock()

    def check(self, key_id: str) -> tuple[bool, float]:
        with self._lock:
            bucket = self._buckets.setdefault(key_id, TokenBucket(self._capacity, self._refill))
            return bucket.take()


rate_limiter = RateLimiter()

# ------------------------------------------------------------- PII redaction

#: Ordered so a 16-digit card number is caught before a looser pattern could split it.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\d{10}\b")),
]


def _luhn_valid(digits: str) -> bool:
    """Standard Luhn check digit algorithm. Without this, a 16-digit order number or
    phone number gets masked as `<CARD_1>` and the user sees garbled placeholders
    in an answer that never touched a real card."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass
class RedactionResult:
    text: str
    redacted: list[tuple[str, str]] = field(default_factory=list)  # (placeholder, original)

    @property
    def count(self) -> int:
        return len(self.redacted)

    def rehydrate(self, text: str) -> str:
        """Swap placeholders back in so the model's answer reads naturally, without
        the model ever having seen the real value it is echoing back."""
        for placeholder, original in self.redacted:
            text = text.replace(placeholder, original)
        return text


def redact(text: str) -> RedactionResult:
    result = RedactionResult(text=text)
    counters: dict[str, int] = {}

    def replace(match: re.Match, kind: str) -> str:
        value = match.group(0)
        if kind == "CARD":
            digits = re.sub(r"[ -]", "", value)
            if not (13 <= len(digits) <= 19) or not _luhn_valid(digits):
                return value  # not a real card number; leave it alone

        counters[kind] = counters.get(kind, 0) + 1
        placeholder = f"<{kind}_{counters[kind]}>"
        result.redacted.append((placeholder, value))
        return placeholder

    for kind, pattern in _PATTERNS:
        text = pattern.sub(lambda m, k=kind: replace(m, k), text)

    result.text = text
    return result


# -------------------------------------------------------- prompt injection

_INJECTION_PATTERNS = re.compile(
    r"\b(ignore (all |the |your )?(previous|prior|above) instructions|"
    r"disregard (all |the |your )?(previous|prior|above)|"
    r"you are now (in )?(dan|developer mode|jailbreak)|"
    r"reveal your (system prompt|instructions)|"
    r"act as (if you (are|have)|an unrestricted)|"
    r"forget (everything|all) (you were told|above))\b",
    re.IGNORECASE,
)


def check_injection(text: str) -> bool:
    """True if the query looks like an attempt to override the system's instructions
    rather than ask it a question. Heuristic and intentionally narrow: the cost of a
    false positive here is a flagged-but-still-answered request (see main.py), so it
    errs toward precision over recall."""
    return bool(_INJECTION_PATTERNS.search(text or ""))
