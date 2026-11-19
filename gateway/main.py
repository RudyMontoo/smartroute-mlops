"""SmartRoute gateway — FastAPI application.

Request path:  classify -> route -> call provider -> log
Later blocks extend this with the cascade verifier, the response cache and the
security middleware; the ordering of those concerns lives in `_handle_chat`.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from gateway import db, metrics
from gateway.config import settings
from gateway.providers import ProviderPool
from gateway.router_service import router_service
from router import policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("smartroute")

pool: ProviderPool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    db.init_db()
    router_service.load()
    pool = ProviderPool()
    log.info("gateway ready: provider=%s model=%s", pool.mode, router_service.info())
    yield
    if pool:
        await pool.aclose()


app = FastAPI(
    title="SmartRoute LLM Gateway",
    description=(
        "Routes each query to a cheap or expensive language model using a trained "
        "difficulty classifier, and measures what that saves."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------- schemas


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override the confidence threshold for this request.",
    )


class RoutingInfo(BaseModel):
    predicted_label: str
    confidence: float
    model: str
    reason: str
    escalated: bool = False
    cached: bool = False
    model_version: str


class UsageInfo(BaseModel):
    input_tokens: int
    output_tokens: int
    cost_usd: float
    baseline_cost_usd: float
    saved_usd: float
    latency_ms: float
    provider: str


class ChatResponse(BaseModel):
    answer: str
    routing: RoutingInfo
    usage: UsageInfo


# --------------------------------------------------------------------- routes


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {
        "status": "ok" if router_service.ready else "degraded",
        "model": router_service.info(),
        "provider": pool.mode if pool else "uninitialised",
        "threshold": settings.router_threshold,
    }


@app.post("/v1/chat", response_model=ChatResponse, tags=["gateway"])
async def chat(request: ChatRequest) -> ChatResponse:
    if pool is None or not router_service.ready:
        raise HTTPException(status_code=503, detail="gateway not ready")
    return await _handle_chat(request)


async def _handle_chat(request: ChatRequest) -> ChatResponse:
    started = time.perf_counter()
    threshold = request.threshold if request.threshold is not None else settings.router_threshold

    label, confidence = router_service.predict(request.query)
    decision = policy.decide(label, confidence, threshold)

    completion = await pool.complete(decision.model, request.query)

    # The counterfactual: what this exact request would have cost had we sent every
    # query to the large model. Measured per request, not extrapolated.
    baseline_cost = policy.cost_usd(
        policy.EXPENSIVE, completion.input_tokens, completion.output_tokens
    )
    elapsed_ms = (time.perf_counter() - started) * 1000

    metrics.REQUESTS.labels(model=decision.model, provider=completion.provider).inc()
    metrics.ROUTING_DECISIONS.labels(predicted_label=label, model=decision.model).inc()
    metrics.LATENCY.labels(model=decision.model).observe(elapsed_ms / 1000)
    metrics.COST.labels(kind="actual").inc(completion.cost_usd)
    metrics.COST.labels(kind="baseline").inc(baseline_cost)
    if completion.provider == "mock" and pool.mode == "groq":
        metrics.PROVIDER_FALLBACKS.inc()

    await db.log_request(
        db.RequestRecord(
            query=request.query,
            predicted_label=label,
            confidence=confidence,
            model=decision.model,
            reason=decision.reason,
            provider=completion.provider,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cost_usd=completion.cost_usd,
            baseline_cost_usd=baseline_cost,
            latency_ms=elapsed_ms,
            model_version=router_service.version,
        )
    )

    return ChatResponse(
        answer=completion.text,
        routing=RoutingInfo(
            predicted_label=label,
            confidence=confidence,
            model=decision.model,
            reason=decision.reason,
            model_version=router_service.version,
        ),
        usage=UsageInfo(
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
            cost_usd=completion.cost_usd,
            baseline_cost_usd=baseline_cost,
            saved_usd=baseline_cost - completion.cost_usd,
            latency_ms=elapsed_ms,
            provider=completion.provider,
        ),
    )


@app.get("/stats", tags=["dashboard"])
async def stats() -> dict:
    return {**db.stats(), "model": router_service.info(), "timeline": db.cost_timeline()}


@app.get("/requests/recent", tags=["dashboard"])
async def recent(limit: int = 25) -> list[dict]:
    return db.recent(min(limit, 200))


@app.get("/metrics", tags=["ops"], include_in_schema=False)
async def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
