# SmartRoute

An LLM gateway that classifies every incoming query as `simple` / `moderate` / `complex`,
routes it to a cheap or expensive model accordingly, verifies the cheap answer and
escalates to the expensive one if it looks wrong, redacts PII before anything leaves the
process, and logs enough to prove — with numbers, not claims — how much that saves and
what it costs in quality.

It is also the vehicle for a full MLOps lifecycle: a trained model behind an API, a
model registry with promotion rules, CI that can fail a build for being a *worse model*
rather than broken code, drift detection against the training distribution, and a
retrain-and-promote path that doesn't require a restart.

## Why a router instead of always calling the big model

Because "always use the expensive model" isn't actually the safe default — it's the
one with no feedback loop. The interesting engineering is in deciding, *per request*,
which model is good enough, and then checking that guess against the real answer. See
[`RESULTS.md`](RESULTS.md) for the measured tradeoff, including a baseline the router
has to beat to justify existing at all (a one-line length heuristic).

## Architecture

```
  browser ──▶ frontend (React+Vite+Recharts, nginx :3000)
                  │ REST
                  ▼
  ┌─────────────────────────────────────────────────────────┐
  │                    gateway  :8000                        │
  │  API-key auth → rate limit → PII redact → injection guard│
  │       → classify (sklearn) → route → cascade escalate   │──▶ Groq gpt-oss-20b / 120b
  └──┬─────────────────────────────┬──────────────┬─────────┘    (mock fallback if no key)
     │ SQLite: requests, audit log │ /metrics       │ loads @champion on boot
     ▼                              ▼                ▼
  data/gateway.db              Prometheus format   MLflow registry :5000

  offline, run once, committed:  eval/  ──▶ RESULTS.md (D1 baseline table, D2 ablation)
  CI gate:                       scripts/quality_gate.py ──▶ fails the build on model regression
  cloud:                         GHCR (gateway + frontend images) + Render (gateway, free tier)
```

## Repo layout

```
router/            prepare_data.py, features.py, train.py, sweep.py, drift.py
                    — the model: data prep, the one shared featurizer, training +
                    MLflow registration + conditional promotion, the ablation sweep,
                    and PSI/KS drift detection, all importing the same feature code
                    so there's no train/serve skew.
gateway/            main.py (FastAPI), providers.py (Groq + mock), cascade.py,
                    security.py (auth/rate-limit/PII/injection), router_service.py
                    (loads @champion from the registry), metrics.py, drift_demo.py
eval/               run_baselines.py (D1: 5 strategies over a held-out set),
                    judge.py (ground-truth match + LLM-as-judge), client.py
                    (a strict, rate-limit-aware Groq client separate from the
                    gateway's auto-fallback one — an eval run must never silently
                    score a mock answer), results/ (cached, committed)
frontend/           React dashboard: hero strip, live request feed, charts, control panel
tests/              75 tests: api, router, drift, cascade, security, quality gate
scripts/            quality_gate.py — the CI model-quality gate
.github/workflows/  ci.yml (lint + test + quality gate + docker build+smoke-test),
                    cd.yml (GHCR publish on green main)
data/                DVC-tracked: queries.csv (train), holdout.csv (eval-only),
                    reference_features.json (drift reference)
```

## Syllabus phase → deliverable map

| Phase | Covered by |
|---|---|
| **1. ML foundation, FastAPI** | `router/train.py` (sklearn `Pipeline` + `ColumnTransformer`), `gateway/main.py` FastAPI service |
| **2. Docker** | `gateway/Dockerfile` (multi-stage: build venv → train a fallback model into the image → slim non-root runtime), `frontend/Dockerfile`, `docker-compose.yml` |
| **3. Testing + CI/CD** | `tests/` (75 tests, pytest + FastAPI `TestClient`); `ci.yml` = ruff + pytest + **model quality gate** + Docker build & smoke-test; `cd.yml` publishes to GHCR on success |
| **4. MLflow tracking + registry** | `mlflow` service; `router/sweep.py` logs a 6-run ablation; `router/train.py` registers versions and manages `@champion`/`@challenger` aliases with conditional promotion (new version only wins if its macro-F1 beats the incumbent) |
| **5. Container registry + cloud deploy** | `cd.yml` → `ghcr.io/<owner>/smartroute-{gateway,frontend}`; `render.yaml` deploys the gateway to Render's free tier; `docker-compose.prod.yml` pulls the published images instead of building |
| **6. Monitoring + drift detection** | `/metrics` (Prometheus exposition), `/drift` (PSI + KS-test vs. the training distribution), `/admin/simulate-drift` (injects shifted traffic so drift visibly fires), drift chart in the React dashboard |
| **7. Automated retraining + promotion** | `POST /admin/retrain` retrains, evaluates, and only promotes `@champion` if the new model's macro-F1 beats the current one — then the gateway hot-swaps the model with no restart |

## The model

**Task:** 3-class query difficulty (`simple` / `moderate` / `complex`), used to route to
`openai/gpt-oss-20b` (cheap) or `openai/gpt-oss-120b` (expensive) on Groq.

**Data is real, not templated.** Templated synthetic queries would make the classifier
circular — it would just relearn its own templates, score ~99%, and tell you nothing.
`router/prepare_data.py` instead builds ~900 queries from public datasets, by source:

| Label | Source | Why that's honestly the difficulty |
|---|---|---|
| `simple` | SQuAD v2 questions | short factual lookups |
| `moderate` | Dolly-15k (open_qa / general_qa / brainstorming) | needs synthesis, not multi-step reasoning |
| `complex` | GSM8K word problems + MBPP code prompts | multi-step reasoning; small models measurably fail |

A 198-row stratified `holdout.csv` is split off *before* training and touched only by
`eval/`. Realistic macro-F1 is ~0.93, not 0.99 — see the ablation in `RESULTS.md`.

**Features** (`router/features.py`, one module imported by training, serving, and
drift — so there's no train/serve skew): TF-IDF word 1–2 grams plus hand-engineered
numeric features (length, word count, `?` count, digit ratio, code-fence presence,
reasoning/code/explain keyword flags).

**The model is deliberately simple.** Sub-5-second training is what makes a CI quality
gate, a live `/admin/retrain`, and a sub-5-second promotion demo all possible in the
first place — a fine-tuned transformer would make the entire MLOps lifecycle
impossible to exercise live. That's a tradeoff, stated rather than hidden.

## Cascade escalation

Classification happens *before* the answer exists, so it's a guess. When the cheap
model is used, `gateway/cascade.py` inspects the response — refusal/hedge patterns, a
suspiciously short answer, a numeric question answered without a number — and retries
on the expensive model if it looks wrong. Both calls are billed, so an escalation that
fires too often shows up as lower measured savings instead of being hidden.

## Security & responsible-AI layer

Request path, in order (`gateway/security.py`, `gateway/main.py`):

1. **API-key auth** — `X-API-Key` checked against SHA-256-hashed keys.
2. **Per-key rate limiting** — token bucket, `429` + `Retry-After`.
3. **PII redaction before anything else sees the text** — regex detectors for email,
   phone, Aadhaar, PAN, and credit card (**Luhn-validated**, so it doesn't flag every
   16-digit number). Redacted spans become typed placeholders (`<EMAIL_1>`) that are
   **rehydrated in the response**, so the user still gets a coherent answer while the
   provider never saw the real value.
4. **Prompt-injection guard** — heuristic detection of instruction-override and
   role-hijack patterns; flagged and logged, not silently dropped.
5. **Audit trail** — every request logs redaction counts and the flag, never the raw
   PII it redacted.

Tested in `tests/test_security.py` with planted PII and injection payloads.

## Gateway API

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/chat` | auth → redact → guard → classify → route → cascade → log → answer + routing metadata |
| GET | `/health` | liveness + current model version/alias/source |
| GET | `/stats` | cost saved, per-model counts, cost timeline |
| GET | `/requests/recent` | live feed for the dashboard |
| GET | `/drift` | per-feature PSI/KS + verdict |
| GET | `/metrics` | Prometheus exposition |
| POST | `/admin/retrain` | retrain → evaluate → conditional promote → hot-swap, no restart |
| POST | `/admin/simulate-drift` | demo affordance — injects shifted synthetic traffic so drift visibly crosses the threshold |

## Running it

```bash
cp .env.example .env    # add GROQ_API_KEY, or leave blank to run entirely on the mock provider
docker compose up --build
# gateway  -> http://localhost:8000/health
# mlflow   -> http://localhost:5000
# frontend -> http://localhost:3000
```

Without Docker:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m router.prepare_data && python -m router.train --no-mlflow
uvicorn gateway.main:app --reload
```

Tests and the model quality gate:

```bash
pytest
python -m router.train --no-mlflow && python -m scripts.quality_gate
```

Reproduce the headline evaluation numbers from the committed cache (no API key needed):

```bash
python -m eval.run_baselines
```

## What's deliberately not here

Render's free tier runs one web service, not a multi-container stack, so the public
deploy is the gateway alone — MLflow, the frontend, and the rest of the compose stack
stay local. That's the normal production split (a stateless API behind a public
endpoint, infra services kept internal), not a corner cut.

A Locust load-test benchmark and a Kubernetes/`kind` deployment were scoped but not
built in the time available; both are straightforward extensions of what's here
(the gateway is already a stateless container with a health check) rather than
architectural gaps.

## Caveat on cost numbers

`router/policy.py`'s per-token prices for `gpt-oss-20b`/`gpt-oss-120b` are placeholder
values, not confirmed against Groq's current pricing page — see the comment in that
file and the note in `RESULTS.md`. The *relative* routing decisions and quality
measurements are unaffected; only the absolute dollar figures would shift.
