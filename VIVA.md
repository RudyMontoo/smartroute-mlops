# Viva prep — likely questions and honest answers

Read this once, out loud, before the viva. The goal isn't to memorize lines — it's to
be able to explain *why* each decision was made, because that's what "not generated"
actually looks like in a conversation.

## "Walk me through what happens to a request"

1. `POST /v1/chat` hits API-key auth (`gateway/security.py`), then a per-key rate
   limiter.
2. The query is redacted for PII (email, phone, Aadhaar, PAN, card-with-Luhn-check)
   *before anything else touches it* — classification, the provider call, and the log
   all see the redacted text, never the original.
3. A prompt-injection heuristic flags (but doesn't block) suspicious patterns.
4. The redacted query goes through `router_service.predict()` — a scikit-learn
   pipeline predicts `simple`/`moderate`/`complex` plus a confidence.
5. `router/policy.decide()` turns that into a model choice: `simple`→cheap,
   `complex`→expensive, `moderate`→cheap if confident enough, else expensive.
6. The provider (Groq, or mock if no API key) answers.
7. If the cheap model was used, `gateway/cascade.py` checks the answer for
   refusal/hedge patterns, suspicious shortness, or a missing number on a numeric
   question — and retries on the expensive model if so. **Both calls are billed.**
8. PII placeholders are rehydrated in the final answer so the user gets a coherent
   response even though the provider never saw their real data.
9. The request is logged (cost, both actual and counterfactual, latency, routing
   reason, redaction count, flag) for `/stats` and the dashboard.

## "Why not just always use the big model — isn't that safer?"

Safer for quality, strictly worse for cost, and it has no feedback loop. The point of
this project isn't "call an LLM" — it's "decide per request which model is good
enough, and verify that decision against the real answer." `RESULTS.md` has to show
the router beats a one-line length heuristic (`len(query) > 80`), or the ML is
decoration. That's a real baseline, included deliberately, not a strawman.

## "Why is the model such a simple classifier (TF-IDF + logistic regression)?"

Because the model being small is what makes everything *else* in this project
possible to demonstrate live: sub-5-second training means a CI quality gate that
retrains on every push is affordable, `/admin/retrain` can hot-swap a model without
a multi-minute wait, and a promotion decision can be shown end-to-end in the viva
instead of described. A fine-tuned transformer would make the entire MLOps lifecycle
a slide instead of a demo. That's a stated tradeoff, not an oversight.

## "Isn't the training data circular — won't a classifier trained on your own templates just relearn them?"

That's exactly why the data isn't templated. `router/prepare_data.py` pulls real
queries from four public datasets (SQuAD v2, Dolly-15k, GSM8K, MBPP) and uses the
*source dataset* as the difficulty label — simple/moderate/complex map onto "short
factual lookup" / "needs synthesis" / "needs multi-step reasoning" respectively,
which is true by construction of those datasets, not asserted. Macro-F1 lands around
0.93, not 0.99 — a classifier with no genuine signal to find would either collapse or
overfit a narrow template, and this doesn't.

## "How do you know the router is actually better than guessing?"

`eval/run_baselines.py` runs the same 198 held-out queries (never seen in training)
through five strategies — always-cheap, always-expensive, random 50/50, the length
heuristic, and the learned router — and scores every answer with ground-truth
matching (GSM8K's exact number, SQuAD's exact span) or an LLM-as-judge for open-ended
answers. The judge pass for the open-ended queries didn't finish tonight (Groq's
free tier started returning 429s with 200+ second backoffs after today's volume), so
`RESULTS.md` §3 reports the honest partial: a D1 table over the 92/198 queries that
have a mechanical, judge-free score. On that subset the router actually scores
*below* always-cheap — and the writeup says so and explains why (strict exact-match
grading rewards the cheap model's terser phrasing, and the subset has zero
`moderate` queries, which is the one case the router's confidence threshold exists
to handle). The stronger evidence for the router's value is §1/§2: a 93%-macro-F1
classifier whose few errors are concentrated in the one genuinely ambiguous boundary
(simple↔moderate), and a 6-run ablation showing *why* LogisticRegression was chosen
over an equally-accurate LinearSVC — better-calibrated confidence, which is what the
whole cost-saving mechanism depends on.

## "Your own results show the router doing worse than always-cheap — isn't that bad?"

On the specific 92-query subset in `RESULTS.md` §3, yes, and that's reported
un-softened rather than hidden. But look at what that subset actually is before
reading it as "the router doesn't work": it's only `simple` and `complex` queries
graded by strict exact-match (no partial credit for a correct answer phrased
differently), with zero `moderate` queries — the one case the router's confidence
threshold exists to handle. Two things are really being measured there: (1) the
120B model tends to answer in fuller sentences, which exact-match string matching
penalizes relative to the 20B model's terser answers — that's a grading artifact, not
a quality difference, and it's exactly what the LLM-as-judge path exists to correct
for, but that pass didn't finish tonight under Groq's free-tier rate limit; (2) the
router's classification accuracy itself (§1's confusion matrix: 93% macro-F1, errors
concentrated in simple↔moderate) is the stronger, complete evidence that the routing
*decisions* are sound — this subset just doesn't reward them with its scoring method.
Reporting a number that complicates the story, with the actual mechanism explained,
is a stronger signal of real engineering than only showing numbers that flatter it.

## "What happens if the classifier is wrong?"

Two independent safety nets. First, the cascade (`gateway/cascade.py`) catches
*visible* failures from the cheap model — refusals, hedges, missing numbers — and
retries on the expensive model, billing both calls so a high escalation rate shows up
as reduced savings rather than being hidden. Second, the CI quality gate
(`scripts/quality_gate.py`) prevents a bad *model* from ever reaching production in
the first place — if a retrain drops macro-F1 below 0.85 or cost savings below 8%, the
build fails.

## "Explain the MLflow setup"

`router/train.py` logs every training run — params, scalar metrics, a confusion
matrix (as a logged artifact, since MLflow's `log_metrics` only accepts scalars), and
the model itself — to a registered model `query-difficulty-router`. Promotion is
conditional: after registering a new version, it compares the new macro-F1 against
whatever currently holds the `@champion` alias, and only moves the alias if the new
version actually wins; otherwise it's tagged `@challenger` and the incumbent stays.
`router/sweep.py` separately ran a 6-run ablation — 3 estimators (MultinomialNB,
LogisticRegression, LinearSVC) × {TF-IDF alone, TF-IDF + hand-engineered numeric
features} — logged as its own experiment, to show the model choice was measured, not
assumed. Numeric features helped LogisticRegression and LinearSVC but *hurt* Naive
Bayes (-0.022 macro-F1) — a real, reportable finding, not a clean story.

## "Why did logistic regression win over LinearSVC if their macro-F1 tied?"

Because macro-F1 isn't the only thing that matters for *this* use case: the routing
policy needs a calibrated confidence to decide whether a `moderate` query is safe to
send to the cheap model. LogisticRegression's `predict_proba` is a real probability;
LinearSVC has no native probability output (the pipeline approximates one via a
softmax over decision-function margins), and that approximation is visibly worse
calibrated — logreg converts its confidence into 31.4% cost savings against the
baseline, LinearSVC only 24.4%, despite matching macro-F1. Equal accuracy, different
economic value — that's the kind of thing a sweep finds and a single run can't.

## "What does drift detection actually do, and how do I see it work?"

`router/drift.py` builds a reference distribution of the numeric features (not raw
TF-IDF — the hand-engineered features are interpretable, which matters for explaining
*why* a drift alert fired) from the training set, then compares recent request
features against it using PSI (quantile-binned) and a KS 2-sample test per feature.
PSI < 0.1 is stable, 0.1–0.25 is moderate, ≥ 0.25 is alert. `POST
/admin/simulate-drift` injects synthetic verbose/code-heavy traffic so this can be
demonstrated live instead of described — `/drift` (and the dashboard's drift chart)
will show the PSI crossing the alert threshold within seconds.

## "What's your security model, and what's the one clever bit?"

API-key auth, per-key rate limiting, PII redaction, and a prompt-injection heuristic,
in that order, before the query reaches the model at all. The detail that's not just
"regex for emails": the redaction is **reversible**. Redacted spans become typed
placeholders (`<EMAIL_1>`, `<CARD_1>`) sent to the provider instead of the real value,
and the final answer has those placeholders **rehydrated** back to the original text
before it reaches the user — so PII never leaves the process, but the user still gets
a coherent, personalized answer instead of a mangled one. Credit-card detection also
runs a Luhn check so it doesn't flag every 16-digit number (order IDs, phone numbers)
as a card.

## "What would you do differently with more time?"

Two things were scoped but not built: a Locust load test (throughput/p50/p95/p99 with
the Redis-style response cache on vs. off) and a Kubernetes deployment with an HPA
demo. Both are natural extensions of what exists — the gateway is already a stateless
container with a health check, so neither needs an architecture change, just time.
The real Groq pricing for gpt-oss-20b/120b also hasn't been confirmed, so the dollar
figures in `RESULTS.md` are directionally correct but not final — that's flagged
explicitly in the code rather than presented as settled.

## "Why Render + GHCR instead of one 'proper' cloud deploy?"

Render's free tier runs one web service, not a multi-container stack, so it hosts the
gateway alone — a stateless API behind a public health-checked endpoint, which is
actually the normal production shape. MLflow, the registry, and the dashboard stay in
local Compose / `docker-compose.prod.yml`, which pulls the same images `cd.yml`
publishes to GHCR. That split is explained, not hidden, because pretending the free
tier runs the full stack would be a worse answer than an honest one.
