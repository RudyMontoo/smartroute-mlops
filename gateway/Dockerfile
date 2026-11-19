# Multi-stage: build the virtualenv once, then ship a slim runtime without pip,
# build tools, or the wheel cache.
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt


# --- model stage -------------------------------------------------------------
# Train a model into the image. At runtime the gateway prefers the @champion from
# the MLflow registry, but baking an artifact in means the container still starts
# and serves if the registry is unreachable -- which is exactly the situation where
# you least want your gateway refusing to boot.
#
# data/queries.csv is DVC-tracked, not committed to git, so a fresh checkout (this
# build, CI, Render) never has it on disk. Rather than wiring DVC remote credentials
# into every build environment, the build regenerates the dataset from the same public
# sources prepare_data.py always uses -- which is the point of a reproducible pipeline:
# the data doesn't need to travel with the code, it needs to be re-derivable from it.
FROM builder AS model

WORKDIR /build
COPY router/ router/
RUN /opt/venv/bin/python -m router.prepare_data \
 && /opt/venv/bin/python -m router.train --no-mlflow \
 && /opt/venv/bin/python -m router.drift


# --- runtime -----------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=model /build/models /app/models
COPY --from=model /build/data/reference_features.json /app/data/reference_features.json

COPY router/ router/
COPY gateway/ gateway/

RUN mkdir -p /app/data && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
