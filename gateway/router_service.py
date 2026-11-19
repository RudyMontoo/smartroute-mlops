"""Serving wrapper around the difficulty classifier.

Loads whichever version currently holds the `@champion` alias in the MLflow registry,
which is what lets a retrain take effect without redeploying the gateway: promote a new
version, call `reload()`, and the next request is served by it. Falls back to the local
joblib artifact if the registry is unreachable, so the service still starts in CI and
on a laptop with no MLflow running.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import joblib

from gateway.config import settings
from gateway.metrics import MODEL_INFO
from router.train import predict_with_confidence

log = logging.getLogger(__name__)


class RouterService:
    def __init__(self) -> None:
        self._model: Any = None
        self._lock = threading.Lock()
        self.version: str = "unknown"
        self.source: str = "none"

    # ------------------------------------------------------------------ loading

    def _load_from_registry(self) -> tuple[Any, str]:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        uri = f"models:/{settings.registered_model_name}@{settings.model_alias}"
        model = mlflow.sklearn.load_model(uri)

        client = mlflow.MlflowClient()
        info = client.get_model_version_by_alias(
            settings.registered_model_name, settings.model_alias
        )
        return model, str(info.version)

    def _load_from_disk(self) -> tuple[Any, str]:
        if not settings.local_model_path.exists():
            raise FileNotFoundError(
                f"no local model at {settings.local_model_path}; run `python -m router.train`"
            )
        return joblib.load(settings.local_model_path), "local"

    def load(self) -> None:
        """(Re)load the serving model. Registry first, disk as a safety net."""
        with self._lock:
            if settings.model_source == "registry":
                try:
                    model, version = self._load_from_registry()
                    self._model, self.version, self.source = model, version, "registry"
                    log.info("loaded %s v%s from MLflow registry", settings.model_alias, version)
                except Exception as exc:  # noqa: BLE001 - degrade rather than refuse to boot
                    log.warning("registry load failed (%s); falling back to local artifact", exc)
                    self._model, self.version = self._load_from_disk()
                    self.source = "local-fallback"
            else:
                self._model, self.version = self._load_from_disk()
                self.source = "local"

        MODEL_INFO.clear()
        MODEL_INFO.labels(version=self.version, alias=settings.model_alias, source=self.source).set(
            1
        )

    # ------------------------------------------------------------------ serving

    @property
    def ready(self) -> bool:
        return self._model is not None

    def predict(self, query: str) -> tuple[str, float]:
        if self._model is None:
            raise RuntimeError("router model not loaded")
        labels, confidences = predict_with_confidence(self._model, [query])
        return labels[0], confidences[0]

    def predict_batch(self, queries: list[str]) -> tuple[list[str], list[float]]:
        if self._model is None:
            raise RuntimeError("router model not loaded")
        return predict_with_confidence(self._model, queries)

    def info(self) -> dict[str, str]:
        return {
            "version": self.version,
            "alias": settings.model_alias,
            "source": self.source,
            "registered_name": settings.registered_model_name,
        }


router_service = RouterService()
