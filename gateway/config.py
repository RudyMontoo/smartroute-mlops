"""Runtime configuration, read from the environment (and .env in development)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- provider ---
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    request_timeout_s: float = 30.0
    max_output_tokens: int = 512

    #: Force the mock provider even when a key is present. Used by the load test so the
    #: benchmark measures this service rather than Groq's latency.
    force_mock: bool = False

    # --- model source ---
    #: "registry" loads the @champion from MLflow so promotions take effect without a
    #: redeploy; "local" reads models/router.joblib, which is what CI and the tests use.
    model_source: str = "registry"
    mlflow_tracking_uri: str = f"sqlite:///{ROOT / 'mlflow.db'}"
    registered_model_name: str = "query-difficulty-router"
    model_alias: str = "champion"
    local_model_path: Path = ROOT / "models" / "router.joblib"

    # --- routing ---
    router_threshold: float = 0.60

    # --- storage ---
    db_path: Path = ROOT / "data" / "gateway.db"

    @property
    def use_mock(self) -> bool:
        return self.force_mock or not self.groq_api_key


settings = Settings()
