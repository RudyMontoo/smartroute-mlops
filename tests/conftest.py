"""Test configuration.

The environment is set up *before* any gateway module is imported, because
`gateway.config` builds its Settings singleton at import time. Tests run against the
local joblib artifact and the mock provider so they never touch MLflow or the network.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="smartroute-tests-"))

os.environ["MODEL_SOURCE"] = "local"
os.environ["FORCE_MOCK"] = "true"
os.environ["DB_PATH"] = str(_TMP / "gateway.db")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from gateway.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    """TestClient as a context manager so FastAPI's lifespan actually runs."""
    with TestClient(app) as test_client:
        yield test_client
