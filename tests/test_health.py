"""
Health endpoint tests.

Validates that the API surface responds before heavy ML tasks run.
"""
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health_ok():
    """Ensure /v1/health responds with status ok."""
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload.get("status") == "ok"
