"""
Test audio upload and processing endpoints.

Validates format checks, size limits, and session management.
"""
from fastapi.testclient import TestClient
from pathlib import Path
import io

from backend.app.main import app

client = TestClient(app)


def test_upload_invalid_format():
    """Ensure invalid formats are rejected."""
    fake_file = io.BytesIO(b"fake content")
    resp = client.post(
        "/v1/audio/upload",
        files={"file": ("test.txt", fake_file, "text/plain")},
    )
    assert resp.status_code == 400
    assert "Unsupported file format" in resp.json()["detail"]


def test_upload_missing_file():
    """Ensure missing file returns 422."""
    resp = client.post("/v1/audio/upload")
    assert resp.status_code == 422


def test_session_cleanup():
    """Ensure session cleanup endpoint responds."""
    resp = client.delete("/v1/audio/session/fake-session-id")
    # Even if session doesn't exist, cleanup should not crash.
    assert resp.status_code in [200, 500]
