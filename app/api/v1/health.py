"""
Health and basic readiness endpoints for API v1.

These endpoints let engineers verify the backend is running before heavy ML tasks.
"""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Simple health check endpoint.

    Returns:
        dict: Basic status to verify the app runs.
    """
    return {"status": "ok"}
