"""
API v1 router definitions.

Aggregates all v1 endpoints (health, session, processing).
"""
from fastapi import APIRouter

from . import health, audio

router = APIRouter()
router.include_router(health.router, tags=["health"])
router.include_router(audio.router, prefix="/audio", tags=["audio"])
