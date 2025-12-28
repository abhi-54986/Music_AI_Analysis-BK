"""
Top-level API router aggregator.

Responsibility:
- Mount versioned routers (v1, future versions).
"""
from fastapi import APIRouter

from .v1 import routes as v1_routes

api_router = APIRouter()
api_router.include_router(v1_routes.router, prefix="/v1")
