"""
FastAPI application entrypoint.

Responsibilities:
- Build and configure the FastAPI app instance.
- Register routers, middleware, and session management helpers.

Note: Endpoint implementations live in versioned routers under app/api.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.router import api_router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance ready to run.
    """
    app = FastAPI(title="Music AI Backend", version="0.1.0")

    _configure_cors(app)
    _register_routes(app)

    return app


def _configure_cors(app: FastAPI) -> None:
    """Attach CORS middleware for local dev and future deploys.

    Args:
        app (FastAPI): Application instance to configure.
    """
    app.add_middleware(
        CORSMiddleware,
        # Allow all origins for public deployment
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _register_routes(app: FastAPI) -> None:
    """Include versioned API routers.

    Args:
        app (FastAPI): Application instance to configure.
    """
    app.include_router(api_router)
    
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "service": "Music AI Backend",
            "version": "0.1.0",
            "endpoints": {
                "health": "/health",
                "upload": "/v1/audio/upload",
                "cleanup": "/v1/audio/session/{session_id}"
            }
        }


app = create_app()
