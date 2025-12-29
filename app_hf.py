"""
Hugging Face Spaces entry point.
This file is required by HF Spaces to run the FastAPI app.
"""
import os
import uvicorn
from app.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
