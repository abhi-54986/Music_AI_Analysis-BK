"""
Configuration helpers for the backend.

Responsibility:
- Centralize environment variables and application constants.
- Provide paths for session-scoped temporary storage.
"""
from pathlib import Path
import os

APP_NAME = "music-ai-backend"

# Base temp directory under the user's TEMP folder.
BASE_TEMP_DIR = Path(os.environ.get("TEMP", os.environ.get("TMP", ""))) / "music-ai" / "sessions"
BASE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
