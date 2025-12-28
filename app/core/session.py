"""
Session management utilities.

Responsibility:
- Create and manage session-scoped directories.
- Provide helpers to clean up temporary files when sessions end.
"""
from __future__ import annotations
from pathlib import Path
import uuid
from .config import BASE_TEMP_DIR

def new_session_id() -> str:
    """Generate a new unique session identifier.

    Returns:
        str: A UUID4-based session ID.
    """
    return str(uuid.uuid4())

def session_dir(session_id: str) -> Path:
    """Get the directory path for a given session.

    Args:
        session_id (str): The session identifier.

    Returns:
        Path: The filesystem path to the session's temp directory.
    """
    d = BASE_TEMP_DIR / session_id
    d.mkdir(parents=True, exist_ok=True)
    return d

def cleanup_session(session_id: str) -> None:
    """Remove the session directory and its contents.

    Args:
        session_id (str): The session identifier.
    """
    d = BASE_TEMP_DIR / session_id
    if d.exists():
        # Remove files and directory recursively.
        for p in d.glob("**/*"):
            if p.is_file():
                p.unlink(missing_ok=True)
        for p in sorted(d.glob("**/*"), reverse=True):
            if p.is_dir():
                p.rmdir()
        d.rmdir()
