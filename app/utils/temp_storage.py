"""
Temporary storage helpers.

Responsibility:
- Provide convenience functions to work with session-scoped files.
- Ensure all writes happen under temporary directories.
"""
from pathlib import Path
from typing import Union
from ..core.session import session_dir


def path_in_session(session_id: str, *parts: Union[str, Path]) -> Path:
    """Construct a path under the session directory.

    Args:
        session_id (str): Session identifier.
        parts (Union[str, Path]): Path components under the session dir.

    Returns:
        Path: Joined path inside the session directory.
    """
    base = session_dir(session_id)
    return base.joinpath(*[str(p) for p in parts])
