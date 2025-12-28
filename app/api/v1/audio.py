"""
Audio upload and processing endpoints (API v1).

Responsibilities:
- Accept audio file uploads (MP3, WAV, M4A).
- Validate formats and size limits.
- Trigger ML processing and return results.
- Provide session cleanup endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from pathlib import Path
import shutil

from ...core.session import new_session_id, session_dir, cleanup_session
from ...models.schemas import AnalysisResponse, ErrorResponse
from ...services.audio_service import process_audio_file

router = APIRouter()

# Allowed audio formats and size limit.
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a"}
MAX_FILE_SIZE_MB = 100


@router.post("/upload", response_model=AnalysisResponse)
async def upload_and_process(
    file: UploadFile = File(...),
    device: str = Query("cpu", description="Device for ML inference: 'cpu' or 'cuda'"),
) -> AnalysisResponse:
    """Upload an audio file, process it, and return analysis results.

    Args:
        file (UploadFile): Audio file to analyze.
        device (str): Processing device (cpu or cuda).

    Returns:
        AnalysisResponse: Complete analysis with metadata, stems, chords, key, tempo, waveform.

    Raises:
        HTTPException: For invalid formats, size limits, or processing errors.
    """
    # Validate file extension.
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {suffix}. Allowed: {ALLOWED_EXTENSIONS}",
        )

    # Create session and save uploaded file.
    session_id = new_session_id()
    sess_dir = session_dir(session_id)
    audio_path = sess_dir / file.filename

    try:
        # Stream file to disk; validate size during write.
        with audio_path.open("wb") as f:
            total_size = 0
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit",
                    )
                f.write(chunk)

        # Process audio through ML pipeline.
        result = process_audio_file(session_id, file.filename, device=device)
        return result

    except FileNotFoundError as e:
        cleanup_session(session_id)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        cleanup_session(session_id)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/stem/{session_id}/{stem_name}")
async def get_stem(session_id: str, stem_name: str) -> FileResponse:
    """Stream a separated stem audio file.

    Args:
        session_id (str): Session identifier.
        stem_name (str): Stem name (vocals, drums, bass, other).

    Returns:
        FileResponse: Audio file stream.

    Raises:
        HTTPException: If session or stem not found.
    """
    # Validate stem name.
    valid_stems = {"vocals", "drums", "bass", "other"}
    if stem_name not in valid_stems:
        raise HTTPException(status_code=400, detail=f"Invalid stem name: {stem_name}")

    # Check if stem file exists.
    sess_dir = session_dir(session_id)
    stem_path = sess_dir / f"{stem_name}.wav"
    
    if not stem_path.exists():
        raise HTTPException(status_code=404, detail=f"Stem not found: {stem_name}")
    
    return FileResponse(
        path=stem_path,
        media_type="audio/wav",
        filename=f"{stem_name}.wav",
    )


@router.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    """Clean up a session and remove all temporary files.

    Args:
        session_id (str): Session identifier to clean up.

    Returns:
        dict: Confirmation message.
    """
    try:
        cleanup_session(session_id)
        return {"status": "ok", "message": f"Session {session_id} cleaned up"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
