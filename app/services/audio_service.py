"""
Audio processing orchestration service.

Responsibilities:
- Coordinate between FastAPI endpoints and ML layer.
- Manage session-scoped temporary files.
- Convert ML outputs to API response models.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

# Add ML package to path for imports.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml" / "src"))

from musicai_ml import analyze_audio
from ..core.session import session_dir
from ..models.schemas import (
    AnalysisResponse,
    AudioMetadata,
    WaveformData,
    ChordSegment,
    StemReferences,
)


def process_audio_file(
    session_id: str,
    audio_filename: str,
    device: str = "cpu",
) -> AnalysisResponse:
    """Process an uploaded audio file and return complete analysis.

    Args:
        session_id (str): Session identifier for temp file management.
        audio_filename (str): Filename of the uploaded audio in the session dir.
        device (str): Device for ML inference ("cpu" or "cuda").

    Returns:
        AnalysisResponse: Structured analysis result with metadata, stems, and insights.
    """
    sess_dir = session_dir(session_id)
    audio_path = sess_dir / audio_filename
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_filename}")

    # Run ML analysis; stems will be written to session dir.
    result = analyze_audio(audio_path, output_dir=sess_dir, device=device)

    # Convert file paths to API URLs for stem streaming.
    stems_dict = result["stems"]
    stems_urls = {}
    for stem_name, stem_path in stems_dict.items():
        if stem_path:
            # Convert local path to streaming URL.
            stems_urls[stem_name] = f"/v1/audio/stem/{session_id}/{stem_name}"
        else:
            stems_urls[stem_name] = None

    # Convert ML result to API response model.
    return AnalysisResponse(
        session_id=session_id,
        metadata=AudioMetadata(**result["metadata"]),
        waveform=WaveformData(**result["waveform"]),
        key=result["key"],
        key_confidence=result["key_confidence"],
        tempo_bpm=result["tempo_bpm"],
        beat_times=result["beat_times"],
        chords=[ChordSegment(**seg) for seg in result["chords"]],
        stems=StemReferences(**stems_urls),
    )
