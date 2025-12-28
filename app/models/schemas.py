"""
Pydantic models for API requests and responses.

These schemas define the contract between the frontend and backend,
ensuring type safety and validation for all audio processing operations.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChordSegment(BaseModel):
    """A time-stamped chord label."""
    time: float = Field(..., description="Time in seconds")
    chord: str = Field(..., description="Chord label (e.g., 'Cmaj', 'Dm', 'N' for no chord)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class WaveformData(BaseModel):
    """Downsampled waveform for UI visualization."""
    sample_rate: int = Field(..., description="Sample rate of the preview")
    channels: int = Field(..., description="Number of channels")
    points: int = Field(..., description="Number of points per channel")
    waveform: List[List[float]] = Field(..., description="Channel-first amplitude data")


class AudioMetadata(BaseModel):
    """Basic metadata about the uploaded audio file."""
    filename: str = Field(..., description="Original filename")
    duration_seconds: float = Field(..., description="Duration in seconds")
    processing_time_seconds: float = Field(..., description="Time taken for analysis")


class StemReferences(BaseModel):
    """References to separated stem files."""
    vocals: Optional[str] = Field(None, description="Path or URL to vocals stem")
    drums: Optional[str] = Field(None, description="Path or URL to drums stem")
    bass: Optional[str] = Field(None, description="Path or URL to bass stem")
    other: Optional[str] = Field(None, description="Path or URL to other stem")


class AnalysisResponse(BaseModel):
    """Complete analysis result returned to the frontend."""
    session_id: str = Field(..., description="Session identifier for cleanup")
    metadata: AudioMetadata
    waveform: WaveformData
    key: str = Field(..., description="Detected musical key")
    key_confidence: float = Field(..., ge=0.0, le=1.0)
    tempo_bpm: float = Field(..., description="Estimated tempo in BPM")
    beat_times: List[float] = Field(..., description="Beat positions in seconds")
    chords: List[ChordSegment] = Field(..., description="Detected chord progression")
    stems: StemReferences = Field(..., description="References to separated stems")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
