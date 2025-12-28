"""
Music AI ML services package.

Provides isolated model loading and inference logic for stem separation,
chord detection, key/BPM extraction, and waveform generation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import time

from .demucs_service import separate_stems
from .chords_service import detect_chords
from .key_bpm_service import analyze_key_and_tempo
from .waveform_service import generate_waveform_preview
from .utils.audio_io import get_duration_seconds


def analyze_audio(
    audio_path: Path,
    output_dir: Path,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Orchestrate full audio analysis: stems, chords, key, tempo, waveform.

    Args:
        audio_path (Path): Path to input audio file.
        output_dir (Path): Directory to write stems and temp outputs.
        device (str): Device for Demucs ("cpu" or "cuda").

    Returns:
        Dict[str, Any]: Complete analysis result with metadata, stems, and insights.
    """
    start = time.time()

    # Metadata.
    duration = get_duration_seconds(audio_path)

    # Waveform for UI rendering.
    waveform = generate_waveform_preview(audio_path)

    # Key and tempo analysis.
    key_tempo = analyze_key_and_tempo(audio_path)

    # Chord detection.
    chords = detect_chords(audio_path)

    # Stem separation (most expensive step).
    stems = separate_stems(audio_path, output_dir, device=device)

    elapsed = time.time() - start

    return {
        "metadata": {
            "filename": audio_path.name,
            "duration_seconds": duration,
            "processing_time_seconds": elapsed,
        },
        "waveform": waveform,
        "key": key_tempo["key"],
        "key_confidence": key_tempo["key_confidence"],
        "tempo_bpm": key_tempo["tempo_bpm"],
        "beat_times": key_tempo["beat_times"],
        "chords": chords["chords"],
        "stems": {name: str(path) for name, path in stems.items()},
    }
