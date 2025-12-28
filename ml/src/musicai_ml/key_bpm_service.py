"""
Key and BPM extraction service.

Inputs:
- Path to an audio file or arrays from preprocessing.

Outputs:
- Musical key with confidence.
- BPM estimate with optional beat grid.

Side Effects:
- None.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import librosa

from .utils.audio_io import load_audio

# Krumhansl key profiles (major/minor) for simple template matching.
_KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _estimate_key(chroma: np.ndarray) -> Dict[str, Any]:
    """Estimate key using chroma template correlation.

    Args:
        chroma (np.ndarray): Chroma matrix shape (12, frames).

    Returns:
        Dict[str, Any]: key name and confidence score.
    """
    profile_major = _KRUMHANSL_MAJOR / np.linalg.norm(_KRUMHANSL_MAJOR)
    profile_minor = _KRUMHANSL_MINOR / np.linalg.norm(_KRUMHANSL_MINOR)

    chroma_mean = chroma.mean(axis=1)
    chroma_norm = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)

    scores = []
    for i in range(12):
        major_score = np.dot(chroma_norm, np.roll(profile_major, i))
        minor_score = np.dot(chroma_norm, np.roll(profile_minor, i))
        scores.append(("major", i, major_score))
        scores.append(("minor", i, minor_score))

    # Pick best scoring mode/tonic pair.
    mode, tonic, best = max(scores, key=lambda x: x[2])
    # Confidence scaled to [0, 1].
    conf = float(max(0.0, min(1.0, (best + 1) / 2)))
    return {"key": f"{_KEY_NAMES[tonic]} {mode}", "confidence": conf}


def analyze_key_and_tempo(audio_path: Path, target_sr: int = 44100) -> Dict[str, Any]:
    """Compute key and tempo estimates using lightweight spectral features.

    Args:
        audio_path (Path): Path to input audio.
        target_sr (int): Sample rate for analysis.

    Returns:
        Dict[str, Any]: tempo (bpm), beat times, key, and confidence scores.
    """
    audio, sr = load_audio(audio_path, target_sr=target_sr, mono=True)
    mono = audio[0]

    # Beat tracking for tempo; returns tempo estimate and beat frames.
    tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=sr, units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Chroma features for key estimation.
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr)
    key_info = _estimate_key(chroma)

    return {
        "tempo_bpm": float(tempo),
        "beat_times": beat_times,
        "key": key_info["key"],
        "key_confidence": key_info["confidence"],
    }
