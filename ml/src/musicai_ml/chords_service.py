"""
Chord detection service suitable for guitar/piano.

Inputs:
- Path to an audio file or stem.

Outputs:
- Time-stamped chord labels.

Side Effects:
- None (pure analysis); may use temporary intermediates.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import librosa
import numpy as np

from .utils.audio_io import load_audio


def compute_chromagram(audio_path: Path, target_sr: int = 44100) -> Dict[str, Any]:
    """Compute chromagram features for downstream chord decoding.

    Args:
        audio_path (Path): Path to input audio.
        target_sr (int): Sample rate for analysis.

    Returns:
        Dict[str, Any]: Contains chromagram (list of lists) and frame times.
    """
    audio, sr = load_audio(audio_path, target_sr=target_sr, mono=True)
    mono = audio[0]

    hop_length = 2048
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

    return {
        "chroma": chroma.tolist(),
        "frame_times": times.tolist(),
        "sample_rate": sr,
        "hop_length": hop_length,
    }


def detect_chords(audio_path: Path, target_sr: int = 44100) -> Dict[str, Any]:
    """Detect chords using chroma-based template matching (pure Python fallback).

    Args:
        audio_path (Path): Path to input audio.
        target_sr (int): Sample rate for analysis.

    Returns:
        Dict[str, Any]: List of chord segments with time, chord label, and confidence.
    """
    audio, sr = load_audio(audio_path, target_sr=target_sr, mono=True)
    mono = audio[0]

    hop_length = 2048
    chroma = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

    # Simple template matching: major/minor triads.
    chord_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    major_template = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)
    minor_template = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float)

    segments = []
    for i, frame_chroma in enumerate(chroma.T):
        best_chord = "N"
        best_score = 0.0
        for root in range(12):
            maj_score = np.dot(frame_chroma, np.roll(major_template, root))
            min_score = np.dot(frame_chroma, np.roll(minor_template, root))
            if maj_score > best_score:
                best_score = maj_score
                best_chord = chord_names[root]
            if min_score > best_score:
                best_score = min_score
                best_chord = chord_names[root] + "m"
        segments.append({
            "time": float(times[i]),
            "chord": best_chord,
            "confidence": float(min(1.0, best_score / 3.0)),
        })

    # Merge consecutive identical chords to reduce UI clutter.
    merged = []
    for seg in segments:
        if merged and merged[-1]["chord"] == seg["chord"]:
            continue
        merged.append(seg)

    return {"chords": merged}
