"""
Waveform generation service.

Inputs:
- Path to an audio file.

Outputs:
- Downsampled amplitude array suitable for UI visualization.

Side Effects:
- None.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np

from .utils.audio_io import load_audio, waveform_preview


def generate_waveform_preview(
    audio_path: Path,
    target_sr: int = 44100,
    max_points: int = 4096,
) -> Dict[str, Any]:
    """Generate a downsampled waveform preview for UI consumption.

    Args:
        audio_path (Path): Path to input audio.
        target_sr (int): Sample rate to normalize the waveform to.
        max_points (int): Maximum points per channel for the preview.

    Returns:
        Dict[str, Any]: Contains sample rate, channel count, and preview array (list of lists).
    """
    audio, sr = load_audio(audio_path, target_sr=target_sr, mono=False)

    # Normalize amplitude to [-1, 1] to keep UI rendering consistent.
    max_abs = np.max(np.abs(audio)) or 1.0
    audio = audio / max_abs

    preview = waveform_preview(audio, max_points=max_points)

    return {
        "sample_rate": sr,
        "channels": preview.shape[0],
        "points": preview.shape[1],
        "waveform": preview.tolist(),
    }
