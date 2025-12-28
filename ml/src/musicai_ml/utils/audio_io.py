"""
Audio I/O utilities for ML services.

Responsibilities:
- Decode audio files using soundfile/audioread (FFmpeg-backed for broad codec support).
- Resample to a target sample rate and preserve mono/stereo as requested.
- Produce downsampled waveform previews for UI rendering.

Notes:
- Returns channel-first float32 arrays for consistency across downstream ML tasks.
- M4A and other compressed formats rely on FFmpeg being present (already installed earlier).
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
import librosa


def load_audio(
    path: Path,
    target_sr: int = 44100,
    mono: bool = False,
) -> Tuple[np.ndarray, int]:
    """Load an audio file, resample, and optionally mix to mono.

    Args:
        path (Path): Path to the audio file.
        target_sr (int): Desired sampling rate for downstream models.
        mono (bool): If True, mix down to mono after loading.

    Returns:
        Tuple[np.ndarray, int]: (audio, sample_rate) where audio is float32 in
        shape (channels, samples).
    """
    # First try libsndfile via soundfile; if unsupported (e.g., M4A),
    # fall back to librosa/audioread which leverages FFmpeg.
    try:
        audio, sr = sf.read(str(path), always_2d=True)
        audio = audio.T.astype(np.float32)  # channel-first
    except Exception:
        # librosa.load returns (channels, samples) when mono=False in recent versions.
        # Use native sampling rate then resample consistently below.
        y, sr = librosa.load(str(path), sr=None, mono=False)
        if y.ndim == 1:
            # Ensure channel-first shape for mono inputs.
            audio = y[np.newaxis, :].astype(np.float32)
        else:
            audio = y.astype(np.float32)

    # Optional mono mix before resampling to reduce computation for chroma/tempo tasks.
    if mono and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0, keepdims=True)

    # Resample per-channel to keep phase aligned between stems.
    if sr != target_sr:
        resampled = []
        for ch in audio:
            resampled.append(librosa.resample(y=ch, orig_sr=sr, target_sr=target_sr))
        audio = np.stack(resampled, axis=0)
        sr = target_sr

    return audio, sr


def get_duration_seconds(path: Path) -> float:
    """Return duration in seconds using soundfile metadata.

    Args:
        path (Path): Audio file path.

    Returns:
        float: Duration in seconds.
    """
    info = sf.info(str(path))
    if info.samplerate <= 0:
        return 0.0
    return info.frames / float(info.samplerate)


def waveform_preview(audio: np.ndarray, max_points: int = 4096) -> np.ndarray:
    """Downsample audio to a compact waveform for UI visualization.

    Args:
        audio (np.ndarray): Channel-first float32 array (channels, samples).
        max_points (int): Maximum number of points to emit per channel.

    Returns:
        np.ndarray: Shape (channels, points) with values in [-1, 1].
    """
    channels, samples = audio.shape
    if samples <= max_points:
        return audio

    # Chunk-based mean preserves envelope while reducing payload size for UI.
    chunk_size = samples // max_points
    if chunk_size == 0:
        chunk_size = 1
    actual_points = samples // chunk_size
    trimmed = audio[:, : actual_points * chunk_size]
    reshaped = trimmed.reshape(channels, actual_points, chunk_size)
    preview = reshaped.mean(axis=2)
    return preview
