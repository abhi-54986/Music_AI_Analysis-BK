"""
Demucs-based stem separation service.

Inputs:
- Path to an audio file in a session-scoped temp directory.

Outputs:
- Paths to stem files (vocals, drums, bass, other) in the same session dir.

Side Effects:
- Writes temporary audio files for stems.

Performance:
- CPU: ~2-4min for a 3min track (htdemucs).
- GPU (CUDA): ~30-60s for a 3min track.
"""
from __future__ import annotations

import torch
from pathlib import Path
from typing import Dict

from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import convert_audio
import soundfile as sf

from .utils.audio_io import load_audio

# Global model cache to avoid reloading on every call.
_MODEL_CACHE = {}


def _get_demucs_model(model_name: str = "htdemucs"):
    """Load or retrieve cached Demucs model.

    Args:
        model_name (str): Demucs model variant.

    Returns:
        Demucs model ready for inference.
    """
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = get_model(model_name)
    return _MODEL_CACHE[model_name]


def separate_stems(
    audio_path: Path,
    output_dir: Path,
    model_name: str = "htdemucs",
    device: str = "cpu",
) -> Dict[str, Path]:
    """Separate audio into stems using Demucs.

    Args:
        audio_path (Path): Path to input audio file.
        output_dir (Path): Directory to write stem files.
        model_name (str): Demucs model variant (default: htdemucs).
        device (str): "cpu" or "cuda" for GPU acceleration.

    Returns:
        Dict[str, Path]: Mapping of stem names to output file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _get_demucs_model(model_name)
    model.to(device)
    model.eval()

    # Load audio; Demucs expects stereo at its native sample rate.
    audio, sr = load_audio(audio_path, target_sr=model.samplerate, mono=False)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)  # (1, channels, samples)

    # Demucs inference returns (1, stems, channels, samples).
    with torch.no_grad():
        stems = apply_model(model, audio_tensor, device=device)

    # Write each stem as a separate WAV file.
    stem_names = model.sources  # e.g., ["drums", "bass", "other", "vocals"]
    stem_paths = {}
    for i, name in enumerate(stem_names):
        stem_audio = stems[0, i].cpu().numpy()  # (channels, samples)
        out_path = output_dir / f"{name}.wav"
        sf.write(str(out_path), stem_audio.T, model.samplerate)
        stem_paths[name] = out_path

    return stem_paths
