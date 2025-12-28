# ML Layer (Python Services)

Purpose: Provide isolated, testable modules for audio ML tasks:
- Stem separation via Demucs
- Chord detection suitable for guitar/piano
- Key detection and BPM estimation
- Waveform data generation for visualization

This package is designed to be imported by the FastAPI backend.
A separate virtual environment is recommended; later steps will add dependencies and implementations.

## Models & Caching
- Demucs default model: `htdemucs` (download via `scripts/download_demucs_weights.py`).
- Weights cache: torch hub cache (e.g., `%USERPROFILE%/.cache/torch`), or set `TORCH_HOME` to a custom path.
- Chord detection: prefer `sonic-annotator` + Chordino if installed; fallback to pure-Python chroma/HMM implementation (added later).

## Setup
- Use the unified backend venv (`backend/.venv`) and install `ml/requirements.txt` (includes backend requirements).
- To prefetch Demucs weights (optional but recommended before first run):
	```powershell
	.\backend\.venv\Scripts\Activate.ps1
	python scripts/download_demucs_weights.py --model htdemucs
	```
