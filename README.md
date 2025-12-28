# Backend (FastAPI)

Purpose: Provide REST APIs for audio upload, session-scoped processing, and returning JSON metadata and temporary stem file references. Orchestrates ML services and enforces privacy constraints.

Responsibilities:
- Validate audio inputs (MP3/WAV/M4A)
- Manage session IDs and temporary storage
- Orchestrate ML services: Demucs, chord detection, key/BPM, waveform
- Return structured JSON responses with timings and warnings
- Implement robust error handling and cleanup

Folders:
- `app/` — FastAPI application package
  - `api/v1/` — Route modules (v1 endpoints)
  - `core/` — Config, session management, middlewares
  - `models/` — Pydantic request/response models
  - `services/` — Orchestration and integration with ML layer
  - `utils/` — Helpers (temp storage, logging)
  - `temp/` — Session-scoped temporary files (auto-cleaned)
- `tests/` — Backend tests (health, upload, processing)

Note: Actual endpoints and models will be added in subsequent steps.

## Environment
- Create venv and install deps:
  ```powershell
  powershell -ExecutionPolicy Bypass -File .\scripts\setup_backend_env.ps1
  .\backend\.venv\Scripts\Activate.ps1
  ```
- (Optional, recommended) Prefetch Demucs weights so first run is faster:
  ```powershell
  python scripts/download_demucs_weights.py --model htdemucs
  ```
- CUDA users: replace torch/torchaudio wheels in `backend/requirements.txt` with the CUDA 12.1 index before install, e.g.:
  ```powershell
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```