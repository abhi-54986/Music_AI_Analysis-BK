# Creates a Python virtual environment for the backend/ML stack and installs dependencies.
# Usage (from repo root):
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_backend_env.ps1

$ErrorActionPreference = "Stop"

$venvPath = "$PSScriptRoot\..\backend\.venv"
$requirements = "$PSScriptRoot\..\backend\requirements.txt"

Write-Host "Creating venv at $venvPath" -ForegroundColor Cyan
python -m venv $venvPath

$activate = "$venvPath\Scripts\Activate.ps1"
Write-Host "Activating venv..." -ForegroundColor Cyan
. $activate

Write-Host "Upgrading pip..." -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "Installing requirements (CPU build of torch by default)..." -ForegroundColor Cyan
# For GPU with CUDA 12.1, replace the torch/torchaudio lines or use the PyTorch extra index.
pip install -r $requirements --extra-index-url https://download.pytorch.org/whl/cpu

Write-Host "Done. To activate later: `n  . $activate" -ForegroundColor Green
