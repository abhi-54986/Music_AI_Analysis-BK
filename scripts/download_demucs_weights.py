"""
Download Demucs model weights for offline use.

Usage (from repo root, after activating backend/.venv):
    python scripts/download_demucs_weights.py --model htdemucs --cache "C:\\path\\to\\cache"

By default, weights are stored under torch hub cache (e.g., %USERPROFILE%/.cache/torch on Windows).
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from demucs.pretrained import get_model


def download_model(model_name: str, cache_dir: Optional[Path] = None) -> None:
    """Download a Demucs model and report cache location.

    Args:
        model_name (str): Name of the Demucs pretrained model (e.g., "htdemucs").
        cache_dir (Optional[Path]): Optional cache directory; if provided, will be
            set via TORCH_HOME for this process.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(cache_dir)

    _ = get_model(model_name)  # Triggers download if missing.
    hub_dir = Path(torch.hub.get_dir()).resolve()
    print(f"Model '{model_name}' available in cache: {hub_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Demucs weights")
    parser.add_argument("--model", default="htdemucs", help="Demucs model name (default: htdemucs)")
    parser.add_argument("--cache", default=None, help="Optional cache directory (default: torch hub cache)")
    args = parser.parse_args()

    cache_dir = Path(args.cache).expanduser().resolve() if args.cache else None
    download_model(args.model, cache_dir)


if __name__ == "__main__":
    main()
