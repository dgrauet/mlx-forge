"""Harvest the LTX-2.5 key fixture — a preset over scripts/harvest_keys.py.

uv run python scripts/harvest_ltx25_keys.py
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = "Lightricks/LTX-2.5"
FILES = [
    "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "vae/ltx-2.5-video-vae-bf16.safetensors",
    "vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
    "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
]
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "ltx_25_keys.json"

if __name__ == "__main__":
    sys.argv = ["harvest_keys.py", "--repo", REPO, "--out", str(FIXTURE)]
    for f in FILES:
        sys.argv += ["--file", f]
    runpy.run_path(str(Path(__file__).with_name("harvest_keys.py")), run_name="__main__")
