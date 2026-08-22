"""Harvest real LTX-2.5 key names into a test fixture.

A sanitizer test is only worth running against the names upstream actually
uses. Reading a 42 GB checkpoint to learn them is not an option in CI, and
inventing them by reading our own code proves nothing — so the names are
captured here, once, and versioned.

Only the safetensors header is read: an 8-byte length prefix followed by the
JSON header, fetched with an HTTP Range request. The tensor data is never
touched, so this costs a few hundred kilobytes against 124 GB of weights.

Keys are reduced by a deterministic rule — keep every key whose numeric path
components are all 0 or 1, plus every key with no numeric component. Block 0
and block 1 together exhibit every naming pattern a repeated stack has, while
blocks 2..47 add nothing but bulk.

Maintenance tool, not a test: network access and an HF login, so CI skips it.

    uv run python scripts/harvest_ltx25_keys.py
"""

import json
import re
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers

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

_INDEX = re.compile(r"\d+")


def keep(key: str) -> bool:
    """Whether this key survives index reduction (all numeric parts are 0 or 1)."""
    return all(int(n) <= 1 for n in _INDEX.findall(key))


def read_header(repo: str, filename: str) -> dict:
    """The safetensors JSON header of a remote file, via two Range requests."""
    url = hf_hub_url(repo, filename)
    base = build_hf_headers()

    def fetch(byte_range: str) -> bytes:
        request = urllib.request.Request(url, headers={**base, "Range": byte_range})
        return urllib.request.urlopen(request).read()

    length = int.from_bytes(fetch("bytes=0-7"), "little")
    return json.loads(fetch(f"bytes=8-{8 + length - 1}"))


def main() -> None:
    out: dict = {}
    for filename in FILES:
        header = read_header(REPO, filename)
        metadata = header.pop("__metadata__", {})
        dtypes: dict[str, int] = {}
        for spec in header.values():
            dtypes[spec["dtype"]] = dtypes.get(spec["dtype"], 0) + 1
        out[filename] = {
            "metadata_keys": sorted(metadata),
            "tensor_count": len(header),
            "dtypes": dtypes,
            "keys": sorted(k for k in header if keep(k)),
        }
        print(f"{filename}: {len(header)} tensors -> {len(out[filename]['keys'])} kept")

    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with open(FIXTURE, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print(f"\nwrote {FIXTURE}")


if __name__ == "__main__":
    main()
