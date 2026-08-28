"""Harvest real upstream key names into a test fixture, for any recipe.

A sanitizer test is only worth running against the names upstream actually
uses; inventing them from our own code proves nothing (six real-conversion
defects in the LTX-2.5 work all traced to invented fixtures). Only the
safetensors header is read — an 8-byte length prefix and the JSON header,
fetched with HTTP Range requests — so this costs kilobytes against any
checkpoint size.

Keys are reduced by a deterministic rule: keep every key whose numeric path
components are all 0 or 1, plus every key with no numeric component. Blocks 0
and 1 exhibit every naming pattern a repeated stack has.

Maintenance tool, not a test: needs network access and an HF login.

    uv run python scripts/harvest_keys.py --repo Lightricks/LTX-2.5 \
        --file vae/ltx-2.5-audio-vae-bf16.safetensors --out tests/fixtures/x.json
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers

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


def harvest(repo: str, files: list[str]) -> dict:
    """The fixture mapping for `files` in `repo` — same shape as ltx_25_keys.json."""
    out: dict = {}
    for filename in files:
        header = read_header(repo, filename)
        metadata = header.pop("__metadata__", {}) or {}
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
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--repo", required=True, help="Hub repo id, e.g. Lightricks/LTX-2.5")
    parser.add_argument(
        "--file", dest="files", action="append", required=True, help="repeatable: file in the repo"
    )
    parser.add_argument("--out", required=True, type=Path, help="fixture JSON to write")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    out = harvest(args.repo, args.files)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
