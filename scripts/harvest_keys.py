"""Harvest real upstream key names into a test fixture, for any recipe.

A sanitizer test is only worth running against the names upstream actually
uses; inventing them from our own code proves nothing (six real-conversion
defects in the LTX-2.5 work all traced to invented fixtures). Only the
safetensors header is read — an 8-byte length prefix and the JSON header,
fetched with HTTP Range requests — so this costs kilobytes against any
checkpoint size.

Keys are reduced by a deterministic rule: keep every key whose all-digit
dot-separated path segments (indices, not digits embedded in a name like
"t5_encoder" or "ssv2_probe") are all 0 or 1. Blocks 0 and 1 exhibit every
naming pattern a repeated stack has.

Maintenance tool, not a test: needs network access and an HF login.

    uv run python scripts/harvest_keys.py --repo Lightricks/LTX-2.5 \
        --file vae/ltx-2.5-audio-vae-bf16.safetensors --out tests/fixtures/x.json

    uv run python scripts/harvest_keys.py --torch models/x-src/model.pth \
        --section state_dict --out tests/fixtures/upstream/x.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers


def keep(key: str) -> bool:
    """Whether this key survives index reduction: every all-digit path segment is 0 or 1."""
    return all(int(seg) <= 1 for seg in key.split(".") if seg.isdigit())


def read_header(repo: str, filename: str) -> dict:
    """The safetensors JSON header of a remote file, via two Range requests."""
    url = hf_hub_url(repo, filename)
    base = build_hf_headers()

    def fetch(byte_range: str) -> bytes:
        request = urllib.request.Request(url, headers={**base, "Range": byte_range})
        return urllib.request.urlopen(request).read()

    length = int.from_bytes(fetch("bytes=0-7"), "little")
    return json.loads(fetch(f"bytes=8-{8 + length - 1}"))


_TORCH_DTYPES = {
    "torch.float32": "F32",
    "torch.float16": "F16",
    "torch.bfloat16": "BF16",
    "torch.float64": "F64",
    "torch.int64": "I64",
    "torch.int32": "I32",
    "torch.int16": "I16",
    "torch.int8": "I8",
    "torch.uint8": "U8",
    "torch.bool": "BOOL",
}


def torch_dtype_name(dtype: object) -> str:
    """The safetensors spelling of a torch dtype, so both tiers compare directly."""
    name = str(dtype)
    if name not in _TORCH_DTYPES:
        raise ValueError(f"unknown torch dtype {name!r}; extend _TORCH_DTYPES")
    return _TORCH_DTYPES[name]


def summarise(header: dict) -> dict:
    """The fixture record for one file's tensors: {name: {"dtype", "shape"}} in, record out."""
    dtypes: dict[str, int] = {}
    for spec in header.values():
        dtypes[spec["dtype"]] = dtypes.get(spec["dtype"], 0) + 1
    return {
        "metadata_keys": [],
        "tensor_count": len(header),
        "dtypes": dtypes,
        "keys": sorted(k for k in header if keep(k)),
    }


def harvest(repo: str, files: list[str]) -> dict:
    """The fixture mapping for `files` in `repo` — same shape as ltx_25_keys.json."""
    out: dict = {}
    for filename in files:
        header = read_header(repo, filename)
        metadata = header.pop("__metadata__", {}) or {}
        record = summarise(header)
        record["metadata_keys"] = sorted(metadata)
        out[filename] = record
        print(f"{filename}: {record['tensor_count']} tensors -> {len(record['keys'])} kept")
    return out


def harvest_torch(path: Path, section: str | None = None) -> dict:
    """The fixture record of a pickled checkpoint, keyed by its basename.

    `weights_only=True`: this reads tensor names, dtypes and shapes and nothing
    else. `section` drills into a container key ("target_encoder",
    "state_dict", ...) so the record holds the keys the recipe actually sees.
    """
    import torch  # ty: ignore[unresolved-import]

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if section is not None:
        if section not in state:
            raise SystemExit(
                f"ERROR: {path} has no section {section!r}; top-level keys: {sorted(state)[:20]}"
            )
        state = state[section]
    header = {
        name: {"dtype": torch_dtype_name(tensor.dtype), "shape": list(tensor.shape)}
        for name, tensor in state.items()
        if hasattr(tensor, "dtype")
    }
    record = summarise(header)
    print(f"{path.name}: {record['tensor_count']} tensors -> {len(record['keys'])} kept")
    return {path.name: record}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", help="Hub repo id, e.g. Lightricks/LTX-2.5")
    source.add_argument("--torch", type=Path, help="local .pt/.pth/.ckpt/.bin to read keys from")
    parser.add_argument(
        "--file", dest="files", action="append", default=[], help="repeatable: file in --repo"
    )
    parser.add_argument("--section", default=None, help="container key to drill into (--torch)")
    parser.add_argument("--out", required=True, type=Path, help="fixture JSON to write")
    args = parser.parse_args(argv)
    if args.repo and not args.files:
        parser.error("--repo needs at least one --file")
    return args


def main() -> None:
    args = parse_args()
    out = harvest_torch(args.torch, args.section) if args.torch else harvest(args.repo, args.files)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(out, handle, indent=2, sort_keys=True)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
