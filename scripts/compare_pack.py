"""Compare a locally converted pack against its published Hub repo, value for value.

Hashes the safetensors DATA section (everything after the header) of each
weight file, so a header-only difference — metadata backfilled later, a
re-serialised header — compares equal, and any changed tensor byte does not.
This is Tier 4 of the process_component factorisation spec: run by hand
after migrating a recipe, result recorded in the PR.

    uv run python scripts/compare_pack.py models/void-model-mlx dgrauet/void-model-mlx
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path

CHUNK = 64 * 1024 * 1024


def data_sha256(path: Path) -> str:
    """SHA-256 of the bytes after the safetensors header."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        handle.seek(8 + header_len)
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def compare(local_dir: Path, remote_files: dict[str, Path]) -> list[tuple[str, str, str]]:
    """Every (name, local sha, remote sha) whose data sections differ or is missing locally."""
    mismatches = []
    for name, remote_path in sorted(remote_files.items()):
        local_path = local_dir / name
        remote_sha = data_sha256(remote_path)
        local_sha = data_sha256(local_path) if local_path.exists() else "<missing>"
        status = "ok" if local_sha == remote_sha else "MISMATCH"
        print(f"  {status:<9} {name}  local {local_sha[:16]}  remote {remote_sha[:16]}")
        if local_sha != remote_sha:
            mismatches.append((name, local_sha, remote_sha))
    return mismatches


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("local_dir", type=Path)
    parser.add_argument("repo_id")
    return parser.parse_args(argv)


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    args = parse_args()
    api = HfApi()
    names = [
        s.rfilename
        for s in (api.model_info(args.repo_id).siblings or [])
        if s.rfilename.endswith(".safetensors")
    ]
    print(f"{args.repo_id}: {len(names)} weight files")
    remote = {name: Path(hf_hub_download(args.repo_id, name)) for name in names}
    mismatches = compare(args.local_dir, remote)
    if mismatches:
        print(f"\n{len(mismatches)} file(s) differ")
        sys.exit(1)
    print("\nall data sections identical")


if __name__ == "__main__":
    main()
