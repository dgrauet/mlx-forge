"""Split a unified safetensors file into per-component files.

Reduces memory usage on constrained machines by allowing each component
to be loaded independently without pulling the entire file into memory.
"""

from __future__ import annotations

import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import cast

import mlx.core as mx
from tqdm import tqdm

from .quantize import _materialize, format_bytes


def split_model(
    model_dir: Path,
    component_map: dict[str, str],
    *,
    source_filename: str = "model.safetensors",
    fallback_filename: str | None = "transformer.safetensors",
) -> dict[str, int]:
    """Split a unified safetensors file into per-component files.

    Args:
        model_dir: Directory containing the model file.
        component_map: Maps weight key prefix -> output filename.
            Example: {"transformer": "transformer.safetensors"}
        source_filename: Name of the unified file to split.
        fallback_filename: Output filename for unmatched keys (None to skip).

    Returns:
        Dict of output filename -> number of tensors saved.
    """
    unified_path = model_dir / source_filename
    if not unified_path.exists():
        print(f"ERROR: {unified_path} not found")
        raise SystemExit(1)

    print(f"Loading: {unified_path}")
    all_weights = cast(dict[str, mx.array], mx.load(str(unified_path)))
    print(f"Loaded {len(all_weights)} tensors")

    # Group weights by output file. Nested so that key/value fall out of
    # scope with this function's frame instead of lingering as split_model
    # locals holding the last-popped tensor alive for the rest of the run.
    def _group(source: dict[str, mx.array]) -> tuple[dict[str, dict[str, mx.array]], dict]:
        grouped: dict[str, dict[str, mx.array]] = defaultdict(dict)
        leftover: dict[str, mx.array] = {}
        for key in list(source):
            value = source.pop(key)
            prefix = key.split(".")[0]
            if prefix in component_map:
                grouped[component_map[prefix]][key] = value
            else:
                leftover[key] = value
        return grouped, leftover

    file_weights, unmatched = _group(all_weights)
    del all_weights

    if unmatched:
        if fallback_filename:
            print(f"WARNING: {len(unmatched)} unmatched keys -> {fallback_filename}")
            for k in sorted(unmatched)[:5]:
                print(f"  {k}")
            file_weights[fallback_filename].update(unmatched)
        else:
            print(f"WARNING: {len(unmatched)} unmatched keys skipped")
    unmatched.clear()

    # Save each component. Popping every key out of all_weights above (and
    # deleting it) means file_weights holds the only references left, so
    # weights.clear() below drops the last reference and gc.collect() +
    # mx.clear_cache() actually reclaim the component's memory.
    result = {}
    sorted_items = sorted(file_weights.items())
    for filename, weights in tqdm(sorted_items, desc="Saving components", leave=False):
        output_path = model_dir / filename
        total_bytes = sum(v.nbytes for v in weights.values())
        tqdm.write(f"Saving: {filename} ({len(weights)} tensors, {format_bytes(total_bytes)})")
        # Lazy (mmap-backed) tensors save as zeros if anything evicts their
        # buffers first — the same rule every recipe's process_component
        # follows; the mmap has kept this path safe by luck, not by design.
        _materialize(*weights.values())
        mx.save_safetensors(str(output_path), weights)
        result[filename] = len(weights)
        weights.clear()
        gc.collect()
        mx.clear_cache()

    # Write marker file — merge with an existing manifest rather than clobber
    # it: convert writes recipe identity, gating and licence provenance into
    # split_model.json, and losing the gating declaration here would let a
    # gated pack's first upload go through open.
    marker = model_dir / "split_model.json"
    info: dict = {}
    if marker.exists():
        with open(marker) as f:
            info = json.load(f)
    info["split"] = True
    info["files"] = dict(result)
    with open(marker, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nSplit complete. Original {source_filename} can be removed to save disk space.")
    print(f"To remove: rm '{unified_path}'")

    return result
