"""Split a unified safetensors file into per-component files.

Reduces memory usage on constrained machines by allowing each component
to be loaded independently without pulling the entire file into memory.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

from .quantize import format_bytes


def split_model(
    model_dir: Path,
    component_map: dict[str, str],
    *,
    source_filename: str = "model.safetensors",
    fallback_component: str | None = "transformer.safetensors",
) -> dict[str, int]:
    """Split a unified safetensors file into per-component files.

    Args:
        model_dir: Directory containing the model file.
        component_map: Maps weight key prefix -> output filename.
            Example: {"transformer": "transformer.safetensors", "vae_decoder": "vae_decoder.safetensors"}
        source_filename: Name of the unified file to split.
        fallback_component: Output file for unmatched keys (None to skip).

    Returns:
        Dict of output filename -> number of tensors saved.
    """
    unified_path = model_dir / source_filename
    if not unified_path.exists():
        raise FileNotFoundError(f"{unified_path} not found")

    print(f"Loading: {unified_path}")
    all_weights = mx.load(str(unified_path))
    print(f"Loaded {len(all_weights)} tensors")

    # Group weights by output file
    file_weights: dict[str, dict[str, mx.array]] = defaultdict(dict)
    unmatched: dict[str, mx.array] = {}

    for key, value in all_weights.items():
        prefix = key.split(".")[0]
        if prefix in component_map:
            output_file = component_map[prefix]
            file_weights[output_file][key] = value
        else:
            unmatched[key] = value

    if unmatched:
        if fallback_component:
            print(f"WARNING: {len(unmatched)} unmatched keys -> {fallback_component}")
            for k in sorted(unmatched)[:5]:
                print(f"  {k}")
            file_weights[fallback_component].update(unmatched)
        else:
            print(f"WARNING: {len(unmatched)} unmatched keys skipped")

    # Save each component
    result = {}
    for filename, weights in sorted(file_weights.items()):
        output_path = model_dir / filename
        total_bytes = sum(v.nbytes for v in weights.values())
        print(f"Saving: {filename} ({len(weights)} tensors, {format_bytes(total_bytes)})")
        mx.save_safetensors(str(output_path), weights)
        result[filename] = len(weights)

    # Write marker file
    marker = model_dir / "split_model.json"
    with open(marker, "w") as f:
        json.dump({
            "split": True,
            "files": {name: count for name, count in result.items()},
        }, f, indent=2)

    print(f"\nSplit complete. Original {source_filename} can be removed to save disk space.")
    print(f"To remove: rm '{unified_path}'")

    return result
