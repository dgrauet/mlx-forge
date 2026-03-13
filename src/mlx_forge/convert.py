"""Generic conversion utilities shared across recipes.

Provides common helpers for downloading, loading, processing, and saving
model components during PyTorch-to-MLX conversion.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)
from tqdm import tqdm

from .quantize import _materialize, quantize_weights


def fmt_size(mb: float) -> str:
    """Format a size in MB to a human-readable string."""
    if mb >= 1000:
        return f"{mb / 1000:.1f} GB"
    return f"{mb:.0f} MB"


def _validate_path_within(filepath: Path, parent: Path) -> Path:
    """Ensure filepath resolves within parent directory (prevents path traversal)."""
    resolved = filepath.resolve()
    parent_resolved = parent.resolve()
    if not str(resolved).startswith(str(parent_resolved) + "/") and resolved != parent_resolved:
        raise ValueError(
            f"Path traversal detected: '{filepath}' resolves outside '{parent}'"
        )
    return resolved


def download_hf_files(
    repo_id: str,
    filenames: list[str],
    download_dir: Path,
) -> None:
    """Download files from HuggingFace Hub with error handling.

    Skips files already present in download_dir.
    """
    download_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        target = download_dir / filename
        _validate_path_within(target, download_dir)
        if target.exists():
            print(f"  Already downloaded: {filename}")
            continue
        try:
            print(f"  Downloading {filename}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=download_dir,
            )
        except RepositoryNotFoundError:
            print(
                f"ERROR: Repository '{repo_id}' not found or access denied.\n"
                "If this is a gated repo, request access and run: huggingface-cli login"
            )
            raise SystemExit(1)
        except LocalEntryNotFoundError:
            print(
                f"ERROR: '{filename}' not in cache and network unavailable.\n"
                "Check your internet connection or download the file manually."
            )
            raise SystemExit(1)
        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 401:
                print("ERROR: Authentication required. Run: huggingface-cli login")
            elif status == 403:
                print(f"ERROR: Access denied to '{repo_id}'.")
            elif status == 404:
                print(f"ERROR: File '{filename}' not found in '{repo_id}'.")
            else:
                print(f"ERROR: HuggingFace Hub request failed: {e}")
            raise SystemExit(1)
        except (OSError, ConnectionError) as e:
            print(f"ERROR: Network error: {e}")
            raise SystemExit(1)


def load_weights(
    checkpoint_dir: Path,
    *,
    index_filename: str = "model.safetensors.index.json",
    single_filename: str = "model.safetensors",
) -> dict[str, mx.array]:
    """Load weights from sharded or single safetensors files.

    If an index file exists, loads shards. Otherwise loads a single file.
    All weights are loaded lazily via mx.load() (memory-mapped).
    """
    index_path = checkpoint_dir / index_filename
    if index_path.exists():
        print("\nLoading sharded weights lazily...")
        weights: dict[str, mx.array] = {}
        with open(index_path) as f:
            index = json.load(f)
        shard_files = sorted(set(index["weight_map"].values()))
        for shard in shard_files:
            shard_path = checkpoint_dir / shard
            _validate_path_within(shard_path, checkpoint_dir)
            print(f"  Loading {shard}...")
            shard_weights = mx.load(str(shard_path))
            weights.update(shard_weights)
        return weights

    single_path = checkpoint_dir / single_filename
    print(f"\nLoading weights lazily from {single_path.name}...")
    return mx.load(str(single_path))


def classify_keys(
    weights: dict[str, mx.array],
    classify_fn: Callable[[str], str | None],
) -> dict[str, list[str]]:
    """Group weight keys by component using a classification function.

    Keys for which classify_fn returns None are skipped.
    """
    keys_by_component: dict[str, list[str]] = {}
    for key in weights:
        comp = classify_fn(key)
        if comp:
            keys_by_component.setdefault(comp, []).append(key)
    return keys_by_component


# Type alias for the optional per-weight transform function.
# Signature: (sanitized_key, weight, component_name) -> transformed_weight
WeightTransform = Callable[[str, mx.array, str], mx.array]


def process_component(
    checkpoint_weights: dict,
    component_name: str,
    keys: list[str],
    output_dir: Path,
    component_prefix: str,
    *,
    sanitizer: Callable[[str], str | None],
    transform: WeightTransform | None = None,
) -> int:
    """Process one component: sanitize keys, optionally transform, materialize, save.

    Args:
        checkpoint_weights: Full checkpoint weight dict.
        component_name: Name of the component (for display).
        keys: List of checkpoint keys belonging to this component.
        output_dir: Directory to write the output safetensors file.
        component_prefix: Prefix to prepend to sanitized keys in output.
        sanitizer: Function to rename keys. Returns None to skip a key.
        transform: Optional per-weight transform (e.g. conv transposition).

    Returns:
        Number of weights saved.
    """
    component_weights = {}

    for key in tqdm(keys, desc=f"  {component_name}", leave=False):
        new_key = sanitizer(key)
        if new_key is None:
            continue

        weight = checkpoint_weights[key]
        if transform is not None:
            weight = transform(new_key, weight, component_name)

        _materialize(weight)
        component_weights[f"{component_prefix}.{new_key}"] = weight

    if not component_weights:
        print(f"  No weights for {component_name}, skipping")
        return 0

    count = len(component_weights)
    output_file = output_dir / f"{component_name}.safetensors"
    print(f"  Saving {count} weights to {output_file.name}...")
    mx.save_safetensors(str(output_file), component_weights)

    del component_weights
    gc.collect()
    mx.clear_cache()
    return count


def quantize_component(
    output_dir: Path,
    component_name: str,
    *,
    bits: int = 8,
    group_size: int = 64,
    should_quantize: Callable[[str, mx.array], bool],
) -> None:
    """Quantize a component's weights in-place.

    Args:
        output_dir: Directory containing the component safetensors file.
        component_name: Name of the component (e.g. "text_model").
        bits: Quantization bits (4 or 8).
        group_size: Quantization group size.
        should_quantize: Predicate deciding which weights to quantize.
    """
    filepath = output_dir / f"{component_name}.safetensors"
    if not filepath.exists():
        print(f"  WARNING: {filepath.name} not found, skipping quantization")
        return

    print(f"\n  Quantizing {component_name} to int{bits} (group_size={group_size})...")
    weights = mx.load(str(filepath))

    result = quantize_weights(
        weights,
        bits=bits,
        group_size=group_size,
        should_quantize=should_quantize,
    )

    print(f"  Saving quantized {component_name} ({len(result)} keys)...")
    mx.save_safetensors(str(filepath), result)

    del result, weights
    gc.collect()
    mx.clear_cache()


def shard_filenames(n: int, prefix: str = "model") -> list[str]:
    """Generate shard filenames for n-shard models, plus the index file."""
    shards = [f"{prefix}-{i:05d}-of-{n:05d}.safetensors" for i in range(1, n + 1)]
    shards.append(f"{prefix}.safetensors.index.json")
    return shards
