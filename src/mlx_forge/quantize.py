"""Generic quantization for MLX models.

Quantizes selected weight tensors (typically Linear .weight matrices) to int4 or int8
using MLX's affine quantization. Non-selected weights are kept in original precision.

CRITICAL: Always materialize non-quantizable tensors BEFORE quantizing others.
mx.quantize() triggers GPU work that can evict memory-mapped lazy tensor buffers,
zeroing them out.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
from tqdm import tqdm


def _materialize(*tensors: mx.array) -> None:
    """Force MLX to materialize tensors (trigger GPU computation).

    This calls mlx.core.eval which is MLX's tensor materialization — NOT Python's eval().
    """
    mx.eval(*tensors)  # noqa: S307 — mlx.core.eval, not builtins.eval


def default_should_quantize(key: str, weight: mx.array, *, min_elements: int = 256) -> bool:
    """Default quantization predicate: quantize 2D+ .weight tensors with enough elements.

    Args:
        key: Weight key name.
        weight: Weight tensor.
        min_elements: Minimum number of elements to quantize.

    Returns:
        True if this weight should be quantized.
    """
    if not key.endswith(".weight"):
        return False
    if weight.ndim < 2:
        return False
    if weight.size < min_elements:
        return False
    if weight.ndim == 2 and min(weight.shape) == 1:
        return False
    return True


def quantize_weights(
    weights: dict[str, mx.array],
    *,
    bits: int = 8,
    group_size: int = 64,
    should_quantize: Callable[[str, mx.array], bool] = default_should_quantize,
) -> dict[str, mx.array]:
    """Quantize selected weights in a dict.

    Args:
        weights: Dict of weight key -> tensor.
        bits: Quantization bits (4 or 8).
        group_size: Quantization group size.
        should_quantize: Predicate function (key, weight) -> bool.

    Returns:
        New dict with quantized weights (includes .scales/.biases for quantized keys).
    """
    to_quantize = {}
    to_keep = {}

    for key, value in weights.items():
        if should_quantize(key, value):
            to_quantize[key] = value
        else:
            to_keep[key] = value

    # CRITICAL: Materialize kept tensors BEFORE quantizing.
    # mx.quantize() GPU work can evict lazy tensor backing buffers.
    if to_keep:
        print(f"  Materializing {len(to_keep)} non-quantizable weights...")
        _materialize(*to_keep.values())

    result = dict(to_keep)
    del to_keep

    for key, weight in tqdm(to_quantize.items(), desc=f"  Quantizing to int{bits}", leave=False):
        if weight.shape[-1] % group_size != 0:
            result[key] = weight
            continue

        _materialize(weight)
        q_weight, scales, biases = mx.quantize(weight, bits=bits, group_size=group_size)
        _materialize(q_weight, scales, biases)

        result[key] = q_weight
        base = key.removesuffix(".weight") if key.endswith(".weight") else key
        result[f"{base}.scales"] = scales
        result[f"{base}.biases"] = biases

        del weight, q_weight, scales, biases

    return result


def quantize_file(
    input_path: Path,
    output_path: Path | None = None,
    *,
    bits: int = 8,
    group_size: int = 64,
    should_quantize: Callable[[str, mx.array], bool] = default_should_quantize,
    config_path: Path | None = None,
) -> Path:
    """Quantize a safetensors file.

    Args:
        input_path: Path to input .safetensors file.
        output_path: Path to output file (defaults to overwriting input).
        bits: Quantization bits.
        group_size: Quantization group size.
        should_quantize: Predicate for which weights to quantize.
        config_path: If set, write quantize_config.json to this path.

    Returns:
        Path to output file.
    """
    if output_path is None:
        output_path = input_path

    print(f"Quantizing {input_path.name} to int{bits} (group_size={group_size})...")
    t0 = time.monotonic()

    weights = mx.load(str(input_path))
    result = quantize_weights(
        weights,
        bits=bits,
        group_size=group_size,
        should_quantize=should_quantize,
    )

    print(f"  Saving {len(result)} keys to {output_path.name}...")
    mx.save_safetensors(str(output_path), result)

    elapsed = time.monotonic() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Write quantize config
    if config_path is not None:
        qconfig = {
            "quantization": {
                "bits": bits,
                "group_size": group_size,
            }
        }
        with open(config_path, "w") as f:
            json.dump(qconfig, f, indent=2)

    del result, weights
    gc.collect()
    mx.clear_cache()

    return output_path


def format_bytes(n: float) -> str:
    """Format bytes as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"
