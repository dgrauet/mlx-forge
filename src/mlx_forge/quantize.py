"""Generic quantization for MLX models.

Quantizes selected weight tensors (typically Linear .weight matrices) to int4 or int8
using MLX's affine quantization. Non-selected weights are kept in original precision.

CRITICAL: Always materialize non-quantizable tensors BEFORE quantizing others.
mx.quantize() triggers GPU work that can evict memory-mapped lazy tensor buffers,
zeroing them out.

CRITICAL: quantize_weights() consumes its `weights` argument (empties the dict as it
goes) to keep peak memory bounded on multi-tens-of-GB checkpoints. See its docstring.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import mlx.core as mx
from tqdm import tqdm


def _materialize(*tensors: mx.array) -> None:
    """Force MLX to materialize tensors (trigger GPU computation).

    This calls mlx.core.eval which is MLX's tensor materialization — NOT Python's eval().
    """
    mx.eval(*tensors)  # noqa: S307 — mlx.core.eval, not builtins.eval


def default_should_quantize(key: str, weight: mx.array, *, min_elements: int = 256) -> bool:
    """Default quantization predicate: 2-D Linear `.weight` matrices only.

    This is the generic `mlx-forge quantize` path, so it enforces the
    project-wide rule (only quantize Linear .weight matrices — never conv,
    norm, embedding layers): conv kernels are 3-D+ in MLX and are rejected by
    rank; embedding tables are 2-D `.weight` too, so they are rejected by the
    conventional "embed" in their key. A recipe that knows its architecture
    supplies its own predicate instead.

    Args:
        key: Weight key name.
        weight: Weight tensor.
        min_elements: Minimum number of elements to quantize.

    Returns:
        True if this weight should be quantized.
    """
    if not key.endswith(".weight"):
        return False
    if weight.ndim != 2:
        return False
    if "embed" in key:
        return False
    if weight.size < min_elements:
        return False
    if min(weight.shape) == 1:
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

    CONSUMES `weights`: this function empties the input dict as it works and
    pops each source tensor from its internal working set right after it is
    quantized (or kept), so no reference to a source tensor survives past the
    iteration that consumes it. Do not reuse `weights` after this call — it
    will be empty.

    This is necessary to keep peak memory bounded. Measured on the LTX-2.5 DiT
    (4091 tensors, 38.0 GB): 37.1 GB of source tensors to quantize, ~0.9 GB to
    keep, ~18.6 GB of int8 result plus scales/biases. If nothing is released
    as it goes, source and result are both live at once — 57.4 GB peak on a
    34 GB machine, which SIGKILLs the process. Releasing each source tensor as
    soon as its replacement exists keeps peak memory at roughly the result
    size plus one tensor in flight.

    Args:
        weights: Dict of weight key -> tensor. Emptied by this call — the
            caller must not use it afterward.
        bits: Quantization bits (4 or 8).
        group_size: Quantization group size.
        should_quantize: Predicate function (key, weight) -> bool.

    Returns:
        New dict with quantized weights (includes .scales/.biases for quantized keys).
    """
    to_quantize = {}
    to_keep = {}

    for key in list(weights.keys()):
        value = weights.pop(key)
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
    to_keep.clear()
    del to_keep

    skipped = []
    total = len(to_quantize)
    keys = list(to_quantize.keys())
    desc = f"  Quantizing to int{bits}"
    for key in tqdm(keys, desc=desc, leave=False):
        weight = to_quantize.pop(key)

        if weight.shape[-1] % group_size != 0:
            skipped.append((key, weight.shape))
            # Materialize before keeping it: this tensor was NOT in the to_keep
            # batch above, and the mx.quantize() calls in the remaining
            # iterations can evict its lazy backing buffer — it would then save
            # as zeros.
            _materialize(weight)
            result[key] = weight
            del weight
            continue

        _materialize(weight)
        q_weight, scales, biases = mx.quantize(weight, bits=bits, group_size=group_size)
        _materialize(q_weight, scales, biases)

        result[key] = q_weight
        base = key.removesuffix(".weight") if key.endswith(".weight") else key
        result[f"{base}.scales"] = scales
        result[f"{base}.biases"] = biases

        # Drop the source tensor now that its quantized replacement exists —
        # this is the release that keeps peak memory at ~result size instead
        # of source-plus-result (see docstring).
        del weight, q_weight, scales, biases

    del to_quantize

    if skipped:
        print(
            f"\n  WARNING: {len(skipped)} weight(s) skipped"
            f" (last dim not divisible by {group_size}):"
        )
        for key, shape in skipped:
            print(f"    {key}: shape={shape}")

    quantized_count = total - len(skipped)
    print(f"  Quantized {quantized_count}/{total} eligible weights")

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
        config_path: If set, the quantization record is written next to it,
            under the canonical `quantize_config.json` name (its parent
            directory is used; its basename is fixed by write_quantize_config).

    Returns:
        Path to output file.
    """
    if output_path is None:
        output_path = input_path

    print(f"Quantizing {input_path.name} to int{bits} (group_size={group_size})...")
    t0 = time.monotonic()

    weights = cast(dict[str, mx.array], mx.load(str(input_path)))
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

    # Record through the shared helper so the shape cannot drift from what
    # read_quantize_config / validate expect. config_path's directory is the
    # record's home; its basename is fixed by the helper.
    if config_path is not None:
        write_quantize_config(config_path.parent, bits=bits, group_size=group_size)

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


QUANTIZE_CONFIG_FILENAME = "quantize_config.json"


def write_quantize_config(
    output_dir: Path,
    *,
    bits: int,
    group_size: int,
    **extra: object,
) -> Path:
    """Record how a model was quantized, next to its weights.

    Every recipe that supports --quantize must call this: `validate` decides
    whether to run the scales/biases checks by looking for this file, so a
    recipe that quantizes without writing it has its quantization silently
    unverified (matrix-game-3.0 recorded quantization only in split_model.json
    and its validate never checked a quantized model).

    Args:
        output_dir: Converted model directory.
        bits: Quantization bit-width.
        group_size: Quantization group size.
        **extra: Recipe-specific fields to record alongside, e.g.
            `skip_components=[...]` or `only_transformer_blocks=True`.

    Returns:
        Path to the written file.
    """
    path = output_dir / QUANTIZE_CONFIG_FILENAME
    with open(path, "w") as f:
        json.dump({"quantization": {"bits": bits, "group_size": group_size, **extra}}, f, indent=2)
    return path


def read_quantize_config(model_dir: Path) -> dict | None:
    """Read a model's quantization record.

    Returns:
        The inner "quantization" mapping, or None if the model is not
        quantized — so callers can write `qconfig = read_quantize_config(d)`
        and treat None as "not quantized" instead of re-deriving the filename.
    """
    path = model_dir / QUANTIZE_CONFIG_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("quantization", {})
