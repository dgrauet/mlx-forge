"""Netflix VOID model conversion recipe.

Converts standalone VOID transformer weights (void_pass1.safetensors, void_pass2.safetensors)
from PyTorch bf16 format to MLX format with optional quantization.

These are CogVideoXTransformer3DModel weights (same architecture as CogVideoX-Fun-V1.5-5b-InP)
but stored as standalone files without config.json or directory structure. The weights contain
only the transformer -- no VAE or T5 (those come from the base CogVideoX model).

Architecture:
  - Two-pass transformer (pass1 + pass2), each ~9.5 GB bf16
  - 1024 keys per pass, 42 transformer blocks
  - patch_embed.proj.weight shape (3072, 384): 48 input channels
    (16 latent + 16 VAE-mask + 16 VAE-video) * patch_volume(8)
  - All weights are Linear (2D) or bias/norm (1D) -- no Conv3d/Conv2d layers
    (CogVideoX-Fun V1.5 uses Linear patch_embed, not Conv3d)

Usage:
    mlx-forge convert void-model --source /path/to/weights/
    mlx-forge convert void-model --source /path/to/weights/ --quantize --bits 8
    mlx-forge convert void-model --source /path/to/weights/ --quantize --bits 4
    mlx-forge validate void-model /path/to/output/
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx

from ..convert import (
    fmt_size,
)
from ..quantize import _materialize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PASS_FILES = ["void_pass1.safetensors", "void_pass2.safetensors"]

# Approximate size per pass in MB (bf16)
_PASS_SIZE_MB = 9_500  # ~9.5 GB each


# ---------------------------------------------------------------------------
# Key sanitization
# ---------------------------------------------------------------------------


def sanitize_key(key: str) -> str | None:
    """Convert a PyTorch transformer key to MLX format.

    VOID transformer keys are already clean -- no prefix stripping or
    renaming needed.
    """
    return key


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def should_quantize_transformer(key: str, weight: mx.array) -> bool:
    """Determine if a transformer weight should be quantized.

    Only quantize Linear .weight matrices in transformer blocks.
    Exclude sensitive layers that harm quality when quantized.
    """
    # Only 2D weight matrices (Linear layers)
    if weight.ndim != 2 or not key.endswith(".weight"):
        return False

    # Strip the pass prefix if present (e.g. "void_pass1.transformer_blocks...")
    bare_key = key

    # Exclude patch embedding (input projection -- expanded for inpainting)
    if "patch_embed" in bare_key:
        return False

    # Exclude timestep/time embedding layers
    if "time_embed" in bare_key or "timestep" in bare_key:
        return False

    # Exclude normalization weights
    if "norm" in bare_key:
        return False

    # Exclude position embedding
    if "pos_embed" in bare_key:
        return False

    # Exclude final output projection
    if "proj_out" in bare_key and "blocks" not in bare_key:
        return False

    # Quantize transformer block weights (attention, ffn, etc.)
    return True


# ---------------------------------------------------------------------------
# Per-pass conversion
# ---------------------------------------------------------------------------


def _convert_pass(
    source_dir: Path,
    output_dir: Path,
    pass_filename: str,
) -> int:
    """Convert one pass file. Returns weight count."""
    pass_name = Path(pass_filename).stem  # e.g. "void_pass1"
    print(f"\n{'=' * 60}")
    print(f"Converting {pass_name}")
    print("=" * 60)

    src_path = source_dir / pass_filename
    if not src_path.exists():
        print(f"  ERROR: {src_path} not found")
        raise SystemExit(1)

    print(f"\nLoading weights from {src_path}...")
    t0 = time.monotonic()
    weights = mx.load(str(src_path))
    print(f"  {len(weights)} keys loaded (lazy) in {time.monotonic() - t0:.1f}s")

    print(f"\nProcessing {len(weights)} keys...")
    t0 = time.monotonic()
    output: dict[str, mx.array] = {}
    for key in weights:
        new_key = sanitize_key(key)
        if new_key is None:
            continue
        weight = weights[key]
        # All VOID weights are Linear (2D) or bias/norm (1D) -- no conv transposition needed
        _materialize(weight)
        output[new_key] = weight

    count = len(output)
    out_file = pass_filename
    print(f"  Saving {count} weights to {out_file}...")
    mx.save_safetensors(str(output_dir / out_file), output)
    elapsed = time.monotonic() - t0
    print(f"  Done: {count} weights saved in {elapsed:.1f}s")

    del output, weights
    gc.collect()
    mx.clear_cache()
    return count


# ---------------------------------------------------------------------------
# Main convert entry point
# ---------------------------------------------------------------------------


def convert(args) -> None:
    """Convert VOID transformer weights to MLX format."""
    if not args.source:
        print("ERROR: --source is required (path to directory containing void_pass1/2.safetensors)")
        raise SystemExit(1)

    source_dir = Path(args.source)
    if not source_dir.is_dir():
        print(f"ERROR: {source_dir} is not a directory")
        raise SystemExit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        suffix = f"-q{args.bits}" if args.quantize else ""
        output_dir = Path("models") / f"void-model-mlx{suffix}"

    if args.dry_run:
        _dry_run(args, output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    total_weights = 0

    # -----------------------------------------------------------------------
    # Convert each pass
    # -----------------------------------------------------------------------
    for pass_file in PASS_FILES:
        total_weights += _convert_pass(source_dir, output_dir, pass_file)

    # -----------------------------------------------------------------------
    # Build config
    # -----------------------------------------------------------------------
    config: dict = {
        "model_type": "void-transformer",
        "source": "netflix-void",
        "architecture": "CogVideoXTransformer3DModel",
        "passes": [Path(f).stem for f in PASS_FILES],
        "notes": {
            "patch_embed": "Linear with in_dim=384 "
            "(in_channels=48 [16 latent + 16 VAE-mask + 16 VAE-video] * patch_volume=8).",
            "base_model": "Uses VAE and T5 from CogVideoX-Fun-V1.5-5b-InP.",
        },
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nSaved config.json")

    # -----------------------------------------------------------------------
    # Optional quantization (transformer weights only)
    # -----------------------------------------------------------------------
    if args.quantize:
        for pass_file in PASS_FILES:
            # quantize_component expects {component_name}.safetensors
            # but our files are named void_pass1.safetensors, so we do it manually
            _quantize_pass(output_dir, pass_file, args.bits, args.group_size)

        qconfig = {
            "quantization": {
                "bits": args.bits,
                "group_size": args.group_size,
            }
        }
        with open(output_dir / "quantize_config.json", "w") as f:
            json.dump(qconfig, f, indent=2)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Conversion complete: {total_weights} total weights")
    print(f"Output: {output_dir}")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            rel = p.relative_to(output_dir)
            print(f"  {rel}: {size_mb:.1f} MB")
    print("\nDone!")


def _quantize_pass(
    output_dir: Path,
    pass_filename: str,
    bits: int,
    group_size: int,
) -> None:
    """Quantize a single pass file in-place."""
    from ..quantize import quantize_weights

    filepath = output_dir / pass_filename
    if not filepath.exists():
        print(f"  WARNING: {filepath.name} not found, skipping quantization")
        return

    pass_name = Path(pass_filename).stem
    print(f"\n  Quantizing {pass_name} to int{bits} (group_size={group_size})...")
    weights = mx.load(str(filepath))

    result = quantize_weights(
        weights,
        bits=bits,
        group_size=group_size,
        should_quantize=should_quantize_transformer,
    )

    print(f"  Saving quantized {pass_name} ({len(result)} keys)...")
    mx.save_safetensors(str(filepath), result)

    del result, weights
    gc.collect()
    mx.clear_cache()


def _dry_run(args, output_dir: Path) -> None:
    """Print conversion plan without executing anything."""
    print("=" * 60)
    print("DRY RUN -- no files will be written")
    print("=" * 60)

    print(f"\nSource:     {args.source}")
    print(f"Output dir: {output_dir}")
    print("\nPass files:")

    total_mb = 0.0
    for pass_file in PASS_FILES:
        size_mb = _PASS_SIZE_MB
        if args.quantize:
            ratio = 16 / args.bits
            size_mb = size_mb / ratio
            print(f"  {pass_file}: ~{fmt_size(size_mb)} (int{args.bits})")
        else:
            print(f"  {pass_file}: ~{fmt_size(size_mb)} (bf16)")
        total_mb += size_mb

    if args.quantize:
        print(f"\nQuantization: int{args.bits}, group_size={args.group_size}")
        print("  Target: transformer block Linear weights only")
        print("  Skipped: patch_embed, time_embed, norm, proj_out")

    print(f"\nEstimated output size: ~{fmt_size(total_mb)}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(args) -> None:
    """Validate converted VOID model weights."""
    from ..validate import (
        ValidationResult,
        count_layer_indices,
        validate_file_exists,
        validate_quantization,
    )

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: {model_dir} does not exist")
        raise SystemExit(1)

    print(f"Validating: {model_dir}")
    result = ValidationResult()

    # Check quantization
    is_quantized = (model_dir / "quantize_config.json").exists()
    if is_quantized:
        with open(model_dir / "quantize_config.json") as f:
            qconfig = json.load(f)
        bits = qconfig.get("quantization", {}).get("bits", "?")
        print(f"Model is quantized: int{bits}")

    # --- File structure ---
    print("\n== File Structure ==")
    for pass_file in PASS_FILES:
        validate_file_exists(model_dir, pass_file, result)
    validate_file_exists(model_dir, "config.json", result)

    # --- Per-pass validation ---
    for pass_file in PASS_FILES:
        pass_name = Path(pass_file).stem
        print(f"\n== {pass_name} Weights ==")
        pass_path = model_dir / pass_file
        if not pass_path.exists():
            continue

        weights = mx.load(str(pass_path))
        keys = set(weights.keys())

        # Quantized models have extra .scales and .biases keys
        base_keys = {k for k in keys if not k.endswith((".scales", ".biases"))}
        result.check(
            len(base_keys) == 1024,
            f"{pass_name}: expected 1024 base keys (got {len(base_keys)})",
        )

        # Check patch_embed input dimension
        pe_key = "patch_embed.proj.weight"
        if pe_key in keys:
            pe_shape = weights[pe_key].shape
            expected_in_dim = 48 * 8  # 384
            if weights[pe_key].ndim == 2:
                in_dim = pe_shape[1]
                result.check(
                    in_dim == expected_in_dim,
                    f"{pass_name}: patch_embed input dim == {expected_in_dim}"
                    f" (got {in_dim}, shape {pe_shape})",
                )

        # Check transformer blocks
        block_indices = count_layer_indices(keys, block_key="transformer_blocks")
        if len(block_indices) > 0:
            result.check(True, f"{pass_name}: {len(block_indices)} transformer blocks found")
        else:
            result.check(False, f"{pass_name}: no transformer blocks found")

        # All base weights should be 2D or 1D (no conv layers)
        high_dim = [k for k in base_keys if weights[k].ndim >= 3]
        result.check(
            len(high_dim) == 0,
            f"{pass_name}: all weights are 1D/2D (no conv, found {len(high_dim)} with ndim>=3)",
        )

        if is_quantized:
            validate_quantization(weights, result, block_key="transformer_blocks")

        total_params = sum(v.size for v in weights.values())
        print(f"  Total {pass_name} parameters: {total_params / 1e9:.2f}B")
        del weights
        gc.collect()
        mx.clear_cache()

    result.summary()
    if not result.passed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
    """Add VOID model convert arguments to a parser."""
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to directory containing void_pass1.safetensors and void_pass2.safetensors "
        "(required).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ./models/void-model-mlx[-q<bits>])",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize transformer weights after conversion",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Quantization bits (default: 8)",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size (default: 64)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview conversion plan without writing anything",
    )


def add_validate_args(parser) -> None:
    """Add VOID model validate arguments to a parser."""
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to converted model directory",
    )


def add_split_args(parser) -> None:
    """Add VOID model split arguments (no-op, model is already split by pass)."""
    parser.add_argument(
        "model_dir",
        type=str,
        help="Model directory containing safetensors files",
    )
