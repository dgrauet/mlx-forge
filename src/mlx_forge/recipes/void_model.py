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
    mlx-forge convert void-model
    mlx-forge convert void-model --quantize --bits 8
    mlx-forge convert void-model --source /path/to/local/weights/
    mlx-forge validate void-model /path/to/output/
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx

from ..convert import (
    add_common_convert_args,
    add_source_arg,
    default_output_dir,
    download_hf_files,
    fmt_size,
    load_safetensors,
    print_output_summary,
    process_component,
    quantization_manifest_fields,
    quantize_component,
    source_download_dir,
    write_split_model,
)
from ..metadata import RecipeMetadata
from ..quantize import read_quantize_config, write_quantize_config

REPO_ID = "netflix/void-model"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PASS_FILES = ["void_pass1.safetensors", "void_pass2.safetensors"]

# Approximate size per pass in MB (bf16)
_PASS_SIZE_MB = 9_500  # ~9.5 GB each


# ---------------------------------------------------------------------------
# Key sanitization
# ---------------------------------------------------------------------------


METADATA = RecipeMetadata(
    name="void-model",
    source=REPO_ID,
    license="apache-2.0",
    # should_quantize_transformer only accepts Linear .weight matrices; saying
    # so opts this model's q4/q8 repos into the fuller quantized card.
    quantization_scope="transformer Linear weights only",
    usage_url="https://github.com/dgrauet/void-model-mlx",
    links=[
        "void-model-mlx (inference): https://github.com/dgrauet/void-model-mlx",
        "VideoX-Fun-mlx (engine): https://github.com/dgrauet/VideoX-Fun-mlx",
        # The base-model link is NOT declared here: it differs per build. The
        # bf16 weights pair with the bf16 CogVideoX, the q4 weights with the q8
        # one. Declaring the bf16 link made every quantized card carry a base
        # model it does not use, alongside the right one. It belongs in each
        # repo's own --link.
    ],
)


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
    weights = load_safetensors(src_path)
    print(f"  {len(weights)} keys loaded (lazy) in {time.monotonic() - t0:.1f}s")

    print(f"\nProcessing {len(weights)} keys...")
    t0 = time.monotonic()
    # All VOID weights are Linear (2D) or bias/norm (1D) -- no conv transposition needed.
    # component_prefix=None: the published packs hold bare keys.
    count = process_component(
        weights,
        pass_name,
        list(weights),
        output_dir,
        None,
        sanitizer=sanitize_key,
        output_filename=pass_filename,
    )
    print(f"  Done: {count} weights saved in {time.monotonic() - t0:.1f}s")

    del weights
    gc.collect()
    mx.clear_cache()
    return count


# ---------------------------------------------------------------------------
# Main convert entry point
# ---------------------------------------------------------------------------


def convert(args) -> None:
    """Convert VOID transformer weights to MLX format."""
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = default_output_dir("void-model", quantize=args.quantize, bits=args.bits)

    # Before any side effect: a dry run must not download or write anything.
    # This used to sit after the download, so previewing the plan for a source
    # that was not local started a multi-GB fetch.
    if args.dry_run:
        _dry_run(args, output_dir)
        return

    if args.source:
        source_dir = Path(args.source)
        if not source_dir.is_dir():
            print(f"ERROR: {source_dir} is not a directory")
            raise SystemExit(1)
    else:
        # Download from HuggingFace
        source_dir = source_download_dir(output_dir)
        print(f"\nDownloading from {REPO_ID}...")
        download_hf_files(REPO_ID, PASS_FILES, source_dir)

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
            quantize_component(
                output_dir,
                Path(pass_file).stem,
                bits=args.bits,
                group_size=args.group_size,
                should_quantize=should_quantize_transformer,
            )

        write_quantize_config(output_dir, bits=args.bits, group_size=args.group_size)

    # Without this, `mlx-forge upload models/void-model-mlx` cannot derive the
    # repo name and refuses to run unless --repo-id is passed by hand. Written
    # once, after quantizing, so `quantized` reflects what actually happened.
    write_split_model(
        output_dir,
        {
            "format": "split",
            "components": [Path(f).stem for f in PASS_FILES],
            **quantization_manifest_fields(
                quantized=args.quantize, bits=args.bits, group_size=args.group_size
            ),
            **METADATA.as_split_fields(),
        },
    )
    print("Saved split_model.json")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Conversion complete: {total_weights} total weights")
    print_output_summary(output_dir)
    print("\nDone!")


def _dry_run(args, output_dir: Path) -> None:
    """Print conversion plan without executing anything."""
    print("=" * 60)
    print("DRY RUN -- no files will be written")
    print("=" * 60)

    source_label = args.source if args.source else f"{REPO_ID} (HuggingFace)"
    print(f"\nSource:     {source_label}")
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
        count_layer_indices,
        finish_validation,
        start_validation,
        validate_file_exists,
        validate_quantization,
    )

    model_dir, result = start_validation(args.model_dir)

    # Check quantization
    qconfig = read_quantize_config(model_dir)
    is_quantized = qconfig is not None
    if qconfig is not None:
        print(f"Model is quantized: int{qconfig.get('bits', '?')}")

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

        weights = load_safetensors(pass_path)
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

    finish_validation(result)


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
    """Add VOID model convert arguments to a parser."""
    add_source_arg(
        parser,
        help="Path to directory containing void_pass1.safetensors and void_pass2.safetensors "
        "(required).",
    )
    add_common_convert_args(
        parser,
        output_default="./models/void-model-mlx[-q<bits>]",
        quantize_help="Quantize transformer weights after conversion",
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


def split(args) -> None:
    print("VOID model is already split by pass during conversion.")
    print("No further splitting needed.")
