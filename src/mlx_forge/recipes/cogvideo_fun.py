"""CogVideoX-Fun-V1.5-5b-InP conversion recipe.

Converts the alibaba-pai/CogVideoX-Fun-V1.5-5b-InP PyTorch checkpoint to MLX split format.
This is a video inpainting model with transformer and VAE components.

The model stores weights in transformer/ and vae/ subdirectories, each with
config.json and safetensors files. Conv3d/Conv2d weights need PyTorch-to-MLX
channels-last transposition; Linear weights pass through unchanged.

Usage:
    mlx-forge convert cogvideo-fun
    mlx-forge convert cogvideo-fun --quantize --bits 8
    mlx-forge convert cogvideo-fun --source /path/to/local/model
    mlx-forge validate cogvideo-fun models/cogvideo-fun-mlx
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx

from ..convert import (
    download_hf_files,
    fmt_size,
    load_weights,
    quantize_component,
)
from ..quantize import _materialize
from ..transpose import transpose_conv
from ..validate import (
    ValidationResult,
    count_layer_indices,
    validate_conv_layout,
    validate_file_exists,
    validate_quantization,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ID = "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP"

COMPONENTS = ["transformer", "vae"]

# Approximate sizes in MB (fp16/bf16)
_COMPONENT_SIZE_MB: dict[str, int] = {
    "transformer": 9_500,  # ~9.5 GB
    "vae": 400,  # ~400 MB
}

# Source files on HuggingFace — transformer may be sharded
_HF_TRANSFORMER_FILES = [
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
]

_HF_VAE_FILES = [
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

# The transformer checkpoint may be split into multiple shards.
# We discover shard filenames dynamically from the index file.

COMPONENT_PREFIX = {
    "transformer": "transformer",
    "vae": "vae",
}


# ---------------------------------------------------------------------------
# Key sanitization
# ---------------------------------------------------------------------------


def sanitize_transformer_key(key: str) -> str | None:
    """Convert a PyTorch transformer key to MLX format.

    CogVideoX-Fun transformer keys are already clean — no prefix stripping needed.
    The safetensors files within transformer/ have bare keys.
    """
    return key


def sanitize_vae_key(key: str) -> str | None:
    """Convert a PyTorch VAE key to MLX format.

    CogVideoX-Fun VAE keys are already clean — no prefix stripping needed.
    The safetensors files within vae/ have bare keys.
    """
    return key


# ---------------------------------------------------------------------------
# Conv transposition
# ---------------------------------------------------------------------------


def maybe_transpose(key: str, value: mx.array, component: str) -> mx.array:
    """Transpose conv weights from PyTorch to MLX channels-last layout if needed.

    Detection strategy:
      - Must be a "weight" key (not bias, not norm)
      - Must have ndim >= 3 (Conv1d=3D, Conv2d=4D, Conv3d=5D)
      - Skip 2D weights (Linear layers)
      - Skip 1D weights (biases, norms)

    Conv3d: (O, I, kD, kH, kW) -> (O, kD, kH, kW, I)
    Conv2d: (O, I, kH, kW) -> (O, kH, kW, I)
    """
    if not key.endswith(".weight"):
        return value
    if value.ndim < 3:
        return value
    # 3D+ weight tensors are conv layers — transpose them
    return transpose_conv(value)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def should_quantize(key: str, weight: mx.array) -> bool:
    """Determine if a weight should be quantized.

    Only quantize Linear .weight matrices in transformer blocks.
    Exclude sensitive layers that harm quality when quantized.
    """
    # Only 2D weight matrices (Linear layers)
    if weight.ndim != 2 or not key.endswith(".weight"):
        return False

    bare_key = key.replace("transformer.", "", 1)

    # Exclude patch embedding (input projection — expanded for inpainting)
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

    # Exclude final output projection / unpatchify
    if "proj_out" in bare_key and "blocks" not in bare_key:
        return False

    # Quantize transformer block weights (attention, ffn, etc.)
    return True


# ---------------------------------------------------------------------------
# Shard discovery
# ---------------------------------------------------------------------------


def _discover_transformer_shards(download_dir: Path) -> list[str]:
    """Read the transformer index file to discover shard filenames."""
    index_path = download_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        return sorted(set(index["weight_map"].values()))
    # Fallback: single file
    single = download_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    if single.exists():
        return ["diffusion_pytorch_model.safetensors"]
    return []


def _get_transformer_hf_files(download_dir: Path) -> list[str]:
    """Get the full list of transformer files to download, including shards."""
    # First download the index to discover shards
    index_path = download_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shards = sorted(set(index["weight_map"].values()))
        return [f"transformer/{s}" for s in shards]
    # If no index, assume single file
    return ["transformer/diffusion_pytorch_model.safetensors"]


# ---------------------------------------------------------------------------
# Component conversion helpers
# ---------------------------------------------------------------------------


def _convert_transformer(
    download_dir: Path,
    output_dir: Path,
    local_source: Path | None = None,
) -> int:
    """Convert the transformer component. Returns weight count."""
    print("\n" + "=" * 60)
    print("Converting transformer")
    print("=" * 60)

    if local_source:
        tf_dir = local_source / "transformer"
    else:
        tf_dir = download_dir / "transformer"

    print(f"\nLoading transformer weights from {tf_dir}...")
    t0 = time.monotonic()
    weights = load_weights(
        tf_dir,
        index_filename="diffusion_pytorch_model.safetensors.index.json",
        single_filename="diffusion_pytorch_model.safetensors",
    )
    print(f"  {len(weights)} keys loaded (lazy) in {time.monotonic() - t0:.1f}s")

    print(f"\nProcessing {len(weights)} transformer keys...")
    t0 = time.monotonic()
    tf_output: dict[str, mx.array] = {}
    for key in weights:
        new_key = sanitize_transformer_key(key)
        if new_key is None:
            continue
        weight = weights[key]
        weight = maybe_transpose(new_key, weight, "transformer")
        _materialize(weight)
        tf_output[f"transformer.{new_key}"] = weight

    count = len(tf_output)
    out_file = "transformer.safetensors"
    print(f"  Saving {count} weights to {out_file}...")
    mx.save_safetensors(str(output_dir / out_file), tf_output)
    elapsed = time.monotonic() - t0
    print(f"  Done: {count} weights saved in {elapsed:.1f}s")

    del tf_output, weights
    gc.collect()
    mx.clear_cache()
    return count


def _convert_vae(
    download_dir: Path,
    output_dir: Path,
    local_source: Path | None = None,
) -> int:
    """Convert the VAE component. Returns weight count."""
    print("\n" + "=" * 60)
    print("Converting VAE")
    print("=" * 60)

    if local_source:
        vae_dir = local_source / "vae"
    else:
        vae_dir = download_dir / "vae"

    print(f"\nLoading VAE weights from {vae_dir}...")
    t0 = time.monotonic()
    weights = load_weights(
        vae_dir,
        index_filename="diffusion_pytorch_model.safetensors.index.json",
        single_filename="diffusion_pytorch_model.safetensors",
    )
    print(f"  {len(weights)} keys loaded (lazy) in {time.monotonic() - t0:.1f}s")

    print(f"\nProcessing {len(weights)} VAE keys...")
    t0 = time.monotonic()
    vae_output: dict[str, mx.array] = {}
    for key in weights:
        new_key = sanitize_vae_key(key)
        if new_key is None:
            continue
        weight = weights[key]
        weight = maybe_transpose(new_key, weight, "vae")
        _materialize(weight)
        vae_output[f"vae.{new_key}"] = weight

    count = len(vae_output)
    out_file = "vae.safetensors"
    print(f"  Saving {count} weights to {out_file}...")
    mx.save_safetensors(str(output_dir / out_file), vae_output)
    elapsed = time.monotonic() - t0
    print(f"  Done: {count} weights saved in {elapsed:.1f}s")

    del vae_output, weights
    gc.collect()
    mx.clear_cache()
    return count


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _build_config(download_dir: Path, local_source: Path | None = None) -> dict:
    """Build output config.json from the source configs."""
    source_dir = local_source if local_source else download_dir

    config: dict = {
        "model_type": "cogvideox-fun-inpaint",
        "source": REPO_ID,
    }

    # Read transformer config
    tf_config_path = source_dir / "transformer" / "config.json"
    if tf_config_path.exists():
        with open(tf_config_path) as f:
            tf_cfg = json.load(f)
        config["transformer"] = tf_cfg

    # Read VAE config
    vae_config_path = source_dir / "vae" / "config.json"
    if vae_config_path.exists():
        with open(vae_config_path) as f:
            vae_cfg = json.load(f)
        config["vae"] = vae_cfg

    return config


# ---------------------------------------------------------------------------
# Main convert entry point
# ---------------------------------------------------------------------------


def convert(args) -> None:
    """Convert CogVideoX-Fun-V1.5-5b-InP to MLX split format."""
    if args.output:
        output_dir = Path(args.output)
    else:
        suffix = f"-q{args.bits}" if args.quantize else ""
        output_dir = Path("models") / f"cogvideo-fun-mlx{suffix}"

    if args.dry_run:
        _dry_run(args, output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine source: local path or HF download
    local_source = Path(args.source) if args.source else None
    download_dir = Path("models") / "cogvideo-fun-src"

    if not local_source:
        # Download from HuggingFace
        print(f"\nDownloading from {REPO_ID}...")

        # First download index + config files
        download_hf_files(REPO_ID, _HF_TRANSFORMER_FILES, download_dir)

        # Discover and download transformer shards
        shard_files = _get_transformer_hf_files(download_dir)
        if shard_files:
            download_hf_files(REPO_ID, shard_files, download_dir)

        # Download VAE files
        download_hf_files(REPO_ID, _HF_VAE_FILES, download_dir)

    total_weights = 0

    # -----------------------------------------------------------------------
    # 1. Transformer
    # -----------------------------------------------------------------------
    total_weights += _convert_transformer(download_dir, output_dir, local_source)

    # -----------------------------------------------------------------------
    # 2. VAE
    # -----------------------------------------------------------------------
    total_weights += _convert_vae(download_dir, output_dir, local_source)

    # -----------------------------------------------------------------------
    # 3. Copy config files
    # -----------------------------------------------------------------------
    source_dir = local_source if local_source else download_dir

    config = _build_config(download_dir, local_source)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nSaved config.json")

    # Copy component configs for reference
    for comp in ("transformer", "vae"):
        src_cfg = source_dir / comp / "config.json"
        if src_cfg.exists():
            dst_cfg = output_dir / f"{comp}_config.json"
            shutil.copy2(str(src_cfg), str(dst_cfg))
            print(f"  Copied {comp}/config.json -> {dst_cfg.name}")

    # Split model manifest
    split_info: dict = {
        "format": "split",
        "components": COMPONENTS,
        "source": REPO_ID,
        "notes": {
            "inpainting": "Transformer patch_embed.proj.weight has 48 input channels "
            "(16 latent + 16 masked latent + 16 mask) for inpainting support.",
        },
    }
    with open(output_dir / "split_model.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # -----------------------------------------------------------------------
    # 4. Optional quantization (transformer only)
    # -----------------------------------------------------------------------
    if args.quantize:
        quantize_component(
            output_dir,
            "transformer",
            bits=args.bits,
            group_size=args.group_size,
            should_quantize=should_quantize,
        )

        split_info["quantized"] = True
        split_info["quantization_bits"] = args.bits
        with open(output_dir / "split_model.json", "w") as f:
            json.dump(split_info, f, indent=2)

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


def _dry_run(args, output_dir: Path) -> None:
    """Print conversion plan without executing anything."""
    print("=" * 60)
    print("DRY RUN — no files will be downloaded or written")
    print("=" * 60)

    source_label = args.source if args.source else f"{REPO_ID} (HuggingFace)"
    print(f"\nSource:     {source_label}")
    print(f"Output dir: {output_dir}")
    print("\nOutput files:")

    total_mb = 0.0
    for comp in COMPONENTS:
        size_mb = _COMPONENT_SIZE_MB[comp]
        if args.quantize and comp == "transformer":
            ratio = 16 / args.bits
            size_mb = size_mb / ratio
            print(f"  {comp}.safetensors: ~{fmt_size(size_mb)} (int{args.bits})")
        else:
            print(f"  {comp}.safetensors: ~{fmt_size(size_mb)} (fp16)")
        total_mb += size_mb

    print("  config.json, split_model.json")
    print("  transformer_config.json, vae_config.json")

    if args.quantize:
        print(f"\nQuantization: int{args.bits}, group_size={args.group_size}")
        print("  Target: transformer block Linear weights only")

    print(f"\nEstimated output size: ~{fmt_size(total_mb)}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(args) -> None:
    """Validate a converted CogVideoX-Fun model."""
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
    for comp in COMPONENTS:
        validate_file_exists(model_dir, f"{comp}.safetensors", result)
    validate_file_exists(model_dir, "config.json", result)
    validate_file_exists(model_dir, "split_model.json", result)

    # --- Config ---
    print("\n== Config Validation ==")
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        result.check(
            config.get("model_type") == "cogvideox-fun-inpaint",
            f"model_type == cogvideox-fun-inpaint (got: {config.get('model_type')})",
        )

    # --- Transformer ---
    print("\n== Transformer Weights ==")
    tf_path = model_dir / "transformer.safetensors"
    if tf_path.exists():
        weights = mx.load(str(tf_path))
        keys = set(weights.keys())

        all_prefixed = all(k.startswith("transformer.") for k in keys)
        result.check(all_prefixed, f"All keys have 'transformer.' prefix ({len(keys)} keys)")

        # Check inpainting patch_embed has expanded input channels
        pe_key = "transformer.patch_embed.proj.weight"
        if pe_key in keys:
            pe_shape = weights[pe_key].shape
            # After Conv3d transpose: (O, kD, kH, kW, I) where I=48 for inpainting
            if weights[pe_key].ndim == 5:
                in_channels = pe_shape[-1]  # channels-last after transpose
                result.check(
                    in_channels == 48,
                    f"patch_embed input channels == 48 for inpainting (got {in_channels},"
                    f" shape {pe_shape})",
                )
            else:
                result.check(
                    False,
                    f"patch_embed.proj.weight expected 5D Conv3d (got {weights[pe_key].ndim}D,"
                    f" shape {pe_shape})",
                )

        # Verify Conv3d weights are in channels-last layout
        validate_conv_layout(weights, result, ndim=5)

        # Check for transformer blocks
        bare_keys = {k.removeprefix("transformer.") for k in keys}
        block_indices = count_layer_indices(bare_keys, block_key="transformer_blocks")
        if len(block_indices) > 0:
            result.check(True, f"{len(block_indices)} transformer blocks found")
        else:
            # Try alternative block naming
            alt_indices = count_layer_indices(bare_keys, block_key="blocks")
            result.check(
                len(alt_indices) > 0,
                f"Transformer blocks found ({len(alt_indices)} via 'blocks')",
            )

        if is_quantized:
            validate_quantization(weights, result, block_key="transformer_blocks")

        total_params = sum(v.size for v in weights.values())
        print(f"  Total transformer parameters: {total_params / 1e9:.2f}B")
        del weights
        gc.collect()
        mx.clear_cache()

    # --- VAE ---
    print("\n== VAE Weights ==")
    vae_path = model_dir / "vae.safetensors"
    if vae_path.exists():
        weights = mx.load(str(vae_path))
        keys = set(weights.keys())

        all_prefixed = all(k.startswith("vae.") for k in keys)
        result.check(all_prefixed, f"All keys have 'vae.' prefix ({len(keys)} keys)")

        # Check encoder and decoder keys present
        enc_keys = [k for k in keys if k.startswith("vae.encoder.")]
        dec_keys = [k for k in keys if k.startswith("vae.decoder.")]
        result.check(len(enc_keys) > 0, f"Encoder keys present ({len(enc_keys)})")
        result.check(len(dec_keys) > 0, f"Decoder keys present ({len(dec_keys)})")

        # Verify conv layout (channels-last)
        validate_conv_layout(weights, result, ndim=5)
        validate_conv_layout(weights, result, ndim=4)

        total_params = sum(v.size for v in weights.values())
        print(f"  Total VAE parameters: {total_params / 1e6:.1f}M")
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
    """Add CogVideoX-Fun convert arguments to a parser."""
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to local model directory (skips HF download). "
        "Must contain transformer/ and vae/ subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ./models/cogvideo-fun-mlx[-q<bits>])",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize transformer after conversion",
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
        help="Preview conversion plan without downloading or writing anything",
    )


def add_validate_args(parser) -> None:
    """Add CogVideoX-Fun validate arguments to a parser."""
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to converted model directory",
    )


def add_split_args(parser) -> None:
    """Add CogVideoX-Fun split arguments to a parser (no-op, model is already split)."""
    parser.add_argument(
        "model_dir",
        type=str,
        help="Model directory containing safetensors files",
    )
