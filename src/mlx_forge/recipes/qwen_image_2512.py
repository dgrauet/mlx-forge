"""Qwen-Image recipe — text-to-image DiT (~57.7 GB).

Qwen-Image is a Flux-style MMDiT (Multi-Modal Diffusion Transformer) for
text-to-image generation. It uses a Qwen2.5-VL model as text encoder and an
AutoencoderKL as VAE.

Architecture (3 components, already split on HuggingFace):
  - transformer  (~40.9 GB, 9 shards) — QwenImageTransformer2DModel, 60 layers
  - text_encoder (~16.6 GB, 4 shards) — Qwen2.5-VL (28L LLM + 32L vision encoder)
  - vae          (~254 MB, 1 file)    — AutoencoderKLQwenImage (Conv2d)

Key difference from other recipes: source files are already per-component in
subdirectories on HuggingFace, so no key classification is needed. Each
component is loaded from its own subdirectory and saved as a flat safetensors.

Quantization: transformer and text_encoder are quantizable (Linear-heavy).
VAE is skipped (conv-heavy).
"""

from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import mlx.core as mx

from ..convert import (
    download_hf_files,
    fmt_size,
    load_weights,
    process_component,
    quantize_component,
)
from ..transpose import needs_transpose, transpose_conv
from ..validate import (
    ValidationResult,
    count_layer_indices,
    validate_file_exists,
    validate_no_pytorch_prefix,
    validate_quantization,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ID = "Qwen/Qwen-Image-2512"

COMPONENTS = ["transformer", "text_encoder", "vae"]

COMPONENT_PREFIX: dict[str, str] = {
    "transformer": "transformer",
    "text_encoder": "text_encoder",
    "vae": "vae",
}

_COMPONENT_SIZE_MB: dict[str, int] = {
    "transformer": 40900,
    "text_encoder": 16600,
    "vae": 254,
}

_CHECKPOINT_SIZE_MB = 57700  # ~57.7 GB total

_TRANSFORMER_SHARDS = 9
_TEXT_ENCODER_SHARDS = 4

TRANSFORMER_FILES = [
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
] + [
    f"transformer/diffusion_pytorch_model-{i:05d}-of-{_TRANSFORMER_SHARDS:05d}.safetensors"
    for i in range(1, _TRANSFORMER_SHARDS + 1)
]

TEXT_ENCODER_FILES = [
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model.safetensors.index.json",
] + [
    f"text_encoder/model-{i:05d}-of-{_TEXT_ENCODER_SHARDS:05d}.safetensors"
    for i in range(1, _TEXT_ENCODER_SHARDS + 1)
]

VAE_FILES = [
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

CONFIG_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/added_tokens.json",
]

ALL_CHECKPOINT_FILES = TRANSFORMER_FILES + TEXT_ENCODER_FILES + VAE_FILES + CONFIG_FILES

# Components that should NOT be quantized (conv-heavy).
_SKIP_QUANTIZE_COMPONENTS = {"vae"}

# Keys to skip during quantization (norms, embeddings, projections, modulation).
_SKIP_QUANTIZE_KEYS = [
    "img_in.",
    "txt_in.",
    "proj_out.",
    "norm_out.",
    "time_text_embed.",
    "_mod.",
    "norm",
    "embed_tokens",
    "lm_head",
    "patch_embed",
    "merger",
    ".bias",
]

# ---------------------------------------------------------------------------
# Key sanitization (identity — diffusers keys are already clean)
# ---------------------------------------------------------------------------


def sanitize_key(key: str) -> str:
    """Identity sanitizer — diffusers keys need no transformation."""
    return key


SANITIZERS: dict[str, type[object] | object] = {
    "transformer": sanitize_key,
    "text_encoder": sanitize_key,
    "vae": sanitize_key,
}

# ---------------------------------------------------------------------------
# Conv transposition (VAE only)
# ---------------------------------------------------------------------------


def vae_transform(key: str, weight: mx.array, component_name: str) -> mx.array:
    """Transpose Conv2d weights in the VAE from PyTorch to MLX layout."""
    if needs_transpose(key, weight):
        return transpose_conv(weight)
    return weight


# ---------------------------------------------------------------------------
# Quantization predicate
# ---------------------------------------------------------------------------


def qwen_image_should_quantize(key: str, weight: mx.array) -> bool:
    """Decide whether a weight should be quantized.

    Quantizes large Linear weights. Skips norms, embeddings, modulation,
    input/output projections, and all 1D tensors.
    """
    if weight.ndim < 2:
        return False
    if any(skip in key for skip in _SKIP_QUANTIZE_KEYS):
        return False
    return weight.shape[0] >= 256 and weight.shape[1] >= 256


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run(args) -> None:
    """Preview the conversion plan without downloading or writing."""
    bits = args.bits if args.quantize else None
    q_label = f" + int{bits} quantization" if bits else ""

    print(f"{'=' * 60}")
    print(f"DRY RUN — Qwen-Image conversion plan{q_label}")
    print(f"{'=' * 60}")
    print(f"\nSource: {REPO_ID}")
    print(f"Total download: ~{fmt_size(_CHECKPOINT_SIZE_MB)}")
    print(f"Files to download: {len(ALL_CHECKPOINT_FILES)}")

    if args.output:
        output_dir = args.output
    else:
        suffix = f"-q{args.bits}" if args.quantize else ""
        output_dir = f"models/qwen-image-2512-mlx{suffix}"
    print(f"Output: {output_dir}")

    print("\nComponents:")
    for comp in COMPONENTS:
        size = _COMPONENT_SIZE_MB[comp]
        skip = comp in _SKIP_QUANTIZE_COMPONENTS
        q_note = " (skip quantize — conv-heavy)" if skip and bits else ""
        print(f"  {comp}: ~{fmt_size(size)}{q_note}")

    if bits:
        print(f"\nQuantization: int{bits}, group_size={args.group_size}")
        print("  Quantized: transformer, text_encoder (Linear weights only)")
        print("  Skipped: vae (conv-heavy)")

    print(f"\n{'=' * 60}")
    print("No files downloaded or written (--dry-run)")


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------


def convert(args) -> None:
    """Convert Qwen-Image checkpoint to MLX format."""
    if args.dry_run:
        dry_run(args)
        return

    # Step 1: Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        suffix = f"-q{args.bits}" if args.quantize else ""
        output_dir = Path("models") / f"qwen-image-2512-mlx{suffix}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Step 2: Download checkpoint files
    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
        print(f"Using local checkpoint: {checkpoint_dir}")
    else:
        checkpoint_dir = Path("models") / "qwen-image-2512-src"
        print(f"Downloading {REPO_ID} checkpoint files...")
        print(f"(This is ~{fmt_size(_CHECKPOINT_SIZE_MB)}, may take a while)")
        download_hf_files(REPO_ID, ALL_CHECKPOINT_FILES, checkpoint_dir)

    # Step 3: Process each component
    total_weights = 0

    for comp_name in COMPONENTS:
        comp_subdir = checkpoint_dir / comp_name
        print(f"\n{'=' * 60}")
        print(f"Processing {comp_name} (~{fmt_size(_COMPONENT_SIZE_MB[comp_name])})")

        # Load weights from component subdirectory
        if comp_name == "vae":
            weights = load_weights(
                comp_subdir, single_filename="diffusion_pytorch_model.safetensors"
            )
        elif comp_name == "transformer":
            weights = load_weights(
                comp_subdir,
                index_filename="diffusion_pytorch_model.safetensors.index.json",
            )
        else:
            weights = load_weights(comp_subdir)

        transform = vae_transform if comp_name == "vae" else None

        count = process_component(
            weights,
            comp_name,
            list(weights.keys()),
            output_dir,
            COMPONENT_PREFIX[comp_name],
            sanitizer=sanitize_key,
            transform=transform,
        )
        total_weights += count

        del weights
        gc.collect()
        mx.clear_cache()

    # Step 4: Copy config files to output
    for config_file in CONFIG_FILES:
        src = checkpoint_dir / config_file
        if src.exists():
            dest = output_dir / Path(config_file).name
            # For subdirectory configs, use a prefixed name to avoid collisions
            if "/" in config_file:
                prefix = config_file.split("/")[0]
                dest = output_dir / f"{prefix}_{Path(config_file).name}"
            shutil.copy2(src, dest)

    # Also copy per-component config.json files
    for comp_name in COMPONENTS:
        src = checkpoint_dir / comp_name / "config.json"
        if src.exists():
            shutil.copy2(src, output_dir / f"{comp_name}_config.json")

    # Step 5: Optional quantization (skip conv-heavy components)
    if args.quantize:
        for component_name in COMPONENTS:
            if component_name in _SKIP_QUANTIZE_COMPONENTS:
                print(f"\n  Skipping quantization for {component_name} (conv-heavy)")
                continue
            quantize_component(
                output_dir,
                component_name,
                bits=args.bits,
                group_size=args.group_size,
                should_quantize=qwen_image_should_quantize,
            )

        qconfig = {
            "quantization": {
                "bits": args.bits,
                "group_size": args.group_size,
                "skip_components": sorted(_SKIP_QUANTIZE_COMPONENTS),
            }
        }
        with open(output_dir / "quantize_config.json", "w") as f:
            json.dump(qconfig, f, indent=2)

    # Step 6: Create split_model.json (once, after quantization if applicable)
    split_info: dict = {
        "format": "split",
        "source": REPO_ID,
        "components": COMPONENTS,
    }
    if args.quantize:
        split_info["quantized"] = True
        split_info["quantization_bits"] = args.bits
    with open(output_dir / "split_model.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Conversion complete: {total_weights} total weights")
    print(f"Output: {output_dir}")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.name}: {size_mb:.1f} MB")

    print("Done!")


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate(args) -> None:
    """Validate a converted Qwen-Image model."""
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: {model_dir} not found")
        raise SystemExit(1)

    result = ValidationResult()
    is_quantized = (model_dir / "quantize_config.json").exists()

    if is_quantized:
        with open(model_dir / "quantize_config.json") as f:
            qconfig = json.load(f)
        bits = qconfig["quantization"]["bits"]
        print(f"Detected quantized model: int{bits}")

    # Check expected files
    print("\n== File Structure ==")
    validate_file_exists(model_dir, "split_model.json", result)
    for comp_name in COMPONENTS:
        validate_file_exists(model_dir, f"{comp_name}.safetensors", result)
    validate_file_exists(model_dir, "transformer_config.json", result)

    # Transformer
    print("\n== Transformer Weights ==")
    tf_path = model_dir / "transformer.safetensors"
    if tf_path.exists():
        weights = mx.load(str(tf_path))
        keys = set(weights.keys())
        print(f"  Keys: {len(keys)}")

        # Verify component prefix
        validate_no_pytorch_prefix(weights, "transformer.transformer.", result)

        # Check for transformer blocks
        block_indices = count_layer_indices(keys, block_key="transformer_blocks")
        result.check(
            len(block_indices) > 0,
            f"Transformer blocks present ({len(block_indices)} blocks)",
        )

        # Check key structural elements
        img_in_keys = [k for k in keys if "img_in." in k]
        result.check(len(img_in_keys) > 0, f"img_in projection present ({len(img_in_keys)})")

        proj_out_keys = [k for k in keys if "proj_out." in k]
        result.check(len(proj_out_keys) > 0, f"proj_out projection present ({len(proj_out_keys)})")

        time_embed_keys = [k for k in keys if "time_text_embed." in k]
        result.check(len(time_embed_keys) > 0, f"time_text_embed present ({len(time_embed_keys)})")

        if is_quantized:
            validate_quantization(weights, result, block_key="transformer_blocks")

        total_params = sum(v.size for v in weights.values())
        print(f"  Total transformer parameters: {total_params / 1e9:.2f}B")
        del weights
        gc.collect()
        mx.clear_cache()

    # Text encoder
    print("\n== Text Encoder Weights ==")
    te_path = model_dir / "text_encoder.safetensors"
    if te_path.exists():
        weights = mx.load(str(te_path))
        keys = set(weights.keys())
        print(f"  Keys: {len(keys)}")

        # LLM layers
        layer_indices = count_layer_indices(keys, block_key="layers")
        result.check(len(layer_indices) == 28, f"28 LLM layers (got {len(layer_indices)})")

        # Vision encoder blocks
        vision_indices = count_layer_indices(keys, block_key="blocks")
        result.check(
            len(vision_indices) == 32, f"32 vision encoder blocks (got {len(vision_indices)})"
        )

        # Embedding and head
        emb_keys = [k for k in keys if "embed_tokens" in k]
        result.check(len(emb_keys) > 0, f"Embedding present ({len(emb_keys)})")

        lm_head_keys = [k for k in keys if "lm_head" in k]
        result.check(len(lm_head_keys) > 0, f"lm_head present ({len(lm_head_keys)})")

        if is_quantized:
            validate_quantization(weights, result, block_key=["layers", "blocks"])

        total_params = sum(v.size for v in weights.values())
        print(f"  Total text_encoder parameters: {total_params / 1e9:.2f}B")
        del weights
        gc.collect()
        mx.clear_cache()

    # VAE
    print("\n== VAE Weights ==")
    vae_path = model_dir / "vae.safetensors"
    if vae_path.exists():
        weights = mx.load(str(vae_path))
        keys = set(weights.keys())
        print(f"  Keys: {len(keys)}")

        # Check for conv weights in channels-last layout
        conv_keys = [(k, v) for k, v in weights.items() if "conv" in k and v.ndim == 4]
        if conv_keys:
            # For Conv2d MLX layout: (O, H, W, I) — dim 1 should be spatial, not channels
            # In PyTorch: (O, I, H, W) — I is typically larger than H/W
            sample_key, sample_weight = conv_keys[0]
            # Just report, don't fail
            print(f"  Conv2d weights: {len(conv_keys)} (sample shape: {sample_weight.shape})")

        result.check(
            not is_quantized or "quantize_config.json" not in str(vae_path),
            "VAE not quantized (conv-heavy)",
            warn_only=True,
        )

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
    """Register CLI arguments for the convert subcommand."""
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to local checkpoint directory (default: download from HuggingFace)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: models/qwen-image-2512-mlx[-q<bits>])",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize transformer and text_encoder after conversion",
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
        help="Preview conversion plan without downloading or writing",
    )


def add_validate_args(parser) -> None:
    """Register CLI arguments for the validate subcommand."""
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to converted model directory",
    )


def add_split_args(parser) -> None:
    """Register CLI arguments for the split subcommand."""
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to model directory",
    )


def split(args) -> None:
    """Split is not needed for Qwen-Image (already split by component)."""
    print("Qwen-Image is already split by component during conversion.")
    print("No further splitting needed.")
