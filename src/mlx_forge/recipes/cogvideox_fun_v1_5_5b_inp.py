"""CogVideoX-Fun-V1.5-5b-InP conversion recipe.

Converts the alibaba-pai/CogVideoX-Fun-V1.5-5b-InP PyTorch checkpoint to MLX split format.
This is a video inpainting model with transformer, text_encoder (T5-XXL), and VAE components.

The model stores weights in transformer/, text_encoder/, and vae/ subdirectories, each with
config.json and safetensors files. Conv3d/Conv2d weights need PyTorch-to-MLX
channels-last transposition; Linear weights pass through unchanged.

Architecture:
  - transformer   (~9.5 GB, 1 file)  — CogVideoXTransformer3DModel, 42 layers
  - text_encoder  (~9.5 GB, 2 shards) — T5EncoderModel (t5-v1_1-xxl), 24 layers
  - vae           (~400 MB, 1 file)   — AutoencoderKLCogVideoX

Pipeline also includes:
  - tokenizer/  — T5Tokenizer (spiece.model + configs)
  - scheduler/  — CogVideoXDDIMScheduler config

Usage:
    mlx-forge convert cogvideox-fun-v1.5-5b-inp
    mlx-forge convert cogvideox-fun-v1.5-5b-inp --quantize --bits 8
    mlx-forge convert cogvideox-fun-v1.5-5b-inp --source /path/to/local/model
    mlx-forge validate cogvideox-fun-v1.5-5b-inp models/cogvideox-fun-v1.5-5b-inp-mlx
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx

from ..convert import (
    add_common_convert_args,
    add_source_arg,
    copy_required_files,
    default_output_dir,
    download_hf_files,
    fmt_size,
    load_safetensors,
    load_weights,
    print_output_summary,
    process_component,
    quantization_manifest_fields,
    quantize_component,
    source_download_dir,
    write_split_model,
)
from ..metadata import RecipeMetadata
from ..quantize import read_quantize_config, write_quantize_config
from ..transpose import transpose_conv
from ..validate import (
    count_layer_indices,
    finish_validation,
    start_validation,
    validate_conv_layout,
    validate_file_exists,
    validate_quantization,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ID = "alibaba-pai/CogVideoX-Fun-V1.5-5b-InP"

COMPONENTS = ["transformer", "text_encoder", "vae"]

# Approximate sizes in MB (fp16/bf16)
_COMPONENT_SIZE_MB: dict[str, int] = {
    "transformer": 9_500,  # ~9.5 GB
    "text_encoder": 9_500,  # ~9.5 GB (T5-XXL)
    "vae": 400,  # ~400 MB
}

_TEXT_ENCODER_SHARDS = 2

# Source files on HuggingFace
_HF_TRANSFORMER_FILES = [
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
]

_HF_TEXT_ENCODER_FILES = [
    "text_encoder/config.json",
    "text_encoder/model.safetensors.index.json",
] + [
    f"text_encoder/model-{i:05d}-of-{_TEXT_ENCODER_SHARDS:05d}.safetensors"
    for i in range(1, _TEXT_ENCODER_SHARDS + 1)
]

_HF_VAE_FILES = [
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

_HF_CONFIG_FILES = [
    "configuration.json",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/added_tokens.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
]

_ALL_HF_FILES = _HF_TRANSFORMER_FILES + _HF_TEXT_ENCODER_FILES + _HF_VAE_FILES + _HF_CONFIG_FILES

COMPONENT_PREFIX = {
    "transformer": "transformer",
    "text_encoder": "text_encoder",
    "vae": "vae",
}

# Components that should NOT be quantized (conv-heavy).
_SKIP_QUANTIZE_COMPONENTS = {"vae"}


# ---------------------------------------------------------------------------
# Key sanitization
# ---------------------------------------------------------------------------


METADATA = RecipeMetadata(
    name="cogvideox-fun-v1.5-5b-inp",
    source=REPO_ID,
    license="apache-2.0",
    # should_quantize_transformer: 2D .weight in the transformer blocks,
    # minus patch/position/timestep embeddings, norms and the output
    # projection; text_encoder and vae are skipped wholesale.
    quantization_scope=(
        "transformer block Linear weights only, leaving the embeddings, "
        "norms and the output projection in bf16"
    ),
    usage_url="https://github.com/dgrauet/VideoX-Fun-mlx",
    # Verbatim from the inference project's own README. Raw string: a
    # trailing backslash must stay a shell continuation, not swallow its
    # newline. {repo_id} is substituted at render time, so one
    # declaration covers every build of the model.
    cli_snippet=r"""
pip install mlx sentencepiece pillow numpy huggingface_hub
pip install git+https://github.com/dgrauet/mlx-arsenal.git
git clone https://github.com/dgrauet/VideoX-Fun-mlx.git && cd VideoX-Fun-mlx

huggingface-cli download {repo_id} --local-dir models/cogvideox-fun

python scripts/quick_infer.py \
    --model-path models/cogvideox-fun \
    --prompt "a beautiful sunset over the ocean" \
    --output sunset.gif
""",
    links=[
        "VideoX-Fun-mlx (inference code): https://github.com/dgrauet/VideoX-Fun-mlx",
        "mlx-forge (conversion tool): https://github.com/dgrauet/mlx-forge",
        "mlx-arsenal (MLX utilities): https://github.com/dgrauet/mlx-arsenal",
    ],
)


def sanitize_transformer_key(key: str) -> str | None:
    """Convert a PyTorch transformer key to MLX format.

    CogVideoX-Fun transformer keys are already clean — no prefix stripping needed.
    """
    return key


def sanitize_text_encoder_key(key: str) -> str | None:
    """Convert a PyTorch text encoder key to MLX format.

    T5 encoder keys are already clean — no prefix stripping needed.
    """
    return key


def sanitize_vae_key(key: str) -> str | None:
    """Convert a PyTorch VAE key to MLX format.

    CogVideoX-Fun VAE keys are already clean — no prefix stripping needed.
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


def should_quantize_transformer(key: str, weight: mx.array) -> bool:
    """Determine if a transformer weight should be quantized.

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
        single_filename="diffusion_pytorch_model.safetensors",
    )
    print(f"  {len(weights)} keys loaded (lazy) in {time.monotonic() - t0:.1f}s")

    print(f"\nProcessing {len(weights)} transformer keys...")
    t0 = time.monotonic()
    count = process_component(
        weights,
        "transformer",
        list(weights),
        output_dir,
        "transformer",
        sanitizer=sanitize_transformer_key,
        transform=maybe_transpose,
    )
    print(f"  Done: {count} weights saved in {time.monotonic() - t0:.1f}s")
    del weights
    gc.collect()
    mx.clear_cache()
    return count


def _convert_text_encoder(
    download_dir: Path,
    output_dir: Path,
    local_source: Path | None = None,
) -> int:
    """Convert the text encoder (T5-XXL) component. Returns weight count."""
    print("\n" + "=" * 60)
    print("Converting text_encoder (T5-XXL)")
    print("=" * 60)

    if local_source:
        te_dir = local_source / "text_encoder"
    else:
        te_dir = download_dir / "text_encoder"

    print(f"\nLoading text_encoder weights from {te_dir}...")
    t0 = time.monotonic()
    weights = load_weights(te_dir)  # uses default model.safetensors.index.json
    print(f"  {len(weights)} keys loaded (lazy) in {time.monotonic() - t0:.1f}s")

    print(f"\nProcessing {len(weights)} text_encoder keys...")
    t0 = time.monotonic()
    # T5 encoder has no conv layers — all weights pass through unchanged
    count = process_component(
        weights,
        "text_encoder",
        list(weights),
        output_dir,
        "text_encoder",
        sanitizer=sanitize_text_encoder_key,
    )
    print(f"  Done: {count} weights saved in {time.monotonic() - t0:.1f}s")
    del weights
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
        single_filename="diffusion_pytorch_model.safetensors",
    )
    print(f"  {len(weights)} keys loaded (lazy) in {time.monotonic() - t0:.1f}s")

    print(f"\nProcessing {len(weights)} VAE keys...")
    t0 = time.monotonic()
    count = process_component(
        weights,
        "vae",
        list(weights),
        output_dir,
        "vae",
        sanitizer=sanitize_vae_key,
        transform=maybe_transpose,
    )
    print(f"  Done: {count} weights saved in {time.monotonic() - t0:.1f}s")
    del weights
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

    # Read per-component configs
    for comp in COMPONENTS:
        comp_config_path = source_dir / comp / "config.json"
        if comp_config_path.exists():
            with open(comp_config_path) as f:
                config[comp] = json.load(f)

    return config


# ---------------------------------------------------------------------------
# Main convert entry point
# ---------------------------------------------------------------------------


def copy_pipeline_configs(source_dir: Path, output_dir: Path) -> None:
    """Copy every _HF_CONFIG_FILES entry, flattening `a/b` to `a_b`.

    Strict on purpose: each listed file is required for a usable artifact (the
    published q8 repo shipped without tokenizer/spiece.model because a silent
    `if src.exists()` skip let an incomplete --source through).
    """
    copy_required_files(source_dir, output_dir, _HF_CONFIG_FILES, flatten=True)


def convert(args) -> None:
    """Convert CogVideoX-Fun-V1.5-5b-InP to MLX split format."""
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = default_output_dir(
            "cogvideox-fun-v1.5-5b-inp", quantize=args.quantize, bits=args.bits
        )

    if args.dry_run:
        _dry_run(args, output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine source: local path or HF download
    local_source = Path(args.source) if args.source else None
    download_dir = source_download_dir(output_dir)

    if not local_source:
        # Download all files from HuggingFace
        print(f"\nDownloading from {REPO_ID}...")
        download_hf_files(REPO_ID, _ALL_HF_FILES, download_dir)

    total_weights = 0

    # -----------------------------------------------------------------------
    # 1. Transformer
    # -----------------------------------------------------------------------
    total_weights += _convert_transformer(download_dir, output_dir, local_source)

    # -----------------------------------------------------------------------
    # 2. Text Encoder (T5-XXL)
    # -----------------------------------------------------------------------
    total_weights += _convert_text_encoder(download_dir, output_dir, local_source)

    # -----------------------------------------------------------------------
    # 3. VAE
    # -----------------------------------------------------------------------
    total_weights += _convert_vae(download_dir, output_dir, local_source)

    # -----------------------------------------------------------------------
    # 4. Copy config files
    # -----------------------------------------------------------------------
    source_dir = local_source if local_source else download_dir

    config = _build_config(download_dir, local_source)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nSaved config.json")

    # Copy per-component configs for reference
    for comp in COMPONENTS:
        src_cfg = source_dir / comp / "config.json"
        if src_cfg.exists():
            dst_cfg = output_dir / f"{comp}_config.json"
            shutil.copy2(str(src_cfg), str(dst_cfg))
            print(f"  Copied {comp}/config.json -> {dst_cfg.name}")

    # Copy pipeline config files (tokenizer, scheduler, model_index)
    copy_pipeline_configs(source_dir, output_dir)

    # Split model manifest — written once, below, after the optional
    # quantization so `quantized` reflects what actually happened.
    split_info: dict = {
        "format": "split",
        "components": COMPONENTS,
        **METADATA.as_split_fields(),
        "notes": {
            "inpainting": "Transformer patch_embed.proj is a Linear with in_dim=264 "
            "(in_channels=33 [16 latent + 16 masked + 1 mask] * patch_volume=8).",
            "text_encoder": "T5-v1.1-XXL encoder (24 layers, d_model=4096).",
        },
    }

    # -----------------------------------------------------------------------
    # 5. Optional quantization (transformer + text_encoder, skip vae)
    # -----------------------------------------------------------------------
    if args.quantize:
        quantize_component(
            output_dir,
            "transformer",
            bits=args.bits,
            group_size=args.group_size,
            should_quantize=should_quantize_transformer,
        )

        skip = set(_SKIP_QUANTIZE_COMPONENTS)
        # T5 text encoder stays in bf16 — quantization degrades quality
        # (24 layers of accumulated error, especially at 4-bit)
        skip.add("text_encoder")

        write_quantize_config(
            output_dir,
            bits=args.bits,
            group_size=args.group_size,
            skip_components=sorted(skip),
        )

    split_info.update(
        quantization_manifest_fields(
            quantized=args.quantize, bits=args.bits, group_size=args.group_size
        )
    )
    write_split_model(output_dir, split_info)

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
    print("DRY RUN — no files will be downloaded or written")
    print("=" * 60)

    source_label = args.source if args.source else f"{REPO_ID} (HuggingFace)"
    print(f"\nSource:     {source_label}")
    print(f"Output dir: {output_dir}")
    print(f"Files to download: {len(_ALL_HF_FILES)}")
    print("\nComponents:")

    total_mb = 0.0
    for comp in COMPONENTS:
        size_mb = _COMPONENT_SIZE_MB[comp]
        skip_q = comp in _SKIP_QUANTIZE_COMPONENTS
        if args.quantize and not skip_q:
            ratio = 16 / args.bits
            size_mb = size_mb / ratio
            print(f"  {comp}.safetensors: ~{fmt_size(size_mb)} (int{args.bits})")
        elif args.quantize and skip_q:
            print(f"  {comp}.safetensors: ~{fmt_size(size_mb)} (fp16, skip quantize)")
        else:
            print(f"  {comp}.safetensors: ~{fmt_size(size_mb)} (fp16)")
        total_mb += size_mb

    print("\nPipeline files:")
    print("  config.json, split_model.json, configuration.json")
    print("  transformer_config.json, text_encoder_config.json, vae_config.json")
    print("  model_index.json")
    print("  scheduler_scheduler_config.json")
    print("  tokenizer_added_tokens.json, tokenizer_tokenizer_config.json")
    print("  tokenizer_special_tokens_map.json, tokenizer_spiece.model")

    if args.quantize:
        print(f"\nQuantization: int{args.bits}, group_size={args.group_size}")
        print("  Target: transformer + text_encoder (Linear weights only)")
        print("  Skipped: vae (conv-heavy)")

    print(f"\nEstimated output size: ~{fmt_size(total_mb)}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(args) -> None:
    """Validate a converted CogVideoX-Fun model."""
    model_dir, result = start_validation(args.model_dir)

    # Check quantization
    qconfig = read_quantize_config(model_dir)
    is_quantized = qconfig is not None
    skip_quantize: set[str] = set(qconfig.get("skip_components", [])) if qconfig else set()
    if qconfig is not None:
        print(f"Model is quantized: int{qconfig.get('bits', '?')}")
        if skip_quantize:
            print(f"  Skipped components: {', '.join(sorted(skip_quantize))}")

    # --- File structure ---
    print("\n== File Structure ==")
    for comp in COMPONENTS:
        validate_file_exists(model_dir, f"{comp}.safetensors", result)
    validate_file_exists(model_dir, "config.json", result)
    validate_file_exists(model_dir, "split_model.json", result)
    validate_file_exists(model_dir, "model_index.json", result)

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
        weights = load_safetensors(tf_path)
        keys = set(weights.keys())

        all_prefixed = all(k.startswith("transformer.") for k in keys)
        result.check(all_prefixed, f"All keys have 'transformer.' prefix ({len(keys)} keys)")

        # Check inpainting patch_embed has correct input dimension
        # CogVideoX-Fun V1.5 uses a Linear patch_embed (not Conv3d):
        #   input_dim = in_channels(33) * patch_size_t(2) * patch_size(2) * patch_size(2) = 264
        #   in_channels = 16 latent + 16 masked latent + 1 mask = 33
        pe_key = "transformer.patch_embed.proj.weight"
        if pe_key in keys:
            pe_shape = weights[pe_key].shape
            expected_in_dim = 33 * 2 * 2 * 2  # 264
            if weights[pe_key].ndim == 2:
                in_dim = pe_shape[1]
                result.check(
                    in_dim == expected_in_dim,
                    f"patch_embed input dim == {expected_in_dim} for inpainting"
                    f" (got {in_dim}, shape {pe_shape})",
                )
            elif weights[pe_key].ndim == 5:
                in_channels = pe_shape[-1]  # channels-last after transpose
                result.check(
                    in_channels == 33,
                    f"patch_embed input channels == 33 for inpainting"
                    f" (got {in_channels}, shape {pe_shape})",
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

    # --- Text Encoder ---
    print("\n== Text Encoder Weights ==")
    te_path = model_dir / "text_encoder.safetensors"
    if te_path.exists():
        weights = load_safetensors(te_path)
        keys = set(weights.keys())

        all_prefixed = all(k.startswith("text_encoder.") for k in keys)
        result.check(all_prefixed, f"All keys have 'text_encoder.' prefix ({len(keys)} keys)")

        # Check encoder blocks (T5-XXL has 24 layers)
        bare_keys = {k.removeprefix("text_encoder.") for k in keys}
        block_indices = count_layer_indices(bare_keys, block_key="encoder.block")
        result.check(
            len(block_indices) == 24,
            f"24 T5 encoder blocks (got {len(block_indices)})",
        )

        # Check shared embedding
        shared_key = "text_encoder.shared.weight"
        result.check(shared_key in keys, "shared embedding present")

        # Check final layer norm
        final_norm_key = "text_encoder.encoder.final_layer_norm.weight"
        result.check(final_norm_key in keys, "encoder.final_layer_norm present")

        if is_quantized and "text_encoder" not in skip_quantize:
            validate_quantization(weights, result, block_key="encoder.block")
        elif is_quantized:
            print("  (text_encoder kept in bf16 — skipping quantization checks)")

        total_params = sum(v.size for v in weights.values())
        print(f"  Total text_encoder parameters: {total_params / 1e9:.2f}B")
        del weights
        gc.collect()
        mx.clear_cache()

    # --- VAE ---
    print("\n== VAE Weights ==")
    vae_path = model_dir / "vae.safetensors"
    if vae_path.exists():
        weights = load_safetensors(vae_path)
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

    finish_validation(result)


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
    """Add CogVideoX-Fun convert arguments to a parser."""
    add_source_arg(
        parser,
        help="Path to local model directory (skips HF download). "
        "Must contain transformer/, text_encoder/, and vae/ subdirectories.",
    )
    add_common_convert_args(
        parser,
        output_default="./models/cogvideox-fun-v1.5-5b-inp-mlx[-q<bits>]",
        quantize_help="Quantize transformer and text_encoder after conversion",
        dry_run_help="Preview conversion plan without downloading or writing anything",
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


def split(args) -> None:
    print("CogVideoX-Fun is already split by component during conversion.")
    print("No further splitting needed.")
