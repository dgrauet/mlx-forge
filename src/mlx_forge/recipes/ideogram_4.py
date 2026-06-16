"""Ideogram 4 FP8 → MLX conversion recipe.

Converts ideogram-ai/ideogram-4-fp8 (float8_e4m3fn weights with per-row float32
scales) to MLX-native bfloat16, with optional int8 quantization.

The source model ships all linear weights in float8_e4m3fn, paired with a per-row
float32 scale tensor (shape: (out_features,)). MLX has no native FP8 dtype, so the
file is loaded with uint8 tensors. Dequantization:

    weight_bf16 = mx.from_fp8(weight_uint8) * scale[:, None]   # per-row multiply

FP8 components: conditional_transformer, unconditional_transformer, text_encoder.
bf16 component:  vae (Flux2 AutoencoderKL, Conv2d only, no FP8).

Usage:
    mlx-forge convert ideogram-4
    mlx-forge convert ideogram-4 --quantize --bits 8
    mlx-forge convert ideogram-4 --source /path/to/local/checkpoint
    mlx-forge validate ideogram-4 models/ideogram-4-mlx
    mlx-forge validate ideogram-4 models/ideogram-4-mlx-q8
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
    process_component,
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

REPO_ID = "ideogram-ai/ideogram-4-fp8"

COMPONENTS = ["conditional_transformer", "unconditional_transformer", "text_encoder", "vae"]

_SKIP_QUANTIZE_COMPONENTS = {"vae"}

_COMPONENT_SIZE_MB = {
    "conditional_transformer": 8_700,
    "unconditional_transformer": 8_700,
    "text_encoder": 8_200,
    "vae": 500,
}

_CHECKPOINT_DOWNLOAD_MB = sum(_COMPONENT_SIZE_MB.values())

COMPONENT_PREFIX = {comp: comp for comp in COMPONENTS}

# ---------------------------------------------------------------------------
# HuggingFace file lists (for auto-download)
# ---------------------------------------------------------------------------

_HF_TRANSFORMER_FILES = [
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
]

_HF_UNCONDITIONAL_TRANSFORMER_FILES = [
    "unconditional_transformer/config.json",
    "unconditional_transformer/diffusion_pytorch_model.safetensors",
    "unconditional_transformer/diffusion_pytorch_model.safetensors.index.json",
]

_HF_TEXT_ENCODER_FILES = [
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder/model.safetensors.index.json",
]

_HF_VAE_FILES = [
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

_HF_CONFIG_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/chat_template.jinja",
]

_ALL_HF_FILES = (
    _HF_TRANSFORMER_FILES
    + _HF_UNCONDITIONAL_TRANSFORMER_FILES
    + _HF_TEXT_ENCODER_FILES
    + _HF_VAE_FILES
    + _HF_CONFIG_FILES
)

# ---------------------------------------------------------------------------
# FP8 dequantization
# ---------------------------------------------------------------------------


def _dequantize_fp8(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Merge float8_e4m3fn weight + per-row float32 scale into bfloat16 tensors.

    MLX loads float8_e4m3fn as uint8 (raw fp8 bits). Each FP8 weight matrix
    key.endswith(".weight") is paired with key.replace(".weight", ".weight_scale"),
    a float32 tensor of shape (out_features,). After dequantization the .weight_scale
    keys are dropped — they are fully consumed here.
    """
    result: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key.endswith(".weight_scale"):
            continue  # consumed when processing the paired .weight tensor
        if value.dtype == mx.uint8 and key.endswith(".weight"):
            scale_key = key[: -len("weight")] + "weight_scale"
            if scale_key in weights:
                scale = weights[scale_key]
                _materialize(value, scale)
                w_f32 = mx.from_fp8(value).astype(mx.float32) * scale[:, None]
                _materialize(w_f32)
                result[key] = w_f32.astype(mx.bfloat16)
                del w_f32
            else:
                result[key] = value
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Key sanitizers
# ---------------------------------------------------------------------------


def sanitize_transformer_key(key: str) -> str:
    """Transformer keys are already in the right format; return as-is."""
    return key


def sanitize_text_encoder_key(key: str) -> str | None:
    """Strip language_model. prefix; skip vision encoder and lm_head keys."""
    if not key.startswith("language_model."):
        return None
    suffix = key[len("language_model.") :]
    if not suffix.startswith(("embed_tokens.", "layers.", "norm.")):
        return None
    return suffix


def sanitize_vae_key(key: str) -> str:
    """VAE keys are already in the right format; return as-is."""
    return key


SANITIZERS = {
    "conditional_transformer": sanitize_transformer_key,
    "unconditional_transformer": sanitize_transformer_key,
    "text_encoder": sanitize_text_encoder_key,
    "vae": sanitize_vae_key,
}

# ---------------------------------------------------------------------------
# Conv transposition (VAE only — transformers and text encoder are all-Linear)
# ---------------------------------------------------------------------------


def maybe_transpose(key: str, value: mx.array, component: str) -> mx.array:
    """Transpose Conv2d weights from PyTorch (O,I,H,W) to MLX (O,H,W,I) layout."""
    if component == "vae" and value.ndim == 4 and "weight" in key:
        return transpose_conv(value)
    return value


# ---------------------------------------------------------------------------
# Quantization predicates
# ---------------------------------------------------------------------------


def _should_quantize_transformer(key: str, weight: mx.array) -> bool:
    """Quantize all 2D linear weights in transformer blocks and projections.

    Norm weights are 1D so the ndim check naturally excludes them.
    """
    return weight.ndim == 2 and key.endswith(".weight")


def _should_quantize_text_encoder(key: str, weight: mx.array) -> bool:
    """Quantize all 2D linear weights; skip the embedding table."""
    if not (weight.ndim == 2 and key.endswith(".weight")):
        return False
    return "embed_tokens" not in key


# ---------------------------------------------------------------------------
# Component conversion helpers
# ---------------------------------------------------------------------------


_FP8_COMPONENTS = {"conditional_transformer", "unconditional_transformer", "text_encoder"}


def _convert_component(
    component: str,
    source_dir: Path,
    output_dir: Path,
) -> int:
    """Load, optionally dequantize FP8, sanitize, transpose, materialize, and save."""
    if component == "conditional_transformer":
        comp_dir = source_dir / "transformer"
        single_file = "diffusion_pytorch_model.safetensors"
    elif component == "unconditional_transformer":
        comp_dir = source_dir / "unconditional_transformer"
        single_file = "diffusion_pytorch_model.safetensors"
    elif component == "text_encoder":
        comp_dir = source_dir / "text_encoder"
        single_file = "model.safetensors"
    else:  # vae
        comp_dir = source_dir / "vae"
        single_file = "diffusion_pytorch_model.safetensors"

    t0 = time.monotonic()
    weights = load_weights(comp_dir, single_filename=single_file)
    print(f"  {len(weights)} keys loaded in {time.monotonic() - t0:.1f}s")

    if component in _FP8_COMPONENTS:
        fp8_count = sum(1 for v in weights.values() if v.dtype == mx.uint8)
        print(f"  Dequantizing {fp8_count} FP8 weight matrices to bfloat16...")
        t1 = time.monotonic()
        weights = _dequantize_fp8(weights)
        print(f"  Dequantization done in {time.monotonic() - t1:.1f}s")

    sanitizer = SANITIZERS[component]
    prefix = COMPONENT_PREFIX[component]

    count = process_component(
        weights,
        component,
        list(weights.keys()),
        output_dir,
        prefix,
        sanitizer=sanitizer,
        transform=maybe_transpose,
    )

    del weights
    gc.collect()
    mx.clear_cache()
    return count


# ---------------------------------------------------------------------------
# Pipeline file copying
# ---------------------------------------------------------------------------


def _copy_pipeline_files(source_dir: Path, output_dir: Path) -> None:
    """Copy tokenizer, scheduler, and model_index files with directory prefix."""
    for config_file in _HF_CONFIG_FILES:
        src = source_dir / config_file
        if not src.exists():
            print(f"  WARNING: {config_file} not found, skipping")
            continue
        if "/" in config_file:
            prefix = config_file.split("/")[0]
            dest = output_dir / f"{prefix}_{Path(config_file).name}"
        else:
            dest = output_dir / Path(config_file).name
        shutil.copy2(str(src), str(dest))
        print(f"  Copied {config_file} → {dest.name}")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _dry_run(args, output_dir: Path) -> None:
    print("=" * 60)
    print("DRY RUN — no files will be downloaded or written")
    print("=" * 60)

    if args.source:
        print(f"\nSource:     {args.source} (local)")
    else:
        print(f"\nSource:     {REPO_ID} (HuggingFace)")
        print(f"Download:   ~{fmt_size(_CHECKPOINT_DOWNLOAD_MB)} → ./models/ideogram-4-src")
        print(f"Files:      {len(_ALL_HF_FILES)} files")

    print(f"\nOutput dir: {output_dir}")
    print("\nOutput files:")
    total_mb = 0.0
    for comp in COMPONENTS:
        size_mb = _COMPONENT_SIZE_MB[comp]
        if args.quantize and comp not in _SKIP_QUANTIZE_COMPONENTS:
            ratio = 16 / args.bits
            size_mb = size_mb / ratio
            label = f"int{args.bits}"
        else:
            label = "bf16"
        print(f"  {comp}.safetensors: ~{fmt_size(size_mb)} ({label})")
        total_mb += size_mb

    print("  config.json, split_model.json, tokenizer_*, scheduler_*, model_index.json")

    if args.quantize:
        print(f"\nQuantization: int{args.bits}, group_size={args.group_size}")
        print("  Target: conditional_transformer, unconditional_transformer, text_encoder")
        print("  Skipped: vae (Conv2d)")

    print(f"\nEstimated output size: ~{fmt_size(total_mb)}")
    if not args.source:
        print(f"Estimated download:   ~{fmt_size(_CHECKPOINT_DOWNLOAD_MB)}")


# ---------------------------------------------------------------------------
# Main convert entry point
# ---------------------------------------------------------------------------


def convert(args) -> None:
    """Convert Ideogram 4 FP8 checkpoint to MLX split format."""
    if args.output:
        output_dir = Path(args.output)
    else:
        suffix = f"-q{args.bits}" if args.quantize else ""
        output_dir = Path("models") / f"ideogram-4-mlx{suffix}"

    if args.dry_run:
        _dry_run(args, output_dir)
        return

    # Resolve source directory
    if args.source:
        source_dir = Path(args.source)
        print(f"Using local source: {source_dir}")
    else:
        source_dir = Path("models") / "ideogram-4-src"
        print(f"Downloading {len(_ALL_HF_FILES)} files from {REPO_ID}...")
        print(f"(~{fmt_size(_CHECKPOINT_DOWNLOAD_MB)} total, may take a while)")
        download_hf_files(REPO_ID, _ALL_HF_FILES, source_dir)
        print(f"Downloaded to: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_weights = 0
    for component in COMPONENTS:
        print(f"\n{'=' * 60}")
        print(f"[{component}]")
        print("=" * 60)
        t0 = time.monotonic()
        count = _convert_component(component, source_dir, output_dir)
        elapsed = time.monotonic() - t0
        total_weights += count
        print(f"  Done: {count} weights saved in {elapsed:.1f}s")

    # Copy transformer configs for downstream use
    for comp_dir_name, comp_name in [
        ("transformer", "conditional_transformer"),
        ("unconditional_transformer", "unconditional_transformer"),
        ("text_encoder", "text_encoder"),
        ("vae", "vae"),
    ]:
        cfg_src = source_dir / comp_dir_name / "config.json"
        if cfg_src.exists():
            shutil.copy2(str(cfg_src), str(output_dir / f"{comp_name}_config.json"))

    # Write config.json
    config = {
        "model_type": "ideogram4",
        "components": COMPONENTS,
        "source": REPO_ID,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Write split_model.json
    split_info: dict = {
        "format": "split",
        "components": COMPONENTS,
        "source": REPO_ID,
        "quantized": False,
    }
    with open(output_dir / "split_model.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Copy pipeline files
    print("\nCopying pipeline files...")
    _copy_pipeline_files(source_dir, output_dir)

    print(f"\n{'=' * 60}")
    print(f"Conversion complete: {total_weights} total weights")
    print(f"Output: {output_dir}")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.name}: {size_mb:.1f} MB")

    # Optional quantization
    if args.quantize:
        print(f"\nQuantizing to int{args.bits} (group_size={args.group_size})...")

        quantize_component(
            output_dir,
            "conditional_transformer",
            bits=args.bits,
            group_size=args.group_size,
            should_quantize=_should_quantize_transformer,
        )
        quantize_component(
            output_dir,
            "unconditional_transformer",
            bits=args.bits,
            group_size=args.group_size,
            should_quantize=_should_quantize_transformer,
        )
        quantize_component(
            output_dir,
            "text_encoder",
            bits=args.bits,
            group_size=args.group_size,
            should_quantize=_should_quantize_text_encoder,
        )

        split_info["quantized"] = True
        split_info["quantization_bits"] = args.bits
        with open(output_dir / "split_model.json", "w") as f:
            json.dump(split_info, f, indent=2)

        qconfig = {
            "quantization": {
                "bits": args.bits,
                "group_size": args.group_size,
                "skip_components": list(_SKIP_QUANTIZE_COMPONENTS),
            }
        }
        with open(output_dir / "quantize_config.json", "w") as f:
            json.dump(qconfig, f, indent=2)

        print("\nFinal files after quantization:")
        for p in sorted(output_dir.iterdir()):
            if p.is_file():
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"  {p.name}: {size_mb:.1f} MB")

    print("\nDone!")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(args) -> None:
    """Validate a converted Ideogram 4 model directory."""
    from ..convert import load_safetensors

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: {model_dir} does not exist")
        raise SystemExit(1)

    print(f"Validating: {model_dir}")
    result = ValidationResult()

    is_quantized = (model_dir / "quantize_config.json").exists()
    if is_quantized:
        with open(model_dir / "quantize_config.json") as f:
            qconfig = json.load(f)
        bits = qconfig.get("quantization", {}).get("bits", "?")
        print(f"Model is quantized: int{bits}")

    # File structure
    print("\n== File Structure ==")
    for fname in [
        "conditional_transformer.safetensors",
        "unconditional_transformer.safetensors",
        "text_encoder.safetensors",
        "vae.safetensors",
        "config.json",
        "split_model.json",
        "model_index.json",
    ]:
        validate_file_exists(model_dir, fname, result)

    for fname in [
        "quantize_config.json",
        "conditional_transformer_config.json",
        "unconditional_transformer_config.json",
        "text_encoder_config.json",
        "vae_config.json",
        "scheduler_scheduler_config.json",
        "tokenizer_tokenizer.json",
        "tokenizer_tokenizer_config.json",
    ]:
        if (model_dir / fname).exists():
            print(f"  \033[92m✓\033[0m {fname} exists (optional)")

    # Validate transformer components
    for comp in ("conditional_transformer", "unconditional_transformer"):
        print(f"\n== {comp} ==")
        tf_path = model_dir / f"{comp}.safetensors"
        if not tf_path.exists():
            continue
        weights = load_safetensors(tf_path)
        keys = set(weights.keys())

        result.check(
            all(k.startswith(f"{comp}.") for k in keys),
            f"All keys have '{comp}.' prefix",
        )

        # Check no FP8 uint8 tensors remain
        fp8_remaining = [k for k, v in weights.items() if v.dtype == mx.uint8]
        result.check(len(fp8_remaining) == 0, "No FP8 (uint8) tensors remaining (dequantized)")

        # Check no weight_scale keys remain
        scale_keys = [k for k in keys if k.endswith(".weight_scale")]
        result.check(len(scale_keys) == 0, "No .weight_scale keys (consumed by dequantize)")

        layer_indices = count_layer_indices(keys, block_key="layers")
        result.check(len(layer_indices) == 34, f"34 transformer layers (got {len(layer_indices)})")

        if is_quantized:
            validate_quantization(weights, result, block_key="layers")

        total_params = sum(v.size for v in weights.values())
        print(f"  Total parameters: {total_params / 1e9:.2f}B")
        del weights
        gc.collect()
        mx.clear_cache()

    # Validate text encoder
    print("\n== text_encoder ==")
    te_path = model_dir / "text_encoder.safetensors"
    if te_path.exists():
        weights = load_safetensors(te_path)
        keys = set(weights.keys())

        result.check(
            all(k.startswith("text_encoder.") for k in keys),
            "All keys have 'text_encoder.' prefix",
        )

        fp8_remaining = [k for k, v in weights.items() if v.dtype == mx.uint8]
        result.check(len(fp8_remaining) == 0, "No FP8 (uint8) tensors remaining")

        scale_keys = [k for k in keys if k.endswith(".weight_scale")]
        result.check(len(scale_keys) == 0, "No .weight_scale keys")

        embed_keys = [k for k in keys if "embed_tokens" in k]
        result.check(len(embed_keys) > 0, f"embed_tokens present ({len(embed_keys)} keys)")

        layer_indices = count_layer_indices(keys, block_key="layers")
        result.check(
            len(layer_indices) == 36, f"36 Qwen3 decoder layers (got {len(layer_indices)})"
        )

        if is_quantized:
            validate_quantization(weights, result, block_key="layers")

        total_params = sum(v.size for v in weights.values())
        print(f"  Total parameters: {total_params / 1e9:.2f}B")
        del weights
        gc.collect()
        mx.clear_cache()

    # Validate VAE
    print("\n== vae ==")
    vae_path = model_dir / "vae.safetensors"
    if vae_path.exists():
        weights = load_safetensors(vae_path)
        keys = set(weights.keys())

        result.check(
            all(k.startswith("vae.") for k in keys),
            "All keys have 'vae.' prefix",
        )

        dec_keys = [k for k in keys if k.startswith("vae.decoder.")]
        enc_keys = [k for k in keys if k.startswith("vae.encoder.")]
        result.check(len(dec_keys) > 0, f"Decoder keys present ({len(dec_keys)})")
        result.check(len(enc_keys) > 0, f"Encoder keys present ({len(enc_keys)})")

        validate_conv_layout(weights, result, ndim=4)

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
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help=(
            "Path to local Ideogram 4 FP8 checkpoint directory (skips HuggingFace download). "
            "Expects transformer/, unconditional_transformer/, text_encoder/, vae/ subdirs."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ./models/ideogram-4-mlx[-q<bits>])",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize linear weights to int4/int8 after FP8 dequantization",
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
    parser.add_argument("model_dir", type=str, help="Path to converted model directory")


def add_split_args(parser) -> None:
    parser.add_argument("model_dir", type=str, help="Model directory (split not applicable)")
