"""Fish Audio S2 Pro conversion recipe.

Converts the fishaudio/s2-pro TTS checkpoint (Dual-AR Qwen3-based) to MLX split format.
Includes all three components: text_model, audio_decoder, and codec (DAC vocoder).

The codec is a Descript Audio Codec (DAC) stored as a separate ``codec.pth`` file
with keys prefixed by ``generator.``.  It contains Conv1d / ConvTranspose1d layers
that need transposition (PyTorch channels-second -> MLX channels-last) and must NOT
be quantized.

Usage:
    mlx-forge convert fish-s2-pro
    mlx-forge convert fish-s2-pro --quantize --bits 8
    mlx-forge validate fish-s2-pro models/fish-s2-pro-mlx
"""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx

from ..convert import (
    classify_keys,
    download_hf_files,
    fmt_size,
    load_weights,
    process_component,
    quantize_component,
    shard_filenames,
)
from ..transpose import transpose_conv
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

REPO_ID = "fishaudio/s2-pro"

COMPONENTS = ["text_model", "audio_decoder", "codec"]

COMPONENT_PREFIX = {
    "text_model": "text_model",
    "audio_decoder": "audio_decoder",
    "codec": "codec",
}

# Approximate sizes in MB for dry-run estimation (bf16)
_COMPONENT_SIZE_MB = {
    "text_model": 8500,
    "audio_decoder": 600,
    "codec": 1870,
}

_CHECKPOINT_SIZE_MB = 11070  # ~9.2 GB (2 shards) + ~1.87 GB (codec.pth)

CHECKPOINT_FILES = shard_filenames(2)

CODEC_FILE = "codec.pth"

CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
]

# ---------------------------------------------------------------------------
# Key classification
# ---------------------------------------------------------------------------


def classify_key(key: str) -> str | None:
    """Classify a weight key into a component name.

    Returns one of: text_model, audio_decoder, codec, or None (skip).

    Codec keys originate from ``codec.pth`` and are prefixed with
    ``generator.`` after loading.  The prefix ``generator.`` is used
    as the classifier trigger — keys like ``encoder.*``, ``decoder.*``,
    and ``quantizer.*`` all live under ``generator.`` in the checkpoint.
    """
    if key.startswith("text_model."):
        return "text_model"
    if key.startswith("audio_decoder."):
        return "audio_decoder"
    if key.startswith("generator."):
        return "codec"
    return None


# ---------------------------------------------------------------------------
# Key sanitization
# ---------------------------------------------------------------------------


def sanitize_text_model_key(key: str) -> str:
    """Strip the text_model.model. prefix."""
    return key.replace("text_model.model.", "")


def sanitize_audio_decoder_key(key: str) -> str:
    """Strip the audio_decoder. prefix."""
    return key.replace("audio_decoder.", "")


def sanitize_codec_key(key: str) -> str:
    """Strip the generator. prefix from codec keys."""
    return key.replace("generator.", "", 1)


SANITIZERS = {
    "text_model": sanitize_text_model_key,
    "audio_decoder": sanitize_audio_decoder_key,
    "codec": sanitize_codec_key,
}


# ---------------------------------------------------------------------------
# Conv transposition (codec)
# ---------------------------------------------------------------------------


def codec_transform(key: str, weight: mx.array, component_name: str) -> mx.array:
    """Transpose Conv1d / ConvTranspose1d weights in the codec from PyTorch to MLX layout.

    PyTorch Conv1d:          (O, I, K) -> MLX: (O, K, I)
    PyTorch ConvTranspose1d: (I, O, K) -> MLX: (O, K, I)
    """
    if weight.ndim != 3:
        return weight
    # ConvTranspose1d layers live inside decoder upsampling blocks
    is_transpose = "upsample" in key and "conv." in key
    return transpose_conv(weight, is_conv_transpose=is_transpose)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

# Components that should NOT be quantized (conv-heavy, no benefit)
_SKIP_QUANTIZE_COMPONENTS = {"codec"}


def fish_s2_should_quantize(key: str, weight: mx.array) -> bool:
    """Only quantize transformer Linear weights (not embeddings, norms, or conv)."""
    return (
        key.endswith(".weight")
        and weight.ndim == 2
        and weight.shape[0] > 1
        and weight.shape[1] > 1
        and weight.size >= 256
        and "embeddings" not in key
        and "norm" not in key
    )


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _dry_run(args, output_dir: Path) -> None:
    """Print conversion plan without executing anything."""
    print("=" * 60)
    print("DRY RUN — no files will be downloaded or written")
    print("=" * 60)

    print(f"\nSource:     {REPO_ID}")
    if not args.checkpoint:
        print(f"Download:   ~{fmt_size(_CHECKPOINT_SIZE_MB)} (2 shards + codec.pth + config)")
        print("            → ./models/fish-s2-pro-src/")

    print(f"Output dir: {output_dir}")

    print("\nOutput files:")
    total_mb = 0.0
    for comp in COMPONENTS:
        size_mb = _COMPONENT_SIZE_MB[comp]
        if args.quantize and comp not in _SKIP_QUANTIZE_COMPONENTS:
            ratio = 16 / args.bits
            size_mb = size_mb / ratio
            label = f"  {comp}.safetensors: ~{fmt_size(size_mb)} (int{args.bits})"
        else:
            label = f"  {comp}.safetensors: ~{fmt_size(size_mb)} (bf16)"
            if args.quantize and comp in _SKIP_QUANTIZE_COMPONENTS:
                label += " (conv-heavy, not quantized)"
        print(label)
        total_mb += size_mb

    print("  config.json, split_model.json, tokenizer.json, ...")

    if args.quantize:
        print(f"\nQuantization: int{args.bits}, group_size={args.group_size}")
        print("  Target: Linear weights only (not embeddings, norms, conv)")
        print("  Skipped: codec (conv-heavy DAC vocoder)")

    print(f"\nEstimated output size: ~{fmt_size(total_mb)}")
    if not args.checkpoint:
        print(f"Estimated download:   ~{fmt_size(_CHECKPOINT_SIZE_MB)}")
        print(f"Estimated total disk: ~{fmt_size(total_mb + _CHECKPOINT_SIZE_MB)}")


# ---------------------------------------------------------------------------
# Main convert entry point
# ---------------------------------------------------------------------------


def convert(args) -> None:
    """Convert Fish Audio S2 Pro checkpoint to MLX split format."""
    if args.output:
        output_dir = Path(args.output)
    else:
        suffix = f"-q{args.bits}" if args.quantize else ""
        output_dir = Path("models") / f"fish-s2-pro-mlx{suffix}"

    if args.dry_run:
        _dry_run(args, output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get checkpoint files
    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
        print(f"Using local checkpoint: {checkpoint_dir}")
    else:
        checkpoint_dir = Path("models") / "fish-s2-pro-src"
        print(f"Downloading {REPO_ID} checkpoint files...")
        download_hf_files(REPO_ID, CHECKPOINT_FILES, checkpoint_dir)
        print("Downloading codec checkpoint...")
        download_hf_files(REPO_ID, [CODEC_FILE], checkpoint_dir)
        print("Downloading config and tokenizer files...")
        download_hf_files(REPO_ID, CONFIG_FILES, checkpoint_dir)

    # Step 2: Copy config and tokenizer files to output dir
    for fname in CONFIG_FILES:
        src = checkpoint_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    # Step 3: Load weights lazily (sharded via index + codec.pth)
    t0 = time.monotonic()
    checkpoint_weights = load_weights(checkpoint_dir)
    print(f"  {len(checkpoint_weights)} transformer keys loaded (lazy)")

    # Load codec weights from codec.pth (PyTorch format, keys under "generator.")
    codec_path = checkpoint_dir / CODEC_FILE
    if codec_path.exists():
        print(f"  Loading codec weights from {CODEC_FILE}...")
        try:
            import torch
        except ImportError:
            print(
                "ERROR: torch is required to load codec.pth\n"
                "Install it with: uv pip install 'mlx-forge[torch]'"
            )
            raise SystemExit(1)

        # SECURITY: weights_only=True restricts unpickling to tensor data only,
        # blocking arbitrary code execution from malicious .pth files.
        print("  (weights_only=True — safe deserialization mode)")
        codec_raw = torch.load(str(codec_path), map_location="cpu", weights_only=True)
        # Unwrap if wrapped in a "state_dict" key
        if isinstance(codec_raw, dict) and "state_dict" in codec_raw:
            codec_raw = codec_raw["state_dict"]
        codec_count = 0
        for k, v in codec_raw.items():
            canon = k if k.startswith("generator.") else f"generator.{k}"
            checkpoint_weights[canon] = mx.array(v.float().numpy())
            codec_count += 1
        print(f"  {codec_count} codec keys loaded")
        del codec_raw
    else:
        print(f"  WARNING: {CODEC_FILE} not found, codec will be skipped")

    # Classify keys
    print("\nClassifying weight keys...")
    keys_by_component = classify_keys(checkpoint_weights, classify_key)

    for comp, keys in sorted(keys_by_component.items()):
        print(f"  {comp}: {len(keys)} keys")
    print(f"  Loaded + classified in {time.monotonic() - t0:.1f}s")

    # Step 4: Process each component
    total_weights = 0
    for component_name in COMPONENTS:
        keys = keys_by_component.get(component_name, [])
        if not keys:
            print(f"\n[{component_name}] No keys found, skipping")
            continue

        component_prefix = COMPONENT_PREFIX[component_name]
        transform = codec_transform if component_name == "codec" else None
        print(f"\n[{component_name}] Processing {len(keys)} keys...")
        t0 = time.monotonic()
        count = process_component(
            checkpoint_weights,
            component_name,
            keys,
            output_dir,
            component_prefix,
            sanitizer=SANITIZERS[component_name],
            transform=transform,
        )
        elapsed = time.monotonic() - t0
        total_weights += count
        print(f"  Done: {count} weights saved in {elapsed:.1f}s")

    del checkpoint_weights
    gc.collect()
    mx.clear_cache()

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
                should_quantize=fish_s2_should_quantize,
            )

        qconfig = {
            "quantization": {
                "bits": args.bits,
                "group_size": args.group_size,
                "target": "linear_weights_only",
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
# Validation
# ---------------------------------------------------------------------------


def validate(args) -> None:
    """Validate a converted Fish Audio S2 Pro model."""
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
    expected = [
        "config.json",
        "split_model.json",
        "text_model.safetensors",
        "audio_decoder.safetensors",
        "codec.safetensors",
    ]
    for fname in expected:
        validate_file_exists(model_dir, fname, result)
    for fname in ["quantize_config.json", "tokenizer.json"]:
        if (model_dir / fname).exists():
            print(f"  \033[92m\u2713\033[0m {fname} exists (optional)")

    # Config
    print("\n== Config Validation ==")
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        result.check(
            config.get("model_type") == "fish_qwen3_omni",
            f"model_type is fish_qwen3_omni (got: {config.get('model_type')})",
        )
        text_cfg = config.get("text_config", {})
        n_layer = text_cfg.get("n_layer")
        result.check(n_layer == 36, f"text n_layer == 36 (got: {n_layer})")
        n_head = text_cfg.get("n_head")
        result.check(n_head == 32, f"text n_head == 32 (got: {n_head})")
        result.check(text_cfg.get("dim") == 2560, f"text dim == 2560 (got: {text_cfg.get('dim')})")

        audio_cfg = config.get("audio_decoder_config", {})
        result.check(
            audio_cfg.get("n_layer") == 4,
            f"audio_decoder n_layer == 4 (got: {audio_cfg.get('n_layer')})",
        )
        result.check(
            audio_cfg.get("num_codebooks") == 10,
            f"num_codebooks == 10 (got: {audio_cfg.get('num_codebooks')})",
        )

    # Text model
    print("\n== Text Model Weights ==")
    tm_path = model_dir / "text_model.safetensors"
    if tm_path.exists():
        weights = mx.load(str(tm_path))
        keys = set(weights.keys())

        validate_no_pytorch_prefix(weights, "text_model.model.", result)

        emb_keys = [k for k in keys if "embeddings.weight" in k]
        result.check(len(emb_keys) > 0, f"Embedding keys present ({len(emb_keys)})")

        layer_indices = count_layer_indices(keys)
        result.check(len(layer_indices) == 36, f"36 transformer layers (got {len(layer_indices)})")

        # QK-norm keys
        qnorm_keys = [k for k in keys if "q_norm" in k]
        result.check(len(qnorm_keys) > 0, f"QK-norm keys present ({len(qnorm_keys)})")

        if is_quantized:
            validate_quantization(weights, result, block_key="layers")

        total_params = sum(v.size for v in weights.values())
        print(f"  Total text_model parameters: {total_params / 1e9:.2f}B")
        del weights
        gc.collect()
        mx.clear_cache()

    # Audio decoder
    print("\n== Audio Decoder Weights ==")
    ad_path = model_dir / "audio_decoder.safetensors"
    if ad_path.exists():
        weights = mx.load(str(ad_path))
        keys = set(weights.keys())

        # Keys are prefixed "audio_decoder." by design (component_prefix)
        # Verify no raw PyTorch double-prefix leaked through
        validate_no_pytorch_prefix(weights, "audio_decoder.audio_decoder.", result)

        cb_keys = [k for k in keys if "codebook_embeddings" in k]
        result.check(len(cb_keys) > 0, f"Codebook embeddings present ({len(cb_keys)})")

        output_keys = [k for k in keys if k.endswith("output.weight")]
        result.check(len(output_keys) > 0, f"Output head present ({len(output_keys)})")

        layer_indices = count_layer_indices(keys)
        result.check(len(layer_indices) == 4, f"4 decoder layers (got {len(layer_indices)})")

        if is_quantized:
            validate_quantization(weights, result, block_key="layers")

        total_params = sum(v.size for v in weights.values())
        print(f"  Total audio_decoder parameters: {total_params / 1e9:.2f}B")
        del weights
        gc.collect()
        mx.clear_cache()

    # Codec (DAC vocoder)
    print("\n== Codec (DAC) Weights ==")
    codec_path = model_dir / "codec.safetensors"
    if codec_path.exists():
        weights = mx.load(str(codec_path))
        keys = set(weights.keys())

        validate_no_pytorch_prefix(weights, "generator.", result)

        # Encoder and decoder sub-networks
        enc_keys = [k for k in keys if k.startswith("codec.encoder.")]
        result.check(len(enc_keys) > 0, f"Encoder keys present ({len(enc_keys)})")

        dec_keys = [k for k in keys if k.startswith("codec.decoder.")]
        result.check(len(dec_keys) > 0, f"Decoder keys present ({len(dec_keys)})")

        # Quantizer keys
        quant_keys = [k for k in keys if "quantizer" in k]
        result.check(len(quant_keys) > 0, f"Quantizer keys present ({len(quant_keys)})")

        # Conv weights should be transposed (channels-last for MLX)
        conv_weights = [(k, v) for k, v in weights.items() if "conv" in k and v.ndim == 3]
        if conv_weights:
            # In MLX channels-last layout for Conv1d: (O, K, I)
            # The last dim should be the input channels (smaller or equal to first)
            sample_key, sample_w = conv_weights[0]
            result.check(
                sample_w.ndim == 3,
                f"Conv weight is 3D after transpose ({sample_key}: {sample_w.shape})",
            )

        # Codec should NOT be quantized (all weights stay bf16/float)
        if is_quantized:
            q_keys = [k for k in keys if ".scales" in k or ".biases" in k]
            result.check(
                len(q_keys) == 0,
                f"Codec not quantized ({len(q_keys)} quantization keys found)",
            )

        total_params = sum(v.size for v in weights.values())
        print(f"  Total codec parameters: {total_params / 1e6:.1f}M")
        del weights
        gc.collect()
        mx.clear_cache()

    result.summary()
    if not result.passed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

FISH_S2_SPLIT_MAP = {
    "text_model": "text_model.safetensors",
    "audio_decoder": "audio_decoder.safetensors",
    "codec": "codec.safetensors",
}


def split(args) -> None:
    """Split a unified Fish S2 Pro model into per-component files."""
    from ..split import split_model

    model_dir = Path(args.model_dir)
    split_model(model_dir, FISH_S2_SPLIT_MAP)


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
    """Add Fish S2 Pro convert arguments."""
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to local checkpoint directory (skips download)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ./models/fish-s2-pro-mlx[-q<bits>])",
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Quantize transformer weights after conversion"
    )
    parser.add_argument(
        "--bits", type=int, default=8, choices=[4, 8], help="Quantization bits (default: 8)"
    )
    parser.add_argument(
        "--group-size", type=int, default=64, help="Quantization group size (default: 64)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview conversion plan without downloading or writing anything",
    )


def add_validate_args(parser) -> None:
    """Add Fish S2 Pro validate arguments."""
    parser.add_argument("model_dir", type=str, help="Path to converted model directory")


def add_split_args(parser) -> None:
    """Add Fish S2 Pro split arguments."""
    parser.add_argument("model_dir", type=str, help="Model directory containing model.safetensors")
