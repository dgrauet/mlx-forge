"""Fish Audio S2 Pro conversion recipe.

Converts the fishaudio/s2-pro TTS checkpoint (Dual-AR Qwen3-based) to MLX split format.
Phase 1: transformer components only (text_model + audio_decoder). Codec not included.

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

from ..convert import classify_keys, download_hf_files, fmt_size, load_weights, process_component
from ..quantize import quantize_weights
from ..validate import (
    ValidationResult,
    validate_file_exists,
    validate_no_pytorch_prefix,
    validate_quantization,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ID = "fishaudio/s2-pro"

COMPONENTS = ["text_model", "audio_decoder"]

COMPONENT_PREFIX = {
    "text_model": "text_model",
    "audio_decoder": "audio_decoder",
}

# Approximate sizes in MB for dry-run estimation (bf16)
_COMPONENT_SIZE_MB = {
    "text_model": 8500,
    "audio_decoder": 600,
}

_CHECKPOINT_SIZE_MB = 9200  # ~9.2 GB download (2 shards)

CHECKPOINT_FILES = [
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
    "model.safetensors.index.json",
]

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

    Returns one of: text_model, audio_decoder, or None (skip).
    """
    if key.startswith("text_model."):
        return "text_model"
    if key.startswith("audio_decoder."):
        return "audio_decoder"
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


SANITIZERS = {
    "text_model": sanitize_text_model_key,
    "audio_decoder": sanitize_audio_decoder_key,
}


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def fish_s2_should_quantize(key: str, weight: mx.array) -> bool:
    """Only quantize transformer Linear weights (not embeddings or norms)."""
    return (
        key.endswith(".weight")
        and weight.ndim == 2
        and weight.shape[0] > 1
        and weight.shape[1] > 1
        and weight.size >= 256
        and "embeddings" not in key
        and "norm" not in key
    )


def quantize_component(
    output_dir: Path,
    component_name: str,
    *,
    bits: int = 8,
    group_size: int = 64,
) -> None:
    """Quantize a component's weights in-place."""
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
        should_quantize=fish_s2_should_quantize,
    )

    print(f"  Saving quantized {component_name} ({len(result)} keys)...")
    mx.save_safetensors(str(filepath), result)

    del result, weights
    gc.collect()
    mx.clear_cache()


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
        print(f"Download:   ~{fmt_size(_CHECKPOINT_SIZE_MB)} (2 shards + config)")
        print("            → ./models/")

    print(f"Output dir: {output_dir}")

    print("\nOutput files:")
    total_mb = 0.0
    for comp in COMPONENTS:
        size_mb = _COMPONENT_SIZE_MB[comp]
        if args.quantize:
            ratio = 16 / args.bits
            size_mb = size_mb / ratio
            label = f"  {comp}.safetensors: ~{fmt_size(size_mb)} (int{args.bits})"
        else:
            label = f"  {comp}.safetensors: ~{fmt_size(size_mb)} (bf16)"
        print(label)
        total_mb += size_mb

    print("  config.json, split_model.json, tokenizer.json, ...")

    if args.quantize:
        print(f"\nQuantization: int{args.bits}, group_size={args.group_size}")
        print("  Target: Linear weights only (not embeddings, norms)")

    print(f"\nEstimated output size: ~{fmt_size(total_mb)}")
    if not args.checkpoint:
        print(f"Estimated download:   ~{fmt_size(_CHECKPOINT_SIZE_MB)}")
        print(f"Estimated total disk: ~{fmt_size(total_mb + _CHECKPOINT_SIZE_MB)}")

    print("\nNote: codec (DAC vocoder) not included — Phase 1 converts transformers only.")


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
        checkpoint_dir = Path("models")
        print(f"Downloading {REPO_ID} checkpoint files...")
        download_hf_files(REPO_ID, CHECKPOINT_FILES, checkpoint_dir)
        print("Downloading config and tokenizer files...")
        download_hf_files(REPO_ID, CONFIG_FILES, checkpoint_dir)

    # Step 2: Copy config and tokenizer files to output dir
    for fname in CONFIG_FILES:
        src = checkpoint_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    # Step 3: Load weights lazily (sharded via index)
    t0 = time.monotonic()
    checkpoint_weights = load_weights(checkpoint_dir)
    print(f"  {len(checkpoint_weights)} keys loaded (lazy)")

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
        print(f"\n[{component_name}] Processing {len(keys)} keys...")
        t0 = time.monotonic()
        count = process_component(
            checkpoint_weights,
            component_name,
            keys,
            output_dir,
            component_prefix,
            sanitizer=SANITIZERS[component_name],
        )
        elapsed = time.monotonic() - t0
        total_weights += count
        print(f"  Done: {count} weights saved in {elapsed:.1f}s")

    del checkpoint_weights
    gc.collect()
    mx.clear_cache()

    # Step 5: Create split_model.json
    split_info: dict = {
        "format": "split",
        "source": REPO_ID,
        "components": COMPONENTS,
    }
    with open(output_dir / "split_model.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Conversion complete: {total_weights} total weights")
    print(f"Output: {output_dir}")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.name}: {size_mb:.1f} MB")

    # Step 6: Optional quantization
    if args.quantize:
        for component_name in COMPONENTS:
            quantize_component(
                output_dir, component_name, bits=args.bits, group_size=args.group_size
            )

        split_info["quantized"] = True
        split_info["quantization_bits"] = args.bits
        with open(output_dir / "split_model.json", "w") as f:
            json.dump(split_info, f, indent=2)

        qconfig = {
            "quantization": {
                "bits": args.bits,
                "group_size": args.group_size,
                "target": "linear_weights_only",
            }
        }
        with open(output_dir / "quantize_config.json", "w") as f:
            json.dump(qconfig, f, indent=2)

        print("\nFinal files after quantization:")
        for p in sorted(output_dir.iterdir()):
            if p.is_file():
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"  {p.name}: {size_mb:.1f} MB")

    print("\nNote: codec (DAC vocoder) not included — Phase 1.")
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

        layer_indices = set()
        for k in keys:
            if "layers." in k:
                parts = k.split("layers.")
                if len(parts) > 1:
                    idx = parts[1].split(".")[0]
                    if idx.isdigit():
                        layer_indices.add(int(idx))
        result.check(len(layer_indices) == 36, f"36 transformer layers (got {len(layer_indices)})")

        # QK-norm keys
        qnorm_keys = [k for k in keys if "q_norm" in k]
        result.check(len(qnorm_keys) > 0, f"QK-norm keys present ({len(qnorm_keys)})")

        if is_quantized:
            validate_quantization(weights, result, block_key="layers")

        total_params = sum(v.size for v in weights.values())
        print(f"  Total text_model parameters: {total_params / 1e9:.2f}B")
        del weights

    # Audio decoder
    print("\n== Audio Decoder Weights ==")
    ad_path = model_dir / "audio_decoder.safetensors"
    if ad_path.exists():
        weights = mx.load(str(ad_path))
        keys = set(weights.keys())

        validate_no_pytorch_prefix(weights, "audio_decoder.", result)

        cb_keys = [k for k in keys if "codebook_embeddings" in k]
        result.check(len(cb_keys) > 0, f"Codebook embeddings present ({len(cb_keys)})")

        output_keys = [k for k in keys if k.endswith("output.weight")]
        result.check(len(output_keys) > 0, f"Output head present ({len(output_keys)})")

        layer_indices = set()
        for k in keys:
            if "layers." in k:
                parts = k.split("layers.")
                if len(parts) > 1:
                    idx = parts[1].split(".")[0]
                    if idx.isdigit():
                        layer_indices.add(int(idx))
        result.check(len(layer_indices) == 4, f"4 decoder layers (got {len(layer_indices)})")

        if is_quantized:
            validate_quantization(weights, result, block_key="layers")

        total_params = sum(v.size for v in weights.values())
        print(f"  Total audio_decoder parameters: {total_params / 1e9:.2f}B")
        del weights

    result.summary()
    if not result.passed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

FISH_S2_SPLIT_MAP = {
    "text_model": "text_model.safetensors",
    "audio_decoder": "audio_decoder.safetensors",
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
