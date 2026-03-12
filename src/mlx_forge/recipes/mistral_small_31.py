"""Mistral Small 3.1 24B conversion recipe.

Converts the mistralai/Mistral-Small-3.1-24B-Instruct-2503 checkpoint to MLX split format.
Includes three components: language_model (dense transformer), vision_tower (Pixtral),
and multimodal_projector.

No conv transposition needed — all layers are Linear/RMSNorm/Embedding.

Usage:
    mlx-forge convert mistral-small-3.1
    mlx-forge convert mistral-small-3.1 --quantize --bits 8
    mlx-forge validate mistral-small-3.1 models/mistral-small-3.1-mlx
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

REPO_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

COMPONENTS = ["language_model", "vision_tower", "multimodal_projector"]

COMPONENT_PREFIX = {
    "language_model": "language_model",
    "vision_tower": "vision_tower",
    "multimodal_projector": "multimodal_projector",
}

# Approximate sizes in MB for dry-run estimation (bf16)
_COMPONENT_SIZE_MB = {
    "language_model": 44000,
    "vision_tower": 800,
    "multimodal_projector": 50,
}

_CHECKPOINT_SIZE_MB = 48000  # ~48 GB (10 shards)

CHECKPOINT_FILES = [
    "model-00001-of-00010.safetensors",
    "model-00002-of-00010.safetensors",
    "model-00003-of-00010.safetensors",
    "model-00004-of-00010.safetensors",
    "model-00005-of-00010.safetensors",
    "model-00006-of-00010.safetensors",
    "model-00007-of-00010.safetensors",
    "model-00008-of-00010.safetensors",
    "model-00009-of-00010.safetensors",
    "model-00010-of-00010.safetensors",
    "model.safetensors.index.json",
]

CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
]

# ---------------------------------------------------------------------------
# Key classification
# ---------------------------------------------------------------------------


def classify_key(key: str) -> str | None:
    """Classify a weight key into a component name.

    Returns one of: language_model, vision_tower, multimodal_projector, or None (skip).
    """
    if key.startswith("language_model."):
        return "language_model"
    if key.startswith("vision_tower."):
        return "vision_tower"
    if key.startswith("multimodal_projector."):
        return "multimodal_projector"
    return None


# ---------------------------------------------------------------------------
# Key sanitization
# ---------------------------------------------------------------------------


def sanitize_language_model_key(key: str) -> str:
    """Strip the language_model.model. prefix; map lm_head specially."""
    if key == "language_model.lm_head.weight":
        return "lm_head.weight"
    return key.replace("language_model.model.", "", 1)


def sanitize_vision_tower_key(key: str) -> str:
    """Strip the vision_tower. prefix."""
    return key.replace("vision_tower.", "", 1)


def sanitize_multimodal_projector_key(key: str) -> str:
    """Strip the multimodal_projector. prefix."""
    return key.replace("multimodal_projector.", "", 1)


SANITIZERS = {
    "language_model": sanitize_language_model_key,
    "vision_tower": sanitize_vision_tower_key,
    "multimodal_projector": sanitize_multimodal_projector_key,
}


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

# Components that should NOT be quantized
_SKIP_QUANTIZE_COMPONENTS = {"vision_tower", "multimodal_projector"}


def mistral_should_quantize(key: str, weight: mx.array) -> bool:
    """Only quantize transformer Linear weights (not embeddings, norms, or lm_head)."""
    return (
        key.endswith(".weight")
        and weight.ndim == 2
        and weight.shape[0] > 1
        and weight.shape[1] > 1
        and weight.size >= 256
        and "embed_tokens" not in key
        and "norm" not in key
        and "lm_head" not in key
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
        should_quantize=mistral_should_quantize,
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
        print(f"Download:   ~{fmt_size(_CHECKPOINT_SIZE_MB)} (10 shards + config)")
        print("            → ./models/mistral-small-3.1-src/")

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
                label += " (small/sensitive, not quantized)"
        print(label)
        total_mb += size_mb

    print("  config.json, split_model.json, tokenizer.json, ...")

    if args.quantize:
        print(f"\nQuantization: int{args.bits}, group_size={args.group_size}")
        print("  Target: Linear weights only (not embeddings, norms, lm_head)")
        print("  Skipped: vision_tower, multimodal_projector (small/sensitive)")

    print(f"\nEstimated output size: ~{fmt_size(total_mb)}")
    if not args.checkpoint:
        print(f"Estimated download:   ~{fmt_size(_CHECKPOINT_SIZE_MB)}")
        print(f"Estimated total disk: ~{fmt_size(total_mb + _CHECKPOINT_SIZE_MB)}")


# ---------------------------------------------------------------------------
# Main convert entry point
# ---------------------------------------------------------------------------


def convert(args) -> None:
    """Convert Mistral Small 3.1 24B checkpoint to MLX split format."""
    if args.output:
        output_dir = Path(args.output)
    else:
        suffix = f"-q{args.bits}" if args.quantize else ""
        output_dir = Path("models") / f"mistral-small-3.1-mlx{suffix}"

    if args.dry_run:
        _dry_run(args, output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get checkpoint files
    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
        print(f"Using local checkpoint: {checkpoint_dir}")
    else:
        checkpoint_dir = Path("models") / "mistral-small-3.1-src"
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

    # Step 6: Optional quantization (skip small/sensitive components)
    if args.quantize:
        for component_name in COMPONENTS:
            if component_name in _SKIP_QUANTIZE_COMPONENTS:
                print(f"\n  Skipping quantization for {component_name} (small/sensitive)")
                continue
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

    print("Done!")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(args) -> None:
    """Validate a converted Mistral Small 3.1 24B model."""
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
        "language_model.safetensors",
        "vision_tower.safetensors",
        "multimodal_projector.safetensors",
    ]
    for fname in expected:
        validate_file_exists(model_dir, fname, result)
    for fname in ["quantize_config.json", "tokenizer.json", "preprocessor_config.json"]:
        if (model_dir / fname).exists():
            print(f"  \033[92m\u2713\033[0m {fname} exists (optional)")

    # Config
    print("\n== Config Validation ==")
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        result.check(
            config.get("model_type") == "mistral3",
            f"model_type is mistral3 (got: {config.get('model_type')})",
        )
        text_cfg = config.get("text_config", {})
        n_layers = text_cfg.get("num_hidden_layers")
        result.check(n_layers == 40, f"text num_hidden_layers == 40 (got: {n_layers})")
        n_heads = text_cfg.get("num_attention_heads")
        result.check(n_heads == 32, f"text num_attention_heads == 32 (got: {n_heads})")
        hidden = text_cfg.get("hidden_size")
        result.check(hidden == 5120, f"text hidden_size == 5120 (got: {hidden})")
        n_kv_heads = text_cfg.get("num_key_value_heads")
        result.check(n_kv_heads == 8, f"text num_key_value_heads == 8 (got: {n_kv_heads})")

        vision_cfg = config.get("vision_config", {})
        result.check(
            vision_cfg.get("num_hidden_layers") == 24,
            f"vision num_hidden_layers == 24 (got: {vision_cfg.get('num_hidden_layers')})",
        )
        result.check(
            vision_cfg.get("num_attention_heads") == 16,
            f"vision num_attention_heads == 16 (got: {vision_cfg.get('num_attention_heads')})",
        )

    # Language model
    print("\n== Language Model Weights ==")
    lm_path = model_dir / "language_model.safetensors"
    if lm_path.exists():
        weights = mx.load(str(lm_path))
        keys = set(weights.keys())

        validate_no_pytorch_prefix(weights, "language_model.model.", result)

        emb_keys = [k for k in keys if "embed_tokens" in k]
        result.check(len(emb_keys) > 0, f"Embedding keys present ({len(emb_keys)})")

        lm_head_keys = [k for k in keys if k == "lm_head.weight"]
        result.check(len(lm_head_keys) > 0, "lm_head.weight present")

        layer_indices = set()
        for k in keys:
            if "layers." in k:
                parts = k.split("layers.")
                if len(parts) > 1:
                    idx = parts[1].split(".")[0]
                    if idx.isdigit():
                        layer_indices.add(int(idx))
        result.check(len(layer_indices) == 40, f"40 transformer layers (got {len(layer_indices)})")

        if is_quantized:
            validate_quantization(weights, result, block_key="layers")

        total_params = sum(v.size for v in weights.values())
        print(f"  Total language_model parameters: {total_params / 1e9:.2f}B")
        del weights

    # Vision tower
    print("\n== Vision Tower Weights ==")
    vt_path = model_dir / "vision_tower.safetensors"
    if vt_path.exists():
        weights = mx.load(str(vt_path))
        keys = set(weights.keys())

        validate_no_pytorch_prefix(weights, "vision_tower.", result)

        layer_indices = set()
        for k in keys:
            if "layers." in k:
                parts = k.split("layers.")
                if len(parts) > 1:
                    idx = parts[1].split(".")[0]
                    if idx.isdigit():
                        layer_indices.add(int(idx))
        result.check(
            len(layer_indices) == 24, f"24 vision transformer layers (got {len(layer_indices)})"
        )

        # Vision tower should NOT be quantized
        if is_quantized:
            q_keys = [k for k in keys if ".scales" in k or ".biases" in k]
            result.check(
                len(q_keys) == 0,
                f"Vision tower not quantized ({len(q_keys)} quantization keys found)",
            )

        total_params = sum(v.size for v in weights.values())
        print(f"  Total vision_tower parameters: {total_params / 1e6:.1f}M")
        del weights

    # Multimodal projector
    print("\n== Multimodal Projector Weights ==")
    mp_path = model_dir / "multimodal_projector.safetensors"
    if mp_path.exists():
        weights = mx.load(str(mp_path))
        keys = set(weights.keys())

        validate_no_pytorch_prefix(weights, "multimodal_projector.", result)

        # Multimodal projector should NOT be quantized
        if is_quantized:
            q_keys = [k for k in keys if ".scales" in k or ".biases" in k]
            result.check(
                len(q_keys) == 0,
                f"Multimodal projector not quantized ({len(q_keys)} quantization keys found)",
            )

        total_params = sum(v.size for v in weights.values())
        print(f"  Total multimodal_projector parameters: {total_params / 1e6:.1f}M")
        del weights

    result.summary()
    if not result.passed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

MISTRAL_SMALL_SPLIT_MAP = {
    "language_model": "language_model.safetensors",
    "vision_tower": "vision_tower.safetensors",
    "multimodal_projector": "multimodal_projector.safetensors",
}


def split(args) -> None:
    """Split a unified Mistral Small 3.1 model into per-component files."""
    from ..split import split_model

    model_dir = Path(args.model_dir)
    split_model(model_dir, MISTRAL_SMALL_SPLIT_MAP)


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
    """Add Mistral Small 3.1 convert arguments."""
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
        help="Output directory (default: ./models/mistral-small-3.1-mlx[-q<bits>])",
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
    """Add Mistral Small 3.1 validate arguments."""
    parser.add_argument("model_dir", type=str, help="Path to converted model directory")


def add_split_args(parser) -> None:
    """Add Mistral Small 3.1 split arguments."""
    parser.add_argument("model_dir", type=str, help="Model directory containing model.safetensors")
