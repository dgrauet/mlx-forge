"""Hunyuan3D-2.1 Shape Generation — MLX conversion recipe.

Converts the DiT (3.3B), ShapeVAE (~330M), and DINOv2 (~300M) components
from a single PyTorch .ckpt checkpoint to MLX safetensors format.

Source: tencent/Hunyuan3D-2.1 on HuggingFace

Usage:
    mlx-forge convert hunyuan3d-2.1
    mlx-forge convert hunyuan3d-2.1 --quantize --bits 8
    mlx-forge validate hunyuan3d-2.1 models/hunyuan3d-2.1-mlx
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import mlx.core as mx

from ..convert import download_hf_files, quantize_component
from ..quantize import _materialize
from ..validate import (
    ValidationResult,
    validate_file_exists,
    validate_no_pytorch_prefix,
    validate_quantization,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_REPO_ID = "tencent/Hunyuan3D-2.1"
CKPT_SUBPATH = "hunyuan3d-dit-v2-1/model.fp16.ckpt"

COMPONENTS = ["dit", "vae", "image_encoder"]

COMPONENT_PREFIX = {
    "dit": "dit",
    "vae": "vae",
    "image_encoder": "image_encoder",
}

# Mapping from ckpt top-level keys to component names
CKPT_SECTIONS = {
    "model": "dit",
    "vae": "vae",
    "conditioner": "image_encoder",
}

# Approximate sizes in MB for dry-run estimation (fp16)
_COMPONENT_SIZE_MB = {
    "dit": 6_600,
    "vae": 650,
    "image_encoder": 600,
}

_CHECKPOINT_SIZE_MB = 7_370  # ~7.37 GB download


# ---------------------------------------------------------------------------
# Key classification
# ---------------------------------------------------------------------------


def classify_key(ckpt_section: str, key: str) -> str | None:
    """Classify a weight key based on its checkpoint section.

    Args:
        ckpt_section: Top-level key in the .ckpt dict ("model", "vae", "conditioner").
        key: The weight key within that section.

    Returns:
        Component name or None to skip.
    """
    return CKPT_SECTIONS.get(ckpt_section)


# ---------------------------------------------------------------------------
# Key sanitization
# ---------------------------------------------------------------------------


def sanitize_dit_key(key: str) -> str | None:
    """Sanitize DiT weight key. Keys are already clean (no prefix to strip)."""
    return key


def sanitize_vae_key(key: str) -> str | None:
    """Sanitize ShapeVAE weight key. Keys are already clean."""
    return key


def sanitize_image_encoder_key(key: str) -> str | None:
    """Sanitize DINOv2 weight key. Strips 'main_image_encoder.model.' prefix."""
    prefix = "main_image_encoder.model."
    if key.startswith(prefix):
        return key[len(prefix) :]
    return None


SANITIZERS = {
    "dit": sanitize_dit_key,
    "vae": sanitize_vae_key,
    "image_encoder": sanitize_image_encoder_key,
}


# ---------------------------------------------------------------------------
# Conv transposition (none needed for this model)
# ---------------------------------------------------------------------------


def maybe_transpose(key: str, value: mx.array, component: str) -> mx.array:
    """Transpose conv weights from PyTorch to MLX layout if needed.

    Hunyuan3D-2.1 uses only Linear layers (no convolutions), so no
    transposition is required.
    """
    return value


# ---------------------------------------------------------------------------
# Quantization predicate
# ---------------------------------------------------------------------------


def should_quantize(key: str, weight: mx.array) -> bool:
    """Determine if a weight should be quantized.

    Quantize: attention projections, MLP/FFN weights, expert FFN weights.
    Exclude: embeddings, norms, MoE gate, final projection, 1D tensors.
    """
    if weight.ndim < 2:
        return False
    if key.endswith(".scales") or key.endswith(".biases"):
        return False

    bare = key.replace("dit.", "", 1)

    # Exclude embeddings
    if "embedder" in bare:
        return False
    # Exclude norms
    if "norm" in bare:
        return False
    # Exclude MoE gate (routing weights are sensitive)
    if ".moe.gate." in bare:
        return False
    # Exclude final output projection
    if "final_layer." in bare:
        return False
    # Exclude pooler
    if "pooler." in bare:
        return False

    # Only quantize .weight tensors in blocks
    return bare.endswith(".weight") and "blocks." in bare


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------


def convert(args) -> None:
    """Convert Hunyuan3D-2.1 Shape Generation checkpoint to MLX format."""
    # 1. Determine output directory
    suffix = f"-q{args.bits}" if args.quantize else ""
    output_dir = Path(args.output) if args.output else Path(f"./models/hunyuan3d-2.1-mlx{suffix}")

    if args.dry_run:
        print(f"Would convert {HF_REPO_ID} to {output_dir}")
        print(f"Components: {', '.join(COMPONENTS)}")
        print(f"Quantize: {args.quantize} (bits={args.bits}, group_size={args.group_size})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Get checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        print(f"Downloading checkpoint from {HF_REPO_ID}...")
        download_dir = Path("./downloads/hunyuan3d-2.1")
        download_hf_files(HF_REPO_ID, [CKPT_SUBPATH], download_dir)
        ckpt_path = download_dir / CKPT_SUBPATH

    # 3. Load checkpoint via torch (it's a .ckpt file)
    print(f"Loading checkpoint from {ckpt_path}...")
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

    # 4. Process each section
    total_weights = 0
    for section_key, component_name in CKPT_SECTIONS.items():
        if section_key not in ckpt:
            print(f"Warning: section '{section_key}' not found in checkpoint, skipping")
            continue

        section_weights = ckpt[section_key]
        print(f"Processing {component_name} ({len(section_weights)} keys)...")

        sanitizer = SANITIZERS[component_name]
        prefix = COMPONENT_PREFIX[component_name]

        # Convert torch tensors to MLX arrays, sanitize keys, transpose
        component_weights: dict[str, mx.array] = {}
        for key, tensor in section_weights.items():
            new_key = sanitizer(key)
            if new_key is None:
                continue
            # Convert torch tensor to numpy then to MLX
            weight = mx.array(tensor.float().numpy())
            weight = weight.astype(mx.float16)
            weight = maybe_transpose(new_key, weight, component_name)
            _materialize(weight)
            component_weights[f"{prefix}.{new_key}"] = weight

        # Save component
        output_file = output_dir / f"{component_name}.safetensors"
        mx.save_safetensors(str(output_file), component_weights)
        count = len(component_weights)
        total_weights += count
        print(f"  Saved {count} weights to {output_file}")

        del component_weights, section_weights
        gc.collect()
        mx.clear_cache()

    del ckpt
    gc.collect()

    # 5. Quantize DiT if requested
    if args.quantize:
        quantize_component(
            output_dir,
            "dit",
            bits=args.bits,
            group_size=args.group_size,
            should_quantize=should_quantize,
        )

    # 6. Write config
    config = {
        "model_type": "hunyuan3d-2.1",
        "stage": "shape",
        "components": COMPONENTS,
        "dit": {
            "hidden_size": 2048,
            "depth": 21,
            "num_heads": 16,
            "in_channels": 64,
            "context_dim": 1024,
            "num_latents": 4096,
            "num_moe_layers": 6,
            "num_experts": 8,
            "moe_top_k": 2,
        },
        "vae": {
            "num_latents": 4096,
            "embed_dim": 64,
            "width": 1024,
            "heads": 16,
            "num_decoder_layers": 16,
            "num_freqs": 8,
        },
        "image_encoder": {
            "type": "dinov2-large",
            "hidden_size": 1024,
            "num_heads": 16,
            "num_layers": 24,
            "patch_size": 14,
            "image_size": 518,
        },
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config written to {config_path}")

    # 7. Write split_model.json
    split_model = {
        "model_type": "hunyuan3d-2.1",
        "components": {name: f"{name}.safetensors" for name in COMPONENTS},
    }
    if args.quantize:
        split_model["quantization"] = {
            "bits": args.bits,
            "group_size": args.group_size,
            "quantized_components": ["dit"],
        }
    split_path = output_dir / "split_model.json"
    with open(split_path, "w") as f:
        json.dump(split_model, f, indent=2)

    print(f"\nConversion complete: {total_weights} total weights -> {output_dir}")


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate(args) -> None:
    """Validate a converted Hunyuan3D-2.1 model directory."""
    model_dir = Path(args.model_dir)
    result = ValidationResult()

    # 1. Check expected files exist
    for fname in [
        "config.json",
        "split_model.json",
        "dit.safetensors",
        "vae.safetensors",
        "image_encoder.safetensors",
    ]:
        validate_file_exists(model_dir, fname, result)

    if not result.passed:
        result.summary()
        raise SystemExit(1)

    # 2. Validate config
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    result.check(config.get("model_type") == "hunyuan3d-2.1", "model_type is hunyuan3d-2.1")
    result.check(config.get("dit", {}).get("depth") == 21, "DiT depth == 21")
    result.check(config.get("dit", {}).get("num_moe_layers") == 6, "num_moe_layers == 6")

    # 3. Validate DiT weights
    dit_weights = dict(mx.load(str(model_dir / "dit.safetensors")))
    dit_keys = list(dit_weights.keys())

    # Check block count
    block_indices = set()
    for k in dit_keys:
        if k.startswith("dit.blocks."):
            parts = k.split(".")
            if len(parts) > 2:
                try:
                    block_indices.add(int(parts[2]))
                except ValueError:
                    pass
    result.check(len(block_indices) == 21, f"21 DiT blocks (got {len(block_indices)})")

    # Check MoE layers exist (blocks 15-20)
    moe_keys = [k for k in dit_keys if ".moe." in k]
    result.check(len(moe_keys) > 0, f"MoE keys present ({len(moe_keys)})")

    # Check MoE expert count
    expert_indices = set()
    for k in moe_keys:
        if ".experts." in k:
            parts = k.split(".experts.")
            if len(parts) > 1:
                try:
                    expert_indices.add(int(parts[1].split(".")[0]))
                except ValueError:
                    pass
    if expert_indices:
        result.check(len(expert_indices) == 8, f"8 MoE experts (got {len(expert_indices)})")

    # Check no raw PyTorch prefixes leaked through
    validate_no_pytorch_prefix(dit_weights, "model.", result)

    # Check quantization if applicable
    split_model = json.loads((model_dir / "split_model.json").read_text())
    if "quantization" in split_model:
        validate_quantization(dit_weights, result, block_key="blocks")

    del dit_weights
    gc.collect()

    # 4. Validate VAE weights
    vae_weights = dict(mx.load(str(model_dir / "vae.safetensors")))
    vae_keys = list(vae_weights.keys())
    # Note: fourier_embedder.frequencies is a buffer computed from config, not a learned weight.
    # It won't appear in the checkpoint — this is expected.
    result.check(any("geo_decoder" in k for k in vae_keys), "VAE has geo_decoder")
    result.check(any("transformer" in k for k in vae_keys), "VAE has transformer layers")
    del vae_weights

    # 5. Validate DINOv2 weights
    dino_weights = dict(mx.load(str(model_dir / "image_encoder.safetensors")))
    dino_keys = list(dino_weights.keys())
    # Should NOT have "main_image_encoder.model." prefix
    result.check(
        not any(k.startswith("image_encoder.main_image_encoder.") for k in dino_keys),
        "DINOv2 prefix properly stripped",
    )
    result.check(any("embeddings" in k for k in dino_keys), "DINOv2 has embeddings")
    result.check(any("encoder.layer" in k for k in dino_keys), "DINOv2 has encoder layers")
    del dino_weights

    result.summary()
    if not result.passed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
    """Add Hunyuan3D-2.1 convert arguments."""
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to local model.fp16.ckpt (skips download)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ./models/hunyuan3d-2.1-mlx[-q<bits>])",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize DiT weights after conversion",
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
    """Add Hunyuan3D-2.1 validate arguments."""
    parser.add_argument("model_dir", type=str, help="Path to converted model directory")
