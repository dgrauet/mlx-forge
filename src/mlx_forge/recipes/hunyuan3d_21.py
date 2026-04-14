"""Hunyuan3D-2.1 — MLX conversion recipe.

Converts both stages from tencent/Hunyuan3D-2.1:
  --stage shape : DiT (3.3B) + ShapeVAE (~330M) + DINOv2-large (~300M)
  --stage paint : PBR UNet2.5D (~1.7B) + SD-VAE (~160M) + DINOv2-giant (~1.3B)

Source: tencent/Hunyuan3D-2.1 on HuggingFace

Usage:
    mlx-forge convert hunyuan3d-2.1 --stage shape
    mlx-forge convert hunyuan3d-2.1 --stage paint
    mlx-forge convert hunyuan3d-2.1 --stage paint --quantize --bits 8
    mlx-forge validate hunyuan3d-2.1 models/hunyuan3d-2.1-mlx
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import mlx.core as mx

from ..convert import download_hf_files, quantize_component
from ..quantize import _materialize
from ..transpose import needs_transpose, transpose_conv
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
# DINOv2-giant is bundled in the paint model as image_encoder/
# (can also be loaded from facebook/dinov2-giant separately)
HF_REPO_DINO_GIANT = "facebook/dinov2-giant"

# --- Shape stage ---
SHAPE_CKPT_SUBPATH = "hunyuan3d-dit-v2-1/model.fp16.ckpt"
SHAPE_COMPONENTS = ["dit", "vae", "image_encoder"]
SHAPE_CKPT_SECTIONS = {"model": "dit", "vae": "vae", "conditioner": "image_encoder"}

# --- Paint stage ---
PAINT_SUBDIR = "hunyuan3d-paintpbr-v2-1"
PAINT_COMPONENTS = ["paint_unet", "paint_vae", "paint_dino"]
PAINT_FILES = [
    f"{PAINT_SUBDIR}/unet/diffusion_pytorch_model.bin",
    f"{PAINT_SUBDIR}/unet/config.json",
    f"{PAINT_SUBDIR}/vae/diffusion_pytorch_model.bin",
    f"{PAINT_SUBDIR}/vae/config.json",
    f"{PAINT_SUBDIR}/image_encoder/model.safetensors",
    f"{PAINT_SUBDIR}/image_encoder/config.json",
]


# ---------------------------------------------------------------------------
# Shape stage — key sanitization (unchanged from original)
# ---------------------------------------------------------------------------


def sanitize_dit_key(key: str) -> str | None:
    return key


def sanitize_shape_vae_key(key: str) -> str | None:
    return key


def sanitize_shape_image_encoder_key(key: str) -> str | None:
    prefix = "main_image_encoder.model."
    if key.startswith(prefix):
        return key[len(prefix) :]
    return None


SHAPE_SANITIZERS = {
    "dit": sanitize_dit_key,
    "vae": sanitize_shape_vae_key,
    "image_encoder": sanitize_shape_image_encoder_key,
}


# ---------------------------------------------------------------------------
# Shape stage — no conv transposition needed
# ---------------------------------------------------------------------------


def shape_maybe_transpose(key: str, value: mx.array, component: str) -> mx.array:
    return value


# ---------------------------------------------------------------------------
# Shape stage — quantization predicate
# ---------------------------------------------------------------------------


def shape_should_quantize(key: str, weight: mx.array) -> bool:
    if weight.ndim < 2:
        return False
    if key.endswith(".scales") or key.endswith(".biases"):
        return False
    bare = key.replace("dit.", "", 1)
    if "embedder" in bare:
        return False
    if "norm" in bare:
        return False
    if ".moe.gate." in bare:
        return False
    if "final_layer." in bare:
        return False
    if "pooler." in bare:
        return False
    return bare.endswith(".weight") and "blocks." in bare


# ---------------------------------------------------------------------------
# Paint stage — key sanitization
# ---------------------------------------------------------------------------


def sanitize_paint_unet_key(key: str) -> str | None:
    """Sanitize HunyuanPaintPBR UNet key.

    The checkpoint wraps BasicTransformerBlock inside Basic2p5DTransformerBlock
    via a `.transformer` attribute. We flatten that indirection so keys map
    directly to the MLX model's parameter tree.

    Handles:
    - `unet_dual.` prefix for dual-stream reference UNet
    - `.transformer.attn1/attn2/ff/norm*` flattening
    - `.to_out.0.` → `.to_out.` (ModuleList)
    - `.to_out.1.` → skip (dropout)
    - `.ff.net.0.proj.` → `.ff.proj_in.` (GEGLU)
    - `.ff.net.2.` → `.ff.proj_out.`
    - `.processor.to_out_mr.0.` → `.processor.to_out_mr.`
    """
    if ".to_out.1." in key:
        return None  # skip dropout

    key = key.replace(".to_out.0.", ".to_out.")
    # Also handle material-specific to_out in processors
    key = key.replace(".to_out_mr.0.", ".to_out_mr.")
    key = key.replace(".to_out_albedo.0.", ".to_out_albedo.")
    key = key.replace(".ff.net.0.proj.", ".ff.proj_in.")
    key = key.replace(".ff.net.2.", ".ff.proj_out.")

    # Flatten .transformer. indirection in 2.5D blocks
    # e.g. transformer_blocks.0.transformer.attn1 → transformer_blocks.0.attn1
    key = key.replace(".transformer.attn1.", ".attn1.")
    key = key.replace(".transformer.attn2.", ".attn2.")
    key = key.replace(".transformer.ff.", ".ff.")
    key = key.replace(".transformer.norm1.", ".norm1.")
    key = key.replace(".transformer.norm2.", ".norm2.")
    key = key.replace(".transformer.norm3.", ".norm3.")

    return key


def sanitize_paint_vae_key(key: str) -> str | None:
    """Sanitize diffusers VAE key (old-style attention names).

    Maps old diffusers naming to our convention while keeping
    the attentions.0 prefix that the MLX model uses.
    """
    replacements = {
        ".mid_block.attentions.0.key.": ".mid_block.attentions.0.to_k.",
        ".mid_block.attentions.0.query.": ".mid_block.attentions.0.to_q.",
        ".mid_block.attentions.0.value.": ".mid_block.attentions.0.to_v.",
        ".mid_block.attentions.0.proj_attn.": ".mid_block.attentions.0.to_out.",
    }
    for old, new in replacements.items():
        if old in key:
            key = key.replace(old, new)
    return key


def sanitize_paint_dino_key(key: str) -> str | None:
    """Sanitize HuggingFace DINOv2-giant key.

    Source format (facebook/dinov2-giant):
        embeddings.cls_token, embeddings.patch_embeddings.projection.*
        encoder.layer.N.attention.attention.{query,key,value}.*
        encoder.layer.N.attention.output.dense.*
        encoder.layer.N.mlp.weights_in.*, encoder.layer.N.mlp.weights_out.*
        encoder.layer.N.norm1.*, encoder.layer.N.norm2.*
        encoder.layer.N.layer_scale1.lambda1, encoder.layer.N.layer_scale2.lambda1
        layernorm.*
    """
    # Skip mask token (not used in inference)
    if "mask_token" in key:
        return None

    key = key.replace("embeddings.patch_embeddings.projection.", "patch_embed.proj.")
    key = key.replace("embeddings.cls_token", "cls_token")
    key = key.replace("embeddings.position_embeddings", "pos_embed")
    key = key.replace("encoder.layer.", "blocks.")
    key = key.replace(".attention.attention.", ".attn.")
    key = key.replace(".attention.output.dense.", ".attn.proj.")
    key = key.replace(".mlp.weights_in.", ".mlp.fc1.")
    key = key.replace(".mlp.weights_out.", ".mlp.fc2.")
    # layernorm at model level
    if key.startswith("layernorm."):
        key = key.replace("layernorm.", "norm.")
    return key


def sanitize_paint_clip_key(key: str) -> str | None:
    """Sanitize CLIP ViT-H image encoder key (bundled in paint model)."""
    # Strip vision_model prefix
    if key.startswith("vision_model."):
        key = key[len("vision_model.") :]
    elif key.startswith("visual_projection."):
        return key  # keep as-is
    else:
        return key
    return key


# ---------------------------------------------------------------------------
# Paint stage — conv transposition
# ---------------------------------------------------------------------------


def paint_maybe_transpose(key: str, value: mx.array, component: str) -> mx.array:
    if needs_transpose(key, value):
        return transpose_conv(value)
    if "patch_embed" in key and "weight" in key and value.ndim == 4:
        return transpose_conv(value)
    return value


# ---------------------------------------------------------------------------
# Paint stage — quantization predicate
# ---------------------------------------------------------------------------


def paint_should_quantize(key: str, weight: mx.array) -> bool:
    if weight.ndim < 2:
        return False
    if key.endswith(".scales") or key.endswith(".biases"):
        return False
    if "conv" in key.lower() and weight.ndim >= 3:
        return False
    if "norm" in key:
        return False
    if "learned_text_clip" in key:
        return False
    if "time_embedding" in key:
        return False
    return key.endswith(".weight") and weight.ndim == 2


# ---------------------------------------------------------------------------
# Paint stage — DINOv2 QKV fusion
# ---------------------------------------------------------------------------


def fuse_dino_qkv(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Fuse separate Q/K/V into single QKV linear."""
    fused = {}
    pending: dict[str, dict[str, mx.array]] = {}

    for key, val in weights.items():
        for proj in ("query", "key", "value"):
            marker = f".attn.{proj}."
            if marker in key:
                group = key.split(marker)[0]
                suffix = key.split(marker)[-1]
                if group not in pending:
                    pending[group] = {}
                pending[group][(proj, suffix)] = val
                break
        else:
            fused[key] = val

    for group, parts in pending.items():
        for suffix in ("weight", "bias"):
            q = parts.get(("query", suffix))
            k = parts.get(("key", suffix))
            v = parts.get(("value", suffix))
            if q is not None and k is not None and v is not None:
                fused[f"{group}.attn.qkv.{suffix}"] = mx.concatenate([q, k, v], axis=0)

    return fused


# ---------------------------------------------------------------------------
# Convert — unified entry point
# ---------------------------------------------------------------------------


def convert(args) -> None:
    """Convert Hunyuan3D-2.1 checkpoint to MLX format."""
    stage = args.stage

    suffix = f"-q{args.bits}" if args.quantize else ""
    output_dir = Path(args.output) if args.output else Path(f"./models/hunyuan3d-2.1-mlx{suffix}")

    if args.dry_run:
        components = SHAPE_COMPONENTS if stage == "shape" else PAINT_COMPONENTS
        print(f"Would convert {HF_REPO_ID} (stage={stage}) to {output_dir}")
        print(f"Components: {', '.join(components)}")
        print(f"Quantize: {args.quantize} (bits={args.bits}, group_size={args.group_size})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if stage == "shape":
        _convert_shape(args, output_dir)
    elif stage == "paint":
        _convert_paint(args, output_dir)
    else:
        raise ValueError(f"Unknown stage: {stage}. Use 'shape' or 'paint'.")


def _convert_shape(args, output_dir: Path) -> None:
    """Convert shape generation stage (DiT + ShapeVAE + DINOv2-large)."""
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        print(f"Downloading shape checkpoint from {HF_REPO_ID}...")
        dl_dir = Path("./downloads/hunyuan3d-2.1")
        download_hf_files(HF_REPO_ID, [SHAPE_CKPT_SUBPATH], dl_dir)
        ckpt_path = dl_dir / SHAPE_CKPT_SUBPATH

    print(f"Loading checkpoint from {ckpt_path}...")
    import torch

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)

    total_weights = 0
    for section_key, component_name in SHAPE_CKPT_SECTIONS.items():
        if section_key not in ckpt:
            print(f"Warning: section '{section_key}' not found, skipping")
            continue

        section_weights = ckpt[section_key]
        print(f"Processing {component_name} ({len(section_weights)} keys)...")

        sanitizer = SHAPE_SANITIZERS[component_name]
        component_weights: dict[str, mx.array] = {}
        for key, tensor in section_weights.items():
            new_key = sanitizer(key)
            if new_key is None:
                continue
            weight = mx.array(tensor.float().numpy())
            weight = weight.astype(mx.float16)
            weight = shape_maybe_transpose(new_key, weight, component_name)
            _materialize(weight)
            component_weights[f"{component_name}.{new_key}"] = weight

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

    if args.quantize:
        quantize_component(
            output_dir,
            "dit",
            bits=args.bits,
            group_size=args.group_size,
            should_quantize=shape_should_quantize,
        )

    _write_shape_config(output_dir, args)
    print(f"\nShape conversion complete: {total_weights} weights -> {output_dir}")


def _load_torch_bin(path: Path) -> dict[str, mx.array]:
    """Load a PyTorch .bin checkpoint as MLX arrays."""
    import torch

    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return {k: mx.array(v.float().numpy()) for k, v in state.items()}


def _convert_component(
    raw: dict[str, mx.array],
    sanitizer,
    transposer,
    component: str,
    out_file: Path,
) -> int:
    """Convert, sanitize, transpose, and save a component."""
    converted = {}
    for key, val in raw.items():
        new_key = sanitizer(key)
        if new_key is None:
            continue
        val = transposer(new_key, val, component)
        val = val.astype(mx.float16)
        _materialize(val)
        converted[new_key] = val

    mx.save_safetensors(str(out_file), converted)
    count = len(converted)
    print(f"  Saved {count} weights to {out_file}")
    del converted
    gc.collect()
    mx.clear_cache()
    return count


def _convert_paint(args, output_dir: Path) -> None:
    """Convert paint stage (UNet2.5D + SD-VAE + DINOv2-giant)."""
    total_weights = 0

    # Download all files if needed
    if not args.local_path:
        dl_dir = Path("./downloads/hunyuan3d-2.1")
        print(f"Downloading paint model from {HF_REPO_ID}...")
        download_hf_files(HF_REPO_ID, PAINT_FILES, dl_dir)
        base_dir = dl_dir / PAINT_SUBDIR
    else:
        base_dir = Path(args.local_path)

    # --- UNet (.bin format, torch) ---
    print("Converting paint UNet...")
    unet_path = base_dir / "unet" / "diffusion_pytorch_model.bin"
    raw = _load_torch_bin(unet_path)
    total_weights += _convert_component(
        raw,
        sanitize_paint_unet_key,
        paint_maybe_transpose,
        "unet",
        output_dir / "paint_unet.safetensors",
    )
    del raw
    gc.collect()

    # --- VAE (.bin format, torch) ---
    print("Converting paint VAE...")
    vae_path = base_dir / "vae" / "diffusion_pytorch_model.bin"
    raw = _load_torch_bin(vae_path)
    total_weights += _convert_component(
        raw,
        sanitize_paint_vae_key,
        paint_maybe_transpose,
        "vae",
        output_dir / "paint_vae.safetensors",
    )
    del raw
    gc.collect()

    # --- CLIP image encoder (bundled in paint model, may not be needed) ---
    clip_path = base_dir / "image_encoder" / "model.safetensors"
    if clip_path.exists():
        print("Converting CLIP image encoder (bundled)...")
        raw = dict(mx.load(str(clip_path)))
        total_weights += _convert_component(
            raw,
            sanitize_paint_clip_key,
            paint_maybe_transpose,
            "clip",
            output_dir / "paint_clip.safetensors",
        )
        del raw
        gc.collect()

    # --- DINOv2-giant (from facebook/dinov2-giant) ---
    print("Converting DINOv2-giant...")
    if args.dino_path:
        dino_path = Path(args.dino_path) / "model.safetensors"
    else:
        dl_dir = Path("./downloads/dinov2-giant")
        download_hf_files(HF_REPO_DINO_GIANT, ["model.safetensors"], dl_dir)
        dino_path = dl_dir / "model.safetensors"

    raw = dict(mx.load(str(dino_path)))
    sanitized = {}
    for key, val in raw.items():
        new_key = sanitize_paint_dino_key(key)
        if new_key is None:
            continue
        val = paint_maybe_transpose(new_key, val, "dino")
        val = val.astype(mx.float16)
        _materialize(val)
        sanitized[new_key] = val
    del raw

    sanitized = fuse_dino_qkv(sanitized)

    out_file = output_dir / "paint_dino.safetensors"
    mx.save_safetensors(str(out_file), sanitized)
    total_weights += len(sanitized)
    print(f"  Saved {len(sanitized)} weights to {out_file}")
    del sanitized
    gc.collect()
    mx.clear_cache()

    # --- Quantize UNet ---
    if args.quantize:
        quantize_component(
            output_dir,
            "paint_unet",
            bits=args.bits,
            group_size=args.group_size,
            should_quantize=paint_should_quantize,
        )

    _write_paint_config(output_dir, args)
    print(f"\nPaint conversion complete: {total_weights} weights -> {output_dir}")


# ---------------------------------------------------------------------------
# Config writers
# ---------------------------------------------------------------------------


def _write_shape_config(output_dir: Path, args) -> None:
    config = {
        "model_type": "hunyuan3d-2.1",
        "stage": "shape",
        "components": SHAPE_COMPONENTS,
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
    _write_config_files(output_dir, config, SHAPE_COMPONENTS, args, "dit")


def _write_paint_config(output_dir: Path, args) -> None:
    config = {
        "model_type": "hunyuan3d-2.1",
        "stage": "paint",
        "components": PAINT_COMPONENTS,
        "paint_unet": {
            "base": "stable-diffusion-2.1",
            "in_channels": 12,
            "out_channels": 4,
            "block_out_channels": [320, 640, 1280, 1280],
            "cross_attention_dim": 1024,
            "attention_head_dim": 8,
            "pbr_settings": ["albedo", "mr"],
            "use_dual_stream": True,
        },
        "paint_vae": {
            "type": "kl",
            "scaling_factor": 0.18215,
            "latent_channels": 4,
            "block_out_channels": [128, 256, 512, 512],
        },
        "paint_dino": {
            "type": "dinov2-giant",
            "embed_dim": 1536,
            "num_heads": 24,
            "depth": 40,
            "patch_size": 14,
            "image_size": 518,
        },
    }
    _write_config_files(output_dir, config, PAINT_COMPONENTS, args, "paint_unet")


def _write_config_files(output_dir, config, components, args, quantize_target):
    config_path = output_dir / "config.json"
    # Merge with existing config if present
    if config_path.exists():
        with open(config_path) as f:
            existing = json.load(f)
        existing.update(config)
        existing["components"] = list(set(existing.get("components", []) + components))
        config = existing

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config written to {config_path}")

    split_model = {
        "model_type": "hunyuan3d-2.1",
        "source": "tencent/Hunyuan3D-2.1",
        "components": {name: f"{name}.safetensors" for name in config["components"]},
    }
    if args.quantize:
        split_model["quantization"] = {
            "bits": args.bits,
            "group_size": args.group_size,
            "quantized_components": [quantize_target],
        }
    split_path = output_dir / "split_model.json"
    with open(split_path, "w") as f:
        json.dump(split_model, f, indent=2)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate(args) -> None:
    """Validate a converted Hunyuan3D-2.1 model directory."""
    model_dir = Path(args.model_dir)
    result = ValidationResult()

    validate_file_exists(model_dir, "config.json", result)
    if not result.passed:
        result.summary()
        raise SystemExit(1)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    result.check(config.get("model_type") == "hunyuan3d-2.1", "model_type is hunyuan3d-2.1")

    # Validate shape components if present
    if "dit" in config.get("components", []):
        _validate_shape(model_dir, config, result)

    # Validate paint components if present
    if "paint_unet" in config.get("components", []):
        _validate_paint(model_dir, config, result)

    result.summary()
    if not result.passed:
        raise SystemExit(1)


def _validate_shape(model_dir: Path, config: dict, result: ValidationResult) -> None:
    for fname in ["dit.safetensors", "vae.safetensors", "image_encoder.safetensors"]:
        validate_file_exists(model_dir, fname, result)

    result.check(config.get("dit", {}).get("depth") == 21, "DiT depth == 21")

    dit_weights = dict(mx.load(str(model_dir / "dit.safetensors")))
    block_indices = {
        int(k.split(".")[2])
        for k in dit_weights
        if k.startswith("dit.blocks.") and k.split(".")[2].isdigit()
    }
    result.check(len(block_indices) == 21, f"21 DiT blocks (got {len(block_indices)})")
    moe_keys = [k for k in dit_weights if ".moe." in k]
    result.check(len(moe_keys) > 0, f"MoE keys present ({len(moe_keys)})")
    validate_no_pytorch_prefix(dit_weights, "model.", result)

    split_model = json.loads((model_dir / "split_model.json").read_text())
    if "quantization" in split_model:
        validate_quantization(dit_weights, result, block_key="blocks")
    del dit_weights
    gc.collect()


def _validate_paint(model_dir: Path, config: dict, result: ValidationResult) -> None:
    for fname in ["paint_unet.safetensors", "paint_vae.safetensors", "paint_dino.safetensors"]:
        validate_file_exists(model_dir, fname, result)

    result.check(config.get("paint_unet", {}).get("in_channels") == 12, "UNet in_channels == 12")

    unet_w = dict(mx.load(str(model_dir / "paint_unet.safetensors")))
    result.check(any("down_blocks" in k for k in unet_w), "UNet has down_blocks")
    result.check(any("up_blocks" in k for k in unet_w), "UNet has up_blocks")
    conv_in = unet_w.get("conv_in.weight")
    if conv_in is not None:
        result.check(conv_in.shape[-1] == 12, f"conv_in 12ch (got {conv_in.shape})")
    del unet_w
    gc.collect()

    vae_w = dict(mx.load(str(model_dir / "paint_vae.safetensors")))
    result.check(any("encoder" in k for k in vae_w), "VAE has encoder")
    result.check(any("decoder" in k for k in vae_w), "VAE has decoder")
    del vae_w

    dino_w = dict(mx.load(str(model_dir / "paint_dino.safetensors")))
    dino_keys = list(dino_w.keys())
    result.check(any("blocks." in k for k in dino_keys), "DINOv2 has blocks")
    result.check(any("qkv" in k or "query" in k for k in dino_keys), "DINOv2 has attention")
    # Check it's actually DINOv2 (1536-dim) not CLIP (1280-dim)
    pos_key = [k for k in dino_keys if "pos_embed" in k]
    if pos_key:
        result.check(dino_w[pos_key[0]].shape[-1] == 1536, "DINOv2 embed_dim == 1536")
    del dino_w


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
    parser.add_argument(
        "--stage",
        type=str,
        default="shape",
        choices=["shape", "paint"],
        help="Conversion stage: 'shape' or 'paint' (default: shape)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to local .ckpt file (shape stage only)"
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Local path to paintpbr model directory (paint stage)",
    )
    parser.add_argument(
        "--dino-path",
        type=str,
        default=None,
        help="Local path to dinov2-giant directory (paint stage)",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")


def add_validate_args(parser) -> None:
    parser.add_argument("model_dir", type=str)
