"""Matrix-Game-3.0 conversion recipe.

Converts the full Skywork/Matrix-Game-3.0 PyTorch checkpoint to MLX split format.
All model variants are converted in a single pass:
  - dit            — DiT base backbone (50-step inference)
  - dit_distilled  — DiT distilled backbone (3-step inference)
  - t5_encoder     — UMT5-XXL text encoder
  - vae            — Wan2.2 VAE (full)
  - vae_lightvae   — MG-LightVAE (pruning 0.5)
  - vae_lightvae_v2 — MG-LightVAE v2 (pruning 0.75)

Usage:
    mlx-forge convert matrix-game-3.0
    mlx-forge convert matrix-game-3.0 --quantize --bits 8
    mlx-forge validate matrix-game-3.0 models/matrix-game-3.0-mlx
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx

from ..convert import (
    download_hf_files,
    fmt_size,
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

REPO_ID = "Skywork/Matrix-Game-3.0"

COMPONENTS = [
    "dit",
    "dit_distilled",
    "t5_encoder",
    "vae",
    "vae_lightvae",
    "vae_lightvae_v2",
]

# Approximate sizes in MB (bf16/fp16)
_COMPONENT_SIZE_MB: dict[str, int] = {
    "dit": 10_500,  # ~10.5 GB
    "dit_distilled": 21_000,  # ~21 GB
    "t5_encoder": 9_000,  # ~9 GB
    "vae": 400,  # ~400 MB (Wan2.2)
    "vae_lightvae": 2_800,  # ~2.8 GB (LightVAE v1, pruning 0.5)
    "vae_lightvae_v2": 850,  # ~850 MB (LightVAE v2, pruning 0.75)
}

# Source files on HuggingFace
_HF_FILES: dict[str, list[str]] = {
    "dit": [
        "base_model/config.json",
        "base_model/diffusion_pytorch_model.safetensors",
    ],
    "dit_distilled": [
        "base_distilled_model/config.json",
        "base_distilled_model/diffusion_pytorch_model.safetensors",
    ],
    "t5_encoder": [
        "models_t5_umt5-xxl-enc-bf16.pth",
    ],
    "vae": [
        "Wan2.2_VAE.pth",
    ],
    "vae_lightvae": [
        "MG-LightVAE.pth",
    ],
    "vae_lightvae_v2": [
        "MG-LightVAE_v2.pth",
    ],
}

TOKENIZER_FILES = [
    "google/umt5-xxl/tokenizer.json",
    "google/umt5-xxl/tokenizer_config.json",
    "google/umt5-xxl/special_tokens_map.json",
    "google/umt5-xxl/spiece.model",
]

# ---------------------------------------------------------------------------
# DiT key sanitization
# ---------------------------------------------------------------------------

# PyTorch uses nn.Sequential for several sub-modules. The MLX port uses
# explicit named Linear layers instead. These mappings handle the conversion.

_DIT_SEQUENTIAL_MAP: list[tuple[str, str]] = [
    # WanModel top-level Sequential -> named Linear
    ("text_embedding.0.", "text_embedding_linear1."),
    ("text_embedding.2.", "text_embedding_linear2."),
    ("time_embedding.0.", "time_embedding_linear1."),
    ("time_embedding.2.", "time_embedding_linear2."),
    ("time_projection.1.", "time_projection_linear1."),
    # WanAttentionBlock.ffn Sequential -> named Linear
    (".ffn.0.", ".ffn_linear1."),
    (".ffn.2.", ".ffn_linear2."),
    # ActionModule.keyboard_embed Sequential -> named Linear
    (".action_model.keyboard_embed.0.", ".action_model.keyboard_embed_linear1."),
    (".action_model.keyboard_embed.2.", ".action_model.keyboard_embed_linear2."),
    # ActionModule.mouse_mlp Sequential -> named Linear
    (".action_model.mouse_mlp.0.", ".action_model.mouse_mlp_linear1."),
    (".action_model.mouse_mlp.2.", ".action_model.mouse_mlp_linear2."),
    (".action_model.mouse_mlp.3.", ".action_model.mouse_mlp_layernorm."),
]


def sanitize_dit_key(key: str) -> str | None:
    """Convert a PyTorch DiT key to MLX format.

    Handles:
      - nn.Sequential index-based keys -> named Linear keys
      - patch_embedding Conv3d -> Linear (weight reshape handled separately)
    """
    for old, new in _DIT_SEQUENTIAL_MAP:
        if old in key:
            key = key.replace(old, new)
            break
    return key


# ---------------------------------------------------------------------------
# T5 key sanitization
# ---------------------------------------------------------------------------


def sanitize_t5_key(key: str) -> str | None:
    """Convert a PyTorch T5 encoder key to MLX format.

    The T5 .pth file stores T5Encoder state_dict directly (encoder_only=True),
    so keys like token_embedding.weight, blocks.0.attn.q.weight, etc. already
    match the MLX T5Encoder attribute structure. No renaming needed.
    """
    return key


# ---------------------------------------------------------------------------
# VAE key sanitization
# ---------------------------------------------------------------------------

# The Wan2.2_VAE.pth file uses diffusers-style keys that must be mapped to
# WanVAE_ attribute names. This mirrors _map_lightvae_key_to_wanvae from
# the PyTorch reference.


def _map_resnet_tail(tail: str) -> str:
    """Map diffusers ResNet sub-keys to WanVAE_ sequential format."""
    if tail.startswith("norm1."):
        return "residual.0." + tail[len("norm1.") :]
    if tail.startswith("conv1."):
        return "residual.2." + tail[len("conv1.") :]
    if tail.startswith("norm2."):
        return "residual.3." + tail[len("norm2.") :]
    if tail.startswith("conv2."):
        return "residual.6." + tail[len("conv2.") :]
    if tail.startswith("conv_shortcut."):
        return "shortcut." + tail[len("conv_shortcut.") :]
    return tail


def _normalize_vae_key(key: str) -> str | None:
    """Map a single VAE key from diffusers format to WanVAE_ format.

    Returns None to skip training-only keys.
    """
    # Skip training-only projection heads
    if key.startswith("dynamic_feature_projection_heads."):
        return None

    # Top-level projections
    if key.startswith("quant_conv."):
        return key.replace("quant_conv.", "conv1.", 1)
    if key.startswith("post_quant_conv."):
        return key.replace("post_quant_conv.", "conv2.", 1)

    # --- Encoder ---
    if key.startswith("encoder.conv_in."):
        return key.replace("encoder.conv_in.", "encoder.conv1.", 1)
    if key.startswith("encoder.mid_block.resnets.0."):
        tail = key[len("encoder.mid_block.resnets.0.") :]
        return "encoder.middle.0." + _map_resnet_tail(tail)
    if key.startswith("encoder.mid_block.attentions.0."):
        return key.replace("encoder.mid_block.attentions.0.", "encoder.middle.1.", 1)
    if key.startswith("encoder.mid_block.resnets.1."):
        tail = key[len("encoder.mid_block.resnets.1.") :]
        return "encoder.middle.2." + _map_resnet_tail(tail)
    if key.startswith("encoder.norm_out."):
        return key.replace("encoder.norm_out.", "encoder.head.0.", 1)
    if key.startswith("encoder.conv_out."):
        return key.replace("encoder.conv_out.", "encoder.head.2.", 1)

    # Encoder down blocks
    if key.startswith("encoder.down_blocks."):
        parts = key.split(".")
        if len(parts) >= 6 and parts[3] == "resnets":
            tail = ".".join(parts[5:])
            return f"encoder.downsamples.{parts[2]}.downsamples.{parts[4]}." + _map_resnet_tail(
                tail
            )
        if len(parts) >= 7 and parts[3] == "downsampler" and parts[4] == "resample":
            return f"encoder.downsamples.{parts[2]}.downsamples.2.resample.{parts[5]}." + ".".join(
                parts[6:]
            )
        if len(parts) >= 6 and parts[3] == "downsampler" and parts[4] == "time_conv":
            return f"encoder.downsamples.{parts[2]}.downsamples.2.time_conv." + ".".join(parts[5:])

    # --- Decoder ---
    if key.startswith("decoder.conv_in."):
        return key.replace("decoder.conv_in.", "decoder.conv1.", 1)
    if key.startswith("decoder.mid_block.resnets.0."):
        tail = key[len("decoder.mid_block.resnets.0.") :]
        return "decoder.middle.0." + _map_resnet_tail(tail)
    if key.startswith("decoder.mid_block.attentions.0."):
        return key.replace("decoder.mid_block.attentions.0.", "decoder.middle.1.", 1)
    if key.startswith("decoder.mid_block.resnets.1."):
        tail = key[len("decoder.mid_block.resnets.1.") :]
        return "decoder.middle.2." + _map_resnet_tail(tail)
    if key.startswith("decoder.norm_out."):
        return key.replace("decoder.norm_out.", "decoder.head.0.", 1)
    if key.startswith("decoder.conv_out."):
        return key.replace("decoder.conv_out.", "decoder.head.2.", 1)

    # Decoder up blocks
    if key.startswith("decoder.up_blocks."):
        parts = key.split(".")
        if len(parts) >= 6 and parts[3] == "resnets":
            tail = ".".join(parts[5:])
            return f"decoder.upsamples.{parts[2]}.upsamples.{parts[4]}." + _map_resnet_tail(tail)
        if len(parts) >= 7 and parts[3] == "upsampler" and parts[4] == "resample":
            return f"decoder.upsamples.{parts[2]}.upsamples.3.resample.{parts[5]}." + ".".join(
                parts[6:]
            )
        if len(parts) >= 6 and parts[3] == "upsampler" and parts[4] == "time_conv":
            return f"decoder.upsamples.{parts[2]}.upsamples.3.time_conv." + ".".join(parts[5:])

    # Already in WanVAE_ naming — pass through
    return key


def sanitize_vae_key(key: str) -> str | None:
    """Convert a PyTorch VAE key to MLX format."""
    return _normalize_vae_key(key)


# ---------------------------------------------------------------------------
# Conv transposition + patch_embedding Conv3d -> Linear
# ---------------------------------------------------------------------------


def maybe_transpose(key: str, value: mx.array, component: str) -> mx.array:
    """Transpose conv weights from PyTorch to MLX layout if needed.

    Special case: the DiT patch_embedding is a Conv3d in PyTorch but a Linear
    in our MLX port. The Conv3d weight (dim, in_dim, pt, ph, pw) must be
    flattened and transposed to Linear format (in_dim*pt*ph*pw, dim).
    """
    if component == "dit":
        # patch_embedding Conv3d -> Linear conversion
        if "patch_embedding.weight" in key and "wancamctrl" not in key:
            if value.ndim == 5:
                # Conv3d: (O, I, D, H, W) -> flatten to (O, I*D*H*W) -> transpose to (I*D*H*W, O)
                o, i, d, h, w = value.shape
                value = value.reshape(o, i * d * h * w)
                value = mx.transpose(value)
                return value
        # DiT is all Linear (no conv to transpose)
        return value

    if component == "t5_encoder":
        # T5 is all Linear, no conv
        return value

    if component == "vae":
        # VAE has Conv3d layers that need transposition
        is_conv = "conv" in key.lower() and "weight" in key
        if is_conv and value.ndim >= 3:
            return transpose_conv(value)
        return value

    return value


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def should_quantize(key: str, weight: mx.array) -> bool:
    """Determine if a weight should be quantized.

    Only quantize Linear .weight matrices in DiT transformer blocks.
    Exclude sensitive layers that harm quality when quantized.
    """
    # Only 2D weight matrices (Linear layers)
    if weight.ndim != 2 or not key.endswith(".weight"):
        return False

    # Quantization artifacts in these layers
    bare_key = key.replace("dit.", "", 1)

    # Exclude timestep/time embedding layers
    if "time_embedding" in bare_key or "time_projection" in bare_key:
        return False

    # Exclude patch embedding (input projection)
    if "patch_embedding" in bare_key:
        return False

    # Exclude final head projection
    if bare_key.startswith("head.") or bare_key == "head.head.weight":
        return False

    # Exclude normalization weights (RMSNorm, LayerNorm)
    if "norm" in bare_key.split(".")[-2:][0]:
        return False

    # Exclude camera injection layers
    if "cam_injector" in bare_key or "cam_scale" in bare_key or "cam_shift" in bare_key:
        return False
    if "c2ws_hidden_states" in bare_key:
        return False

    # Exclude modulation parameters (not Linear but can appear as 2D)
    if "modulation" in bare_key:
        return False

    # Exclude text embedding (small, sensitive)
    if "text_embedding" in bare_key:
        return False

    # Quantize everything else (blocks.N.self_attn.*, blocks.N.cross_attn.*,
    # blocks.N.ffn_linear*, blocks.N.action_model.*)
    return True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _build_config(dit_config_path: Path) -> dict:
    """Build output config.json from the diffusers config."""
    with open(dit_config_path) as f:
        src = json.load(f)

    return {
        "model_type": src.get("model_type", "ti2v"),
        "dim": src.get("dim", 3072),
        "ffn_dim": src.get("ffn_dim", 14336),
        "freq_dim": src.get("freq_dim", 256),
        "in_dim": src.get("in_dim", 48),
        "out_dim": src.get("out_dim", 48),
        "text_len": src.get("text_len", 512),
        "num_heads": src.get("num_heads", 24),
        "num_layers": src.get("num_layers", 30),
        "eps": src.get("eps", 1e-6),
        "sigma_theta": src.get("sigma_theta", 0.8),
        "action_config": src.get("action_config", {}),
        "use_text_crossattn": src.get("use_text_crossattn", True),
        "is_action_model": src.get("is_action_model", True),
        "source": REPO_ID,
    }


# ---------------------------------------------------------------------------
# Component conversion helpers
# ---------------------------------------------------------------------------


def _convert_dit(args, download_dir: Path, output_dir: Path, component: str, hf_subdir: str) -> int:
    """Download, convert and save a DiT variant. Returns weight count."""
    print("\n" + "=" * 60)
    print(f"Converting {component}")
    print("=" * 60)

    if args.dit_checkpoint and component == "dit":
        dit_path = args.dit_checkpoint
    else:
        print(f"Downloading {component} from {REPO_ID}...")
        download_hf_files(REPO_ID, _HF_FILES[component], download_dir)
        dit_path = str(download_dir / hf_subdir / "diffusion_pytorch_model.safetensors")

    print(f"\nLoading {component} weights lazily from {dit_path}...")
    t0 = time.monotonic()
    dit_weights = mx.load(dit_path)
    print(f"  {len(dit_weights)} keys loaded (lazy) in {time.monotonic() - t0:.1f}s")

    print(f"\nProcessing {len(dit_weights)} {component} keys...")
    t0 = time.monotonic()
    dit_output: dict[str, mx.array] = {}
    for key in dit_weights:
        new_key = sanitize_dit_key(key)
        if new_key is None:
            continue
        weight = dit_weights[key]
        weight = maybe_transpose(new_key, weight, "dit")
        _materialize(weight)
        dit_output[f"{component}.{new_key}"] = weight

    count = len(dit_output)
    out_file = f"{component}.safetensors"
    print(f"  Saving {count} weights to {out_file}...")
    mx.save_safetensors(str(output_dir / out_file), dit_output)
    elapsed = time.monotonic() - t0
    print(f"  Done: {count} weights saved in {elapsed:.1f}s")

    del dit_output, dit_weights
    gc.collect()
    mx.clear_cache()
    return count


def _convert_t5_pth(pth_path: str, output_dir: Path) -> int:
    """Load T5 .pth, convert to mx arrays, save as safetensors. Returns weight count."""
    print(f"\nLoading T5 weights from {pth_path}...")
    t0 = time.monotonic()
    try:
        import torch
    except ImportError:
        print("ERROR: torch is required to load .pth files\nInstall it with: uv pip install torch")
        raise SystemExit(1)

    print("  (weights_only=True — safe deserialization mode)")
    t5_raw = torch.load(pth_path, map_location="cpu", weights_only=True)
    t5_raw = _extract_state_dict(t5_raw)
    print(f"  {len(t5_raw)} keys loaded in {time.monotonic() - t0:.1f}s")

    print(f"\nProcessing {len(t5_raw)} T5 keys...")
    t0 = time.monotonic()
    t5_output: dict[str, mx.array] = {}
    for key, value in t5_raw.items():
        new_key = sanitize_t5_key(key)
        if new_key is None:
            continue
        weight = mx.array(value.float().numpy())
        t5_output[f"t5_encoder.{new_key}"] = weight

    count = len(t5_output)
    print(f"  Saving {count} weights to t5_encoder.safetensors...")
    mx.save_safetensors(str(output_dir / "t5_encoder.safetensors"), t5_output)
    elapsed = time.monotonic() - t0
    print(f"  Done: {count} weights saved in {elapsed:.1f}s")

    del t5_output, t5_raw
    gc.collect()
    mx.clear_cache()
    return count


def _extract_state_dict(checkpoint: dict) -> dict:
    """Recursively unwrap a PyTorch checkpoint to find the flat state_dict.

    Some checkpoints nest the actual weights under keys like 'state_dict',
    'gen_model', or 'generator'. This mirrors _extract_checkpoint_state_dict
    from the Matrix-Game-3 reference code.
    """
    wrapper_keys = ("state_dict", "gen_model", "generator")
    for wk in wrapper_keys:
        if isinstance(checkpoint, dict) and wk in checkpoint:
            return _extract_state_dict(checkpoint[wk])
    return checkpoint


def _convert_vae_pth(pth_path: str, output_path: Path, prefix: str) -> int:
    """Load a VAE .pth, sanitize keys, transpose convs, and save as safetensors.

    Returns the number of weights saved.
    """
    print(f"\nLoading VAE weights from {pth_path}...")
    t0 = time.monotonic()
    try:
        import torch
    except ImportError:
        print("ERROR: torch is required to load .pth files\nInstall it with: uv pip install torch")
        raise SystemExit(1)

    print("  (weights_only=True — safe deserialization mode)")
    vae_raw = torch.load(pth_path, map_location="cpu", weights_only=True)
    vae_raw = _extract_state_dict(vae_raw)
    print(f"  {len(vae_raw)} keys loaded in {time.monotonic() - t0:.1f}s")

    print(f"\nProcessing {len(vae_raw)} VAE keys...")
    t0 = time.monotonic()
    vae_output: dict[str, mx.array] = {}
    for key, value in vae_raw.items():
        new_key = sanitize_vae_key(key)
        if new_key is None:
            continue
        weight = mx.array(value.float().numpy())
        weight = maybe_transpose(new_key, weight, "vae")
        _materialize(weight)
        vae_output[f"{prefix}.{new_key}"] = weight

    count = len(vae_output)
    print(f"  Saving {count} weights to {output_path.name}...")
    mx.save_safetensors(str(output_path), vae_output)
    elapsed = time.monotonic() - t0
    print(f"  Done: {count} weights saved in {elapsed:.1f}s")

    del vae_output, vae_raw
    gc.collect()
    mx.clear_cache()
    return count


# ---------------------------------------------------------------------------
# Main convert entry point
# ---------------------------------------------------------------------------


def convert(args) -> None:
    """Convert all Matrix-Game-3.0 variants to MLX split format."""
    if args.output:
        output_dir = Path(args.output)
    else:
        suffix = f"-q{args.bits}" if args.quantize else ""
        output_dir = Path("models") / f"matrix-game-3.0-mlx{suffix}"

    if args.dry_run:
        _dry_run(args, output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = Path("models") / "matrix-game-3.0-src"

    total_weights = 0

    # -----------------------------------------------------------------------
    # 1. DiT base (diffusers safetensors)
    # -----------------------------------------------------------------------
    total_weights += _convert_dit(args, download_dir, output_dir, "dit", "base_model")

    # -----------------------------------------------------------------------
    # 2. DiT distilled (diffusers safetensors)
    # -----------------------------------------------------------------------
    total_weights += _convert_dit(
        args, download_dir, output_dir, "dit_distilled", "base_distilled_model"
    )

    # -----------------------------------------------------------------------
    # 3. T5 encoder (.pth)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Converting T5 encoder")
    print("=" * 60)

    if args.t5_checkpoint:
        t5_path = args.t5_checkpoint
    else:
        print(f"Downloading T5 from {REPO_ID}...")
        download_hf_files(REPO_ID, _HF_FILES["t5_encoder"], download_dir)
        t5_path = str(download_dir / "models_t5_umt5-xxl-enc-bf16.pth")

    total_weights += _convert_t5_pth(t5_path, output_dir)

    # -----------------------------------------------------------------------
    # 4. VAE — Wan2.2 (full)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Converting Wan2.2 VAE")
    print("=" * 60)

    if args.vae_checkpoint:
        vae_wan_path = args.vae_checkpoint
    else:
        print(f"Downloading Wan2.2 VAE from {REPO_ID}...")
        download_hf_files(REPO_ID, _HF_FILES["vae"], download_dir)
        vae_wan_path = str(download_dir / "Wan2.2_VAE.pth")

    total_weights += _convert_vae_pth(vae_wan_path, output_dir / "vae.safetensors", "vae")

    # -----------------------------------------------------------------------
    # 5. VAE — MG-LightVAE (pruning 0.5)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Converting MG-LightVAE")
    print("=" * 60)

    print(f"Downloading MG-LightVAE from {REPO_ID}...")
    download_hf_files(REPO_ID, _HF_FILES["vae_lightvae"], download_dir)
    total_weights += _convert_vae_pth(
        str(download_dir / "MG-LightVAE.pth"),
        output_dir / "vae_lightvae.safetensors",
        "vae_lightvae",
    )

    # -----------------------------------------------------------------------
    # 6. VAE — MG-LightVAE v2 (pruning 0.75)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Converting MG-LightVAE v2")
    print("=" * 60)

    print(f"Downloading MG-LightVAE v2 from {REPO_ID}...")
    download_hf_files(REPO_ID, _HF_FILES["vae_lightvae_v2"], download_dir)
    total_weights += _convert_vae_pth(
        str(download_dir / "MG-LightVAE_v2.pth"),
        output_dir / "vae_lightvae_v2.safetensors",
        "vae_lightvae_v2",
    )

    # -----------------------------------------------------------------------
    # 7. Tokenizer files
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Syncing tokenizer files")
    print("=" * 60)

    if not args.skip_tokenizer:
        download_hf_files(REPO_ID, TOKENIZER_FILES, download_dir)
        import shutil

        for fname in TOKENIZER_FILES:
            src = download_dir / fname
            dst = output_dir / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(str(src), str(dst))
                print(f"  Copied {fname}")

    # -----------------------------------------------------------------------
    # 8. Config (from base_model config.json)
    # -----------------------------------------------------------------------
    dit_config_path = download_dir / "base_model" / "config.json"
    if dit_config_path.exists():
        config = _build_config(dit_config_path)
    else:
        config = _build_config_defaults()

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nSaved config.json")

    # Split model manifest
    split_info: dict = {
        "format": "split",
        "components": COMPONENTS,
        "source": REPO_ID,
    }
    with open(output_dir / "split_model.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # -----------------------------------------------------------------------
    # 9. Optional quantization (both DiT variants)
    # -----------------------------------------------------------------------
    if args.quantize:
        for dit_comp in ("dit", "dit_distilled"):
            print(f"\nQuantizing {dit_comp} to int{args.bits} (group_size={args.group_size})...")
            quantize_component(
                output_dir,
                dit_comp,
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


def _build_config_defaults() -> dict:
    """Fallback config when diffusers config.json is not available."""
    return {
        "model_type": "ti2v",
        "dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "in_dim": 48,
        "out_dim": 48,
        "text_len": 512,
        "num_heads": 24,
        "num_layers": 30,
        "eps": 1e-6,
        "sigma_theta": 0.8,
        "action_config": {
            "blocks": list(range(15)),
            "enable_keyboard": True,
            "enable_mouse": True,
            "heads_num": 16,
            "hidden_size": 128,
            "img_hidden_size": 3072,
            "keyboard_dim_in": 6,
            "keyboard_hidden_dim": 1024,
            "mouse_dim_in": 2,
            "mouse_hidden_dim": 1024,
            "mouse_qk_dim_list": [8, 28, 28],
            "patch_size": [1, 2, 2],
            "qk_norm": True,
            "qkv_bias": False,
            "rope_dim_list": [8, 28, 28],
            "rope_theta": 256,
            "vae_time_compression_ratio": 4,
            "windows_size": 3,
        },
        "use_text_crossattn": True,
        "is_action_model": True,
        "source": REPO_ID,
    }


def _dry_run(args, output_dir: Path) -> None:
    """Print conversion plan without executing anything."""
    print("=" * 60)
    print("DRY RUN — no files will be downloaded or written")
    print("=" * 60)

    print(f"\nSource:     {REPO_ID} (HuggingFace)")
    print(f"Output dir: {output_dir}")
    print("\nOutput files:")

    total_mb = 0.0
    for comp in COMPONENTS:
        size_mb = _COMPONENT_SIZE_MB[comp]
        if args.quantize and comp.startswith("dit"):
            ratio = 16 / args.bits
            size_mb = size_mb / ratio
            print(f"  {comp}.safetensors: ~{fmt_size(size_mb)} (int{args.bits})")
        else:
            print(f"  {comp}.safetensors: ~{fmt_size(size_mb)} (fp16)")
        total_mb += size_mb

    print("  config.json, split_model.json")
    print("  google/umt5-xxl/ (tokenizer files)")

    if args.quantize:
        print(f"\nQuantization: int{args.bits}, group_size={args.group_size}")
        print("  Target: DiT blocks Linear weights only (both base and distilled)")

    print(f"\nEstimated output size: ~{fmt_size(total_mb)}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_dit(
    model_dir: Path,
    component: str,
    result: ValidationResult,
    is_quantized: bool,
) -> None:
    """Validate a DiT component (dit or dit_distilled)."""
    print(f"\n== {component} Weights ==")
    dit_path = model_dir / f"{component}.safetensors"
    if not dit_path.exists():
        return

    weights = mx.load(str(dit_path))
    keys = set(weights.keys())

    prefix = f"{component}."
    all_prefixed = all(k.startswith(prefix) for k in keys)
    result.check(all_prefixed, f"All keys have '{prefix}' prefix ({len(keys)} keys)")

    # Check no un-sanitized Sequential keys remain
    result.check(
        not any(".ffn.0." in k or ".ffn.2." in k for k in keys),
        "No un-sanitized ffn Sequential keys",
    )
    result.check(
        not any("text_embedding.0." in k or "text_embedding.2." in k for k in keys),
        "No un-sanitized text_embedding Sequential keys",
    )
    result.check(
        not any("time_embedding.0." in k or "time_embedding.2." in k for k in keys),
        "No un-sanitized time_embedding Sequential keys",
    )

    # Check patch_embedding is Linear (2D weight)
    pe_key = f"{component}.patch_embedding.weight"
    if pe_key in keys:
        pe_shape = weights[pe_key].shape
        result.check(
            weights[pe_key].ndim == 2,
            f"patch_embedding.weight is 2D Linear (shape: {pe_shape})",
        )

    # Block count
    bare_keys = {k.removeprefix(prefix) for k in keys}
    block_indices = count_layer_indices(bare_keys, block_key="blocks")
    result.check(
        len(block_indices) == 30,
        f"30 transformer blocks (got {len(block_indices)})",
    )

    # Action model present
    action_keys = [k for k in keys if "action_model" in k]
    result.check(len(action_keys) > 0, f"Action model keys present ({len(action_keys)})")

    # Camera injection layers
    cam_keys = [k for k in keys if "cam_injector" in k or "cam_scale" in k or "cam_shift" in k]
    result.check(len(cam_keys) > 0, f"Camera injection keys present ({len(cam_keys)})")

    # Modulation parameters
    mod_keys = [k for k in keys if "modulation" in k]
    result.check(len(mod_keys) > 0, f"Modulation parameters present ({len(mod_keys)})")

    if is_quantized:
        validate_quantization(weights, result, block_key="blocks")

    total_params = sum(v.size for v in weights.values())
    print(f"  Total {component} parameters: {total_params / 1e9:.2f}B")
    del weights
    gc.collect()
    mx.clear_cache()


def _validate_vae(
    model_dir: Path,
    component: str,
    result: ValidationResult,
) -> None:
    """Validate a VAE component (vae, vae_lightvae, vae_lightvae_v2)."""
    print(f"\n== {component} Weights ==")
    vae_path = model_dir / f"{component}.safetensors"
    if not vae_path.exists():
        return

    weights = mx.load(str(vae_path))
    keys = set(weights.keys())

    prefix = f"{component}."
    all_prefixed = all(k.startswith(prefix) for k in keys)
    result.check(all_prefixed, f"All keys have '{prefix}' prefix ({len(keys)} keys)")

    enc_keys = [k for k in keys if k.startswith(f"{component}.encoder.")]
    dec_keys = [k for k in keys if k.startswith(f"{component}.decoder.")]
    result.check(len(enc_keys) > 0, f"Encoder keys present ({len(enc_keys)})")
    result.check(len(dec_keys) > 0, f"Decoder keys present ({len(dec_keys)})")

    validate_conv_layout(weights, result, ndim=5)

    total_params = sum(v.size for v in weights.values())
    print(f"  Total {component} parameters: {total_params / 1e6:.1f}M")
    del weights
    gc.collect()
    mx.clear_cache()


def validate(args) -> None:
    """Validate a converted Matrix-Game-3.0 model."""
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
            config.get("num_layers") == 30,
            f"num_layers == 30 (got: {config.get('num_layers')})",
        )
        result.check(
            config.get("num_heads") == 24,
            f"num_heads == 24 (got: {config.get('num_heads')})",
        )
        result.check(
            config.get("dim") == 3072,
            f"dim == 3072 (got: {config.get('dim')})",
        )
        result.check(
            config.get("in_dim") == 48,
            f"in_dim == 48 (got: {config.get('in_dim')})",
        )

    # --- DiT variants ---
    _validate_dit(model_dir, "dit", result, is_quantized)
    _validate_dit(model_dir, "dit_distilled", result, is_quantized)

    # --- T5 weights ---
    print("\n== T5 Encoder Weights ==")
    t5_path = model_dir / "t5_encoder.safetensors"
    if t5_path.exists():
        weights = mx.load(str(t5_path))
        keys = set(weights.keys())

        all_prefixed = all(k.startswith("t5_encoder.") for k in keys)
        result.check(all_prefixed, f"All keys have 't5_encoder.' prefix ({len(keys)} keys)")

        bare_keys = {k.removeprefix("t5_encoder.") for k in keys}
        result.check("token_embedding.weight" in bare_keys, "token_embedding present")
        result.check("norm.weight" in bare_keys, "Final norm present")

        block_indices = count_layer_indices(bare_keys, block_key="blocks")
        result.check(
            len(block_indices) == 24,
            f"24 encoder blocks (got {len(block_indices)})",
        )

        total_params = sum(v.size for v in weights.values())
        print(f"  Total T5 parameters: {total_params / 1e9:.2f}B")
        del weights
        gc.collect()
        mx.clear_cache()

    # --- VAE variants ---
    _validate_vae(model_dir, "vae", result)
    _validate_vae(model_dir, "vae_lightvae", result)
    _validate_vae(model_dir, "vae_lightvae_v2", result)

    result.summary()
    if not result.passed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
    """Add Matrix-Game-3.0 convert arguments to a parser."""
    parser.add_argument(
        "--dit-checkpoint",
        type=str,
        default=None,
        help="Path to local base DiT .safetensors checkpoint (skips download)",
    )
    parser.add_argument(
        "--t5-checkpoint",
        type=str,
        default=None,
        help="Path to local T5 .pth checkpoint (skips download)",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=str,
        default=None,
        help="Path to local Wan2.2 VAE .pth checkpoint (skips download)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ./models/matrix-game-3.0-mlx[-q<bits>])",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize both DiT variants after conversion",
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
    parser.add_argument(
        "--skip-tokenizer",
        action="store_true",
        help="Skip copying tokenizer files",
    )


def add_validate_args(parser) -> None:
    """Add Matrix-Game-3.0 validate arguments to a parser."""
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to converted model directory",
    )


def add_split_args(parser) -> None:
    """Add Matrix-Game-3.0 split arguments to a parser."""
    parser.add_argument(
        "model_dir",
        type=str,
        help="Model directory containing unified safetensors",
    )
