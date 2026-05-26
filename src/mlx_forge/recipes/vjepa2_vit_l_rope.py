"""V-JEPA 2.1 ViT-L RoPE encoder conversion recipe.

Converts a Meta V-JEPA 2.1 ViT-L (RoPE) vision encoder checkpoint from the
PyTorch ``torch.hub`` ``.pt`` format into split MLX safetensors loadable by the
``vjepa2-core-mlx`` runtime (``vjepa2_core_mlx.utils.weights.from_pretrained``).

Target config (the only one ported first):
    embed_dim=1024, depth=24, num_heads=16, patch_size=16, qkv_bias=True,
    mlp_ratio=4, norm_layer=LayerNorm(eps=1e-6), use_rope=True.
    head_dim = 1024 // 16 = 64.

Architecture / key facts (verified by diffing the upstream PyTorch module's
``state_dict()`` against the MLX module's flattened ``parameters()`` — 300/300
keys, zero orphans on either side):

  * Single component ("encoder"); the released encoders ship as one state_dict.
  * The ONLY weight whose layout differs is ``patch_embed.proj.weight``:
      - image encoder: Conv2d  ``(O, I, Kh, Kw)`` -> ``(O, Kh, Kw, I)``  (ndim 4)
      - video encoder: Conv3d  ``(O, I, Kt, Kh, Kw)`` -> ``(O, Kt, Kh, Kw, I)`` (ndim 5)
    Dispatched generically on ndim via ``mlx_forge.transpose.transpose_conv``.
  * ``patch_embed.proj.bias`` and every Linear / LayerNorm weight & bias are
    byte-identical — no transpose.
  * Learnable bare params ``img_mod_embed`` / ``video_mod_embed`` (shape
    ``(1, 1, embed_dim)``) carry over unchanged.
  * ``blocks.*`` (24 transformer blocks) and ``norms_block.*`` (4 hierarchical
    norms) keep their dotted indices — identical between ``nn.ModuleList``
    (upstream) and MLX plain lists.

Checkpoint format note:
    Meta currently ships V-JEPA 2.1 encoders ONLY as torch ``.pt`` state dicts
    (NOT on the HF Hub, NOT as MLX safetensors). This recipe therefore requires
    a user-supplied local ``--source`` path; the canonical torch.hub URL is left
    as a TODO until verified (see ``TORCH_HUB_CHECKPOINT_URL``).

Usage::

    mlx-forge convert vjepa2-vit-l-rope --source /path/to/vitl_rope.pt
    mlx-forge convert vjepa2-vit-l-rope --source /path/to/vitl_rope.pt --quantize --bits 8
    mlx-forge validate vjepa2-vit-l-rope /path/to/output/
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from ..quantize import _materialize, quantize_weights
from ..transpose import transpose_conv

# Canonical Meta CDN checkpoint for the ViT-L RoPE V-JEPA 2.1 encoder (the
# state_dict has ema_encoder + predictor; we take ema_encoder). Not on the HF
# Hub. Resolved from src/hub/backbones.py (VJEPA_BASE_URL + ARCH_NAME_MAP).
TORCH_HUB_CHECKPOINT_URL: str | None = (
    "https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt"
)

#: Output component / filename. Single-component encoder.
COMPONENT_NAME = "encoder"
OUTPUT_FILENAME = f"{COMPONENT_NAME}.safetensors"

#: Conv weight keys whose layout differs (Conv2d image / Conv3d video). The
#: released ViT-L 2.1 encoder has TWO (img_temporal_dim_size=1): the tubelet
#: patch embed and the single-frame image patch embed. Both are Conv3d here.
CONV_WEIGHT_KEYS = ("patch_embed.proj.weight", "patch_embed_img.proj.weight")

#: Expected key count at the released ViT-L RoPE config (with patch_embed_img;
#: sanity check during validate). 300 core + 2 for the second patch embed.
EXPECTED_KEY_COUNT = 302

#: Keys that are legitimately all-zero in the released checkpoint, so they must
#: NOT trip the materialization-bug guard. ViT-L 2.1 uses n_output_distillation=1
#: -> only norms_block[3] is trained; norms_block.{0,1,2} stay at LayerNorm init
#: (bias all-zero, weight all-ones). Verified against the source ema_encoder.
KNOWN_ZERO_KEYS = frozenset({"norms_block.0.bias", "norms_block.1.bias", "norms_block.2.bias"})

#: Container keys Meta checkpoints commonly nest the encoder under. "ema_encoder"
#: is the canonical released (EMA) encoder and MUST take precedence over the
#: online "encoder" — upstream loads checkpoint_key="ema_encoder" for ViT-L 2.1.
_CONTAINER_KEYS = ("ema_encoder", "encoder", "target_encoder", "model", "state_dict")

#: Parameter-name prefixes commonly wrapped onto every key.
_KNOWN_PREFIXES = ("module.", "encoder.", "target_encoder.", "backbone.")


# --------------------------------------------------------------------------- #
# Checkpoint loading + unwrapping
# --------------------------------------------------------------------------- #


def _load_torch_checkpoint(src_path: Path) -> dict[str, Any]:
    """Load a Meta ``.pt`` checkpoint and unwrap it to the bare encoder dict."""
    import torch  # local import: torch is a conversion-time dependency only

    print(f"\nLoading torch checkpoint from {src_path}...")
    t0 = time.monotonic()
    raw = torch.load(str(src_path), map_location="cpu", weights_only=False)
    print(f"  loaded in {time.monotonic() - t0:.1f}s")

    state = _unwrap_state_dict(raw)
    print(f"  {len(state)} encoder keys after unwrapping")
    return state


def _unwrap_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Unwrap a container dict and strip a shared key prefix."""
    # (1) Container dict (weights nested alongside optimizer/epoch state).
    import torch

    if state_dict and not any(isinstance(v, torch.Tensor) for v in state_dict.values()):
        for ck in _CONTAINER_KEYS:
            inner = state_dict.get(ck)
            if isinstance(inner, dict) and inner:
                state_dict = inner
                break

    # (2) Strip a prefix shared by all keys.
    keys = list(state_dict.keys())
    for prefix in _KNOWN_PREFIXES:
        if keys and all(k.startswith(prefix) for k in keys):
            state_dict = {k[len(prefix) :]: v for k, v in state_dict.items()}
            keys = list(state_dict.keys())
    return state_dict


# --------------------------------------------------------------------------- #
# Key sanitization / transform
# --------------------------------------------------------------------------- #


def sanitize_key(key: str) -> str | None:
    """Map a PyTorch encoder key -> MLX key.

    V-JEPA 2.1 encoder keys are already byte-aligned with the MLX module
    (verified zero-orphan), so no renaming is needed. Returns ``None`` to drop a
    key (none are dropped for the encoder).
    """
    return key


def _to_mx(value: Any) -> mx.array:
    """Convert a torch.Tensor to an mx.array (fp32)."""
    if isinstance(value, mx.array):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().float().numpy()
    return mx.array(value)


def transform_weight(new_key: str, weight: mx.array) -> mx.array:
    """Transpose conv (patch-embed) weights PyTorch->MLX; Linear projs (ndim 2) untouched.

    Matches any ``*.proj.weight`` with conv rank (ndim 4/5) so both the tubelet
    patch embed and the single-frame ``patch_embed_img`` are handled; attention
    ``proj.weight`` is a Linear (ndim 2) and is left alone.
    """
    if new_key.endswith(".proj.weight") and weight.ndim in (4, 5):
        return transpose_conv(weight)
    return weight


# --------------------------------------------------------------------------- #
# Quantization scope
# --------------------------------------------------------------------------- #


def should_quantize_encoder(key: str, weight: mx.array) -> bool:
    """Quantize only transformer-block Linear weights.

    Keeps the patch embedding, modality embeds, all norms, and biases at full
    precision (sensitive / tiny), matching the house policy.
    """
    if weight.ndim != 2 or not key.endswith(".weight"):
        return False
    if "blocks." not in key:  # only the 24 transformer blocks
        return False
    if "norm" in key:
        return False
    # qkv / proj / mlp.fc1 / mlp.fc2 inside blocks -> quantize.
    return True


# --------------------------------------------------------------------------- #
# Main convert entry point
# --------------------------------------------------------------------------- #


def convert(args) -> None:
    """Convert a V-JEPA 2.1 ViT-L RoPE encoder checkpoint to MLX safetensors."""
    if not args.source:
        print(
            "ERROR: --source is required.\n"
            "Meta ships V-JEPA 2.1 encoders only as torch .pt checkpoints "
            "(not on the HF Hub). Pass the local .pt path via --source.\n"
            "(TODO: TORCH_HUB_CHECKPOINT_URL is not yet verified.)"
        )
        raise SystemExit(1)

    src_path = Path(args.source)
    if not src_path.exists():
        print(f"ERROR: {src_path} does not exist")
        raise SystemExit(1)

    if args.output:
        output_dir = Path(args.output)
    else:
        suffix = f"-q{args.bits}" if args.quantize else ""
        output_dir = Path("models") / f"vjepa2-vit-l-rope-mlx{suffix}"

    if args.dry_run:
        _dry_run(args, src_path, output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load + unwrap -----
    state = _load_torch_checkpoint(src_path)

    # ----- Sanitize + transform + materialize -----
    print(f"\nProcessing {len(state)} keys...")
    t0 = time.monotonic()
    out_weights: dict[str, mx.array] = {}
    transposed = 0
    for key in state:
        new_key = sanitize_key(key)
        if new_key is None:
            continue
        w = _to_mx(state[key])
        before = w.shape
        w = transform_weight(new_key, w)
        if w.shape != before:
            transposed += 1
        _materialize(w)  # CRITICAL: lazy tensors save as zeros otherwise
        out_weights[new_key] = w

    count = len(out_weights)
    print(f"  {count} weights, {transposed} conv weight(s) transposed")

    out_file = output_dir / OUTPUT_FILENAME
    print(f"  Saving to {out_file.name}...")
    mx.save_safetensors(str(out_file), out_weights)
    print(f"  Done in {time.monotonic() - t0:.1f}s")

    del state
    gc.collect()
    mx.clear_cache()

    # ----- Config -----
    config = {
        "model_type": "vjepa2-vit-l-rope",
        "source": "facebookresearch/vjepa2 (app/vjepa_2_1)",
        "variant": "vit-l-rope",
        "architecture": "VisionTransformer",
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "patch_size": 16,
        "mlp_ratio": 4,
        "use_rope": True,
        "layernorm_eps": 1e-6,
        "num_frames": 64,
        "tubelet_size": 2,
        "img_size": 384,
        "img_temporal_dim_size": 1,
        "interpolate_rope": True,
        "components": [COMPONENT_NAME],
        "notes": {
            "conv_transpose": "both patch-embed convs transposed (O,I,*K)->(O,*K,I).",
            "modality_embeds": "img_mod_embed / video_mod_embed carried over.",
            "checkpoint_key": "ema_encoder (EMA), keys stripped of module./backbone.",
        },
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nSaved config.json")

    # split_model.json so `mlx-forge upload` can derive the repo card.
    split_info = {
        "model_name": "vjepa2-vit-l-rope-mlx",
        "components": {COMPONENT_NAME: OUTPUT_FILENAME},
        "quantized": bool(args.quantize),
    }
    with open(output_dir / "split_model.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # ----- Optional quantization -----
    if args.quantize:
        _quantize_encoder(output_dir, args.bits, args.group_size)
        with open(output_dir / "quantize_config.json", "w") as f:
            json.dump(
                {"quantization": {"bits": args.bits, "group_size": args.group_size}},
                f,
                indent=2,
            )

    # ----- Summary -----
    print(f"\n{'=' * 60}")
    print(f"Conversion complete: {count} weights")
    print(f"Output: {output_dir}")
    for p in sorted(output_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(output_dir)}: {p.stat().st_size / (1024 * 1024):.1f} MB")
    print("\nDone!")


def _quantize_encoder(output_dir: Path, bits: int, group_size: int) -> None:
    """Quantize the encoder safetensors in-place (transformer Linears only)."""
    from ..convert import load_safetensors

    filepath = output_dir / OUTPUT_FILENAME
    if not filepath.exists():
        print(f"  WARNING: {filepath.name} not found, skipping quantization")
        return

    print(f"\n  Quantizing encoder to int{bits} (group_size={group_size})...")
    weights = load_safetensors(filepath)
    result = quantize_weights(
        weights,
        bits=bits,
        group_size=group_size,
        should_quantize=should_quantize_encoder,
    )
    print(f"  Saving quantized encoder ({len(result)} keys)...")
    mx.save_safetensors(str(filepath), result)
    del result, weights
    gc.collect()
    mx.clear_cache()


def _dry_run(args, src_path: Path, output_dir: Path) -> None:
    """Print the conversion plan without writing anything."""
    print("=" * 60)
    print("DRY RUN -- no files will be written")
    print("=" * 60)
    print(f"\nSource:     {src_path}")
    print(f"Output dir: {output_dir}")
    print(f"Component:  {COMPONENT_NAME} -> {OUTPUT_FILENAME}")
    print("\nTranspose:  patch_embed.proj.weight (Conv2d/Conv3d -> channels-last)")
    if args.quantize:
        print(f"\nQuantization: int{args.bits}, group_size={args.group_size}")
        print("  Target: transformer block Linear weights only")
        print("  Skipped: patch_embed, modality embeds, norms, biases")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate(args) -> None:
    """Validate converted V-JEPA 2.1 encoder weights."""
    from ..convert import load_safetensors
    from ..validate import (
        ValidationResult,
        count_layer_indices,
        validate_file_exists,
        validate_quantization,
    )

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: {model_dir} does not exist")
        raise SystemExit(1)

    print(f"Validating: {model_dir}")
    result = ValidationResult()

    is_quantized = (model_dir / "quantize_config.json").exists()
    if is_quantized:
        with open(model_dir / "quantize_config.json") as f:
            bits = json.load(f).get("quantization", {}).get("bits", "?")
        print(f"Model is quantized: int{bits}")

    print("\n== File Structure ==")
    validate_file_exists(model_dir, OUTPUT_FILENAME, result)
    validate_file_exists(model_dir, "config.json", result)

    enc_path = model_dir / OUTPUT_FILENAME
    if enc_path.exists():
        print("\n== Encoder Weights ==")
        weights = load_safetensors(enc_path)
        keys = set(weights.keys())
        base_keys = {k for k in keys if not k.endswith((".scales", ".biases"))}

        result.check(
            len(base_keys) == EXPECTED_KEY_COUNT,
            f"expected {EXPECTED_KEY_COUNT} base keys (got {len(base_keys)})",
        )

        # Conv weights must be channels-last: last axis == in_chans (3).
        for pe in CONV_WEIGHT_KEYS:
            if pe in keys:
                shape = tuple(weights[pe].shape)
                result.check(
                    weights[pe].ndim in (4, 5) and shape[-1] == 3,
                    f"{pe} channels-last (in_chans last); shape {shape}",
                )

        # Modality embeds present.
        for me in ("img_mod_embed", "video_mod_embed"):
            result.check(me in keys, f"{me} present")

        # 24 transformer blocks + 4 hierarchical norms.
        n_blocks = len(count_layer_indices(keys, block_key="blocks"))
        result.check(n_blocks == 24, f"24 transformer blocks (got {n_blocks})")

        # No all-zero tensors (catches the materialization bug), excluding the
        # known untrained LayerNorm biases (see KNOWN_ZERO_KEYS).
        zero_keys = [
            k
            for k in base_keys
            if k not in KNOWN_ZERO_KEYS and float(mx.abs(weights[k]).sum().item()) == 0.0
        ]
        result.check(
            len(zero_keys) == 0,
            f"no unexpected all-zero tensors (found {len(zero_keys)}: {zero_keys[:3]})",
        )

        if is_quantized:
            validate_quantization(weights, result, block_key="blocks")

        del weights
        gc.collect()
        mx.clear_cache()

    result.summary()
    if not result.passed:
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# CLI argument registration
# --------------------------------------------------------------------------- #


def add_convert_args(parser) -> None:
    """Register convert arguments."""
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to the Meta V-JEPA 2.1 ViT-L RoPE encoder .pt checkpoint (required).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: ./models/vjepa2-vit-l-rope-mlx[-q<bits>])",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize transformer block Linear weights after conversion",
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
        help="Preview the conversion plan without writing anything",
    )


def add_validate_args(parser) -> None:
    """Register validate arguments."""
    parser.add_argument("model_dir", type=str, help="Path to converted model directory")


def add_split_args(parser) -> None:
    """Register split arguments (no-op: single-component model)."""
    parser.add_argument("model_dir", type=str, help="Model directory containing safetensors")
