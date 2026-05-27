"""V-JEPA 2.1 ViT-L RoPE dual-component conversion recipe.

Converts a Meta V-JEPA 2.1 ViT-L (RoPE) checkpoint from the PyTorch
``torch.hub`` ``.pt`` format into split MLX safetensors loadable by the
``vjepa2-core-mlx`` runtime (``vjepa2_core_mlx.utils.weights.from_pretrained``).

Two components are extracted from the same ``.pt`` file:
  * **encoder** (``ema_encoder``): 302 keys, written to ``encoder.safetensors``.
  * **predictor** (``predictor``): 162 keys, written to ``predictor.safetensors``.

Target encoder config:
    embed_dim=1024, depth=24, num_heads=16, patch_size=16, qkv_bias=True,
    mlp_ratio=4, norm_layer=LayerNorm(eps=1e-6), use_rope=True.
    head_dim = 1024 // 16 = 64.

Target predictor config:
    predictor_embed_dim=384, predictor_depth=12, num_heads=16,
    predictor_proj shape (1664, 384), 8 mask tokens.

Architecture / key facts (encoder, verified 300/300 keys zero-orphan):

  * The ONLY weight whose layout differs is ``patch_embed.proj.weight``:
      - image encoder: Conv2d  ``(O, I, Kh, Kw)`` -> ``(O, Kh, Kw, I)``  (ndim 4)
      - video encoder: Conv3d  ``(O, I, Kt, Kh, Kw)`` -> ``(O, Kt, Kh, Kw, I)`` (ndim 5)
    Dispatched generically on ndim via ``mlx_forge.transpose.transpose_conv``.
  * ``patch_embed.proj.bias`` and every Linear / LayerNorm weight & bias are
    byte-identical — no transpose.
  * Learnable bare params ``img_mod_embed`` / ``video_mod_embed`` (shape
    ``(1, 1, embed_dim)``) carry over unchanged.

Predictor key facts:
  * All keys prefixed ``module.backbone.`` in the raw checkpoint — stripped here.
  * All weights are Linear / LayerNorm / bare params — ZERO convs, no transpose.
  * 8 mask tokens (``mask_tokens.{0..7}``): indices 1-7 are all-zero (legitimate
    ``zero_init`` from training); index 0 is non-zero.

Checkpoint format note:
    Meta ships V-JEPA 2.1 only as torch ``.pt`` state dicts (NOT on HF Hub).
    This recipe requires a user-supplied local ``--source`` path.

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

# --------------------------------------------------------------------------- #
# Encoder component constants
# --------------------------------------------------------------------------- #

#: Encoder output filename.
ENCODER_COMPONENT = "encoder"
OUTPUT_FILENAME = f"{ENCODER_COMPONENT}.safetensors"

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

# --------------------------------------------------------------------------- #
# Predictor component constants
# --------------------------------------------------------------------------- #

#: Predictor output filename.
PREDICTOR_COMPONENT = "predictor"
PREDICTOR_OUTPUT_FILENAME = f"{PREDICTOR_COMPONENT}.safetensors"

#: Expected key count for the predictor component (verified against checkpoint).
#: predictor_embed(2) + predictor_blocks.0-11(12*10=120) + predictor_norm(2)
#: + predictor_proj(2) + predictor_proj_context(2) + mask_tokens.0-7(8)
#: + img_mod_embed(1) + video_mod_embed(1) = 138 bare + 24 extra = 162 total.
PREDICTOR_EXPECTED_KEY_COUNT = 162

#: Prefix stripped from all raw predictor keys in the released checkpoint.
_PREDICTOR_PREFIX = "module.backbone."

#: Predictor mask tokens are ``zero_init`` from training — indices 1-7 are
#: legitimately all-zero in this checkpoint. Allowlisted to avoid tripping
#: the materialization-bug guard. Index 0 is non-zero and must NOT be listed.
PREDICTOR_KNOWN_ZERO_KEYS = frozenset({f"mask_tokens.{i}" for i in range(1, 8)})

#: Container keys Meta checkpoints nest the encoder under. "ema_encoder" MUST
#: take precedence over "encoder" — upstream loads checkpoint_key="ema_encoder".
_CONTAINER_KEYS = ("ema_encoder", "encoder", "target_encoder", "model", "state_dict")

#: Parameter-name prefixes commonly wrapped onto every key.
_KNOWN_PREFIXES = ("module.", "encoder.", "target_encoder.", "backbone.")


# --------------------------------------------------------------------------- #
# Checkpoint loading + unwrapping
# --------------------------------------------------------------------------- #


def _load_torch_checkpoint(src_path: Path) -> dict[str, Any]:
    """Load a Meta ``.pt`` checkpoint; return the full raw dict (not unwrapped)."""
    import torch  # ty: ignore[unresolved-import]

    print(f"\nLoading torch checkpoint from {src_path}...")
    t0 = time.monotonic()
    raw = torch.load(str(src_path), map_location="cpu", weights_only=False)
    print(f"  loaded in {time.monotonic() - t0:.1f}s")
    return raw


def _unwrap_encoder(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract and unwrap the encoder state dict from the raw checkpoint."""
    import torch  # ty: ignore[unresolved-import]

    state_dict = raw

    # (1) Container dict: weights nested alongside optimizer / epoch state.
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

    print(f"  {len(state_dict)} encoder keys after unwrapping")
    return state_dict


def _unwrap_predictor(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract and unwrap the predictor state dict from the raw checkpoint.

    The predictor is stored under ``raw["predictor"]`` with all keys prefixed
    by ``module.backbone.``.  No further container nesting.
    """
    pred = raw.get("predictor")
    if not pred:
        raise KeyError("'predictor' key not found in checkpoint")

    # Strip the shared module.backbone. prefix (defensive: also handles missing).
    stripped: dict[str, Any] = {}
    for k, v in pred.items():
        new_k = k[len(_PREDICTOR_PREFIX) :] if k.startswith(_PREDICTOR_PREFIX) else k
        stripped[new_k] = v

    print(f"  {len(stripped)} predictor keys after unwrapping")
    return stripped


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


def should_quantize_predictor(key: str, weight: mx.array) -> bool:
    """Quantize Linear weights in predictor transformer blocks only.

    Full-precision: predictor_embed, predictor_proj, predictor_proj_context,
    predictor_norm, mask_tokens, img_mod_embed, video_mod_embed, all biases.
    Quantized: qkv / proj / mlp.fc1 / mlp.fc2 inside predictor_blocks.*.
    """
    if weight.ndim != 2 or not key.endswith(".weight"):
        return False
    if "predictor_blocks." not in key:
        return False
    if "norm" in key:
        return False
    return True


# --------------------------------------------------------------------------- #
# Main convert entry point
# --------------------------------------------------------------------------- #


def convert(args) -> None:
    """Convert a V-JEPA 2.1 ViT-L RoPE checkpoint to dual-component MLX safetensors."""
    if not args.source:
        print(
            "ERROR: --source is required.\n"
            "Meta ships V-JEPA 2.1 only as torch .pt checkpoints (not on HF Hub).\n"
            "Pass the local .pt path via --source."
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

    # ----- Load the raw checkpoint once (shared by both components) -----
    raw = _load_torch_checkpoint(src_path)

    # ================================================================== #
    # Component 1: encoder (ema_encoder)
    # ================================================================== #
    enc_state = _unwrap_encoder(raw)

    print(f"\nProcessing encoder ({len(enc_state)} keys)...")
    t0 = time.monotonic()
    enc_weights: dict[str, mx.array] = {}
    transposed = 0
    for key in enc_state:
        new_key = sanitize_key(key)
        if new_key is None:
            continue
        w = _to_mx(enc_state[key])
        before = w.shape
        w = transform_weight(new_key, w)
        if w.shape != before:
            transposed += 1
        _materialize(w)  # CRITICAL: lazy tensors save as zeros otherwise
        enc_weights[new_key] = w

    enc_count = len(enc_weights)
    print(f"  {enc_count} weights, {transposed} conv weight(s) transposed")

    enc_file = output_dir / OUTPUT_FILENAME
    print(f"  Saving to {enc_file.name}...")
    mx.save_safetensors(str(enc_file), enc_weights)
    print(f"  Done in {time.monotonic() - t0:.1f}s")

    del enc_state, enc_weights
    gc.collect()
    mx.clear_cache()

    # ================================================================== #
    # Component 2: predictor
    # ================================================================== #
    pred_state = _unwrap_predictor(raw)

    print(f"\nProcessing predictor ({len(pred_state)} keys)...")
    t0 = time.monotonic()
    pred_weights: dict[str, mx.array] = {}
    for key, val in pred_state.items():
        # No conv transpose needed — predictor is all Linear / LayerNorm / bare params.
        w = _to_mx(val)
        _materialize(w)  # CRITICAL: lazy tensors save as zeros otherwise
        pred_weights[key] = w

    pred_count = len(pred_weights)
    print(f"  {pred_count} weights (no conv transpose)")

    pred_file = output_dir / PREDICTOR_OUTPUT_FILENAME
    print(f"  Saving to {pred_file.name}...")
    mx.save_safetensors(str(pred_file), pred_weights)
    print(f"  Done in {time.monotonic() - t0:.1f}s")

    del raw, pred_state, pred_weights
    gc.collect()
    mx.clear_cache()

    # ================================================================== #
    # Metadata
    # ================================================================== #
    config = {
        "model_type": "vjepa2-vit-l-rope",
        "source": "facebookresearch/vjepa2 (app/vjepa_2_1)",
        "variant": "vit-l-rope",
        "architecture": "VisionTransformer",
        # Encoder params
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
        # Predictor params
        "predictor_embed_dim": 384,
        "predictor_depth": 12,
        "predictor_proj_dim": 1664,
        "predictor_num_mask_tokens": 8,
        "components": [ENCODER_COMPONENT, PREDICTOR_COMPONENT],
        "notes": {
            "conv_transpose": "both patch-embed convs transposed (O,I,*K)->(O,*K,I).",
            "modality_embeds": "img_mod_embed / video_mod_embed in both encoder + predictor.",
            "checkpoint_key": "ema_encoder (EMA); predictor keys stripped of module.backbone.",
        },
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nSaved config.json")

    # split_model.json so `mlx-forge upload` can derive the repo card.
    split_info = {
        "model_name": "vjepa2-vit-l-rope-mlx",
        "components": {
            ENCODER_COMPONENT: OUTPUT_FILENAME,
            PREDICTOR_COMPONENT: PREDICTOR_OUTPUT_FILENAME,
        },
        "quantized": bool(args.quantize),
    }
    with open(output_dir / "split_model.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # ----- Optional quantization -----
    if args.quantize:
        _quantize_encoder(output_dir, args.bits, args.group_size)
        _quantize_predictor(output_dir, args.bits, args.group_size)
        with open(output_dir / "quantize_config.json", "w") as f:
            json.dump(
                {"quantization": {"bits": args.bits, "group_size": args.group_size}},
                f,
                indent=2,
            )

    # ----- Summary -----
    print(f"\n{'=' * 60}")
    print(f"Conversion complete: encoder={enc_count}, predictor={pred_count} weights")
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
        print(f"  WARNING: {filepath.name} not found, skipping encoder quantization")
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


def _quantize_predictor(output_dir: Path, bits: int, group_size: int) -> None:
    """Quantize the predictor safetensors in-place (predictor_blocks Linears only)."""
    from ..convert import load_safetensors

    filepath = output_dir / PREDICTOR_OUTPUT_FILENAME
    if not filepath.exists():
        print(f"  WARNING: {filepath.name} not found, skipping predictor quantization")
        return

    print(f"\n  Quantizing predictor to int{bits} (group_size={group_size})...")
    weights = load_safetensors(filepath)
    result = quantize_weights(
        weights,
        bits=bits,
        group_size=group_size,
        should_quantize=should_quantize_predictor,
    )
    print(f"  Saving quantized predictor ({len(result)} keys)...")
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
    print(f"Component 1: {ENCODER_COMPONENT} -> {OUTPUT_FILENAME}")
    print(f"Component 2: {PREDICTOR_COMPONENT} -> {PREDICTOR_OUTPUT_FILENAME}")
    print("\nTranspose (encoder only): patch_embed.proj.weight (Conv2d/Conv3d -> channels-last)")
    print("Predictor: all Linear / LayerNorm / bare params, no conv transpose")
    if args.quantize:
        print(f"\nQuantization: int{args.bits}, group_size={args.group_size}")
        print("  Encoder target:   transformer block Linear weights only")
        print("  Predictor target: predictor_blocks.* Linear weights only")
        print("  Skipped:          patch_embed, modality embeds, norms, biases, mask_tokens")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate(args) -> None:
    """Validate converted V-JEPA 2.1 encoder + predictor weights."""
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
    # Predictor is optional — warn but do not hard-fail if absent.
    pred_path = model_dir / PREDICTOR_OUTPUT_FILENAME
    if not pred_path.exists():
        result.check(False, f"{PREDICTOR_OUTPUT_FILENAME} present", warn_only=True)

    # ================================================================== #
    # Encoder
    # ================================================================== #
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
        enc_zero_keys = [
            k
            for k in base_keys
            if k not in KNOWN_ZERO_KEYS and float(mx.abs(weights[k]).sum().item()) == 0.0
        ]
        result.check(
            len(enc_zero_keys) == 0,
            f"no unexpected all-zero encoder tensors "
            f"(found {len(enc_zero_keys)}: {enc_zero_keys[:3]})",
        )

        if is_quantized:
            validate_quantization(weights, result, block_key="blocks")

        del weights
        gc.collect()
        mx.clear_cache()

    # ================================================================== #
    # Predictor
    # ================================================================== #
    if pred_path.exists():
        print("\n== Predictor Weights ==")
        pred_weights = load_safetensors(pred_path)
        pred_keys = set(pred_weights.keys())
        pred_base_keys = {k for k in pred_keys if not k.endswith((".scales", ".biases"))}

        result.check(
            len(pred_base_keys) == PREDICTOR_EXPECTED_KEY_COUNT,
            f"expected {PREDICTOR_EXPECTED_KEY_COUNT} predictor base keys "
            f"(got {len(pred_base_keys)})",
        )

        # predictor_proj shape must be (1664, 384).
        if "predictor_proj.weight" in pred_keys:
            pshape = tuple(pred_weights["predictor_proj.weight"].shape)
            result.check(
                pshape == (1664, 384),
                f"predictor_proj.weight shape (1664, 384) (got {pshape})",
            )

        # predictor_proj_context shape must also be (1664, 384).
        if "predictor_proj_context.weight" in pred_keys:
            pcshape = tuple(pred_weights["predictor_proj_context.weight"].shape)
            result.check(
                pcshape == (1664, 384),
                f"predictor_proj_context.weight shape (1664, 384) (got {pcshape})",
            )

        # 12 predictor_blocks.
        n_pred_blocks = len(count_layer_indices(pred_keys, block_key="predictor_blocks"))
        result.check(
            n_pred_blocks == 12,
            f"12 predictor_blocks (got {n_pred_blocks})",
        )

        # 8 mask tokens present.
        for i in range(8):
            result.check(f"mask_tokens.{i}" in pred_keys, f"mask_tokens.{i} present")

        # No unexpected all-zero tensors. mask_tokens.1-7 are legitimately zero
        # (zero_init from training); mask_tokens.0 and everything else must be non-zero.
        pred_zero_keys = [
            k
            for k in pred_base_keys
            if k not in PREDICTOR_KNOWN_ZERO_KEYS
            and float(mx.abs(pred_weights[k]).sum().item()) == 0.0
        ]
        result.check(
            len(pred_zero_keys) == 0,
            f"no unexpected all-zero predictor tensors "
            f"(found {len(pred_zero_keys)}: {pred_zero_keys[:3]})",
        )

        if is_quantized:
            validate_quantization(pred_weights, result, block_key="predictor_blocks")

        del pred_weights
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
