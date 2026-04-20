"""ERNIE-Image recipe — text-to-image DiT (~31.6 GB).

ERNIE-Image is Baidu's 8B single-stream DiT for text-to-image generation.
Two variants share the same architecture:
  - baidu/ERNIE-Image       : SFT, 50-step
  - baidu/ERNIE-Image-Turbo : distilled 8-step

Architecture (3 components, per-subdirectory on HuggingFace):
  - transformer  (~16.1 GB, 2 shards) — ErnieImageTransformer2DModel, 36 layers, 4096 hidden
  - text_encoder (~7.7 GB, 1 file)    — Mistral3Model (multimodal; we use only the text path)
  - vae          (~168 MB, 1 file)    — AutoencoderKLFlux2 (Flux 2 VAE, Conv2d-heavy)

Key translations performed at conversion time (matches the translations used by
`ernie-image-mlx` at runtime — see its `tests/parity/test_model_parity.py` for
provenance):

  Transformer (PyTorch → MLX):
    - `x_embedder.proj.weight` (O, I, 1, 1) → squeeze to (O, I) — patch_size=1 Conv2d
      collapses to a Linear on the MLX side.
    - `time_embedding.linear_1.*` → `time_embedding.linear1.*` (drop underscore to
      match the `mlx_arsenal.diffusion.TimestepEmbedding` naming).
    - `time_embedding.linear_2.*` → `time_embedding.linear2.*`
    - `adaLN_modulation.1.*` → `adaLN_modulation.linear.*` (PT Sequential[1] → MLX
      flat module with a `linear` attribute).
    - `layers.N.self_attention.to_out.0.weight` → `layers.N.self_attention.to_out_0.weight`

  VAE (PyTorch → MLX):
    - All Conv2d weights transposed (O, I, H, W) → (O, H, W, I) via the standard
      `mlx_forge.transpose.transpose_conv` helper.
    - `bn.running_mean / running_var` are kept as-is (channels-last BatchNorm stats
      used by the ERNIE-Image pipeline for latent normalization).

  Text encoder (Mistral3):
    - `vision_tower.*` and `multi_modal_projector.*` keys dropped (text-only path
      consumes only `language_model.*` keys).
    - No further renames — mlx-lm's `mistral3.Model.sanitize` expects the
      language_model prefix, which is already present in the HF checkpoint.

Quantization: transformer Linears only. VAE and text_encoder stay in fp16/bf16.
"""

from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path

import mlx.core as mx

from ..convert import (
    download_hf_files,
    fmt_size,
    load_weights,
    process_component,
    quantize_component,
)
from ..transpose import needs_transpose, transpose_conv
from ..validate import (
    ValidationResult,
    count_layer_indices,
    validate_file_exists,
    validate_no_pytorch_prefix,
    validate_quantization,
)

# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

REPO_SFT = "baidu/ERNIE-Image"
REPO_TURBO = "baidu/ERNIE-Image-Turbo"

# Both variants publish the same file layout.
COMPONENTS = ["transformer", "text_encoder", "vae"]

COMPONENT_PREFIX: dict[str, str] = {
    "transformer": "transformer",
    "text_encoder": "text_encoder",
    "vae": "vae",
}

_COMPONENT_SIZE_MB: dict[str, int] = {
    "transformer": 16100,
    "text_encoder": 7700,
    "vae": 168,
}

_CHECKPOINT_SIZE_MB = 31600

_TRANSFORMER_SHARDS = 2

TRANSFORMER_FILES = [
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
] + [
    f"transformer/diffusion_pytorch_model-{i:05d}-of-{_TRANSFORMER_SHARDS:05d}.safetensors"
    for i in range(1, _TRANSFORMER_SHARDS + 1)
]

TEXT_ENCODER_FILES = [
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
]

VAE_FILES = [
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

CONFIG_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "pe_tokenizer/tokenizer_config.json",
]

# Baidu's ERNIE-Image repo only publishes `tokenizer_config.json` for the text
# encoder — the actual tokenizer.json / special_tokens_map.json live upstream.
# Ministral3 + Pixtral tokenizers are identical (Tekken, vocab_size=131072) —
# pull the community Pixtral repo which is public and Apache 2.0.
TOKENIZER_SOURCE_REPO = "mistral-community/pixtral-12b"
TOKENIZER_SOURCE_FILES = ["tokenizer.json", "special_tokens_map.json"]

ALL_CHECKPOINT_FILES = TRANSFORMER_FILES + TEXT_ENCODER_FILES + VAE_FILES + CONFIG_FILES

# ---------------------------------------------------------------------------
# Sanitization — translates PyTorch keys to MLX names (matches runtime loader)
# ---------------------------------------------------------------------------


def _sanitize_transformer_key(key: str) -> str:
    """Rename a transformer key from diffusers PT naming to ernie-image-mlx MLX naming."""
    # TimestepEmbedding: PT `linear_1/_2` → mlx-arsenal `linear1/2`
    key = key.replace("time_embedding.linear_1.", "time_embedding.linear1.")
    key = key.replace("time_embedding.linear_2.", "time_embedding.linear2.")

    # AdaLN modulation: PT nn.Sequential index [1] → MLX flat attribute `linear`
    key = key.replace("adaLN_modulation.1.", "adaLN_modulation.linear.")

    # Attention output: PT ModuleList[0] → MLX flat name
    key = key.replace(".self_attention.to_out.0.", ".self_attention.to_out_0.")

    return key


def _sanitize_identity(key: str) -> str:
    """No-op — VAE and text_encoder keys are already clean."""
    return key


SANITIZERS: dict[str, object] = {
    "transformer": _sanitize_transformer_key,
    "text_encoder": _sanitize_identity,
    "vae": _sanitize_identity,
}


# ---------------------------------------------------------------------------
# Value transforms (conv transpose + patch-embed squeeze)
# ---------------------------------------------------------------------------


def _transformer_transform(key: str, weight: mx.array, component_name: str) -> mx.array:
    """Squeeze the patch-embed conv weight to a Linear.

    PT `x_embedder.proj.weight` has shape `(hidden_size, in_channels, 1, 1)` because
    the reference uses `Conv2d(kernel=1)`. At `patch_size=1` this is a pointwise
    Linear — the MLX port uses `nn.Linear` to keep the runtime path simpler, so
    we drop the two trailing singleton axes here.
    """
    if key == "x_embedder.proj.weight":
        if weight.ndim == 4 and weight.shape[2] == 1 and weight.shape[3] == 1:
            return weight.squeeze(axis=(2, 3))
    return weight


def _vae_transform(key: str, weight: mx.array, component_name: str) -> mx.array:
    """Transpose Conv2d weights (O,I,H,W) → (O,H,W,I). BN running stats pass through."""
    if needs_transpose(key, weight):
        return transpose_conv(weight)
    return weight


TRANSFORMS: dict[str, object] = {
    "transformer": _transformer_transform,
    "text_encoder": None,
    "vae": _vae_transform,
}


# ---------------------------------------------------------------------------
# Text-encoder key filter (drop vision tower + projector)
# ---------------------------------------------------------------------------

_TEXT_ENCODER_SKIP_PREFIXES = (
    "vision_tower.",
    "multi_modal_projector.",
)


def _filter_text_encoder_keys(weights: dict) -> dict:
    """Drop keys belonging to the Pixtral vision tower (we only run the text path)."""
    return {
        k: v
        for k, v in weights.items()
        if not any(k.startswith(p) for p in _TEXT_ENCODER_SKIP_PREFIXES)
    }


# ---------------------------------------------------------------------------
# Quantization predicate
# ---------------------------------------------------------------------------

# Quantize transformer + text_encoder; VAE stays fp16 (conv-heavy).
_SKIP_QUANTIZE_COMPONENTS = {"vae"}

_SKIP_QUANTIZE_KEYS = [
    # Small projection Linears — quantizing hurts quality for little size win.
    "x_embedder.",
    "text_proj.",
    "time_embedding.",
    "adaLN_modulation.",
    "final_norm.",
    "final_linear.",
    "pos_embed.",
    # All norms.
    "norm",
    ".bias",
]


def ernie_image_should_quantize(key: str, weight: mx.array) -> bool:
    """Quantize only the block Linear weights (attention + FFN)."""
    if weight.ndim < 2:
        return False
    if any(skip in key for skip in _SKIP_QUANTIZE_KEYS):
        return False
    return weight.shape[0] >= 256 and weight.shape[1] >= 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_repo_id(args) -> str:
    variant = (getattr(args, "variant", None) or "turbo").lower()
    if variant == "sft":
        return REPO_SFT
    if variant == "turbo":
        return REPO_TURBO
    raise SystemExit(f"Unknown variant {variant!r} (expected 'sft' or 'turbo')")


def _default_output_dir(variant: str, quantize: bool, bits: int) -> Path:
    suffix = f"-q{bits}" if quantize else ""
    return Path("models") / f"ernie-image-{variant}-mlx{suffix}"


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run(args) -> None:
    variant = (getattr(args, "variant", None) or "turbo").lower()
    repo_id = _resolve_repo_id(args)
    bits = args.bits if args.quantize else None
    q_label = f" + int{bits} quantization" if bits else ""

    print("=" * 60)
    print(f"DRY RUN — ERNIE-Image conversion plan ({variant}){q_label}")
    print("=" * 60)
    print(f"\nSource: {repo_id}")
    print(f"Total download: ~{fmt_size(_CHECKPOINT_SIZE_MB)}")
    print(f"Files to download: {len(ALL_CHECKPOINT_FILES)}")

    output_dir = args.output or _default_output_dir(variant, args.quantize, args.bits)
    print(f"Output: {output_dir}")

    print("\nComponents:")
    for comp in COMPONENTS:
        size = _COMPONENT_SIZE_MB[comp]
        skip = comp in _SKIP_QUANTIZE_COMPONENTS
        q_note = " (skip quantize)" if skip and bits else ""
        print(f"  {comp}: ~{fmt_size(size)}{q_note}")

    if bits:
        print(f"\nQuantization: int{bits}, group_size={args.group_size}")
        print("  Quantized: transformer, text_encoder (Linear weights only)")
        print("  Skipped: vae (conv-heavy)")

    print("\nKey translations:")
    print("  transformer: x_embedder conv → Linear squeeze, AdaLN flatten,")
    print("               TimestepEmbedding name alignment (linear_N → linearN),")
    print("               to_out.0 → to_out_0")
    print("  vae:         Conv2d layout transpose (O,I,H,W) → (O,H,W,I)")
    print("  text_encoder: drop vision_tower.* and multi_modal_projector.* keys")

    print("\n" + "=" * 60)
    print("No files downloaded or written (--dry-run)")


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------


def convert(args) -> None:
    if args.dry_run:
        dry_run(args)
        return

    variant = (getattr(args, "variant", None) or "turbo").lower()
    repo_id = _resolve_repo_id(args)

    output_dir = (
        Path(args.output) if args.output else _default_output_dir(variant, args.quantize, args.bits)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Variant: {variant}")
    print(f"Output directory: {output_dir}")

    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
        print(f"Using local checkpoint: {checkpoint_dir}")
    else:
        checkpoint_dir = Path("models") / f"ernie-image-{variant}-src"
        print(f"Downloading {repo_id} checkpoint files...")
        print(f"(~{fmt_size(_CHECKPOINT_SIZE_MB)}, may take a while)")
        download_hf_files(repo_id, ALL_CHECKPOINT_FILES, checkpoint_dir)

    total_weights = 0
    for comp_name in COMPONENTS:
        comp_subdir = checkpoint_dir / comp_name
        print("\n" + "=" * 60)
        print(f"Processing {comp_name} (~{fmt_size(_COMPONENT_SIZE_MB[comp_name])})")

        if comp_name == "transformer":
            weights = load_weights(
                comp_subdir,
                index_filename="diffusion_pytorch_model.safetensors.index.json",
            )
        elif comp_name == "text_encoder":
            weights = load_weights(comp_subdir, single_filename="model.safetensors")
            weights = _filter_text_encoder_keys(weights)
        else:
            weights = load_weights(
                comp_subdir, single_filename="diffusion_pytorch_model.safetensors"
            )

        count = process_component(
            weights,
            comp_name,
            list(weights.keys()),
            output_dir,
            COMPONENT_PREFIX[comp_name],
            sanitizer=SANITIZERS[comp_name],
            transform=TRANSFORMS[comp_name],
        )
        total_weights += count

        del weights
        gc.collect()
        mx.clear_cache()

    # Copy config files
    for config_file in CONFIG_FILES:
        src = checkpoint_dir / config_file
        if src.exists():
            dest = output_dir / Path(config_file).name
            if "/" in config_file:
                prefix = config_file.split("/")[0]
                dest = output_dir / f"{prefix}_{Path(config_file).name}"
            shutil.copy2(src, dest)
    for comp_name in COMPONENTS:
        src = checkpoint_dir / comp_name / "config.json"
        if src.exists():
            shutil.copy2(src, output_dir / f"{comp_name}_config.json")

    # Pull tokenizer files from the upstream Pixtral repo — Baidu only publishes
    # tokenizer_config.json, not the actual `tokenizer.json`. Community Pixtral
    # ships the exact same tokenizer (Tekken, vocab_size=131072).
    print(f"\n  Fetching tokenizer files from {TOKENIZER_SOURCE_REPO}...")
    from huggingface_hub import hf_hub_download

    for fname in TOKENIZER_SOURCE_FILES:
        try:
            p = hf_hub_download(TOKENIZER_SOURCE_REPO, fname)
            shutil.copy2(p, output_dir / fname)
            print(f"    {fname}")
        except Exception as exc:  # noqa: BLE001
            print(f"    WARN: could not fetch {fname}: {exc}")

    # Quantize
    if args.quantize:
        for component_name in COMPONENTS:
            if component_name in _SKIP_QUANTIZE_COMPONENTS:
                print(f"\n  Skipping quantization for {component_name}")
                continue
            quantize_component(
                output_dir,
                component_name,
                bits=args.bits,
                group_size=args.group_size,
                should_quantize=ernie_image_should_quantize,
            )
        qconfig = {
            "quantization": {
                "bits": args.bits,
                "group_size": args.group_size,
                "skip_components": sorted(_SKIP_QUANTIZE_COMPONENTS),
            }
        }
        with open(output_dir / "quantize_config.json", "w") as f:
            json.dump(qconfig, f, indent=2)

    split_info: dict = {
        "format": "split",
        "source": repo_id,
        "variant": variant,
        "components": COMPONENTS,
    }
    if args.quantize:
        split_info["quantized"] = True
        split_info["quantization_bits"] = args.bits
    with open(output_dir / "split_model.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Conversion complete: {total_weights} total weights")
    print(f"Output: {output_dir}")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.name}: {size_mb:.1f} MB")

    print("Done!")


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate(args) -> None:
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: {model_dir} not found")
        raise SystemExit(1)

    result = ValidationResult()
    is_quantized = (model_dir / "quantize_config.json").exists()

    print("\n== File Structure ==")
    validate_file_exists(model_dir, "split_model.json", result)
    for comp_name in COMPONENTS:
        validate_file_exists(model_dir, f"{comp_name}.safetensors", result)
    validate_file_exists(model_dir, "transformer_config.json", result)

    print("\n== Transformer Weights ==")
    tf_path = model_dir / "transformer.safetensors"
    if tf_path.exists():
        weights = mx.load(str(tf_path))
        keys = set(weights.keys())
        print(f"  Keys: {len(keys)}")
        validate_no_pytorch_prefix(weights, "transformer.transformer.", result)

        block_indices = count_layer_indices(keys, block_key="layers")
        result.check(
            len(block_indices) >= 1,
            f"Transformer blocks present ({len(block_indices)} blocks)",
        )

        result.check(
            any("x_embedder.proj.weight" in k for k in keys),
            "x_embedder.proj.weight present",
        )
        result.check(
            any("time_embedding.linear1." in k for k in keys),
            "time_embedding.linear1 present (renamed)",
        )
        result.check(
            any("adaLN_modulation.linear." in k for k in keys),
            "adaLN_modulation.linear present (flattened)",
        )
        result.check(
            any(".self_attention.to_out_0." in k for k in keys),
            "to_out_0 present (renamed from to_out.0)",
        )

        if is_quantized:
            validate_quantization(weights, result, block_key="layers")

    print("\n== Text-encoder Weights ==")
    te_path = model_dir / "text_encoder.safetensors"
    if te_path.exists():
        weights = mx.load(str(te_path))
        keys = set(weights.keys())
        result.check(
            not any(k.startswith("vision_tower.") for k in keys),
            "vision_tower.* keys dropped",
        )
        result.check(
            not any(k.startswith("multi_modal_projector.") for k in keys),
            "multi_modal_projector.* keys dropped",
        )

    print("\n== VAE Weights ==")
    vae_path = model_dir / "vae.safetensors"
    if vae_path.exists():
        weights = mx.load(str(vae_path))
        keys = set(weights.keys())
        print(f"  Keys: {len(keys)}")

        # Structural keys (the recipe prepends `vae.` to every key).
        result.check(
            "vae.encoder.conv_in.weight" in keys,
            "encoder.conv_in.weight present",
        )
        result.check(
            "vae.decoder.conv_out.weight" in keys,
            "decoder.conv_out.weight present",
        )
        result.check(
            "vae.quant_conv.weight" in keys and "vae.post_quant_conv.weight" in keys,
            "quant_conv + post_quant_conv present",
        )
        result.check(
            any("vae.encoder.mid_block.attentions.0." in k for k in keys),
            "encoder mid-block self-attention present",
        )

        # Channels-last Conv2d layout: input-channels is the last axis.
        k = "vae.encoder.conv_in.weight"
        if k in keys:
            w = weights[k]
            result.check(
                w.ndim == 4 and w.shape[-1] == 3,
                f"encoder.conv_in channels-last, in=3 ({tuple(w.shape)})",
            )
        k = "vae.decoder.conv_out.weight"
        if k in keys:
            w = weights[k]
            result.check(
                w.ndim == 4 and w.shape[0] == 3,
                f"decoder.conv_out produces 3 channels ({tuple(w.shape)})",
            )

        # Latent BatchNorm stats used by the pipeline for DiT-space renormalization.
        result.check(
            "vae.bn.running_mean" in keys and "vae.bn.running_var" in keys,
            "bn.running_mean + bn.running_var present (latent normalization)",
        )
        mean = weights.get("vae.bn.running_mean")
        var = weights.get("vae.bn.running_var")
        if mean is not None and var is not None:
            result.check(
                mean.shape == var.shape,
                f"bn stats aligned (shape={tuple(mean.shape)})",
            )
            # Unbroken running stats are never the 0/1 init defaults.
            import mlx.core as mx_

            is_init_mean = bool(mx_.all(mean == 0).item())
            is_init_var = bool(mx_.all(var == 1).item())
            result.check(
                not (is_init_mean and is_init_var),
                "bn stats are trained (non-default)",
            )

    print("\n" + "=" * 60)
    result.summary()


# ---------------------------------------------------------------------------
# Argument registration
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
    parser.add_argument(
        "--variant",
        choices=["sft", "turbo"],
        default="turbo",
        help="Which variant to convert: 'sft' (50-step) or 'turbo' (8-step, default)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to local checkpoint directory (default: download from HuggingFace)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: models/ernie-image-<variant>-mlx[-q<bits>])",
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Quantize transformer after conversion"
    )
    parser.add_argument(
        "--bits", type=int, default=8, choices=[4, 8], help="Quantization bits (default: 8)"
    )
    parser.add_argument(
        "--group-size", type=int, default=64, help="Quantization group size (default: 64)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview conversion plan")


def add_validate_args(parser) -> None:
    parser.add_argument("model_dir", type=str, help="Path to converted model directory")


def add_split_args(parser) -> None:
    parser.add_argument("model_dir", type=str, help="Path to model directory")


def split(args) -> None:
    print("ERNIE-Image is already split by component during conversion.")
    print("No further splitting needed.")
