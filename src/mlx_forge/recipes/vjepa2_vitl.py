"""V-JEPA 2.0 ViT-L — encoder + predictor + all attentive probes conversion recipe.

Architecture (2 always present + up to 3 optional components, local .pt files):

  encoder         (~814 MB, 292 keys)  — V-JEPA 2.0 ViT-L target encoder
                  24 transformer blocks, embed_dim=1024, num_heads=16, RoPE
                  Single PatchEmbed3D conv (5D weight → transpose to channels-last)
                  Single final `norm` LayerNorm — no per-block `norm` wrappers

  predictor       (~88 MB, 160 keys)   — V-JEPA 2.0 world-model predictor
                  12 transformer blocks, predictor_embed_dim=384, RoPE
                  predictor_embed (1024→384), predictor_proj (384→1024), 10 mask_tokens
                  Lives under `predictor` in the SAME vitl.pt as the encoder.
                  No conv: mask_tokens are (1,1,384) embeddings, NOT transposed.

  ssv2_probe      (~134 MB, 51 keys)   — AttentiveClassifier, Something-Something v2
                  pooler.* (1 cross-attn block + 3 self-attn blocks)
                  linear.*  head (174 classes)
                  query_tokens (1, 1, 1024)

  diving48_probe  (~134 MB, 51 keys)   — AttentiveClassifier, Diving-48
                  Same pooler architecture
                  linear.* head (48 classes)
                  query_tokens (1, 1, 1024)

  ek100_probe     (~148 MB, 55 keys)   — AttentiveClassifier, Epic-Kitchens-100
                  Same pooler architecture
                  verb_classifier.* (97 classes) + noun_classifier.* (289 classes)
                  action_classifier.* (3568 classes) — NO single `linear`
                  query_tokens (1, 3, 1024)

Key sanitization:
  encoder:   strip `module.backbone.` prefix
             transpose patch_embed.proj.weight (O,I,D,H,W) → (O,D,H,W,I)
  predictor: strip `module.backbone.` prefix  (no conv, no transpose)
  probes:    strip `module.` prefix  (no conv, no transpose)

Quantization: encoder/predictor block Linears + pooler/blocks Linears.
  Keep full precision: norms, biases, query_tokens, classifier heads (small),
  predictor_embed / predictor_proj / mask_tokens.

Usage (encoder + predictor + all probes):
    mlx-forge convert vjepa2-vitl \\
        --source          ~/Work/.vjepa2-weights/vitl.pt \\
        --ssv2-source     ~/Work/.vjepa2-weights/ssv2-vitl.pt \\
        --diving48-source ~/Work/.vjepa2-weights/diving48-vitl-256.pt \\
        --ek100-source    ~/Work/.vjepa2-weights/ek100-vitl-256.pt \\
        --output          ~/Work/.vjepa2-weights/vjepa-2.0-vitl-mlx

Usage (encoder + predictor only):
    mlx-forge convert vjepa2-vitl --source ~/Work/.vjepa2-weights/vitl.pt

Validate:
    mlx-forge validate vjepa2-vitl ~/Work/.vjepa2-weights/vjepa-2.0-vitl-mlx
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx

from ..convert import fmt_size, load_safetensors
from ..quantize import _materialize, quantize_weights
from ..transpose import transpose_conv
from ..validate import (
    ValidationResult,
    count_layer_indices,
    validate_conv_layout,
    validate_file_exists,
    validate_no_pytorch_prefix,
    validate_quantization,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All possible components (encoder + predictor always present in vitl.pt)
_ALL_COMPONENTS = ["encoder", "predictor", "ssv2_probe", "diving48_probe", "ek100_probe"]

# Probe name → CLI dest attribute name (argparse converts hyphens to underscores)
_PROBE_ARGS: dict[str, str] = {
    "ssv2": "ssv2_source",
    "diving48": "diving48_source",
    "ek100": "ek100_source",
}

# Probe name → output safetensors stem
_PROBE_OUTPUT: dict[str, str] = {
    "ssv2": "ssv2_probe",
    "diving48": "diving48_probe",
    "ek100": "ek100_probe",
}

_COMPONENT_SIZE_MB: dict[str, int] = {
    "encoder": 814,
    "predictor": 88,
    "ssv2_probe": 134,
    "diving48_probe": 134,
    "ek100_probe": 148,
}

# Encoder architecture
_ENCODER_DEPTH = 24
_ENCODER_EMBED_DIM = 1024
_ENCODER_NUM_HEADS = 16
_ENCODER_EXPECTED_KEYS = 292  # non-quantized base keys

# Predictor architecture (V-JEPA 2.0 world model, stored in vitl.pt["predictor"])
_PREDICTOR_DEPTH = 12
_PREDICTOR_EMBED_DIM = 384
_PREDICTOR_EXPECTED_KEYS = 160  # non-quantized base keys

# Probe shared architecture
_PROBE_SELF_ATTN_BLOCKS = 3

# Keys to stay full precision in the encoder
_ENCODER_SKIP_QUANT = (
    "patch_embed",  # single 5D conv — ndim check catches it anyway
    "norm",  # LayerNorm weight/bias
    ".bias",
)

# Keys to stay full precision in the predictor
_PREDICTOR_SKIP_QUANT = (
    "predictor_embed",  # input projection 1024→384
    "predictor_proj",  # output projection 384→1024
    "predictor_norm",  # final LayerNorm
    "mask_tokens",  # learnable (1,1,384) token embeddings, not Linear
    "norm1",  # per-block LayerNorm
    "norm2",
    ".bias",
)

# Keys to stay full precision in any probe
_PROBE_SKIP_QUANT = (
    "query_tokens",  # bare parameter, not a Linear
    "norm",  # LayerNorm weight/bias
    ".bias",
    # Classifier heads — all end with *_classifier.* or linear.*
    # They are small and accuracy-sensitive; keep fp.
    "linear.",
    "verb_classifier.",
    "noun_classifier.",
    "action_classifier.",
)


# ---------------------------------------------------------------------------
# Key sanitization
# ---------------------------------------------------------------------------


def _sanitize_encoder_key(key: str) -> str:
    """Strip `module.backbone.` prefix from encoder keys."""
    prefix = "module.backbone."
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


def _sanitize_probe_key(key: str) -> str:
    """Strip `module.` prefix from probe keys."""
    prefix = "module."
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


# ---------------------------------------------------------------------------
# Weight transforms
# ---------------------------------------------------------------------------


def _encoder_transform(key: str, weight: mx.array) -> mx.array:
    """Transpose the single PatchEmbed3D conv weight (O,I,D,H,W) → (O,D,H,W,I)."""
    if key == "patch_embed.proj.weight" and weight.ndim == 5:
        return transpose_conv(weight)
    return weight


# ---------------------------------------------------------------------------
# Quantization predicates
# ---------------------------------------------------------------------------


def _encoder_should_quantize(key: str, weight: mx.array) -> bool:
    """Quantize only block Linear .weight matrices in the encoder."""
    if weight.ndim != 2 or not key.endswith(".weight"):
        return False
    if any(s in key for s in _ENCODER_SKIP_QUANT):
        return False
    return "blocks." in key


def _predictor_should_quantize(key: str, weight: mx.array) -> bool:
    """Quantize only Linear .weight matrices inside predictor_blocks.*."""
    if weight.ndim != 2 or not key.endswith(".weight"):
        return False
    if any(s in key for s in _PREDICTOR_SKIP_QUANT):
        return False
    return "predictor_blocks." in key


def _probe_should_quantize(key: str, weight: mx.array) -> bool:
    """Quantize only attention/MLP Linear .weight matrices in the pooler blocks."""
    if weight.ndim != 2 or not key.endswith(".weight"):
        return False
    if any(s in key for s in _PROBE_SKIP_QUANT):
        return False
    return "pooler." in key


# ---------------------------------------------------------------------------
# Probe head detection
# ---------------------------------------------------------------------------


def _detect_probe_heads(raw: dict[str, mx.array]) -> dict[str, int]:
    """Detect classifier heads present in a probe and return head→num_classes map.

    Strips the `module.` prefix from raw keys before inspection.
    Recognized head key patterns: `linear`, `verb_classifier`, `noun_classifier`,
    `action_classifier`.
    """
    head_stems = ("linear", "verb_classifier", "noun_classifier", "action_classifier")
    heads: dict[str, int] = {}
    for raw_key, tensor in raw.items():
        k = _sanitize_probe_key(raw_key)
        for stem in head_stems:
            if k == f"{stem}.weight":
                heads[stem] = tensor.shape[0]
    return heads


# ---------------------------------------------------------------------------
# Per-component loaders
# ---------------------------------------------------------------------------


def _load_pt_encoder(source_path: Path) -> dict[str, mx.array]:
    """Load target_encoder from a V-JEPA 2.0 .pt checkpoint."""
    try:
        import torch  # ty: ignore[unresolved-import]
    except ImportError:
        raise SystemExit("torch is required to load .pt files: pip install torch")

    print(f"  Loading encoder from {source_path.name} (torch.load)...")
    ckpt = torch.load(str(source_path), map_location="cpu", weights_only=True)
    if "target_encoder" not in ckpt:
        raise SystemExit(
            f"'target_encoder' key not found in {source_path.name}. "
            f"Available keys: {list(ckpt.keys())}"
        )
    raw: dict = dict(ckpt["target_encoder"])
    print(f"  Loaded {len(raw)} raw keys")
    return {k: mx.array(v.float().numpy()) for k, v in raw.items()}


def _load_pt_predictor(source_path: Path) -> dict[str, mx.array] | None:
    """Load the `predictor` state dict from the SAME vitl.pt as the encoder.

    Returns None (with a warning) if the checkpoint has no `predictor` key, so a
    non-standard checkpoint degrades to encoder-only instead of crashing.
    Uses mmap so only the predictor's ~160 tensors are materialized, not the 5 GB.
    """
    try:
        import torch  # ty: ignore[unresolved-import]
    except ImportError:
        raise SystemExit("torch is required to load .pt files: pip install torch")

    print(f"  Loading predictor from {source_path.name} (torch.load, mmap)...")
    ckpt = torch.load(str(source_path), map_location="cpu", mmap=True, weights_only=True)
    if "predictor" not in ckpt:
        print(f"  [WARN] no 'predictor' key in {source_path.name}; skipping predictor")
        return None
    raw: dict = dict(ckpt["predictor"])
    print(f"  Loaded {len(raw)} raw keys")
    return {k: mx.array(v.float().numpy()) for k, v in raw.items()}


def _load_pt_probe(probe_path: Path) -> dict[str, mx.array]:
    """Load classifiers[0] from a V-JEPA 2.0 probe .pt checkpoint."""
    try:
        import torch  # ty: ignore[unresolved-import]
    except ImportError:
        raise SystemExit("torch is required to load .pt files: pip install torch")

    print(f"  Loading probe from {probe_path.name} (torch.load)...")
    ckpt = torch.load(str(probe_path), map_location="cpu", weights_only=True)
    if "classifiers" not in ckpt or len(ckpt["classifiers"]) == 0:
        raise SystemExit(
            f"'classifiers' list not found or empty in {probe_path.name}. "
            f"Available keys: {list(ckpt.keys())}"
        )
    raw: dict = dict(ckpt["classifiers"][0])
    print(f"  Loaded {len(raw)} raw keys")
    return {k: mx.array(v.float().numpy()) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Per-component converters
# ---------------------------------------------------------------------------


def _process_encoder(source_path: Path, output_dir: Path) -> int:
    """Convert encoder weights → encoder.safetensors. Returns key count."""
    print(f"\n{'=' * 60}")
    print(f"Processing encoder (~{fmt_size(_COMPONENT_SIZE_MB['encoder'])})")
    t0 = time.monotonic()

    raw = _load_pt_encoder(source_path)
    print(f"  Sanitizing {len(raw)} keys...")
    output: dict[str, mx.array] = {}
    for key, weight in raw.items():
        new_key = _sanitize_encoder_key(key)
        weight = _encoder_transform(new_key, weight)
        _materialize(weight)
        output[f"encoder.{new_key}"] = weight

    count = len(output)
    out_path = output_dir / "encoder.safetensors"
    print(f"  Saving {count} weights → {out_path.name}...")
    mx.save_safetensors(str(out_path), output)
    print(f"  Done in {time.monotonic() - t0:.1f}s")

    del output, raw
    gc.collect()
    mx.clear_cache()
    return count


def _process_predictor(source_path: Path, output_dir: Path) -> int:
    """Convert predictor weights → predictor.safetensors. Returns key count (0 if absent).

    Shares the `module.backbone.` sanitizer with the encoder. No conv transpose:
    the only ndim>=3 tensors are mask_tokens (1,1,384) learnable embeddings.
    """
    print(f"\n{'=' * 60}")
    print(f"Processing predictor (~{fmt_size(_COMPONENT_SIZE_MB['predictor'])})")
    t0 = time.monotonic()

    raw = _load_pt_predictor(source_path)
    if raw is None:
        return 0

    print(f"  Sanitizing {len(raw)} keys...")
    output: dict[str, mx.array] = {}
    for key, weight in raw.items():
        new_key = _sanitize_encoder_key(key)  # strips `module.backbone.`
        _materialize(weight)  # no transpose — predictor has no conv weights
        output[f"predictor.{new_key}"] = weight

    count = len(output)
    out_path = output_dir / "predictor.safetensors"
    print(f"  Saving {count} weights → {out_path.name}...")
    mx.save_safetensors(str(out_path), output)
    print(f"  Done in {time.monotonic() - t0:.1f}s")

    del output, raw
    gc.collect()
    mx.clear_cache()
    return count


def _process_probe(
    probe_name: str,
    probe_path: Path,
    output_dir: Path,
) -> tuple[int, dict[str, int]]:
    """Convert a probe checkpoint → <name>_probe.safetensors.

    Returns (key_count, heads_dict) where heads_dict maps head_stem → num_classes.
    """
    comp_name = _PROBE_OUTPUT[probe_name]
    size_mb = _COMPONENT_SIZE_MB.get(comp_name, 134)
    print(f"\n{'=' * 60}")
    print(f"Processing {comp_name} (~{fmt_size(size_mb)})")
    t0 = time.monotonic()

    raw = _load_pt_probe(probe_path)

    # Detect heads BEFORE stripping keys
    heads = _detect_probe_heads(raw)
    print(f"  Detected classifier heads: {heads}")

    print(f"  Sanitizing {len(raw)} keys...")
    output: dict[str, mx.array] = {}
    for key, weight in raw.items():
        new_key = _sanitize_probe_key(key)
        _materialize(weight)
        output[f"{comp_name}.{new_key}"] = weight

    count = len(output)
    out_path = output_dir / f"{comp_name}.safetensors"
    print(f"  Saving {count} weights → {out_path.name}...")
    mx.save_safetensors(str(out_path), output)
    print(f"  Done in {time.monotonic() - t0:.1f}s")

    del output, raw
    gc.collect()
    mx.clear_cache()
    return count, heads


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def _quantize_component(
    comp_name: str,
    should_quantize,
    output_dir: Path,
    bits: int,
    group_size: int,
) -> None:
    filepath = output_dir / f"{comp_name}.safetensors"
    print(f"\n  Quantizing {comp_name} to int{bits} (group_size={group_size})...")
    weights = load_safetensors(filepath)
    result = quantize_weights(
        weights,
        bits=bits,
        group_size=group_size,
        should_quantize=should_quantize,
    )
    print(f"  Saving quantized {comp_name} ({len(result)} keys)...")
    mx.save_safetensors(str(filepath), result)
    del result, weights
    gc.collect()
    mx.clear_cache()


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------


def convert(args) -> None:  # noqa: C901
    """Convert V-JEPA 2.0 ViT-L encoder (+ optional probes) to MLX safetensors."""
    # Resolve encoder source (required)
    if not args.source:
        raise SystemExit(
            "--source <vitl.pt> is required: no public HuggingFace release for V-JEPA 2.0 yet."
        )
    source_path = Path(args.source).expanduser()
    if not source_path.exists():
        raise SystemExit(f"Encoder source not found: {source_path}")

    # Resolve optional probe sources
    probe_paths: dict[str, Path] = {}
    for probe_name, dest_attr in _PROBE_ARGS.items():
        raw_val = getattr(args, dest_attr, None)
        if raw_val:
            p = Path(raw_val).expanduser()
            if not p.exists():
                raise SystemExit(f"Probe source not found: {p}")
            probe_paths[probe_name] = p

    # Resolve output directory
    if args.output:
        output_dir = Path(args.output).expanduser()
    else:
        suffix = f"-q{args.bits}" if args.quantize else ""
        output_dir = Path("models") / f"vjepa-2.0-vitl-mlx{suffix}"

    if args.dry_run:
        _dry_run(args, source_path, probe_paths, output_dir)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Convert encoder
    enc_count = _process_encoder(source_path, output_dir)
    total_weights = enc_count

    # Convert predictor (from the SAME vitl.pt; skipped only if absent)
    pred_count = _process_predictor(source_path, output_dir)
    total_weights += pred_count

    # Convert each provided probe; accumulate heads info for config
    probe_info: dict[str, dict] = {}
    for probe_name, probe_path in probe_paths.items():
        probe_count, heads = _process_probe(probe_name, probe_path, output_dir)
        total_weights += probe_count
        probe_info[probe_name] = {
            "component": _PROBE_OUTPUT[probe_name],
            "heads": heads,
            "key_count": probe_count,
        }

    # Compute COMPONENTS list in insertion order (encoder, predictor, then probes)
    components = ["encoder"]
    if pred_count:
        components.append("predictor")
    components += [info["component"] for info in probe_info.values()]

    # Build config.json
    probes_cfg: dict[str, dict] = {}
    for probe_name, info in probe_info.items():
        probes_cfg[probe_name] = {
            "component": info["component"],
            "heads": info["heads"],
        }

    config: dict = {
        "model_type": "vjepa2-vitl",
        "encoder": {
            "embed_dim": _ENCODER_EMBED_DIM,
            "depth": _ENCODER_DEPTH,
            "num_heads": _ENCODER_NUM_HEADS,
            "use_rope": True,
            "patch_embed": "PatchEmbed3D",
            "patch_embed_kernel": [2, 16, 16],
        },
        "probes": probes_cfg,
        "components": components,
    }
    if pred_count:
        config["predictor"] = {
            "embed_dim": _PREDICTOR_EMBED_DIM,
            "depth": _PREDICTOR_DEPTH,
            "use_rope": True,
        }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("\nSaved config.json")

    # Build split_model.json
    split_info: dict[str, str] = {comp: f"{comp}.safetensors" for comp in components}
    with open(output_dir / "split_model.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print("Saved split_model.json")

    # Optional quantization
    if args.quantize:
        _quantize_component(
            "encoder", _encoder_should_quantize, output_dir, args.bits, args.group_size
        )
        if pred_count:
            _quantize_component(
                "predictor", _predictor_should_quantize, output_dir, args.bits, args.group_size
            )
        for info in probe_info.values():
            _quantize_component(
                info["component"], _probe_should_quantize, output_dir, args.bits, args.group_size
            )

        qconfig: dict = {
            "quantization": {
                "bits": args.bits,
                "group_size": args.group_size,
                "skip_keys_encoder": list(_ENCODER_SKIP_QUANT),
                "skip_keys_predictor": list(_PREDICTOR_SKIP_QUANT),
                "skip_keys_probe": list(_PROBE_SKIP_QUANT),
            }
        }
        with open(output_dir / "quantize_config.json", "w") as f:
            json.dump(qconfig, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Conversion complete: {total_weights} total weights")
    print(f"Output: {output_dir}")
    for p in sorted(output_dir.iterdir()):
        if p.is_file():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  {p.name}: {size_mb:.1f} MB")
    print("Done!")


def _dry_run(
    args,
    source_path: Path,
    probe_paths: dict[str, Path],
    output_dir: Path,
) -> None:
    bits = args.bits if args.quantize else None
    q_label = f" + int{bits} quantization" if bits else ""

    print("=" * 60)
    print(f"DRY RUN — V-JEPA 2.0 ViT-L conversion{q_label}")
    print("=" * 60)
    print(f"\nEncoder source:   {source_path}")
    for probe_name, probe_path in probe_paths.items():
        print(f"{probe_name:14s} source: {probe_path}")
    print(f"Output directory: {output_dir}")

    print("\nComponents:")
    print(f"  encoder: ~{fmt_size(_COMPONENT_SIZE_MB['encoder'])}")
    print(f"  predictor: ~{fmt_size(_COMPONENT_SIZE_MB['predictor'])} (from the same vitl.pt)")
    for probe_name in probe_paths:
        comp = _PROBE_OUTPUT[probe_name]
        print(f"  {comp}: ~{fmt_size(_COMPONENT_SIZE_MB.get(comp, 134))}")

    print("\nKey translations:")
    print("  encoder:   strip 'module.backbone.' prefix")
    print("             transpose patch_embed.proj.weight (O,I,D,H,W) → (O,D,H,W,I)")
    print("  predictor: strip 'module.backbone.' prefix (no conv, no transpose)")
    if probe_paths:
        print("  probes:    strip 'module.' prefix (no conv, no transpose)")
        print("             heads auto-detected from classifier weight shapes")

    if bits:
        print(f"\nQuantization: int{bits}, group_size={args.group_size}")
        print("  encoder:   blocks.*.attn/mlp Linear .weight (skip norms, patch_embed, biases)")
        print("  predictor: predictor_blocks.*.attn/mlp Linear .weight")
        print("             (skip predictor_embed/proj, mask_tokens, norms, biases)")
        print("  probes:    pooler.* Linear .weight (skip heads, query_tokens, norms)")

    print("\n" + "=" * 60)
    print("No files downloaded or written (--dry-run)")


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def validate(args) -> None:  # noqa: C901
    """Validate a converted V-JEPA 2.0 ViT-L model directory."""
    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.exists():
        print(f"ERROR: {model_dir} not found")
        raise SystemExit(1)

    result = ValidationResult()
    is_quantized = (model_dir / "quantize_config.json").exists()
    if is_quantized:
        print("  [INFO] Quantized model detected")

    # Load config to discover which probes were converted
    config_path = model_dir / "config.json"
    components_present: list[str] = []
    probe_cfg: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        components_present = cfg.get("components", [])
        probe_cfg = cfg.get("probes", {})
    else:
        # Fallback: infer from files on disk
        components_present = ["encoder"]
        if (model_dir / "predictor.safetensors").exists():
            components_present.append("predictor")
        for probe_name, comp in _PROBE_OUTPUT.items():
            if (model_dir / f"{comp}.safetensors").exists():
                components_present.append(comp)

    print(f"  Components: {components_present}")

    # --- File structure ---
    print("\n== File Structure ==")
    validate_file_exists(model_dir, "config.json", result)
    validate_file_exists(model_dir, "split_model.json", result)
    for comp in components_present:
        validate_file_exists(model_dir, f"{comp}.safetensors", result)

    # --- Encoder ---
    print("\n== Encoder Weights ==")
    enc_path = model_dir / "encoder.safetensors"
    if enc_path.exists():
        weights = load_safetensors(enc_path)
        keys = set(weights.keys())
        base_keys = {k for k in keys if not k.endswith((".scales", ".biases"))}
        print(f"  Keys: {len(base_keys)} (base)")

        validate_no_pytorch_prefix(weights, "module.", result)
        validate_no_pytorch_prefix(weights, "backbone.", result)

        result.check(
            len(base_keys) == _ENCODER_EXPECTED_KEYS,
            f"encoder key count == {_ENCODER_EXPECTED_KEYS} (got {len(base_keys)})",
        )

        block_indices = count_layer_indices(keys, block_key="blocks")
        result.check(
            len(block_indices) == _ENCODER_DEPTH,
            f"encoder has {_ENCODER_DEPTH} transformer blocks (got {len(block_indices)})",
        )

        norm_keys = [k for k in base_keys if k == "encoder.norm.weight"]
        result.check(len(norm_keys) == 1, "single final norm present (encoder.norm.weight)")
        result.check(
            not any("norms_block" in k for k in base_keys),
            "no norms_block keys (single-norm architecture confirmed)",
        )

        pe_key = "encoder.patch_embed.proj.weight"
        if pe_key in keys:
            w = weights[pe_key]
            result.check(
                w.ndim == 5 and w.shape[-1] == 3,
                f"patch_embed.proj.weight channels-last (O,D,H,W,I), in_channels=3"
                f" (shape={tuple(w.shape)})",
            )
        validate_conv_layout(weights, result, ndim=5)

        # Spot-check: no all-zero base weights
        zero_keys = [
            k
            for k in list(base_keys)[:10]
            if weights[k].ndim >= 1 and float(mx.max(mx.abs(weights[k])).item()) == 0.0
        ]
        result.check(
            len(zero_keys) == 0,
            f"no all-zero weights in spot-check ({len(zero_keys)} found)",
            warn_only=True,
        )

        if is_quantized:
            validate_quantization(weights, result, block_key="blocks")

        total_params = sum(v.size for v in weights.values())
        print(f"  Total encoder parameters: {total_params / 1e6:.1f}M")

        del weights
        gc.collect()
        mx.clear_cache()

    # --- Predictor ---
    if "predictor" in components_present:
        print("\n== Predictor Weights ==")
        pred_path = model_dir / "predictor.safetensors"
        if pred_path.exists():
            weights = load_safetensors(pred_path)
            keys = set(weights.keys())
            base_keys = {k for k in keys if not k.endswith((".scales", ".biases"))}
            print(f"  Keys: {len(base_keys)} (base)")

            validate_no_pytorch_prefix(weights, "module.", result)
            validate_no_pytorch_prefix(weights, "backbone.", result)

            result.check(
                len(base_keys) == _PREDICTOR_EXPECTED_KEYS,
                f"predictor key count == {_PREDICTOR_EXPECTED_KEYS} (got {len(base_keys)})",
            )

            block_indices = count_layer_indices(keys, block_key="predictor_blocks")
            result.check(
                len(block_indices) == _PREDICTOR_DEPTH,
                f"predictor has {_PREDICTOR_DEPTH} blocks (got {len(block_indices)})",
            )

            for stem in ("predictor.predictor_embed.weight", "predictor.predictor_proj.weight"):
                result.check(stem in keys, f"{stem} present")

            # Predictor has no conv: assert nothing got conv-transposed.
            result.check(
                not any(weights[k].ndim >= 4 for k in base_keys),
                "no >=4D tensors in predictor (mask_tokens stay (1,1,D), no conv transpose)",
            )

            if is_quantized:
                validate_quantization(weights, result, block_key="predictor_blocks")

            total_params = sum(v.size for v in weights.values())
            print(f"  Total predictor parameters: {total_params / 1e6:.1f}M")

            del weights
            gc.collect()
            mx.clear_cache()

    # --- Probes (generic, iterate over whatever is present) ---
    # Build a reverse map: comp_name → probe_name (for config lookup)
    comp_to_probe: dict[str, str] = {v: k for k, v in _PROBE_OUTPUT.items()}

    for comp_name in components_present:
        if comp_name in ("encoder", "predictor"):
            continue  # validated in their dedicated blocks above

        probe_name = comp_to_probe.get(comp_name, comp_name)
        print(f"\n== {comp_name} Weights ==")
        probe_path = model_dir / f"{comp_name}.safetensors"
        if not probe_path.exists():
            continue

        weights = load_safetensors(probe_path)
        keys = set(weights.keys())
        base_keys = {k for k in keys if not k.endswith((".scales", ".biases"))}
        print(f"  Keys: {len(base_keys)} (base)")

        validate_no_pytorch_prefix(weights, "module.", result)

        # Cross-attention and self-attention structure
        xattn_q = f"{comp_name}.pooler.cross_attention_block.xattn.q.weight"
        xattn_kv = f"{comp_name}.pooler.cross_attention_block.xattn.kv.weight"
        qt_key = f"{comp_name}.pooler.query_tokens"
        result.check(xattn_q in keys, f"{comp_name}: cross_attention_block.xattn.q.weight present")
        result.check(
            xattn_kv in keys, f"{comp_name}: cross_attention_block.xattn.kv.weight present"
        )
        result.check(qt_key in keys, f"{comp_name}: pooler.query_tokens present")

        sa_indices = count_layer_indices(keys, block_key="pooler.blocks")
        result.check(
            len(sa_indices) == _PROBE_SELF_ATTN_BLOCKS,
            f"{comp_name}: {_PROBE_SELF_ATTN_BLOCKS} self-attn blocks (got {len(sa_indices)})",
        )

        # Classifier head(s) — at least one must be present
        head_stems = ("linear", "verb_classifier", "noun_classifier", "action_classifier")
        found_heads: dict[str, int] = {}
        for stem in head_stems:
            wk = f"{comp_name}.{stem}.weight"
            if wk in keys:
                found_heads[stem] = weights[wk].shape[0]

        head_list = list(found_heads.keys())
        result.check(
            len(found_heads) > 0,
            f"{comp_name}: at least one classifier head present (found: {head_list})",
        )
        print(f"  Classifier heads: {found_heads}")

        # Cross-check against config if available
        if probe_name in probe_cfg:
            expected_heads = probe_cfg[probe_name].get("heads", {})
            for stem, expected_classes in expected_heads.items():
                actual = found_heads.get(stem)
                result.check(
                    actual == expected_classes,
                    f"{comp_name}: {stem}.weight num_classes == {expected_classes} (got {actual})",
                )

        # No all-zero weights (full scan for probes — they're small)
        zero_keys = [
            k
            for k in base_keys
            if weights[k].ndim >= 1 and float(mx.max(mx.abs(weights[k])).item()) == 0.0
        ]
        result.check(
            len(zero_keys) == 0,
            f"{comp_name}: no all-zero weights (found {len(zero_keys)})",
            warn_only=len(zero_keys) <= 2,
        )
        for k in zero_keys:
            print(f"    All-zero key: {k}")

        if is_quantized:
            validate_quantization(weights, result, block_key=["pooler.blocks", "cross_attention"])

        total_params = sum(v.size for v in weights.values())
        print(f"  Total {comp_name} parameters: {total_params / 1e6:.1f}M")

        del weights
        gc.collect()
        mx.clear_cache()

    print("\n" + "=" * 60)
    result.summary()
    if not result.passed:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Argument registration
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
    """Register convert arguments for the vjepa2-vitl recipe."""
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to vitl.pt (V-JEPA 2.0 ViT-L encoder checkpoint). Required.",
    )
    parser.add_argument(
        "--ssv2-source",
        type=str,
        default=None,
        dest="ssv2_source",
        help="Path to ssv2-vitl.pt (Something-Something v2 AttentiveClassifier). Optional.",
    )
    parser.add_argument(
        "--diving48-source",
        type=str,
        default=None,
        dest="diving48_source",
        help="Path to diving48-vitl-256.pt (Diving-48 AttentiveClassifier). Optional.",
    )
    parser.add_argument(
        "--ek100-source",
        type=str,
        default=None,
        dest="ek100_source",
        help="Path to ek100-vitl-256.pt (Epic-Kitchens-100 AttentiveClassifier). Optional.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: models/vjepa-2.0-vitl-mlx[-q<bits>])",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize encoder + probe block Linears after conversion",
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
        help="Preview conversion plan without writing files",
    )


def add_validate_args(parser) -> None:
    """Register validate arguments for the vjepa2-vitl recipe."""
    parser.add_argument("model_dir", type=str, help="Path to converted model directory")


def add_split_args(parser) -> None:
    """Register split arguments (no-op — model is split by component during convert)."""
    parser.add_argument("model_dir", type=str, help="Path to model directory")


def split(args) -> None:
    print("vjepa2-vitl is already split by component during conversion.")
    print("No further splitting needed.")
