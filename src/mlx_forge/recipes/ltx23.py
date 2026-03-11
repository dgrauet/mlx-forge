"""LTX-2.3 conversion recipe.

Converts the official Lightricks/LTX-2.3 PyTorch checkpoint to MLX split format.
Handles: transformer, connector, VAE decoder/encoder, audio VAE, vocoder.

Usage:
    mlx-forge convert ltx23
    mlx-forge convert ltx23 --quantize --bits 8
    mlx-forge validate ltx23 ~/.cache/huggingface/hub/ltx23-mlx
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download

from ..quantize import _materialize, quantize_weights
from ..transpose import transpose_conv
from ..validate import (
    ValidationResult,
    validate_conv_layout,
    validate_file_exists,
    validate_no_pytorch_prefix,
    validate_quantization,
)

# ---------------------------------------------------------------------------
# Component classification
# ---------------------------------------------------------------------------

COMPONENTS = ["transformer", "connector", "vae_decoder", "vae_encoder", "audio_vae", "vocoder"]

COMPONENT_PREFIX = {
    "transformer": "transformer",
    "connector": "connector",
    "vae_decoder": "vae_decoder",
    "vae_encoder": "vae_encoder",
    "audio_vae": "audio_vae",
    "vocoder": "vocoder",
}


def classify_key(key: str) -> str | None:
    """Classify a PyTorch weight key into a component name.

    Returns one of: transformer, connector, vae_decoder, vae_encoder,
    audio_vae, vocoder, vae_shared_stats, or None (skip).
    """
    if key.startswith("model.diffusion_model."):
        suffix = key[len("model.diffusion_model."):]
        if suffix.startswith("video_embeddings_connector.") or suffix.startswith(
            "audio_embeddings_connector."
        ):
            return "connector"
        return "transformer"
    if key.startswith("vae.per_channel_statistics."):
        return "vae_shared_stats"
    if key.startswith("vae.encoder."):
        return "vae_encoder"
    if key.startswith("vae.decoder."):
        return "vae_decoder"
    if key.startswith("audio_vae."):
        return "audio_vae"
    if key.startswith("vocoder."):
        return "vocoder"
    if key.startswith("text_embedding_projection."):
        return "connector"
    return None


# ---------------------------------------------------------------------------
# Key sanitization
# ---------------------------------------------------------------------------


def sanitize_transformer_key(key: str) -> str:
    """Convert a PyTorch transformer key to MLX format."""
    k = key.replace("model.diffusion_model.", "")
    k = k.replace(".to_out.0.", ".to_out.")
    k = k.replace(".ff.net.0.proj.", ".ff.proj_in.")
    k = k.replace(".ff.net.2.", ".ff.proj_out.")
    k = k.replace(".audio_ff.net.0.proj.", ".audio_ff.proj_in.")
    k = k.replace(".audio_ff.net.2.", ".audio_ff.proj_out.")
    k = k.replace(".linear_1.", ".linear1.")
    k = k.replace(".linear_2.", ".linear2.")
    return k


def sanitize_connector_key(key: str) -> str:
    """Convert a PyTorch connector/text_embedding key to MLX format."""
    if key.startswith("model.diffusion_model."):
        return key.replace("model.diffusion_model.", "")
    return key


def sanitize_vae_decoder_key(key: str) -> str | None:
    """Convert a PyTorch VAE decoder key to MLX format."""
    if key.startswith("vae.per_channel_statistics."):
        if "mean-of-means" in key:
            return "per_channel_statistics.mean"
        if "std-of-means" in key:
            return "per_channel_statistics.std"
        return None
    if key.startswith("vae.decoder."):
        return key.replace("vae.decoder.", "")
    return None


def sanitize_vae_encoder_key(key: str) -> str | None:
    """Convert a PyTorch VAE encoder key to MLX format."""
    if key.startswith("vae.per_channel_statistics."):
        if "mean-of-means" in key:
            return "per_channel_statistics._mean_of_means"
        if "std-of-means" in key:
            return "per_channel_statistics._std_of_means"
        return None
    if key.startswith("vae.encoder."):
        return key.replace("vae.encoder.", "")
    return None


def sanitize_audio_vae_key(key: str) -> str | None:
    """Convert a PyTorch audio VAE key to MLX format."""
    if key.startswith("audio_vae.decoder."):
        return key.replace("audio_vae.decoder.", "")
    if key.startswith("audio_vae.per_channel_statistics."):
        if "mean-of-means" in key:
            return "per_channel_statistics._mean_of_means"
        if "std-of-means" in key:
            return "per_channel_statistics._std_of_means"
        return None
    return None


def sanitize_vocoder_key(key: str) -> str | None:
    """Convert a PyTorch vocoder key to MLX format."""
    if key.startswith("vocoder."):
        return key.replace("vocoder.", "")
    return None


SANITIZERS = {
    "transformer": sanitize_transformer_key,
    "connector": sanitize_connector_key,
    "vae_decoder": sanitize_vae_decoder_key,
    "vae_encoder": sanitize_vae_encoder_key,
    "audio_vae": sanitize_audio_vae_key,
    "vocoder": sanitize_vocoder_key,
}


# ---------------------------------------------------------------------------
# Conv transposition
# ---------------------------------------------------------------------------


def maybe_transpose(key: str, value: mx.array, component: str) -> mx.array:
    """Transpose conv weights from PyTorch to MLX layout if needed."""
    if component == "transformer":
        return value  # All Linear, no conv

    is_conv = "conv" in key.lower() and "weight" in key
    if not is_conv:
        return value

    is_conv_transpose = component == "vocoder" and "ups" in key
    return transpose_conv(value, is_conv_transpose=is_conv_transpose)


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------


def extract_config(checkpoint_path: str) -> dict:
    """Read model config from safetensors file metadata."""
    _, metadata = mx.load(checkpoint_path, return_metadata=True)

    model_version = metadata.get("model_version", "unknown")
    is_v2 = model_version.startswith("2.3")

    config = {
        "model_version": model_version,
        "is_v2": is_v2,
        "model_type": "AudioVideo",
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "in_channels": 128,
        "out_channels": 128,
        "num_layers": 48,
        "cross_attention_dim": 4096,
        "caption_channels": None if is_v2 else 3840,
        "apply_gated_attention": is_v2,
        "cross_attention_adaln": is_v2,
        "audio_num_attention_heads": 32,
        "audio_attention_head_dim": 64,
        "audio_in_channels": 128,
        "audio_out_channels": 128,
        "audio_cross_attention_dim": 2048,
        "positional_embedding_theta": 10000.0,
        "positional_embedding_max_pos": [20, 2048, 2048],
        "audio_positional_embedding_max_pos": [20],
        "timestep_scale_multiplier": 1000,
        "av_ca_timestep_scale_multiplier": 1000,
        "norm_eps": 1e-6,
        # Connector defaults — CRITICAL: wrong defaults cause scrambled text embeddings.
        # Embeddings1DConnector needs these exact values for correct RoPE frequencies.
        # positional_embedding_max_pos=[1] or rope_type=INTERLEAVED → model ignores prompts.
        "connector_positional_embedding_max_pos": [4096],
        "connector_rope_type": "SPLIT",
    }

    if "config" in metadata:
        try:
            embedded = json.loads(metadata["config"])
            config["embedded_config"] = embedded
        except json.JSONDecodeError:
            pass

    return config


# ---------------------------------------------------------------------------
# Component processing
# ---------------------------------------------------------------------------


def process_component(
    checkpoint_weights: dict,
    component_name: str,
    keys: list[str],
    output_dir: Path,
    component_prefix: str,
) -> int:
    """Process one component: extract keys, sanitize, transpose, save.

    Returns number of weights saved.
    """
    sanitizer = SANITIZERS[component_name]
    component_weights = {}

    print(f"  Processing {len(keys)} tensors...")
    for key in keys:
        new_key = sanitizer(key)
        if new_key is None:
            continue

        weight = checkpoint_weights[key]
        weight = maybe_transpose(new_key, weight, component_name)

        # Force-materialize — mx.save_safetensors may not handle lazy cross-file refs
        _materialize(weight)
        component_weights[f"{component_prefix}.{new_key}"] = weight

    if not component_weights:
        print(f"  No weights for {component_name}, skipping")
        return 0

    count = len(component_weights)
    output_file = output_dir / f"{component_name}.safetensors"
    print(f"  Saving {count} weights to {output_file.name}...")
    mx.save_safetensors(str(output_file), component_weights)

    del component_weights
    gc.collect()
    mx.clear_cache()
    return count


def process_shared_stats(
    checkpoint_weights: dict,
    keys: list[str],
    output_dir: Path,
) -> None:
    """Load shared VAE per_channel_statistics and append to decoder/encoder files."""
    for key in keys:
        stat_tensor = checkpoint_weights[key]
        _materialize(stat_tensor)

        # Append to decoder
        decoder_path = output_dir / "vae_decoder.safetensors"
        if decoder_path.exists():
            decoder_weights = mx.load(str(decoder_path))
            for k in decoder_weights:
                _materialize(decoder_weights[k])
        else:
            decoder_weights = {}

        if "mean-of-means" in key:
            decoder_weights["vae_decoder.per_channel_statistics.mean"] = stat_tensor
        elif "std-of-means" in key:
            decoder_weights["vae_decoder.per_channel_statistics.std"] = stat_tensor
        mx.save_safetensors(str(decoder_path), decoder_weights)
        del decoder_weights

        # Append to encoder
        encoder_path = output_dir / "vae_encoder.safetensors"
        if encoder_path.exists():
            encoder_weights = mx.load(str(encoder_path))
            for k in encoder_weights:
                _materialize(encoder_weights[k])
        else:
            encoder_weights = {}

        if "mean-of-means" in key:
            encoder_weights["vae_encoder.per_channel_statistics._mean_of_means"] = stat_tensor
        elif "std-of-means" in key:
            encoder_weights["vae_encoder.per_channel_statistics._std_of_means"] = stat_tensor
        mx.save_safetensors(str(encoder_path), encoder_weights)
        del encoder_weights

        del stat_tensor
        gc.collect()
        mx.clear_cache()


# ---------------------------------------------------------------------------
# Quantization (LTX-2.3 specific)
# ---------------------------------------------------------------------------


def ltx23_should_quantize(key: str, weight: mx.array) -> bool:
    """Only quantize transformer_blocks Linear weights."""
    bare_key = key.replace("transformer.", "", 1)
    return (
        "transformer_blocks" in bare_key
        and bare_key.endswith(".weight")
        and weight.ndim == 2
        and not bare_key.endswith(".scales")
        and not bare_key.endswith(".biases")
    )


def quantize_transformer(output_dir: Path, *, bits: int = 8, group_size: int = 64) -> None:
    """Quantize transformer weights in-place."""
    tf_file = output_dir / "transformer.safetensors"
    if not tf_file.exists():
        print("ERROR: transformer.safetensors not found")
        return

    print(f"\nQuantizing transformer to int{bits} (group_size={group_size})...")
    weights = mx.load(str(tf_file))

    result = quantize_weights(
        weights, bits=bits, group_size=group_size, should_quantize=ltx23_should_quantize,
    )

    print(f"  Saving quantized transformer ({len(result)} keys)...")
    mx.save_safetensors(str(tf_file), result)

    qconfig = {
        "quantization": {
            "bits": bits,
            "group_size": group_size,
            "only_transformer_blocks": True,
        }
    }
    with open(output_dir / "quantize_config.json", "w") as f:
        json.dump(qconfig, f, indent=2)

    del result, weights
    gc.collect()
    mx.clear_cache()
    print("  Quantization complete")


# ---------------------------------------------------------------------------
# Main convert entry point
# ---------------------------------------------------------------------------


def convert(args) -> None:
    """Convert LTX-2.3 PyTorch checkpoint to MLX split format."""
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path.home() / ".cache/huggingface/hub" / f"ltx23-mlx-{args.variant}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        print(f"Using local checkpoint: {checkpoint_path}")
    else:
        filename = f"ltx-2.3-22b-{args.variant}.safetensors"
        print(f"Downloading {filename} from Lightricks/LTX-2.3...")
        print("(This is ~46 GB, may take a while)")
        checkpoint_path = hf_hub_download(repo_id="Lightricks/LTX-2.3", filename=filename)
        print(f"Downloaded to: {checkpoint_path}")

    # Step 2: Extract config
    print("\nExtracting config from safetensors metadata...")
    config = extract_config(checkpoint_path)
    print(f"  Model version: {config['model_version']}")
    print(f"  Is V2 (2.3): {config['is_v2']}")
    print(f"  Gated attention: {config['apply_gated_attention']}")

    config_out = {k: v for k, v in config.items() if k != "embedded_config"}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_out, f, indent=2)
    if "embedded_config" in config:
        with open(output_dir / "embedded_config.json", "w") as f:
            json.dump(config["embedded_config"], f, indent=2)

    # Step 3: Load weights lazily
    print("\nLoading weights lazily via mx.load...")
    t0 = time.monotonic()
    checkpoint_weights = mx.load(checkpoint_path)
    print(f"  {len(checkpoint_weights)} keys loaded (lazy)")

    # Classify keys
    print("\nClassifying weight keys...")
    keys_by_component: dict[str, list[str]] = {}
    for key in checkpoint_weights:
        comp = classify_key(key)
        if comp:
            keys_by_component.setdefault(comp, []).append(key)

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
            checkpoint_weights, component_name, keys, output_dir, component_prefix,
        )
        elapsed = time.monotonic() - t0
        total_weights += count
        print(f"  Done: {count} weights saved in {elapsed:.1f}s")

    # Handle shared stats
    shared_keys = keys_by_component.get("vae_shared_stats", [])
    if shared_keys:
        print(f"\n[shared stats] Processing {len(shared_keys)} per_channel_statistics keys...")
        process_shared_stats(checkpoint_weights, shared_keys, output_dir)

    del checkpoint_weights
    gc.collect()
    mx.clear_cache()

    # Step 5: Create split_model.json
    split_info = {
        "format": "split",
        "model_version": config["model_version"],
        "components": COMPONENTS,
        "source": "Lightricks/LTX-2.3",
        "variant": args.variant,
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
        quantize_transformer(output_dir, bits=args.bits, group_size=args.group_size)

        split_info["quantized"] = True
        split_info["quantization_bits"] = args.bits
        with open(output_dir / "split_model.json", "w") as f:
            json.dump(split_info, f, indent=2)

        print("\nFinal files after quantization:")
        for p in sorted(output_dir.iterdir()):
            if p.is_file():
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"  {p.name}: {size_mb:.1f} MB")

    print("\nDone!")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate(args) -> None:
    """Validate a converted LTX-2.3 model."""
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

    # File structure
    print("\n== File Structure ==")
    expected = [
        "config.json", "split_model.json", "transformer.safetensors",
        "connector.safetensors", "vae_decoder.safetensors", "vae_encoder.safetensors",
        "audio_vae.safetensors", "vocoder.safetensors",
    ]
    for fname in expected:
        validate_file_exists(model_dir, fname, result)
    for fname in ["quantize_config.json", "embedded_config.json"]:
        if (model_dir / fname).exists():
            print(f"  \033[92m\u2713\033[0m {fname} exists (optional)")

    # Config
    print("\n== Config Validation ==")
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        result.check(config.get("model_version", "").startswith("2.3"),
                     f"Model version is 2.3.x (got: {config.get('model_version')})")
        result.check(config.get("is_v2") is True, "is_v2 flag is True")
        result.check(config.get("apply_gated_attention") is True, "apply_gated_attention is True")
        result.check(config.get("cross_attention_adaln") is True, "cross_attention_adaln is True")
        result.check(config.get("caption_channels") is None, "caption_channels is None (V2)")
        result.check(config.get("num_layers") == 48,
                     f"num_layers == 48 (got: {config.get('num_layers')})")
        result.check(config.get("num_attention_heads") == 32, "num_attention_heads == 32")
        result.check(config.get("attention_head_dim") == 128, "attention_head_dim == 128")

        # Connector defaults — CRITICAL: wrong values cause scrambled text embeddings
        result.check(
            config.get("connector_positional_embedding_max_pos") == [4096],
            f"connector_positional_embedding_max_pos == [4096] "
            f"(got: {config.get('connector_positional_embedding_max_pos')})",
        )
        result.check(
            config.get("connector_rope_type") == "SPLIT",
            f"connector_rope_type == SPLIT (got: {config.get('connector_rope_type')})",
        )

    # Transformer
    print("\n== Transformer Weights ==")
    tf_path = model_dir / "transformer.safetensors"
    if tf_path.exists():
        weights = mx.load(str(tf_path))
        keys = set(weights.keys())

        validate_no_pytorch_prefix(weights, "model.diffusion_model.", result)
        result.check(not any(".ff.net." in k for k in keys), "No un-sanitized FF keys")
        result.check(not any(".to_out.0." in k for k in keys), "No un-sanitized to_out keys")

        gate_keys = [k for k in keys if "to_gate_logits" in k]
        result.check(len(gate_keys) > 0, f"Gated attention keys present ({len(gate_keys)})")

        prompt_adaln = [k for k in keys if "prompt_adaln_single" in k]
        result.check(
            len(prompt_adaln) > 0,
            f"prompt_adaln_single keys present ({len(prompt_adaln)})",
        )

        block_indices = set()
        for k in keys:
            if "transformer_blocks." in k:
                parts = k.split("transformer_blocks.")
                if len(parts) > 1:
                    idx = parts[1].split(".")[0]
                    if idx.isdigit():
                        block_indices.add(int(idx))
        result.check(len(block_indices) == 48, f"48 transformer blocks (got {len(block_indices)})")

        if is_quantized:
            validate_quantization(weights, result, block_key="transformer_blocks")

        sst_keys = [k for k in keys if "scale_shift_table" in k
                    and "prompt" not in k and "audio_prompt" not in k]
        if sst_keys:
            shape = weights[sst_keys[0]].shape
            result.check(shape[0] == 9, f"scale_shift_table has 9 params (got shape {shape})")

        # last_scale_shift_table is NOT in the original weights — initialized to zeros
        # at model load time. Its absence is expected and correct.
        lsst_keys = [k for k in keys if "last_scale_shift_table" in k]
        result.check(
            len(lsst_keys) == 0,
            "last_scale_shift_table absent (expected — initialized to zeros at load time)",
            warn_only=True,
        )

        total_params = sum(v.size for v in weights.values())
        print(f"  Total transformer parameters: {total_params / 1e9:.2f}B")
        del weights

    # Connector
    print("\n== Connector Weights ==")
    conn_path = model_dir / "connector.safetensors"
    if conn_path.exists():
        weights = mx.load(str(conn_path))
        keys = set(weights.keys())
        result.check(any("video_embeddings_connector" in k for k in keys),
                     "Video connector keys present")
        result.check(any("audio_embeddings_connector" in k for k in keys),
                     "Audio connector keys present")
        result.check(any("text_embedding_projection" in k for k in keys),
                     "Text projection keys present", warn_only=True)
        del weights

    # VAE decoder/encoder
    for component in ["vae_decoder", "vae_encoder"]:
        print(f"\n== {component} Weights ==")
        vae_path = model_dir / f"{component}.safetensors"
        if vae_path.exists():
            weights = mx.load(str(vae_path))
            validate_no_pytorch_prefix(weights, "vae.", result)
            stats_keys = [k for k in weights if "per_channel_statistics" in k]
            result.check(
                len(stats_keys) >= 2,
                f"Per-channel statistics present ({len(stats_keys)})",
            )
            validate_conv_layout(weights, result, ndim=5)
            del weights

    # Audio VAE
    print("\n== Audio VAE Weights ==")
    avae_path = model_dir / "audio_vae.safetensors"
    if avae_path.exists():
        weights = mx.load(str(avae_path))
        bad = [k for k in weights if "audio_vae.decoder." in k]
        result.check(len(bad) == 0,
                     f"No PyTorch 'audio_vae.decoder.' prefix (found {len(bad)})")
        del weights

    # Vocoder
    print("\n== Vocoder Weights ==")
    voc_path = model_dir / "vocoder.safetensors"
    if voc_path.exists():
        weights = mx.load(str(voc_path))
        has_prefix = all(k.startswith("vocoder.") for k in weights)
        result.check(has_prefix, f"All keys have 'vocoder.' prefix ({len(weights)} keys)")
        conv1d = [(k, v) for k, v in weights.items() if "weight" in k and v.ndim == 3]
        result.check(len(conv1d) > 0, f"Conv1d weights present ({len(conv1d)})")
        del weights

    # Cross-reference
    if hasattr(args, "source") and args.source:
        _cross_reference(model_dir, args.source, result)

    result.summary()
    if not result.passed:
        raise SystemExit(1)


def _cross_reference(model_dir: Path, source_path: str, result: ValidationResult) -> None:
    """Compare converted weights against source checkpoint."""
    print("\n== Cross-Reference with Source ==")

    source_weights = mx.load(source_path)
    print(f"  Source checkpoint: {len(source_weights)} keys")

    classified = sum(1 for k in source_weights if classify_key(k) is not None)
    result.check(classified > 0, f"Classified {classified}/{len(source_weights)} source keys")

    # Spot-check tensor values
    print("\n  Spot-checking tensor values...")
    spot_checks = [
        "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight",
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight",
        "model.diffusion_model.transformer_blocks.0.attn1.to_gate_logits.weight",
    ]

    tf_weights = mx.load(str(model_dir / "transformer.safetensors"))

    for src_key in spot_checks:
        if src_key not in source_weights:
            print(f"  \033[93m\u26a0\033[0m Source key not found: {src_key}")
            continue

        mlx_key = src_key.replace("model.diffusion_model.", "transformer.")
        mlx_key = mlx_key.replace(".linear_1.", ".linear1.")
        mlx_key = mlx_key.replace(".linear_2.", ".linear2.")

        if mlx_key not in tf_weights:
            result.check(False, f"MLX key not found: {mlx_key}")
            continue

        src_tensor = source_weights[src_key]
        mlx_tensor = tf_weights[mlx_key]

        if mlx_tensor.dtype == mx.uint32:
            print(f"  \033[92m\u2713\033[0m {src_key.split('.')[-2]} — quantized")
            continue

        if src_tensor.dtype != mlx_tensor.dtype:
            src_tensor = src_tensor.astype(mlx_tensor.dtype)

        max_diff = mx.max(mx.abs(src_tensor - mlx_tensor)).item()
        key_name = f"{src_key.split('.')[-2]}.{src_key.split('.')[-1]}"
        result.check(max_diff < 1e-5, f"{key_name} — max diff: {max_diff:.2e}")

    del tf_weights, source_weights


# ---------------------------------------------------------------------------
# Split (LTX-2.3 component map for legacy unified models)
# ---------------------------------------------------------------------------

LTX23_SPLIT_MAP = {
    "transformer": "transformer.safetensors",
    "connector": "connector.safetensors",
    "text_embedding_projection": "connector.safetensors",
    "vae_decoder": "vae_decoder.safetensors",
    "vae_encoder": "vae_encoder.safetensors",
    "vocoder": "vocoder.safetensors",
    "audio_vae": "audio_vae.safetensors",
}


def split(args) -> None:
    """Split a unified LTX model into per-component files."""
    from ..split import split_model
    model_dir = Path(args.model_dir)
    split_model(model_dir, LTX23_SPLIT_MAP)


# ---------------------------------------------------------------------------
# CLI argument registration
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
    """Add LTX-2.3 convert arguments to a parser."""
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to local .safetensors checkpoint (skips download)")
    parser.add_argument("--variant", type=str, default="distilled",
                        choices=["distilled", "dev"],
                        help="Model variant (default: distilled)")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: ~/.cache/huggingface/hub/ltx23-mlx-<variant>)",
    )
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize transformer after conversion")
    parser.add_argument("--bits", type=int, default=8, choices=[4, 8],
                        help="Quantization bits (default: 8)")
    parser.add_argument("--group-size", type=int, default=64,
                        help="Quantization group size (default: 64)")


def add_validate_args(parser) -> None:
    """Add LTX-2.3 validate arguments to a parser."""
    parser.add_argument("model_dir", type=str, help="Path to converted model directory")
    parser.add_argument("--source", type=str, default=None,
                        help="Path to source checkpoint for cross-reference")


def add_split_args(parser) -> None:
    """Add LTX-2.3 split arguments to a parser."""
    parser.add_argument("model_dir", type=str,
                        help="Model directory containing model.safetensors")
