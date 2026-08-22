"""LTX-2.5: 22B audio-video DiT, Gemma-4 text encoder, two video VAEs.

Unlike LTX-2.3, whose 46 GB monolith this repo splits itself, LTX-2.5 arrives
already split by Lightricks into one file per role. Classification is therefore
per-file, not global — and it has to be: the two video VAE files expose the
same top-level prefixes (`decoder.`, `encoder.`, `per_channel_statistics.`) for
different architectures, so the discriminating information is which file a key
came from, never the key itself.

`SOURCE_FILES` is the single source of truth for what upstream contains: the
download list, the dry-run estimate and the conversion loop all read it, rather
than being maintained separately and drifting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ..metadata import RecipeMetadata
from ..transpose import transpose_conv

# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------

METADATA = RecipeMetadata(
    name="ltx-2.5",
    source="Lightricks/LTX-2.5",
    license="other",
    license_name="ltx-2-community-license-agreement",
    # The August 11 2026 "LTX-2.x" agreement — the one LTX-2.5's own weights
    # carry in their safetensors __metadata__, which is not the January text
    # our 2.3 packs ship. Deliberately the plain-text LICENSE, not the
    # LICENSE.md the upstream card links: it is the rendering whose bytes we
    # distribute, and the only one convert() can check against the weights.
    license_link="https://github.com/Lightricks/LTX-2/blob/main/LICENSE",
    license_file="LICENSE",
    # LTX-2.5 publishes no LICENSE on the Hub, so the copy comes from GitHub.
    license_source="github:Lightricks/LTX-2/LICENSE",
    quantization_scope="transformer block and text-encoder Linear weights",
    gated=True,
)

UPSTREAM_REPO = "Lightricks/LTX-2.5"


# ---------------------------------------------------------------------------
# Source table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceFile:
    """One upstream file and what this recipe makes of it.

    Args:
        path: Path inside the upstream repo.
        components: Output component names this file produces.
        classify: Maps one of this file's keys to a component name, or None to
            skip it. None when `converter` handles the file wholesale.
        size_mb: Approximate download size, used by --dry-run.
        required: A missing required file aborts; a missing optional one warns.
        converter: Bespoke conversion for a file the common loop cannot serve.
            Signature: (source_path, output_dir, header_metadata) -> None.
    """

    path: str
    components: tuple[str, ...]
    size_mb: int
    classify: Callable[[str], str | None] | None = None
    required: bool = True
    converter: Callable | None = None


# SOURCE_FILES is filled in over Tasks 7-11; the DiT, text encoder and
# duration-head/upscaler entries are added by later tasks.
SOURCE_FILES: tuple[SourceFile, ...] = ()


# ---------------------------------------------------------------------------
# Video VAE — two files, same prefixes, different architectures
# ---------------------------------------------------------------------------


def classify_video_vae_key(key: str, suffix: str) -> str | None:
    """Route a video VAE key to `vae_encoder{suffix}` or `vae_decoder{suffix}`.

    `suffix` is "_conv" or "_av" and comes from the SOURCE_FILES entry, because
    the key alone cannot say which of the two files it belongs to. Neither
    component is ever named plainly: a pack whose `vae_decoder.safetensors`
    could be either architecture is a silent mis-load waiting to happen.

    per_channel_statistics belongs to both halves and is reported as the
    decoder's here; the encoder picks it up through its own sanitizer, which is
    how LTX-2.3 does it and why neither ships a separate stats file.
    """
    if key.startswith("encoder."):
        return f"vae_encoder{suffix}"
    if key.startswith("decoder.") or key.startswith("per_channel_statistics."):
        return f"vae_decoder{suffix}"
    return None


def sanitize_vae_decoder_key(key: str) -> str | None:
    """Convert a video VAE decoder key to MLX format."""
    if key.startswith("per_channel_statistics."):
        if "mean-of-means" in key:
            return "per_channel_statistics.mean"
        if "std-of-means" in key:
            return "per_channel_statistics.std"
        return None
    if key.startswith("decoder."):
        return key[len("decoder.") :]
    return None


def sanitize_vae_encoder_key(key: str) -> str | None:
    """Convert a video VAE encoder key to MLX format."""
    if key.startswith("per_channel_statistics."):
        if "mean-of-means" in key:
            return "per_channel_statistics._mean_of_means"
        if "std-of-means" in key:
            return "per_channel_statistics._std_of_means"
        return None
    if key.startswith("encoder."):
        return key[len("encoder.") :]
    return None


# ---------------------------------------------------------------------------
# Audio — one upstream file, two components
# ---------------------------------------------------------------------------


def classify_audio_key(key: str) -> str | None:
    """Route an audio VAE file key to `audio_vae` or `vocoder`."""
    if key.startswith("audio_vae."):
        return "audio_vae"
    if key.startswith("vocoder."):
        return "vocoder"
    return None


def sanitize_audio_vae_key(key: str) -> str | None:
    """Convert an audio VAE key to MLX format, keeping the decoder/encoder split.

    Identical to LTX-2.3's, which tests/test_ltx25_sanitizer_parity.py pins.
    """
    if not key.startswith("audio_vae."):
        return None
    suffix = key[len("audio_vae.") :]
    if suffix.startswith("per_channel_statistics."):
        if "mean-of-means" in suffix:
            return "per_channel_statistics._mean_of_means"
        if "std-of-means" in suffix:
            return "per_channel_statistics._std_of_means"
        return None
    if suffix.startswith("decoder.") or suffix.startswith("encoder."):
        return suffix
    return None


def sanitize_vocoder_key(key: str) -> str | None:
    """Convert a vocoder key to MLX format.

    Uses replace() rather than a single prefix strip to match LTX-2.3's
    sanitizer byte for byte: some vocoder submodules are themselves named
    "vocoder" (e.g. "vocoder.vocoder.act_post.act.alpha"), so the two
    approaches diverge on those keys, and parity with 2.3 is the point.
    """
    if key.startswith("vocoder."):
        return key.replace("vocoder.", "")
    return None


# ---------------------------------------------------------------------------
# DiT — the 22B transformer and its connectors
# ---------------------------------------------------------------------------

UPSTREAM_TRANSFORMERS = {
    "dev": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "distilled": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
}

VARIANT_FILENAMES = {
    "dev": "transformer-dev.safetensors",
    "distilled": "transformer-distilled.safetensors",
}

_CONNECTOR_STACKS = ("video_embeddings_connector.", "audio_embeddings_connector.")


def classify_dit_key(key: str) -> str | None:
    """Route a DiT key to `transformer` or `connector`.

    Only the two `*_embeddings_connector` stacks are the connector. 366 DiT
    keys contain "prompt" or "connector" as a substring — the per-block
    `prompt_scale_shift_table` buffers among them — and every one of those
    belongs to the transformer.
    """
    if not key.startswith("model.diffusion_model."):
        return None
    suffix = key[len("model.diffusion_model.") :]
    if suffix.startswith(_CONNECTOR_STACKS):
        return "connector"
    return "transformer"


def sanitize_transformer_key(key: str) -> str:
    """Convert a DiT key to MLX format."""
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
    """Convert a connector key to MLX format: only the container prefix goes."""
    return key.replace("model.diffusion_model.", "")


def _is_conv_buffer(key: str, value: mx.array) -> bool:
    """A register_buffer with conv-like layout — vocoder filters and STFT bases."""
    if value.ndim < 3:
        return False
    suffix = key.rsplit(".", 1)[-1]
    return suffix == "filter" or suffix.endswith("_basis")


def maybe_transpose(key: str, value: mx.array, component: str) -> mx.array:
    """Transpose conv weights from PyTorch to MLX layout when the key is one."""
    if component in ("transformer", "connector", "text_encoder"):
        return value  # all Linear
    if _is_conv_buffer(key, value):
        return transpose_conv(value)
    is_conv = (
        "conv" in key.lower() or (component == "vocoder" and "ups" in key)
    ) and "weight" in key
    if not is_conv:
        return value
    return transpose_conv(value, is_conv_transpose=component == "vocoder" and "ups" in key)


# ---------------------------------------------------------------------------
# Duration head — 15 tensors, per-file classification
# ---------------------------------------------------------------------------


UPSCALER_FILES = {
    "spatial_upscaler_x2_v1_0": (
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    ),
    "temporal_upscaler_x2_v1_0": (
        "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"
    ),
}

DURATION_HEAD_FILE = "model_patches/ltx-2.5-duration-head-bf16.safetensors"

LORA_FILES = {
    "distilled-450": "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors",
}


def classify_duration_head_key(key: str) -> str | None:
    """Route a duration-head key. The file holds one component and nothing else."""
    return "duration_head" if key.startswith("duration_head.") else None


def sanitize_duration_head_key(key: str) -> str | None:
    """Convert a duration-head key to MLX format."""
    if key.startswith("duration_head."):
        return key[len("duration_head.") :]
    return None


def _is_upscaler_conv_weight(key: str, weight: mx.array) -> bool:
    """Whether an upscaler tensor is a conv weight needing the layout swap."""
    return "conv" in key.lower() and key.endswith(".weight") and weight.ndim >= 3


# ---------------------------------------------------------------------------
# Quantisation — two components, different rules
# ---------------------------------------------------------------------------

from ..quantize import write_quantize_config  # noqa: E402
from .ltx_25_text_encoder import should_quantize_gemma  # noqa: E402

QUANTIZED_COMPONENTS = {
    "transformer": "transformer_blocks Linear weights",
    "text_encoder": "Gemma-4 attention and MLP Linear weights",
}


def ltx25_should_quantize(key: str, weight: mx.array) -> bool:
    """Only transformer_blocks Linear weights — LTX-2.3's rule, unchanged."""
    bare_key = key.replace("transformer.", "", 1)
    return (
        "transformer_blocks" in bare_key
        and bare_key.endswith(".weight")
        and weight.ndim == 2
        and not bare_key.endswith(".scales")
        and not bare_key.endswith(".biases")
    )


def write_ltx25_quantize_config(output_dir: Path, *, bits: int, group_size: int) -> Path:
    """Record how this pack was quantised, per component.

    LTX-2.3 records a single `only_transformer_blocks` flag, which cannot
    describe two components quantised under different rules. The runtime has to
    rebuild each without guessing.
    """
    return write_quantize_config(
        output_dir,
        bits=bits,
        group_size=group_size,
        components=dict(QUANTIZED_COMPONENTS),
    )


__all__ = [
    "METADATA",
    "UPSTREAM_REPO",
    "SourceFile",
    "SOURCE_FILES",
    "classify_video_vae_key",
    "sanitize_vae_decoder_key",
    "sanitize_vae_encoder_key",
    "classify_audio_key",
    "sanitize_audio_vae_key",
    "sanitize_vocoder_key",
    "UPSTREAM_TRANSFORMERS",
    "VARIANT_FILENAMES",
    "classify_dit_key",
    "sanitize_transformer_key",
    "sanitize_connector_key",
    "maybe_transpose",
    "UPSCALER_FILES",
    "DURATION_HEAD_FILE",
    "LORA_FILES",
    "classify_duration_head_key",
    "sanitize_duration_head_key",
    "QUANTIZED_COMPONENTS",
    "ltx25_should_quantize",
    "write_ltx25_quantize_config",
    "should_quantize_gemma",
]
