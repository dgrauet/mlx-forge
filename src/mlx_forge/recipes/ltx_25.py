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

import gc
import hashlib
import json
import re
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers

from ..convert import (
    add_common_convert_args,
    classify_keys,
    default_output_dir,
    download_hf_files,
    ensure_license_file,
    fmt_size,
    load_safetensors,
    print_output_summary,
    process_component,
    quantize_component,
    write_split_model,
)
from ..metadata import RecipeMetadata
from ..quantize import read_quantize_config, write_quantize_config
from ..transpose import transpose_conv
from ..validate import (
    ValidationResult,
    finish_validation,
    start_validation,
    validate_conv_layout,
    validate_file_exists,
    validate_quantization,
)
from .ltx_25_text_encoder import (
    ASSET_FILENAMES,
    TEXT_ENCODER_FILE,
    convert_text_encoder,
    should_quantize_gemma,
)

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


# SOURCE_FILES is assembled at the end of this module, once every classifier
# and converter it references has been defined.


# ---------------------------------------------------------------------------
# Video VAE — two files, same prefixes, different architectures
# ---------------------------------------------------------------------------


def classify_video_vae_key(key: str, suffix: str) -> str | None:
    """Route a video VAE key to `vae_encoder{suffix}` or `vae_decoder{suffix}`.

    `suffix` is "_conv" or "_av" and comes from the SOURCE_FILES entry, because
    the key alone cannot say which of the two files it belongs to. Neither
    component is ever named plainly: a pack whose `vae_decoder.safetensors`
    could be either architecture is a silent mis-load waiting to happen.

    per_channel_statistics belongs to both halves, but a classifier hands each
    key to exactly one component, so this routes it to the decoder only; the
    caller (`_share_video_vae_statistics`, in the `convert()` loop) is what
    gives the encoder its own copy afterwards, before either output file is
    written. LTX-2.3 reaches the same result differently: it classifies these
    keys into a third component, `vae_shared_stats`, and `process_shared_stats`
    appends them to the decoder and encoder files after both already exist on
    disk (load, add the key, re-save — each file twice). That shape fits 2.3,
    which processes its monolith through a `component -> keys` dict per whole
    checkpoint; per-file classification here means encoder and decoder keys
    are already sitting in the same in-memory `keys_by_component` before
    either is written, so duplicating the keys there is simpler than 2.3's
    read-modify-write.
    """
    if key.startswith("encoder."):
        return f"vae_encoder{suffix}"
    if key.startswith("decoder.") or key.startswith("per_channel_statistics."):
        return f"vae_decoder{suffix}"
    return None


def _share_video_vae_statistics(
    keys_by_component: dict[str, list[str]], components: tuple[str, ...]
) -> None:
    """Give the encoder its own copy of the decoder's per_channel_statistics keys.

    `classify_video_vae_key` can only route a key to one component, so it
    hands `per_channel_statistics.*` to the decoder alone. Both halves need
    the statistics — an encoder without them is missing the normalisation its
    LTX-2.3 counterpart carries, which for the image/video conditioning path
    is a load failure or unnormalised latents, not a naming nit. This
    duplicates those keys into the encoder's list before `process_component`
    writes either file, called from `convert()` right after `classify_keys`.

    A no-op for any `components` tuple that is not a video VAE's
    `(vae_encoder*, vae_decoder*)` pair: every other source's components are
    named otherwise, so `encoder_name`/`decoder_name` stay `None`.
    """
    encoder_name = next((c for c in components if c.startswith("vae_encoder")), None)
    decoder_name = next((c for c in components if c.startswith("vae_decoder")), None)
    if encoder_name is None or decoder_name is None:
        return
    stats_keys = [
        key
        for key in keys_by_component.get(decoder_name, [])
        if key.startswith("per_channel_statistics.")
    ]
    if stats_keys:
        keys_by_component.setdefault(encoder_name, []).extend(stats_keys)


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
    """Convert a vocoder key to MLX format — 2.3's emitted layout, exactly.

    The upstream file holds three children under a shared "vocoder."
    container: the main HiFiGAN-style generator, itself named "vocoder"
    (667 tensors), its sibling "bwe_generator" (557), and "mel_stft" (3).
    The published LTX-2 vocoder contract — what ltx-2-mlx and the 2.3 packs
    ship and load — flattens the main generator to the component root:
    `vocoder.act_post...`, `bwe_generator.act_post...`, `mel_stft...`.

    An earlier revision of this recipe kept the main generator nested
    (`vocoder.vocoder.act_post...`), reading the extra level as sibling
    symmetry worth preserving; the loader crashed on the 667 unknown
    parameters. Same architecture, same names — the flat base IS the
    contract, and test_ltx25_sanitizer_parity pins it against ltx_23.

    Implemented as two positional strips rather than 2.3's
    `key.replace("vocoder.", "")`: the result is identical on every real
    key, but a hypothetical interior ".vocoder." deeper in a name cannot be
    mangled by it.
    """
    if not key.startswith("vocoder."):
        return None
    k = key[len("vocoder.") :]
    if k.startswith("vocoder."):
        k = k[len("vocoder.") :]
    return k


# ---------------------------------------------------------------------------
# DiT — the 22B transformer and its connectors
# ---------------------------------------------------------------------------

UPSTREAM_TRANSFORMERS = {
    "dev": "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors",
    "distilled": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
}

#: num_layers in the checkpoint-embedded transformer config, common to both
#: variants; verified on the real packs (block indices 0..47).
TRANSFORMER_BLOCK_COUNT = 48

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
    """Whether an upscaler tensor is a conv weight needing the layout swap.

    Matches ltx_23.py's rule (rank >= 3 and a `.weight`), not a name match:
    both LTX-2.5 upscalers have a `upsampler.0.weight` that is a real Conv2d
    ([4096, 1024, 3, 3]) or Conv3d ([1024, 512, 3, 3, 3]) weight but does not
    contain "conv" in its key, so a substring check on the name misses it.
    Every rank>=3 tensor in both upscaler checkpoints is in fact a `.weight`,
    so the plain suffix check is sufficient; there is no `.kernel`-style fixed
    buffer in either file the way ltx_23's BlurDownsample has, so unlike
    ltx_23 there is no branch for one here.
    """
    if weight.ndim < 3:
        return False
    return key.endswith(".weight")


def transpose_upscaler_weight(key: str, value: mx.array, component: str) -> mx.array:
    """Transform for the two upscaler components: transpose by rank, not by name.

    `maybe_transpose` cannot serve these: its conv detection is a substring
    match on the key ("conv" in key.lower()), and the upscalers' conv weight is
    named `upsampler.0.weight`, which that substring check misses. See
    `_is_upscaler_conv_weight` for why rank is the only reliable signal here.
    """
    if _is_upscaler_conv_weight(key, value):
        return transpose_conv(value)
    return value


# ---------------------------------------------------------------------------
# Quantisation — two components, different rules
# ---------------------------------------------------------------------------

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


def write_ltx25_quantize_config(
    output_dir: Path, *, bits: int, group_size: int, skip_shared: bool
) -> Path:
    """Record how this pack was quantised, per component actually written.

    LTX-2.3 records a single `only_transformer_blocks` flag, which cannot
    describe two components quantised under different rules. The runtime has
    to rebuild each without guessing.

    `skip_shared` mirrors `convert()`'s own guard around text-encoder
    quantisation (`if not args.skip_shared`): a `--skip-shared` pack contains
    no text encoder, so recording `QUANTIZED_COMPONENTS["text_encoder"]`
    unconditionally would claim a component the pack does not ship — the
    delta workflow's own manifest documented rule this function's docstring
    already stated but did not enforce.
    """
    components = dict(QUANTIZED_COMPONENTS)
    if skip_shared:
        del components["text_encoder"]
    return write_quantize_config(
        output_dir,
        bits=bits,
        group_size=group_size,
        components=components,
    )


#: The sanitizer for each component `classify_dit_key`/`classify_video_vae_key`/etc.
#: can produce. `ltx_23.py` keeps the same shape as a module-level SANITIZERS
#: dict; the upscalers' checkpoints have bare keys, so their sanitizer is the
#: identity function.
SANITIZERS: dict[str, Callable[[str], str | None]] = {
    "vae_encoder_conv": sanitize_vae_encoder_key,
    "vae_decoder_conv": sanitize_vae_decoder_key,
    "vae_encoder_av": sanitize_vae_encoder_key,
    "vae_decoder_av": sanitize_vae_decoder_key,
    "audio_vae": sanitize_audio_vae_key,
    "vocoder": sanitize_vocoder_key,
    "duration_head": sanitize_duration_head_key,
    "spatial_upscaler_x2_v1_0": lambda key: key,
    "temporal_upscaler_x2_v1_0": lambda key: key,
    "transformer": sanitize_transformer_key,
    "connector": sanitize_connector_key,
}

#: Which upstream transformer file a variant name maps to, inverted — used to
#: pick VARIANT_FILENAMES's output name while iterating SOURCE_FILES, which
#: only carries the path.
_TRANSFORMER_VARIANT_BY_PATH = {path: variant for variant, path in UPSTREAM_TRANSFORMERS.items()}


# ---------------------------------------------------------------------------
# Source table — the single source of truth for what upstream contains
# ---------------------------------------------------------------------------

#: Upstream files in conversion order. Small components first, so a wrong
#: classifier surfaces in minutes rather than after the 42 GB downloads.
SOURCE_FILES: tuple[SourceFile, ...] = (
    SourceFile(
        path="vae/ltx-2.5-video-vae-conv-bf16.safetensors",
        components=("vae_encoder_conv", "vae_decoder_conv"),
        size_mb=1_450,
        classify=partial(classify_video_vae_key, suffix="_conv"),
    ),
    SourceFile(
        path="vae/ltx-2.5-video-vae-bf16.safetensors",
        components=("vae_encoder_av", "vae_decoder_av"),
        size_mb=1_470,
        classify=partial(classify_video_vae_key, suffix="_av"),
    ),
    SourceFile(
        path="vae/ltx-2.5-audio-vae-bf16.safetensors",
        components=("audio_vae", "vocoder"),
        size_mb=370,
        classify=classify_audio_key,
    ),
    SourceFile(
        path=DURATION_HEAD_FILE,
        components=("duration_head",),
        size_mb=4,
        classify=classify_duration_head_key,
    ),
    SourceFile(
        path=UPSCALER_FILES["spatial_upscaler_x2_v1_0"],
        components=("spatial_upscaler_x2_v1_0",),
        size_mb=1_000,
        # This file holds exactly one component, so the classifier ignores
        # its argument and always returns that component's name.
        classify=lambda key: "spatial_upscaler_x2_v1_0",
    ),
    SourceFile(
        path=UPSCALER_FILES["temporal_upscaler_x2_v1_0"],
        components=("temporal_upscaler_x2_v1_0",),
        size_mb=260,
        # Same reasoning as the spatial upscaler above.
        classify=lambda key: "temporal_upscaler_x2_v1_0",
    ),
    SourceFile(
        path=TEXT_ENCODER_FILE,
        components=("text_encoder",),
        size_mb=26_300,
        converter=convert_text_encoder,
    ),
    SourceFile(
        path=UPSTREAM_TRANSFORMERS["dev"],
        components=("transformer", "connector"),
        size_mb=42_020,
        classify=classify_dit_key,
    ),
    SourceFile(
        path=UPSTREAM_TRANSFORMERS["distilled"],
        components=("transformer", "connector"),
        size_mb=42_020,
        classify=classify_dit_key,
    ),
)

#: Files copied through untouched, not converted.
PASSTHROUGH_FILES = {"lora": LORA_FILES["distilled-450"], "size_mb": 8_900}

#: Measured on the published LTX-2.3 pack, whose connector is the same shape.
#: Defined above its use so download_size_mb/output_size_mb read top-to-bottom.
_CONNECTOR_SIZE_MB = 6_340

#: Components that do not vary with which transformer variant produced them,
#: and which --skip-shared therefore omits from a delta pack. This is the
#: single source of truth `convert()` reads to decide what to skip writing —
#: see the `component_name in SHARED_COMPONENTS` guard in its write loop —
#: not a description of the effect for a reader to take on faith.
#:
#: `connector` is here despite riding inside the same downloaded file as
#: `transformer` (both come from UPSTREAM_TRANSFORMERS' safetensors): dev and
#: distilled carry byte-identical connectors, verified via
#: `connector_fingerprint` on every run with two variants, so it is shared by
#: any reasonable definition even though it has no source file of its own.
#: `_selected_sources` cannot skip *downloading* it — it is bundled with the
#: transformer weights the delta workflow needs regardless — so this
#: membership is what makes convert() skip *writing* it out instead.
SHARED_COMPONENTS = frozenset(
    {
        "vae_encoder_conv",
        "vae_decoder_conv",
        "vae_encoder_av",
        "vae_decoder_av",
        "audio_vae",
        "vocoder",
        "duration_head",
        "spatial_upscaler_x2_v1_0",
        "temporal_upscaler_x2_v1_0",
        "text_encoder",
        "connector",
    }
)


def _selected_sources(variants: list[str], *, skip_shared: bool) -> list[SourceFile]:
    """The SOURCE_FILES entries a run with these options actually reads."""
    wanted_transformers = {UPSTREAM_TRANSFORMERS[v] for v in variants}
    out = []
    for source in SOURCE_FILES:
        if source.path in UPSTREAM_TRANSFORMERS.values():
            if source.path in wanted_transformers:
                out.append(source)
        elif not skip_shared:
            out.append(source)
    return out


def download_size_mb(
    variants: list[str], *, skip_shared: bool, lora: list[str] | None = None
) -> int:
    """Approximate download for this run, in MB, LoRA included.

    `lora` mirrors `convert()`'s `--lora`: passed through to `_selected_loras`
    so the estimate is derived from the very same selection the copy step
    uses, rather than a second, looser rule that could disagree with it —
    exactly the kind of drift `SOURCE_FILES` exists to prevent. `None` (the
    default) reproduces `_selected_loras`'s own default: every declared LoRA
    when `variants` includes `"dev"`, none otherwise.
    """
    total = sum(s.size_mb for s in _selected_sources(variants, skip_shared=skip_shared))
    if _selected_loras(variants, skip_shared=skip_shared, lora=lora):
        # PASSTHROUGH_FILES mixes a str (the lora path) and an int (its size)
        # as values, so the dict's inferred value type is `str | int`; the
        # key always holds an int, hence the explicit conversion rather than
        # a cast. LORA_FILES has exactly one entry today, so "any selected"
        # and "the flat size" agree; a second LoRA would need a per-file size.
        total += int(PASSTHROUGH_FILES["size_mb"])
    return total


def output_size_mb(variants: list[str], *, skip_shared: bool, lora: list[str] | None = None) -> int:
    """Approximate bf16 output for this run, in MB.

    Smaller than the download because the connector is written once however
    many transformer variants carry a copy of it.
    """
    total = download_size_mb(variants, skip_shared=skip_shared, lora=lora)
    extra_variants = max(0, len(variants) - 1)
    return total - extra_variants * _CONNECTOR_SIZE_MB


# ---------------------------------------------------------------------------
# Licence and connector checks
# ---------------------------------------------------------------------------


def _embedded_config_payload(raw_config: str, model_version: str | None) -> dict:
    """The embedded_config.json payload: the upstream config, plus two
    consumer-driven touches.

    Verbatim except for exactly these, both requested by the consuming
    runtime (ltx-2-mlx) after end-to-end validation:

    * `model_version` is added at the root — the upstream ecosystem gates
      behaviour on it (e.g. ANCESTRAL_SAMPLER_SINCE_VERSION), and the value
      comes from the same checkpoint header as the config itself.
    * `transformer.text_encoder_norm_type` is lowercased to the 2.3 casing
      ("per_token_rms") that every published LTX config uses; 2.5's header
      carries the enum name ("PER_TOKEN_RMS").

    Nothing else is renamed, filtered, or added.
    """
    config = json.loads(raw_config)
    norm_type = config.get("transformer", {}).get("text_encoder_norm_type")
    if isinstance(norm_type, str):
        config["transformer"]["text_encoder_norm_type"] = norm_type.lower()
    if model_version:
        config["model_version"] = model_version
    return config


def read_header_metadata(path: Path) -> dict:
    """The `__metadata__` mapping of a local safetensors file.

    Read from the header alone — the file may be 42 GB.
    """
    with open(path, "rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(length))
    # `or {}`, not a .get default: mx.save_safetensors writes an explicit
    # `"__metadata__": null` when no metadata is passed, and .get's default
    # does not apply to a present-but-null key.
    return header.get("__metadata__") or {}


def _remote_header_metadata(repo: str, filename: str) -> dict:
    """The `__metadata__` mapping of an upstream file, read via HTTP Range requests.

    Fetches only the 8-byte length prefix and the JSON header that follows —
    a few kilobytes against a checkpoint that can be 42 GB — never the tensor
    data itself. Mirrors `scripts/harvest_ltx25_keys.py`'s `read_header`,
    which established this approach against this same repository; this is
    the same two-request shape, just returning `__metadata__` instead of the
    reduced key list that script harvests.

    Used only when the file is not already sitting in the run's source
    download directory (`read_header_metadata` handles that case straight
    off disk); `--config-only` is `_config_only`'s only caller.
    """
    url = hf_hub_url(repo, filename)
    base = build_hf_headers()

    def fetch(byte_range: str) -> bytes:
        request = urllib.request.Request(url, headers={**base, "Range": byte_range})
        return urllib.request.urlopen(request).read()

    length = int.from_bytes(fetch("bytes=0-7"), "little")
    header = json.loads(fetch(f"bytes=8-{8 + length - 1}"))
    return header.get("__metadata__", {})


def verify_embedded_license(header_metadata: dict, license_path: Path) -> None:
    """Check the LICENSE we ship against the one the weights carry.

    LTX-2.5 publishes no LICENSE on the Hub, so the copy comes from GitHub.
    This is what makes that defensible: the text we distribute is provably the
    agreement attached to the weights we converted.

    The comparison is normalised — `rstrip` per line — because the two
    renderings differ by trailing whitespace and a final newline (34 441 bytes
    on GitHub against 34 562 embedded, same 580 lines, identical once trailing
    whitespace is removed). That is a weaker guarantee than byte equality, and
    it is stated here rather than only in a design document: whitespace is not
    terms, but a check that tolerates anything tolerates too much, so nothing
    else — leading indentation included — is normalised away.

    Raises:
        SystemExit: The checkpoint carries no licence, or its text differs.
    """
    embedded = header_metadata.get("license")
    if not embedded:
        raise SystemExit(
            "ERROR: this checkpoint carries no licence in its safetensors metadata, "
            "so the copy we ship cannot be checked against it. Refusing to convert."
        )

    shipped_lines = [line.rstrip() for line in license_path.read_text().splitlines()]
    embedded_lines = [line.rstrip() for line in embedded.splitlines()]
    if shipped_lines != embedded_lines:
        raise SystemExit(
            f"ERROR: {license_path} does not match the agreement embedded in the weights. "
            "Either upstream revised its terms, or the copy came from somewhere "
            "undocumented; both need a decision, not a silent overwrite."
        )


def _verify_license_if_carried(header_metadata: dict, license_path: Path) -> bool:
    """Verify the licence against one file's metadata, if that file carries one.

    Not every LTX-2.5 checkpoint embeds the licence text: of the nine
    SOURCE_FILES entries, the temporal upscaler (`['config']`) and the text
    encoder (`['format', 'gemma_config']`) do not, only seven do. Demanding it
    of every file — `verify_embedded_license`'s original per-file contract —
    means `convert()` cannot finish a real run: it aborts on the temporal
    upscaler after several components have already been downloaded and
    written.

    `verify_embedded_license` itself is unchanged: given a metadata mapping
    that does carry a licence, it still raises on a mismatch exactly as
    before. What moves is the decision to call it at all — skip a file with
    no `license` key rather than treating its absence as a failure here;
    `_require_license_verified` is where "silence must not read as agreement"
    now applies, at the level of the whole run rather than of one file.

    Returns:
        True if this file's metadata carried a licence (and it verified —
        a mismatch still raises `SystemExit`, it does not return False).
        False if this file carries no licence to check, so `convert()`
        should move on without treating that as a failure.
    """
    if not header_metadata.get("license"):
        return False
    verify_embedded_license(header_metadata, license_path)
    return True


def _require_license_verified(verified_any: bool, license_path: Path) -> None:
    """Abort if nothing in the run ever verified `license_path` against a checkpoint.

    The run-level half of the guarantee `_verify_license_if_carried` leaves
    open: skipping a file with no licence in its metadata is fine as long as
    some *other* file in the pack did carry one and got checked. A pack where
    nothing ever did is a `LICENSE` nobody verified — silence must not read as
    agreement, at this level exactly as it did at the single-file level
    before this function existed.

    Raises:
        SystemExit: No file processed in this run carried a licence to check.
    """
    if not verified_any:
        raise SystemExit(
            f"ERROR: no file converted in this run carried a licence in its "
            f"safetensors metadata, so {license_path} was never verified against "
            "anything upstream attaches to these weights. Refusing to convert "
            "with an unverified copy."
        )


def connector_fingerprint(weights: dict[str, mx.array]) -> str:
    """A hash of the connector tensors, used to compare two DiT variants.

    LTX-2.3 extracts shared components from the first variant only, assuming
    dev and distilled agree. The assumption has never been checked. Here it is,
    for a few seconds against a 42 GB file.
    """
    digest = hashlib.sha256()
    for key in sorted(k for k in weights if classify_dit_key(k) == "connector"):
        digest.update(key.encode())
        tensor = weights[key].astype(mx.float32)
        # mx.array implements the buffer protocol at runtime but its stubs
        # don't declare __buffer__, so ty can't see it satisfies Buffer.
        digest.update(bytes(memoryview(tensor)))  # ty: ignore[invalid-argument-type]
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
    """Register `mlx-forge convert ltx-2.5` arguments."""
    add_common_convert_args(
        parser,
        output_default="./models/ltx-2.5-mlx[-q<bits>]",
        quantize_help="Quantize transformer block and text-encoder Linear weights",
        dry_run_help="Preview the plan, with download and output footprints, downloading nothing",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(UPSTREAM_TRANSFORMERS),
        help="Transformer variant to convert; repeat for several (default: both)",
    )
    parser.add_argument(
        "--skip-shared",
        action="store_true",
        help="Convert only the transformers, for the delta workflow",
    )
    parser.add_argument(
        "--lora",
        action="append",
        choices=sorted(LORA_FILES),
        help="LoRA to sync, copied as-is (default: all)",
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help=(
            "Backfill embedded_config.json into an existing pack without "
            "reconverting: reads only safetensors headers (local file if present "
            "under the run's source directory, otherwise an HTTP range request), "
            "never a weight file"
        ),
    )


def _selected_loras(variants: list[str], *, skip_shared: bool, lora: list[str] | None) -> list[str]:
    """Which LORA_FILES entries this run copies.

    Takes `variants`/`skip_shared`/`lora` directly rather than an `args`
    object, so `download_size_mb` can call it with the same three values
    `convert()` does and the estimate cannot drift from the actual selection.

    `--skip-shared` always wins: the delta workflow's whole point is not
    re-shipping what the remote copy already has. Short of that, an explicit
    `--lora` always wins too, including for a distilled-only pack — an
    explicit choice overrides any default reasoning. Otherwise, default to
    every entry in `LORA_FILES` only when `"dev"` is among the variants: the
    distilled-* LoRAs are meant to be applied to the *dev* transformer, and a
    distilled checkpoint has the distillation baked in and never loads one,
    so bundling it by default for a distilled-only pack would be dead weight
    (8.9 GB per pack). This rule is inherited from `ltx_23._effective_lora_names`'s
    reading of the same artefact one version earlier
    (`ltx-2.3-22b-distilled-lora-384` / `ltx-2.5-22b-distilled-lora-450`) —
    2.3 has shipped on that reading — not from an upstream statement specific
    to LTX-2.5's LoRA.
    """
    if skip_shared:
        return []
    if lora is not None:
        return list(lora)
    if "dev" in variants:
        return sorted(LORA_FILES)
    print(
        "\n[lora] Skipping distilled LoRA(s) — package has no 'dev' variant "
        "(distilled transformers have the distillation baked in)."
    )
    return []


def _dry_run(args, output_dir: Path, variants: list[str]) -> None:
    """Print the plan and what it will cost, without downloading anything."""
    download = download_size_mb(variants, skip_shared=args.skip_shared, lora=args.lora)
    output = output_size_mb(variants, skip_shared=args.skip_shared, lora=args.lora)
    print(f"\nWould convert {UPSTREAM_REPO} -> {output_dir}")
    print(f"  variants: {', '.join(variants)}")
    for source in _selected_sources(variants, skip_shared=args.skip_shared):
        print(f"  {fmt_size(source.size_mb):>10}  {source.path} -> {', '.join(source.components)}")
    print(f"\n  download: {fmt_size(download)}")
    print(f"  output:   {fmt_size(output)} (bf16)")
    print(f"  free space needed: {fmt_size(download + output)}")


def _config_only(args, output_dir: Path, variants: list[str]) -> None:
    """Backfill embedded_config.json into an existing pack, header bytes only.

    For each selected variant's transformer, reads its safetensors header —
    a few kilobytes, never the 42 GB tensor data — from the run's source
    download directory if that file is already there (`read_header_metadata`,
    the same one `convert()` uses), or via an HTTP Range request otherwise
    (`_remote_header_metadata`). Never opens, downloads, or writes a
    `.safetensors` file: this is a JSON emission, not a conversion.

    Compares every selected variant's raw config the same way `convert()`'s
    own loop does, so a two-variant backfill gets the same "abort naming the
    disagreement" guarantee a normal run does — there being only one already
    on disk (as with the two real packs this backfills) does not exempt this
    path from checking, since --variant can still select both.

    Raises:
        SystemExit: The target directory does not exist or contains no
            split_model.json (what makes it a converted pack). This guard runs
            before any directory creation, so a mistyped path produces an error
            without silently creating an empty directory.
    """
    # Guard against --config-only on a non-existent or incomplete directory.
    # This must run BEFORE any mkdir() call, so if it fails, no directory is
    # created that did not exist before.
    split_model_path = output_dir / "split_model.json"
    if not split_model_path.exists():
        raise SystemExit(
            f"ERROR: {output_dir} does not contain split_model.json, so it is not a\n"
            "converted pack. --config-only backfills a pack that already exists; it\n"
            "cannot create one. Check --output for a typo, or run a normal conversion\n"
            "(without --config-only) to produce the pack first.\n"
            "Creating the directory does not help: the check is for split_model.json."
        )

    download_dir = _source_download_dir(output_dir)
    dit_config_raw: str | None = None
    dit_config_variant: str | None = None
    dit_model_version: str | None = None
    for variant in variants:
        path = UPSTREAM_TRANSFORMERS[variant]
        local = download_dir / path
        if local.exists():
            print(f"  Reading header from local file: {local}")
            header_metadata = read_header_metadata(local)
        else:
            print(f"  Fetching header over HTTP: {path}")
            header_metadata = _remote_header_metadata(UPSTREAM_REPO, path)
        dit_model_version = header_metadata.get("model_version") or dit_model_version
        raw_config = header_metadata.get("config")
        if not raw_config:
            raise SystemExit(
                f"ERROR: {path} carries no 'config' in its safetensors metadata, "
                "so embedded_config.json cannot be backfilled from it."
            )
        if dit_config_raw is None:
            dit_config_raw = raw_config
            dit_config_variant = variant
        elif raw_config != dit_config_raw:
            raise SystemExit(
                f"ERROR: the {dit_config_variant} and {variant} transformers carry "
                "different embedded configs; this recipe writes one "
                "embedded_config.json, which would be wrong for one of them. "
                "Backfill them into separate directories."
            )

    assert dit_config_raw is not None  # variants is never empty

    config_path = output_dir / "embedded_config.json"
    if args.dry_run:
        print(
            f"\nWould write {config_path} "
            f"({len(dit_config_raw)} bytes of upstream config, from {', '.join(variants)})"
        )
        return

    with open(config_path, "w") as handle:
        json.dump(
            _embedded_config_payload(dit_config_raw, dit_model_version),
            handle,
            indent=2,
        )
    print(f"\nWrote {config_path}")


def _source_download_dir(output_dir: Path) -> Path:
    """Where `convert()` downloads upstream checkpoints for this run.

    A sibling of `output_dir`, not a child of it (mirrors ltx_23's
    `models/ltx-2.3-src`) — `upload.iter_model_files` walks `output_dir`
    recursively with no exclusion for a nested source cache, so a download
    directory placed inside the pack would ship 76+ GB of gated upstream
    checkpoints to the public mirror alongside the converted weights.
    `--output` can point anywhere, so this is derived from `output_dir`
    rather than hardcoding "models/"; keeping it alongside `output_dir`
    (rather than under a fixed tmp root) keeps the 42 GB downloads on the
    same filesystem as the eventual output.
    """
    return output_dir.parent / f"{output_dir.name}-src"


def convert(args) -> None:
    """Convert LTX-2.5 to MLX, one upstream file at a time."""
    variants = args.variant or sorted(UPSTREAM_TRANSFORMERS)
    output_dir = (
        Path(args.output)
        if args.output
        else default_output_dir("ltx-2.5", quantize=args.quantize, bits=args.bits)
    )

    if args.config_only:
        # A JSON-only backfill for a pack that already exists: no download
        # plan to preview, no weights to write — _config_only handles
        # --dry-run itself rather than sharing _dry_run's download/output
        # size estimate, which describes a full conversion this path never
        # does.
        _config_only(args, output_dir, variants)
        return

    if args.dry_run:
        _dry_run(args, output_dir, variants)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = _source_download_dir(output_dir)

    # Built once and carried through the whole run: ensure_license_file records
    # license_provenance into this same dict, and write_split_model below must
    # see that record rather than a fresh dict that never received it — a
    # throwaway `info` here would silently drop the provenance from the
    # manifest, and, since a fresh dict carries no prior record, would also
    # make ensure_license_file refetch from GitHub on every source file.
    info = dict(METADATA.as_split_fields())

    # Not every SOURCE_FILES entry embeds the licence (the temporal upscaler
    # and the text encoder do not — see _verify_license_if_carried), so this
    # is checked against whichever files do carry it, not only the first.
    # SOURCE_FILES orders the small conv video VAE first and it does carry
    # the licence, so a mismatch there still surfaces within minutes, before
    # the 42 GB transformer downloads — _require_license_verified below is
    # what catches a run where nothing ever carried one to check at all.
    license_path = output_dir / "LICENSE"
    license_verified = False

    connector_seen: str | None = None
    # The DiT's own scheduler/transformer config, embedded verbatim in every
    # UPSTREAM_TRANSFORMERS file's __metadata__["config"] (confirmed on both
    # dev and distilled — see _dit_config below). Tracked the same way as
    # connector_seen just above: computed for every selected transformer,
    # dev and distilled included, and compared rather than trusted to agree,
    # because a pack carries exactly one embedded_config.json.
    dit_config_raw: str | None = None
    dit_config_variant: str | None = None
    dit_model_version: str | None = None
    for source in _selected_sources(variants, skip_shared=args.skip_shared):
        download_hf_files(UPSTREAM_REPO, [source.path], download_dir)
        local = download_dir / source.path
        header_metadata = read_header_metadata(local)

        if not license_path.exists():
            ensure_license_file(output_dir, info)
        if _verify_license_if_carried(header_metadata, license_path):
            license_verified = True

        if source.converter is not None:
            source.converter(local, output_dir, header_metadata)
        else:
            weights = load_safetensors(local)

            # The fingerprint is computed and compared on every run that
            # carries a connector, --skip-shared included: it is what proves
            # dev and distilled agree, and it costs seconds against an
            # already-downloaded 42 GB file. Only the *writing* is
            # conditional — write_connector picks the first variant to carry
            # it (the second only compares), and the SHARED_COMPONENTS guard
            # below additionally suppresses it for a delta pack.
            write_connector = False
            if "connector" in source.components:
                fingerprint = connector_fingerprint(weights)
                if connector_seen is None:
                    connector_seen = fingerprint
                    write_connector = True
                elif fingerprint != connector_seen:
                    raise SystemExit(
                        "ERROR: the dev and distilled transformers carry different "
                        "connectors; this recipe writes one, which would be wrong "
                        "for the other. Convert them into separate directories."
                    )

            if source.path in UPSTREAM_TRANSFORMERS.values():
                variant = _TRANSFORMER_VARIANT_BY_PATH[source.path]
                dit_model_version = header_metadata.get("model_version") or dit_model_version
                raw_config = header_metadata.get("config")
                if not raw_config:
                    raise SystemExit(
                        f"ERROR: {source.path} carries no 'config' in its safetensors "
                        "metadata, so embedded_config.json cannot be written from it."
                    )
                if dit_config_raw is None:
                    dit_config_raw = raw_config
                    dit_config_variant = variant
                elif raw_config != dit_config_raw:
                    raise SystemExit(
                        f"ERROR: the {dit_config_variant} and {variant} transformers "
                        "carry different embedded configs; this recipe writes one "
                        "embedded_config.json, which would be wrong for one of them. "
                        "Convert them into separate directories."
                    )

            # Every SOURCE_FILES entry without a converter must declare a
            # classify function (test_every_entry_can_convert_itself pins
            # this); asserting it here, rather than trusting the type alone,
            # is what lets classify_keys take a non-optional callable.
            assert source.classify is not None, f"{source.path} has neither classify nor converter"
            keys_by_component = classify_keys(weights, source.classify)
            _share_video_vae_statistics(keys_by_component, source.components)
            for component_name, keys in keys_by_component.items():
                if component_name == "connector" and not write_connector:
                    continue
                # The single place --skip-shared's contract is enforced for
                # a component that rides inside a file downloaded anyway
                # (the connector, bundled with the transformer): the file
                # still had to be fetched and classified, but a component
                # named in SHARED_COMPONENTS is never written to a delta
                # pack. Sources _selected_sources already dropped entirely
                # (vae_*, audio_vae, vocoder, duration_head, the upscalers,
                # text_encoder) never reach this loop under --skip-shared, so
                # this check is a no-op for them and load-bearing only here.
                if args.skip_shared and component_name in SHARED_COMPONENTS:
                    continue

                transform = (
                    transpose_upscaler_weight
                    if component_name in UPSCALER_FILES
                    else maybe_transpose
                )
                output_filename = None
                if component_name == "transformer":
                    variant = _TRANSFORMER_VARIANT_BY_PATH[source.path]
                    output_filename = VARIANT_FILENAMES[variant]

                process_component(
                    weights,
                    component_name,
                    keys,
                    output_dir,
                    component_prefix=component_name,
                    sanitizer=SANITIZERS[component_name],
                    transform=transform,
                    output_filename=output_filename,
                    # The upstream ecosystem gates behaviour on model_version
                    # (e.g. ANCESTRAL_SAMPLER_SINCE_VERSION); carry it — and
                    # the config blob — from this source file's header into
                    # every file converted from it. Only what the source
                    # actually carries is propagated, verbatim.
                    metadata={
                        k: header_metadata[k]
                        for k in ("model_version", "config")
                        if header_metadata.get(k)
                    }
                    or None,
                )

            # The upscalers carry their own config in the checkpoint's
            # __metadata__["config"], not in a separate file upstream.
            if source.path in UPSCALER_FILES.values():
                config = header_metadata.get("config")
                if config:
                    component_name = source.components[0]
                    config_path = output_dir / f"{component_name}_config.json"
                    with open(config_path, "w") as handle:
                        # Wrapped under "config" like the 2.3 recipe emits its
                        # upscaler configs, so consumers see a single shape.
                        json.dump({"config": json.loads(config)}, handle, indent=2)

            del weights
            gc.collect()
            mx.clear_cache()

    _require_license_verified(license_verified, license_path)

    # The consuming runtime (ltx-2-mlx) reads the DiT config through
    # LTXModelConfig.from_checkpoint_dir(), which looks for
    # embedded_config.json (then config.json) at the pack root; without it,
    # that reader falls back to LTX-2.3 defaults, silently, which are wrong
    # for 2.5. Written through _embedded_config_payload (verbatim except the
    # two consumer-driven touches its docstring lists),
    # once every selected transformer's raw config has been checked to agree
    # (dit_config_raw is None only if no transformer was selected, which
    # SOURCE_FILES makes impossible: --variant always names at least one).
    if dit_config_raw is not None:
        config_path = output_dir / "embedded_config.json"
        with open(config_path, "w") as handle:
            json.dump(
                _embedded_config_payload(dit_config_raw, dit_model_version),
                handle,
                indent=2,
            )

    # LoRAs ship as-is: no conversion, just downloaded and copied under their
    # upstream basename. `_selected_loras` empties this under --skip-shared or
    # for a distilled-only pack with no explicit --lora — see its docstring.
    lora_synced: list[str] = []
    for lora_name in _selected_loras(variants, skip_shared=args.skip_shared, lora=args.lora):
        filename = LORA_FILES[lora_name]
        dest = output_dir / Path(filename).name
        if dest.exists():
            print(f"\n[lora:{lora_name}] {dest.name} already exists, skipping")
            lora_synced.append(dest.name)
            continue
        download_hf_files(UPSTREAM_REPO, [filename], download_dir)
        shutil.copy2(download_dir / filename, dest)
        print(f"  Synced {dest.name} ({dest.stat().st_size / (1024**2):.0f} MB)")
        lora_synced.append(dest.name)

    if args.quantize:
        for variant in variants:
            quantize_component(
                output_dir,
                f"transformer ({variant})",
                bits=args.bits,
                group_size=args.group_size,
                should_quantize=ltx25_should_quantize,
                filename=VARIANT_FILENAMES[variant],
            )
        if not args.skip_shared:
            quantize_component(
                output_dir,
                "text_encoder",
                bits=args.bits,
                group_size=args.group_size,
                should_quantize=should_quantize_gemma,
            )
        write_ltx25_quantize_config(
            output_dir, bits=args.bits, group_size=args.group_size, skip_shared=args.skip_shared
        )

    # Reuse the same `info` the loop above passed to ensure_license_file, so
    # its license_provenance record reaches the manifest instead of being
    # computed into a dict nobody writes.
    info.update(
        {
            "transformer_variants": variants,
            "quantized": bool(args.quantize),
            "quantization_bits": args.bits if args.quantize else None,
            "delta": bool(args.skip_shared),
            "lora": lora_synced,
        }
    )
    write_split_model(output_dir, info)
    print_output_summary(output_dir, header="LTX-2.5 conversion complete")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

#: What a correct pack holds, per component. Cheap identity checks: swapping
#: the two video VAE files changes nothing a key name would reveal (both
#: expose the same `encoder.`/`decoder.` prefixes), but the decoder tensor
#: counts differ by 226 — a swapped pair is detectable without loading a
#: single weight.
#: The two video VAE pairs each sum to their file's harvested `tensor_count`
#: plus 2: `_share_video_vae_statistics` duplicates the 2 `per_channel_statistics`
#: tensors into the encoder, so they are written into both output files and
#: counted in both entries here (conv: 170 + 2 = 86 + 86; av: 396 + 2 = 86 + 312).
#: `test_video_vae_counts_reconcile_with_the_fixture` pins this against
#: tests/fixtures/ltx_25_keys.json rather than trusting the arithmetic by eye.
#: Covers every entry in SHARED_COMPONENTS (test_covers_every_shared_component
#: pins this). text_encoder's 681 is the checkpoint's 686 tensors minus the 5
#: U8 assets `sanitize_text_encoder_key` refuses, both harvested from
#: tests/fixtures/ltx_25_keys.json; the two upscalers hold 72 each. connector's
#: 258 was measured on a real converted pack (`mlx-forge convert ltx-2.5
#: --variant dev --skip-shared --quantize`), not harvested from the fixture —
#: unlike the other entries here, the connector has no SOURCE_FILES entry of
#: its own to harvest a header count from; it only ever appears bundled
#: inside a transformer file.
EXPECTED_TENSOR_COUNTS = {
    "vae_encoder_conv": 86,
    "vae_decoder_conv": 86,
    "vae_encoder_av": 86,
    "vae_decoder_av": 312,
    "audio_vae": 102,
    "vocoder": 1227,
    "duration_head": 15,
    "spatial_upscaler_x2_v1_0": 72,
    "temporal_upscaler_x2_v1_0": 72,
    "text_encoder": 681,
    "connector": 258,
}

#: The five files `convert_text_encoder`/`extract_assets` write from the
#: Gemma-4 checkpoint's embedded U8 tensors — tokenizer.json among them at
#: 32 MB. A pack whose tokenizer silently failed to extract loads and then
#: generates nonsense, so their presence and parseability is worth asserting.
TEXT_ENCODER_ASSET_FILES = tuple(ASSET_FILENAMES.values())

#: Components whose upstream keys are checked for a leaked PyTorch prefix
#: *underneath* the component prefix `process_component` writes (every key
#: is `f"{component}.{sanitized}"`, so one leading `f"{component}."` is
#: always present and correct — only a second occurrence, in what remains
#: after stripping that one, means the sanitizer failed). The video VAE
#: sanitizers strip "encoder."/"decoder." from the front of every key,
#: leaving nothing PyTorch-shaped to look for, so they are not in this list.
#:
#: `vocoder` is included since the sanitizer flattens the main generator to
#: the component root (the published LTX-2 layout): an emitted
#: `vocoder.vocoder.*` key now means exactly what this check exists to catch
#: — a sanitizer that failed to strip the internal level. That defect shipped
#: once (the loader crashed on 667 unknown parameters), which is why the
#: check gates it now.
_LEAKED_PREFIX_COMPONENTS = ("audio_vae", "duration_head", "vocoder")

#: Components with conv weights worth layout-checking, and at what rank.
#: Video VAEs are Conv3d; audio VAE and vocoder are Conv1d/Conv2d family but
#: share the ndim=4 check ltx_23.py uses for its own audio components.
_CONV_NDIM_BY_COMPONENT = {
    "vae_encoder_conv": 5,
    "vae_decoder_conv": 5,
    "vae_encoder_av": 5,
    "vae_decoder_av": 5,
    "audio_vae": 4,
    "vocoder": 4,
}


def _validate_no_leaked_pytorch_prefix(
    weights: dict[str, mx.array], component: str, result: ValidationResult
) -> None:
    """Check that no key still carries `component`'s upstream prefix underneath
    the component prefix `process_component` writes.

    `process_component` stores every key as `f"{component}.{sanitized}"`, so
    a correctly-sanitized key legitimately starts with `f"{component}."`
    once — that leading occurrence is the component prefix, not a survival.
    Only a *second* occurrence, found in what remains after stripping that
    one leading prefix, means the sanitizer left the original PyTorch prefix
    in place (e.g. `audio_vae.audio_vae.decoder...`).

    Args:
        weights: Dict of weight keys -> tensors, as written to the component
            file.
        component: Component name, matching both its own key prefix and the
            upstream prefix its sanitizer is meant to strip.
        result: ValidationResult to record into.
    """
    prefix = f"{component}."
    bad_keys = [k for k in weights if k.startswith(prefix) and prefix in k[len(prefix) :]]
    result.check(
        len(bad_keys) == 0,
        f"No PyTorch prefix '{prefix}' remaining beneath the component prefix "
        f"(found {len(bad_keys)})",
    )
    for k in bad_keys[:5]:
        print(f"    Bad key: {k}")


def add_validate_args(parser) -> None:
    """Register `mlx-forge validate ltx-2.5` arguments."""
    parser.add_argument("model_dir", type=str, help="Converted model directory")


def validate(args) -> None:
    """Check a converted LTX-2.5 directory.

    Two things are worth understanding rather than just reading off the
    checks below:

    The tensor counts in `EXPECTED_TENSOR_COUNTS` distinguish the two video
    VAEs, whose files expose identical top-level prefixes for different
    architectures — the decoders differ by 226 tensors, so a swapped pair is
    caught without loading a single weight.

    The text encoder's five assets (`TEXT_ENCODER_ASSET_FILES`) are files
    extracted from U8 tensors at conversion time; a pack whose tokenizer
    silently failed to extract loads and then generates nonsense, so their
    presence and parseability is checked explicitly.
    """
    model_dir, result = start_validation(args.model_dir)

    manifest_path = model_dir / "split_model.json"
    split_info: dict = {}
    if manifest_path.exists():
        split_info = json.loads(manifest_path.read_text())
    delta = bool(split_info.get("delta"))
    if delta:
        print("[INFO] Delta mode (skipping shared component checks)")

    if not delta:
        print("\n== Text Encoder Assets ==")
        for filename in TEXT_ENCODER_ASSET_FILES:
            path = model_dir / filename
            result.check(
                path.exists() and path.stat().st_size > 0,
                f"{filename} present and non-empty",
            )
        for filename in TEXT_ENCODER_ASSET_FILES:
            if not filename.endswith(".json"):
                continue
            path = model_dir / filename
            if not path.exists():
                continue
            try:
                json.loads(path.read_text())
                result.check(True, f"{filename} parses as JSON")
            except ValueError:
                result.check(False, f"{filename} parses as JSON")

        print("\n== VAE and Component Weights ==")
        qconfig = read_quantize_config(model_dir)
        quantized_components = set(qconfig.get("components", {})) if qconfig is not None else set()
        for component, expected in EXPECTED_TENSOR_COUNTS.items():
            filename = f"{component}.safetensors"
            if not validate_file_exists(model_dir, filename, result):
                continue
            weights = load_safetensors(model_dir / filename)
            if component in QUANTIZED_COMPONENTS and component in quantized_components:
                # EXPECTED_TENSOR_COUNTS holds bf16 counts. Quantization
                # turns each quantized tensor into three entries (the
                # weight, plus .scales and .biases), so a quantized
                # component legitimately holds more tensors than its bf16
                # count. Derive the expected count from what actually got
                # quantized rather than skipping the check — this still
                # asserts the component holds every original tensor *and*
                # that each quantized one gained exactly one .scales and
                # one .biases.
                scale_keys = [k for k in weights if k.endswith(".scales")]
                bias_keys = [k for k in weights if k.endswith(".biases")]
                result.check(
                    len(scale_keys) > 0,
                    f"{component} is quantized ({len(scale_keys)} .scales keys)",
                )
                result.check(
                    len(scale_keys) == len(bias_keys),
                    f"{component}: equal .scales and .biases count "
                    f"({len(scale_keys)} vs {len(bias_keys)})",
                )
                expected_quantized = expected + 2 * len(scale_keys)
                result.check(
                    len(weights) == expected_quantized,
                    f"{component} holds {expected_quantized} tensors when quantized "
                    f"(found {len(weights)})",
                )
                if component == "text_encoder":
                    # The transformer's scales/biases pairing is checked in the
                    # variant loop below; the other quantized component gets
                    # the same check. 328 .scales on the real q8 pack, all
                    # under model.layers.*.
                    validate_quantization(weights, result, block_key="layers")
            else:
                result.check(
                    len(weights) == expected,
                    f"{component} holds {expected} tensors (found {len(weights)})",
                )
            if component in _LEAKED_PREFIX_COMPONENTS:
                _validate_no_leaked_pytorch_prefix(weights, component, result)
            ndim = _CONV_NDIM_BY_COMPONENT.get(component)
            if ndim:
                validate_conv_layout(weights, result, ndim=ndim)
            del weights
            gc.collect()
            mx.clear_cache()

    print("\n== Embedded Config ==")
    if validate_file_exists(model_dir, "embedded_config.json", result):
        embedded = json.loads((model_dir / "embedded_config.json").read_text())
        result.check(
            bool(embedded.get("model_version")),
            "embedded_config.json carries model_version",
        )

    print("\n== Transformer Variants ==")
    variants = split_info.get("transformer_variants") or list(VARIANT_FILENAMES)
    qconfig = read_quantize_config(model_dir)
    if qconfig is not None:
        print(f"Model is quantized: int{qconfig.get('bits', '?')}")
    for variant in variants:
        filename = VARIANT_FILENAMES[variant]
        if not validate_file_exists(model_dir, filename, result):
            continue
        header_md = read_header_metadata(model_dir / filename)
        result.check(
            bool(header_md.get("model_version")),
            f"{variant}: safetensors header carries model_version",
        )
        weights = load_safetensors(model_dir / filename)
        # Structural checks, quantized or not: a sanitizer regression is
        # invisible to every other check here. The patterns are exactly what
        # sanitize_transformer_key strips, verified absent on the real packs.
        leaked = [
            k
            for k in weights
            if "model.diffusion_model." in k
            or ".ff.net." in k
            or ".to_out.0." in k
            or ".linear_1." in k
        ]
        result.check(
            len(leaked) == 0,
            f"{variant}: no unsanitized upstream key patterns (found {len(leaked)})",
        )
        for k in leaked[:5]:
            print(f"    Bad key: {k}")
        block_indices = {
            int(m.group(1))
            for k in weights
            for m in [re.search(r"transformer_blocks\.(\d+)\.", k)]
            if m
        }
        result.check(
            len(block_indices) == TRANSFORMER_BLOCK_COUNT,
            f"{variant}: {TRANSFORMER_BLOCK_COUNT} transformer blocks (found {len(block_indices)})",
        )
        if qconfig is not None:
            validate_quantization(weights, result, block_key="transformer_blocks")
        del weights
        gc.collect()
        mx.clear_cache()

    finish_validation(result)


# ---------------------------------------------------------------------------
# Split — a required no-op
# ---------------------------------------------------------------------------


def add_split_args(parser) -> None:
    """Register `mlx-forge split ltx-2.5` arguments."""
    parser.add_argument("model_dir", type=str, help="Converted model directory")


def split(args) -> None:
    """No-op: LTX-2.5 arrives already split upstream.

    `split()` is part of the recipe contract enforced by `cli.py` and
    `tests/test_recipe_contract.py`; a recipe that omits it produces an
    `AttributeError` traceback instead of an explanation. LTX-2.3's 46 GB
    monolith needs this repo's own splitting step — LTX-2.5 does not, because
    Lightricks already ships one upstream file per role and `convert()`
    writes each component directly.
    """
    print(
        "LTX-2.5 ships already split by Lightricks — one upstream file per role — "
        "so `convert` writes the components directly and there is nothing to split."
    )


__all__ = [
    "METADATA",
    "UPSTREAM_REPO",
    "SourceFile",
    "SOURCE_FILES",
    "PASSTHROUGH_FILES",
    "SHARED_COMPONENTS",
    "download_size_mb",
    "output_size_mb",
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
    "SANITIZERS",
    "transpose_upscaler_weight",
    "read_header_metadata",
    "verify_embedded_license",
    "connector_fingerprint",
    "add_convert_args",
    "convert",
    "EXPECTED_TENSOR_COUNTS",
    "TEXT_ENCODER_ASSET_FILES",
    "add_validate_args",
    "validate",
    "add_split_args",
    "split",
]
