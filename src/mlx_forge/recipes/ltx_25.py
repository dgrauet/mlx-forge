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
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlx.core as mx

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
from ..quantize import write_quantize_config
from ..transpose import transpose_conv
from .ltx_25_text_encoder import (
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

#: Components that exist independently of any transformer variant, and which
#: --skip-shared omits for the delta workflow.
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


def read_header_metadata(path: Path) -> dict:
    """The `__metadata__` mapping of a local safetensors file.

    Read from the header alone — the file may be 42 GB.
    """
    with open(path, "rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(length))
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


def convert(args) -> None:
    """Convert LTX-2.5 to MLX, one upstream file at a time."""
    variants = args.variant or sorted(UPSTREAM_TRANSFORMERS)
    output_dir = (
        Path(args.output)
        if args.output
        else default_output_dir("ltx-2.5", quantize=args.quantize, bits=args.bits)
    )

    if args.dry_run:
        _dry_run(args, output_dir, variants)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = output_dir / ".source"

    # Built once and carried through the whole run: ensure_license_file records
    # license_provenance into this same dict, and write_split_model below must
    # see that record rather than a fresh dict that never received it — a
    # throwaway `info` here would silently drop the provenance from the
    # manifest, and, since a fresh dict carries no prior record, would also
    # make ensure_license_file refetch from GitHub on every source file.
    info = dict(METADATA.as_split_fields())

    connector_seen: str | None = None
    for source in _selected_sources(variants, skip_shared=args.skip_shared):
        download_hf_files(UPSTREAM_REPO, [source.path], download_dir)
        local = download_dir / source.path
        header_metadata = read_header_metadata(local)

        # The licence travels with the weights: check it against what we ship,
        # on the first file, before any of the long work.
        license_path = output_dir / "LICENSE"
        if not license_path.exists():
            ensure_license_file(output_dir, info)
        verify_embedded_license(header_metadata, license_path)

        if source.converter is not None:
            source.converter(local, output_dir, header_metadata)
        else:
            weights = load_safetensors(local)

            # The connector is written from the first variant that carries
            # it; the second variant only compares its fingerprint against
            # the first, never rewriting the file the first one wrote.
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

            # Every SOURCE_FILES entry without a converter must declare a
            # classify function (test_every_entry_can_convert_itself pins
            # this); asserting it here, rather than trusting the type alone,
            # is what lets classify_keys take a non-optional callable.
            assert source.classify is not None, f"{source.path} has neither classify nor converter"
            keys_by_component = classify_keys(weights, source.classify)
            for component_name, keys in keys_by_component.items():
                if component_name == "connector" and not write_connector:
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
                )

            # The upscalers carry their own config in the checkpoint's
            # __metadata__["config"], not in a separate file upstream.
            if source.path in UPSCALER_FILES.values():
                config = header_metadata.get("config")
                if config:
                    component_name = source.components[0]
                    config_path = output_dir / f"{component_name}_config.json"
                    with open(config_path, "w") as handle:
                        json.dump(json.loads(config), handle, indent=2)

            del weights
            gc.collect()
            mx.clear_cache()

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
        write_ltx25_quantize_config(output_dir, bits=args.bits, group_size=args.group_size)

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
]
