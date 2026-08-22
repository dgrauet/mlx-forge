"""LTX-2.5's Gemma-4 12B unified text encoder.

Upstream ships it inside the LTX-2.5 repo under a name of its own —
`gemma4-12b-with-proj-ltx-2.5` — because it is not a stock Gemma: it is an LTX
fine-tune with vision and audio branches, whose architecture string is
`Gemma4UnifiedForConditionalGeneration`. It therefore belongs to this recipe
rather than to a general Gemma conversion.

Its checkpoint carries five files as U8 tensors, the tokenizer among them at
32 MB. They are extracted to real files: nothing downstream can read a
tokenizer out of a safetensors entry.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import mlx.core as mx

from ..convert import load_safetensors, process_component

TEXT_ENCODER_FILE = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"

#: U8 tensor key -> the filename it becomes. `hf_asset__X` carries its own
#: filename after the prefix; `tokenizer_json` predates that convention.
ASSET_FILENAMES = {
    "tokenizer_json": "tokenizer.json",
    "hf_asset__tokenizer_config.json": "tokenizer_config.json",
    "hf_asset__chat_template.jinja": "chat_template.jinja",
    "hf_asset__generation_config.json": "generation_config.json",
    "hf_asset__processor_config.json": "processor_config.json",
}

_UNQUANTISED_PREFIXES = (
    "vision_model.",
    "text_embedding_projection.",
    "audio_projector.",
    "multi_modal_projector.",
)

_QUANTISED_SUFFIXES = (
    ".self_attn.q_proj.weight",
    ".self_attn.k_proj.weight",
    ".self_attn.v_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.gate_proj.weight",
    ".mlp.up_proj.weight",
    ".mlp.down_proj.weight",
)


def classify_text_encoder_key(key: str) -> str | None:
    """Route a text-encoder key to `text_encoder` or `text_encoder_asset`."""
    if key in ASSET_FILENAMES:
        return "text_encoder_asset"
    return "text_encoder"


def sanitize_text_encoder_key(key: str) -> str | None:
    """Convert a Gemma-4 weight key to MLX format.

    Upstream already uses the flat HuggingFace naming mlx-lm expects, so the
    only work is refusing the asset tensors, which are files and not weights.
    """
    if key in ASSET_FILENAMES:
        return None
    return key


def extract_assets(weights: dict[str, mx.array], output_dir: Path) -> list[Path]:
    """Write the five U8 asset tensors out as real files.

    Raises:
        SystemExit: An asset is missing. A pack whose tokenizer silently failed
            to extract loads and then produces nonsense; stopping here is the
            cheaper failure.
    """
    missing = [name for key, name in ASSET_FILENAMES.items() if key not in weights]
    if missing:
        raise SystemExit(
            "ERROR: the text encoder checkpoint is missing embedded assets: "
            + ", ".join(sorted(missing))
        )

    written: list[Path] = []
    for key, filename in ASSET_FILENAMES.items():
        tensor = weights[key].astype(mx.uint8)
        # mx.array implements the buffer protocol at runtime but its stubs
        # don't declare __buffer__, so ty can't see it satisfies Buffer.
        payload = bytes(memoryview(tensor))  # ty: ignore[invalid-argument-type]
        target = output_dir / filename
        target.write_bytes(payload)
        print(f"  Extracted {filename} ({len(payload) / 1024:.0f} KB)")
        written.append(target)
    return written


def convert_text_encoder(source_path: Path, output_dir: Path, header_metadata: dict) -> None:
    """Convert the Gemma-4 checkpoint: weights to one file, assets to five.

    A SOURCE_FILES hook rather than a classifier, because the five U8 tensors
    are files and the common loop only knows how to write weights.

    Args:
        source_path: The downloaded checkpoint.
        output_dir: The converted model directory.
        header_metadata: The checkpoint's `__metadata__`, whose `gemma_config`
            becomes text_encoder_config.json.
    """
    weights = load_safetensors(source_path)

    extract_assets(weights, output_dir)

    config = header_metadata.get("gemma_config")
    if config:
        with open(output_dir / "text_encoder_config.json", "w") as handle:
            json.dump(json.loads(config), handle, indent=2)

    keys = [k for k in weights if classify_text_encoder_key(k) == "text_encoder"]
    process_component(
        weights,
        "text_encoder",
        keys,
        output_dir,
        component_prefix="text_encoder",
        sanitizer=sanitize_text_encoder_key,
    )

    del weights
    gc.collect()
    mx.clear_cache()


def should_quantize_gemma(key: str, weight: mx.array) -> bool:
    """Only the transformer stack's Linear weights.

    Excluded on purpose: the embedding table (Gemma is a feature extractor
    here, so a quantised table shifts the starting point of every layer and the
    error surfaces as drifting conditioning, not as slightly worse text), the
    six projectors (the whole conditioning passes through them, for a
    negligible weight), and the vision branch (nine tensors).

    `quantize_component` reloads the file `process_component` wrote under
    `component_prefix="text_encoder"`, so `key` here is
    `text_encoder.vision_model...`, not the bare upstream name
    `_UNQUANTISED_PREFIXES` was written against. Strip the prefix first —
    mirrors `ltx25_should_quantize`'s `key.replace("transformer.", "", 1)` for
    the same reason.
    """
    bare_key = key.replace("text_encoder.", "", 1)
    if bare_key.startswith(_UNQUANTISED_PREFIXES):
        return False
    if bare_key.endswith((".scales", ".biases")):
        return False
    if weight.ndim != 2:
        return False
    return bare_key.endswith(_QUANTISED_SUFFIXES)
