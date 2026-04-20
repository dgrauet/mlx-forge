"""ERNIE-Image Prompt Enhancer recipe — standalone Ministral3ForCausalLM.

The Prompt Enhancer is a 3B Mistral-family CausalLM that expands short user
prompts into detailed Chinese visual descriptions before the text encoder.
Baidu ships it alongside each ERNIE-Image variant in a ``pe/`` subfolder, but
the weights are byte-identical across Turbo / SFT — hosting it as its own
repository avoids ~7 GB of duplication per image variant.

Layout produced by this recipe::

    models/ernie-image-pe-mlx[-q<bits>]/
    ├── pe.safetensors          # Ministral3 CausalLM weights (optionally quantized)
    ├── pe_config.json          # copy of pe/config.json
    ├── chat_template.jinja     # copy of pe/chat_template.jinja (Chinese system prompt)
    ├── tokenizer.json          # copy of pe_tokenizer/tokenizer.json (Mistral chat tokens)
    ├── tokenizer_config.json   # copy of pe_tokenizer/tokenizer_config.json
    ├── generation_config.json  # copy of pe/generation_config.json
    ├── split_model.json        # mlx-forge metadata
    └── quantize_config.json    # only when --quantize

Key translations: none. The HF checkpoint for ``Ministral3ForCausalLM`` already
uses ``model.layers.*`` / ``model.embed_tokens.*`` / ``model.norm.*``, which
matches mlx-lm's ``ministral3.Model`` one-to-one — no renames or transposes.

Tied word embeddings: ``tie_word_embeddings=true`` in ``pe/config.json``. Baidu
ships a redundant ``lm_head.weight`` anyway (some HF export tools do this even
when tied); the recipe drops it at conversion time to save ~768 MB fp16. mlx-lm
computes the tied lm_head on the fly via ``embed_tokens.as_linear(...)``.
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
from ..validate import (
    ValidationResult,
    validate_file_exists,
    validate_quantization,
)

# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------

# Turbo and SFT ship identical PE weights — pick Turbo as the canonical source.
REPO_SOURCE = "baidu/ERNIE-Image-Turbo"

PE_FILES = [
    "pe/config.json",
    "pe/model.safetensors",
    "pe/chat_template.jinja",
    "pe/generation_config.json",
]

# Tokenizer ships in pe_tokenizer/ (NOT pe/) per model_index.json:
#   "pe_tokenizer": ["transformers", "TokenizersBackend"]
# The tokenizer_config.json file is identical in both subfolders, but the
# actual vocab (tokenizer.json) lives ONLY in pe_tokenizer/.
PE_TOKENIZER_FILES = [
    "pe_tokenizer/tokenizer.json",
    "pe_tokenizer/tokenizer_config.json",
]

ALL_FILES = PE_FILES + PE_TOKENIZER_FILES

_PE_SIZE_MB = 7700  # ~7.1 GB fp16 single-shard safetensors


# ---------------------------------------------------------------------------
# Sanitization — none needed, HF keys already match mlx-lm layout
# ---------------------------------------------------------------------------


def _sanitize_pe_key(key: str) -> str:
    return key


SANITIZERS = {"pe": _sanitize_pe_key}
TRANSFORMS = {"pe": None}


# ---------------------------------------------------------------------------
# Quantization predicate
# ---------------------------------------------------------------------------


def ernie_image_pe_should_quantize(key: str, weight: mx.array) -> bool:
    """Quantize block Linears + the embed_tokens table.

    Matches the text-encoder recipe's behavior. ``embed_tokens`` IS quantized:
    MLX's ``QuantizedEmbedding`` supports ``as_linear`` natively, so the tied
    lm_head path keeps working. Skipping it would waste ~768 MB (131072 × 3072).
    Norms and biases stay fp16/bf16.
    """
    if weight.ndim < 2:
        return False
    if "norm" in key or ".bias" in key:
        return False
    return weight.shape[0] >= 256 and weight.shape[1] >= 256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_output_dir(quantize: bool, bits: int) -> Path:
    suffix = f"-q{bits}" if quantize else ""
    return Path("models") / f"ernie-image-pe-mlx{suffix}"


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def dry_run(args) -> None:
    q_label = f" + int{args.bits} quantization" if args.quantize else ""
    print("=" * 60)
    print(f"DRY RUN — ERNIE-Image Prompt Enhancer conversion plan{q_label}")
    print("=" * 60)
    print(f"\nSource: {REPO_SOURCE} (pe/ + pe_tokenizer/)")
    print(f"Total download: ~{fmt_size(_PE_SIZE_MB)}")
    print(f"Files to download: {len(ALL_FILES)}")

    output_dir = args.output or _default_output_dir(args.quantize, args.bits)
    print(f"Output: {output_dir}")

    print("\nKey translations: none (HF layout matches mlx-lm ministral3 one-to-one)")
    print("Tied embeddings: redundant lm_head.weight dropped at conversion time (~768 MB fp16)")
    if args.quantize:
        print(f"\nQuantization: int{args.bits}, group_size={args.group_size}")
        print("  Quantized: block Linears (attention + MLP projections) + embed_tokens")
        print("  Skipped: all norms + biases (not Linear-compatible)")

    print("\n" + "=" * 60)
    print("No files downloaded or written (--dry-run)")


# ---------------------------------------------------------------------------
# Convert
# ---------------------------------------------------------------------------


def convert(args) -> None:
    if args.dry_run:
        dry_run(args)
        return

    output_dir = Path(args.output) if args.output else _default_output_dir(args.quantize, args.bits)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if args.checkpoint:
        checkpoint_dir = Path(args.checkpoint)
        print(f"Using local checkpoint: {checkpoint_dir}")
    else:
        checkpoint_dir = Path("models") / "ernie-image-turbo-src"
        print(f"Downloading {REPO_SOURCE} PE files...")
        print(f"(~{fmt_size(_PE_SIZE_MB)}, may take a while)")
        download_hf_files(REPO_SOURCE, ALL_FILES, checkpoint_dir)

    pe_subdir = checkpoint_dir / "pe"
    pe_tok_subdir = checkpoint_dir / "pe_tokenizer"

    print("\n" + "=" * 60)
    print(f"Processing pe (~{fmt_size(_PE_SIZE_MB)})")

    weights = load_weights(pe_subdir, single_filename="model.safetensors")

    # `tie_word_embeddings=True` in pe/config.json — some HF export tools ship both
    # `lm_head.weight` and `model.embed_tokens.weight` even though they're tied.
    # mlx-lm drops `lm_head.weight` at load time; save ~768 MB (131072 × 3072 × 2 B)
    # by dropping it at conversion time too.
    keys = [k for k in weights if k != "lm_head.weight"]

    count = process_component(
        weights,
        "pe",
        keys,
        output_dir,
        "pe",
        sanitizer=SANITIZERS["pe"],
        transform=TRANSFORMS["pe"],
    )

    del weights
    gc.collect()
    mx.clear_cache()

    # Copy metadata + tokenizer + chat template
    shutil.copy2(pe_subdir / "config.json", output_dir / "pe_config.json")
    if (pe_subdir / "generation_config.json").exists():
        shutil.copy2(pe_subdir / "generation_config.json", output_dir / "generation_config.json")
    if (pe_subdir / "chat_template.jinja").exists():
        shutil.copy2(pe_subdir / "chat_template.jinja", output_dir / "chat_template.jinja")

    # Tokenizer — copy from pe_tokenizer/ (where the actual vocab lives).
    for tok_file in ("tokenizer.json", "tokenizer_config.json"):
        src = pe_tok_subdir / tok_file
        if src.exists():
            shutil.copy2(src, output_dir / tok_file)

    if args.quantize:
        quantize_component(
            output_dir,
            "pe",
            bits=args.bits,
            group_size=args.group_size,
            should_quantize=ernie_image_pe_should_quantize,
        )
        qconfig = {
            "quantization": {
                "bits": args.bits,
                "group_size": args.group_size,
                "skip_components": [],
            }
        }
        with open(output_dir / "quantize_config.json", "w") as f:
            json.dump(qconfig, f, indent=2)

    split_info: dict = {
        "format": "split",
        "source": REPO_SOURCE + "/pe",
        "components": ["pe"],
    }
    if args.quantize:
        split_info["quantized"] = True
        split_info["quantization_bits"] = args.bits
    with open(output_dir / "split_model.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Conversion complete: {count} weights")
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
    validate_file_exists(model_dir, "pe.safetensors", result)
    validate_file_exists(model_dir, "pe_config.json", result)
    validate_file_exists(model_dir, "tokenizer.json", result)
    validate_file_exists(model_dir, "tokenizer_config.json", result)
    validate_file_exists(model_dir, "chat_template.jinja", result)

    print("\n== PE Weights ==")
    pe_path = model_dir / "pe.safetensors"
    if pe_path.exists():
        weights = mx.load(str(pe_path))
        keys = set(weights.keys())
        print(f"  Keys: {len(keys)}")

        # All keys should be prefixed with `pe.` by the recipe.
        result.check(
            all(k.startswith("pe.") for k in keys),
            "All keys prefixed with 'pe.'",
        )

        # Expected backbone structure.
        result.check(
            any("pe.model.embed_tokens.weight" == k for k in keys),
            "model.embed_tokens present",
        )
        result.check(
            any("pe.model.norm.weight" == k for k in keys),
            "model.norm present",
        )
        # Ministral3 has 26 layers — check the first and last are present.
        result.check(
            any("pe.model.layers.0.self_attn.q_proj" in k for k in keys),
            "layers.0 present",
        )
        result.check(
            any("pe.model.layers.25.self_attn.q_proj" in k for k in keys),
            "layers.25 present (26 total)",
        )

        # Tied embeddings: no separate lm_head.weight expected (dropped by sanitize).
        has_lm_head = any(k == "pe.lm_head.weight" for k in keys)
        result.check(
            not has_lm_head,
            "lm_head.weight absent (tied to embed_tokens)",
        )

        if is_quantized:
            validate_quantization(weights, result, block_key="layers")

    print("\n" + "=" * 60)
    result.summary()


# ---------------------------------------------------------------------------
# Argument registration
# ---------------------------------------------------------------------------


def add_convert_args(parser) -> None:
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
        help="Output directory (default: models/ernie-image-pe-mlx[-q<bits>])",
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Quantize block Linears after conversion"
    )
    parser.add_argument(
        "--bits", type=int, default=4, choices=[4, 8], help="Quantization bits (default: 4)"
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
    print("ERNIE-Image PE is a single-component model — no splitting needed.")
