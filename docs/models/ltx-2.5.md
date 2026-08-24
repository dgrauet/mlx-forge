# LTX-2.5

[Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) is a gated audio-video DiT
(48 transformer blocks) shipped **pre-split** upstream: nine independent checkpoint files
instead of one unified safetensors. This page documents the LTX-2.5 recipe: CLI usage,
components, gating, and known gotchas.

## Quick Start

```bash
# Convert everything: both DiT variants + all shared components (~76 GB download)
mlx-forge convert ltx-2.5

# Convert + quantize (DiT transformer blocks + Gemma-4 text encoder)
mlx-forge convert ltx-2.5 --quantize --bits 8

# One variant only
mlx-forge convert ltx-2.5 --variant dev

# Delta workflow: only the transformers, for adding a variant to an existing repo
mlx-forge convert ltx-2.5 --variant dev --skip-shared --output models/ltx-2.5-delta

# Backfill embedded_config.json into an existing pack (headers only, no weights read)
mlx-forge convert ltx-2.5 --config-only --output models/ltx-2.5-mlx

# Preview the plan with download/output footprints
mlx-forge convert ltx-2.5 --quantize --bits 8 --dry-run

# Validate (auto-detects delta packs)
mlx-forge validate ltx-2.5 models/ltx-2.5-mlx
```

## CLI Options

### Convert

| Flag | Default | Description |
|------|---------|-------------|
| `--variant` | *(both)* | Transformer variant to convert (`dev`, `distilled`); repeat for several |
| `--output` | `models/ltx-2.5-mlx[-q<bits>]` | Output directory |
| `--quantize` | off | Quantize transformer-block and text-encoder Linear weights |
| `--bits` | `8` | Quantization bits (`4` or `8`) |
| `--group-size` | `64` | Quantization group size |
| `--skip-shared` | off | Delta mode: only the transformers (marks `"delta": true`) |
| `--lora` | *(all)* | LoRA file(s) to sync, copied as-is |
| `--config-only` | off | Emit `embedded_config.json` into an existing pack without reconverting |
| `--dry-run` | off | Preview the plan, downloading nothing |

### Validate

| Flag | Default | Description |
|------|---------|-------------|
| `model_dir` | *(required)* | Path to converted model directory |

`split` is a printing no-op: upstream ships the model already split.

## Components

The declarative `SOURCE_FILES` table routes nine upstream files; each is classified
per-file because the two video VAE files expose identical top-level prefixes for
different architectures.

| Component | Notes |
|---|---|
| `transformer-{dev,distilled}` | 48-block audio-video DiT; connector split out |
| `connector` | Identical between variants (fingerprint-checked at conversion) |
| `text_encoder` | Gemma-4; tokenizer/config assets extracted from embedded U8 tensors |
| `vae_encoder_conv` / `vae_decoder_conv`, `vae_encoder_av` / `vae_decoder_av` | Two video VAEs; per-channel statistics shared to both encoder and decoder |
| `audio_vae`, `vocoder`, `duration_head` | Audio stack |
| `spatial_upscaler_x2_v1_0`, `temporal_upscaler_x2_v1_0` | Upscalers |

The DiT config embedded in the checkpoints is emitted verbatim as
`embedded_config.json` (transformer + scheduler sections, identical across variants).

## Gating and Licence

- Upstream is **gated**; the recipe declares `gated` for our mirrors, and the first
  upload refuses to run without `--set-gated` so an open mirror cannot be created by
  accident.
- The licence text is fetched from GitHub (`license_source`) and verified against the
  copy embedded in the checkpoints that carry one; each pack ships the agreement its
  own weights carry.

## Sources directory

Downloads land in a sibling of the output directory (`<output>-src`), never inside the
pack — `upload` walks the pack recursively and must not push upstream weights.

## Gotchas

- Quantization consumes ~20.6 GB peak (measured); `quantize_weights` empties its input
  dict as it goes — do not reuse it after the call.
- `EXPECTED_TENSOR_COUNTS` holds bf16 counts; validate derives quantized expectations
  from the actual `.scales` present.
- The delta workflow (convert → validate → upload `--add-only` → `--card-only`) is
  documented in the repo-level CLAUDE.md.
