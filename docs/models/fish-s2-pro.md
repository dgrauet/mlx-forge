# Fish Audio S2 Pro

[fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) — 5B parameter Dual-AR text-to-speech model based on a modified Qwen3-4B backbone.

> **Phase 1**: MLX Forge converts the transformer components (text_model + audio_decoder). The DAC codec (vocoder) is not yet supported.

## Quick Start

```bash
# Convert (downloads ~9.2 GB)
mlx-forge convert fish-s2-pro

# Convert + quantize
mlx-forge convert fish-s2-pro --quantize --bits 8

# Preview what will happen
mlx-forge convert fish-s2-pro --dry-run

# Convert from local checkpoint directory
mlx-forge convert fish-s2-pro --checkpoint ./models/

# Validate
mlx-forge validate fish-s2-pro models/fish-s2-pro-mlx
```

## CLI Options

### Convert

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(download)* | Path to local checkpoint directory |
| `--output` | `models/fish-s2-pro-mlx[-q<bits>]` | Output directory |
| `--quantize` | off | Quantize weights after conversion |
| `--bits` | `8` | Quantization bits (`4` or `8`) |
| `--group-size` | `64` | Quantization group size |
| `--dry-run` | off | Preview plan without downloading or writing |

## Architecture

```
 ┌─────────────────────────────────────────────────────────┐
 │  Fish S2 Pro (fish_qwen3_omni)                          │
 │                                                         │
 │  ┌───────────────────────┐  ┌─────────────────────────┐ │
 │  │  text_model (Slow AR) │  │  audio_decoder (Fast AR)│ │
 │  │  36 layers, ~4B       │  │  4 layers, ~400M        │ │
 │  │  Qwen3-based          │  │  Qwen3-based            │ │
 │  └───────────┬───────────┘  └──────────┬──────────────┘ │
 │              │                         │                │
 │              │  semantic tokens        │  10 codebook   │
 │              │  (1 per timestep)       │  tokens/step   │
 │              │                         │                │
 │              └─────────┬───────────────┘                │
 │                        │                                │
 │              ┌─────────▼─────────┐                      │
 │              │  DAC Codec        │  NOT YET CONVERTED   │
 │              │  (vocoder)        │  1.87 GB             │
 │              │  Conv1d encoder/  │                      │
 │              │  decoder + RVQ    │                      │
 │              └───────────────────┘                      │
 └─────────────────────────────────────────────────────────┘
```

### Dual-AR Inference Flow

1. **Slow AR (text_model)**: autoregressively generates one semantic codebook token per timestep
2. **Fast AR (audio_decoder)**: at each timestep, generates 9 residual codebook tokens (total 10 codebooks)
3. **DAC Codec**: converts 10-codebook token sequence to audio waveform

## Components

### text_model (Slow AR)

Modified Qwen3-4B transformer with audio extensions.

| Parameter | Value |
|-----------|-------|
| Layers | 36 |
| Heads | 32 (8 KV heads, GQA) |
| Head dim | 128 |
| Hidden dim | 2560 |
| Intermediate (FFN) | 9728 |
| Vocab size | 155,776 |
| RoPE base | 1,000,000 |
| QK-norm | Yes |
| Tied embeddings | Yes (no separate lm_head) |

**Per-layer weight keys:**
```
layers.{i}.attention.wqkv.weight      # fused QKV projection (Linear)
layers.{i}.attention.wo.weight         # output projection (Linear)
layers.{i}.attention.q_norm.weight     # Q-norm (RMSNorm)
layers.{i}.attention.k_norm.weight     # K-norm (RMSNorm)
layers.{i}.attention_norm.weight       # pre-attention norm (RMSNorm)
layers.{i}.feed_forward.w1.weight      # SwiGLU gate (Linear)
layers.{i}.feed_forward.w2.weight      # SwiGLU down (Linear)
layers.{i}.feed_forward.w3.weight      # SwiGLU up (Linear)
layers.{i}.ffn_norm.weight             # pre-FFN norm (RMSNorm)
```

**Root keys:**
```
embeddings.weight    # token embeddings (155,776 x 2560)
norm.weight          # final RMSNorm
```

### audio_decoder (Fast AR)

Smaller transformer predicting residual codebook tokens.

| Parameter | Value |
|-----------|-------|
| Layers | 4 |
| Heads | 32 (8 KV heads, GQA) |
| Head dim | 128 |
| Hidden dim | 2560 |
| Intermediate (FFN) | 9728 |
| Num codebooks | 10 |
| Vocab size | 4,096 |
| QK-norm | No |
| Tied embeddings | No (separate output head) |

**Root keys:**
```
codebook_embeddings.weight    # codebook embeddings (10 x 4096)
embeddings.weight             # input embeddings
norm.weight                   # final RMSNorm
output.weight                 # output head (Linear)
```

## Key Classification

| Original prefix | Component | Sanitized prefix |
|-----------------|-----------|-----------------|
| `text_model.model.*` | text_model | *(strip `text_model.model.`)* |
| `audio_decoder.*` | audio_decoder | *(strip `audio_decoder.`)* |

No conv transposition needed — all layers are Linear, RMSNorm, or Embedding.

## Quantization Strategy

**Quantized** (Linear weights, 2D, >= 256 elements):
- `*.attention.wqkv.weight`
- `*.attention.wo.weight`
- `*.feed_forward.w1.weight`
- `*.feed_forward.w2.weight`
- `*.feed_forward.w3.weight`
- `audio_decoder.output.weight`

**NOT quantized**:
- `*.embeddings.weight` — embedding tables
- `*.codebook_embeddings.weight` — codebook embedding
- `*norm*.weight` — all RMSNorm layers

## Output Files

| File | Content |
|------|---------|
| `text_model.safetensors` | Slow AR weights (~8.5 GB fp16, ~4.3 GB int8) |
| `audio_decoder.safetensors` | Fast AR weights (~600 MB fp16, ~300 MB int8) |
| `config.json` | Model configuration (copied from source) |
| `tokenizer.json` | Tokenizer (copied from source) |
| `tokenizer_config.json` | Tokenizer config (copied from source) |
| `special_tokens_map.json` | Special tokens (copied from source) |
| `split_model.json` | MLX Forge split metadata |
| `quantize_config.json` | Quantization config (if quantized) |

## Checkpoint Format

The source checkpoint is **sharded** (2 safetensors files + index):
- `model-00001-of-00002.safetensors` (~5 GB) — text_model layers 0-20
- `model-00002-of-00002.safetensors` (~4.1 GB) — text_model layers 21-35 + audio_decoder
- `model.safetensors.index.json` — shard index

MLX Forge loads shards sequentially via the index file.

## Phase 2 Roadmap (DAC Codec)

The DAC codec (1.87 GB) is not yet converted. It requires:

- Loading a PyTorch `.pth` checkpoint
- Conv1d / ConvTranspose1d transposition (already supported in `transpose.py`)
- Weight normalization fusion (`weight_g` + `weight_v` -> `weight`)
- Custom `Snake1d` activation implementation for MLX
- Residual Vector Quantization (RVQ) codebook handling

The codec is needed for end-to-end audio generation but not for the transformer weight conversion.

## Validation Checks

| Check | What it verifies |
|-------|-----------------|
| File structure | All expected files exist |
| Config `model_type` | `fish_qwen3_omni` |
| Text config | 36 layers, 32 heads, dim 2560 |
| Audio config | 4 layers, 10 codebooks |
| text_model keys | No `text_model.model.` prefix, embedding present, 36 layers, QK-norm present |
| audio_decoder keys | No `audio_decoder.` prefix, codebook embeddings, output head, 4 layers |
| Quantization | Scales/biases pairs for quantized weights |
