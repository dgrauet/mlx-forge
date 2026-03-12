# Mistral Small 3.1 24B

[mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) — 24B parameter multimodal (vision + language) model with a Pixtral vision encoder and dense transformer LLM.

> MLX Forge converts all three components: language_model (dense transformer), vision_tower (Pixtral), and multimodal_projector.

## Quick Start

```bash
# Convert (downloads ~48 GB)
mlx-forge convert mistral-small

# Convert + quantize
mlx-forge convert mistral-small --quantize --bits 8

# Preview what will happen
mlx-forge convert mistral-small --dry-run

# Convert from local checkpoint directory
mlx-forge convert mistral-small --checkpoint ./models/

# Validate
mlx-forge validate mistral-small models/mistral-small-mlx
```

## CLI Options

### Convert

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(download)* | Path to local checkpoint directory |
| `--output` | `models/mistral-small-mlx[-q<bits>]` | Output directory |
| `--quantize` | off | Quantize weights after conversion |
| `--bits` | `8` | Quantization bits (`4` or `8`) |
| `--group-size` | `64` | Quantization group size |
| `--dry-run` | off | Preview plan without downloading or writing |

## Architecture

```
 ┌──────────────────────────────────────────────────────────┐
 │  Mistral Small 3.1 (mistral3)                            │
 │                                                          │
 │  ┌──────────────────────────┐  ┌───────────────────────┐ │
 │  │  language_model           │  │  vision_tower         │ │
 │  │  40 layers, ~44 GB        │  │  24 layers, ~800 MB   │ │
 │  │  Dense transformer        │  │  Pixtral encoder      │ │
 │  └────────────┬─────────────┘  └──────────┬────────────┘ │
 │               │                           │              │
 │               │  text tokens              │  image       │
 │               │                           │  patches     │
 │               │                           │              │
 │               └──────────┬────────────────┘              │
 │                          │                               │
 │               ┌──────────▼──────────┐                    │
 │               │  multimodal_        │  ~50 MB (bf16)     │
 │               │  projector          │  Linear projection │
 │               │                     │  NOT quantized     │
 │               └─────────────────────┘                    │
 └──────────────────────────────────────────────────────────┘
```

### Inference Flow

1. **Vision tower (Pixtral)**: encodes image patches into visual embeddings
2. **Multimodal projector**: projects visual embeddings into the language model's hidden space
3. **Language model**: dense transformer generates text tokens conditioned on text + visual inputs

## Components

### language_model

Dense transformer with grouped-query attention (GQA).

| Parameter | Value |
|-----------|-------|
| Layers | 40 |
| Heads | 32 (8 KV heads, GQA) |
| Head dim | 128 |
| Hidden dim | 5120 |
| Intermediate (FFN) | 32768 |
| Vocab size | 131,072 |
| RoPE base | 1,000,000,000 |
| Tied embeddings | No (separate lm_head) |

**Per-layer weight keys:**
```
layers.{i}.self_attn.q_proj.weight       # Q projection (Linear)
layers.{i}.self_attn.k_proj.weight       # K projection (Linear)
layers.{i}.self_attn.v_proj.weight       # V projection (Linear)
layers.{i}.self_attn.o_proj.weight       # output projection (Linear)
layers.{i}.input_layernorm.weight        # pre-attention norm (RMSNorm)
layers.{i}.mlp.gate_proj.weight          # SwiGLU gate (Linear)
layers.{i}.mlp.down_proj.weight          # SwiGLU down (Linear)
layers.{i}.mlp.up_proj.weight            # SwiGLU up (Linear)
layers.{i}.post_attention_layernorm.weight  # pre-FFN norm (RMSNorm)
```

**Root keys:**
```
embed_tokens.weight    # token embeddings (131,072 x 5120)
norm.weight            # final RMSNorm
lm_head.weight         # output head (Linear, NOT quantized)
```

### vision_tower (Pixtral)

Pixtral vision transformer encoding image patches.

| Parameter | Value |
|-----------|-------|
| Layers | 24 |
| Heads | 16 |
| Head dim | 64 |
| Hidden dim | 1024 |
| Intermediate (FFN) | 4096 |
| Patch size | 14 |
| Image size | 1540 |

### multimodal_projector

Linear projection layer mapping vision embeddings to language model hidden space.

## Key Classification

| Original prefix | Component | Sanitized prefix |
|-----------------|-----------|-----------------|
| `language_model.model.*` | language_model | *(strip `language_model.model.`)* |
| `language_model.lm_head.*` | language_model | `lm_head.*` |
| `vision_tower.*` | vision_tower | *(strip `vision_tower.`)* |
| `multimodal_projector.*` | multimodal_projector | *(strip `multimodal_projector.`)* |

All components use Linear, RMSNorm, and Embedding layers only. No conv transposition is needed.

## Quantization Strategy

**Quantized** (Linear weights, 2D, >= 256 elements, language_model only):
- `layers.{i}.self_attn.q_proj.weight`
- `layers.{i}.self_attn.k_proj.weight`
- `layers.{i}.self_attn.v_proj.weight`
- `layers.{i}.self_attn.o_proj.weight`
- `layers.{i}.mlp.gate_proj.weight`
- `layers.{i}.mlp.down_proj.weight`
- `layers.{i}.mlp.up_proj.weight`

**NOT quantized**:
- `embed_tokens.weight` — embedding table
- `lm_head.weight` — output head (sensitive to quantization)
- `*norm*.weight` — all RMSNorm layers
- All vision_tower weights — entire component skipped (small, sensitive)
- All multimodal_projector weights — entire component skipped (small, sensitive)

## Output Files

| File | Content |
|------|---------|
| `language_model.safetensors` | Dense transformer weights (~44 GB bf16, ~22 GB int8) |
| `vision_tower.safetensors` | Pixtral vision encoder weights (~800 MB bf16, not quantized) |
| `multimodal_projector.safetensors` | Projection weights (~50 MB bf16, not quantized) |
| `config.json` | Model configuration (copied from source) |
| `tokenizer.json` | Tokenizer (copied from source) |
| `tokenizer_config.json` | Tokenizer config (copied from source) |
| `special_tokens_map.json` | Special tokens (copied from source) |
| `preprocessor_config.json` | Vision preprocessor config (copied from source) |
| `split_model.json` | MLX Forge split metadata |
| `quantize_config.json` | Quantization config (if quantized) |

## Checkpoint Format

The source checkpoint is **sharded** (10 safetensors files + index, ~48 GB total):
- `model-00001-of-00010.safetensors` through `model-00010-of-00010.safetensors`
- `model.safetensors.index.json` — shard index

MLX Forge loads shards sequentially via the index file. No optional dependencies required — all weights are in safetensors format.

## Validation Checks

| Check | What it verifies |
|-------|-----------------|
| File structure | All expected files exist |
| Config `model_type` | `mistral3` |
| Text config | 40 layers, 32 heads, hidden_size 5120, 8 KV heads |
| Vision config | 24 layers, 16 heads |
| language_model keys | No `language_model.model.` prefix, embedding present, lm_head present, 40 layers |
| vision_tower keys | No `vision_tower.` prefix, 24 layers, NOT quantized |
| multimodal_projector keys | No `multimodal_projector.` prefix, NOT quantized |
| Quantization | Scales/biases pairs for quantized weights in language_model |
