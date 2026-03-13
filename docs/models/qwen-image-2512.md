# Qwen-Image

[Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) is a ~57B-parameter text-to-image diffusion model using a Flux-style MMDiT (Multi-Modal Diffusion Transformer) architecture.

## Quick Start

```bash
# Convert (~57.7 GB download)
mlx-forge convert qwen-image-2512

# Convert + quantize in one pass
mlx-forge convert qwen-image-2512 --quantize --bits 8

# Preview conversion plan (no download, no writes)
mlx-forge convert qwen-image-2512 --dry-run

# Convert from a local checkpoint directory
mlx-forge convert qwen-image-2512 --checkpoint /path/to/qwen-image-2512-dir

# Validate a converted model
mlx-forge validate qwen-image-2512 models/qwen-image-2512-mlx
```

## CLI Options

### Convert

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(download)* | Path to local checkpoint directory |
| `--output` | `models/qwen-image-2512-mlx` | Output directory |
| `--quantize` | off | Quantize transformer and text_encoder after conversion |
| `--bits` | `8` | Quantization bits (`4` or `8`) |
| `--group-size` | `64` | Quantization group size |
| `--dry-run` | off | Preview conversion plan without downloading or writing |

### Validate

| Flag | Default | Description |
|------|---------|-------------|
| `model_dir` | *(required)* | Path to converted model directory |

## Model Architecture

Qwen-Image consists of three components, already split by directory on HuggingFace:

```
                ┌─────────────────────────────────────┐
                │       HuggingFace Repository        │
                │          (~57.7 GB total)            │
                └──────────────┬──────────────────────┘
                               │ per-directory download
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌────────────┐    ┌──────────────┐      ┌──────────┐
   │ Transformer │    │ Text Encoder │      │   VAE    │
   │   MMDiT     │    │  Qwen2.5-VL  │      │ AutoKL   │
   │  60 layers  │    │ 28L LLM +    │      │  Conv2d  │
   │  24 heads   │    │ 32L vision   │      │          │
   └────────────┘    └──────────────┘      └──────────┘
     ~40.9 GB           ~16.6 GB             ~254 MB
     (9 shards)         (4 shards)          (1 file)
```

### Component Details

#### Transformer (`QwenImageTransformer2DModel`)

- 60 layers, 24 attention heads, 128 head dim
- Joint attention dim: 3584
- Flux-style MMDiT with joint image-text blocks
- Keys: `transformer_blocks.*.attn.*`, `img_mlp.*`, `txt_mlp.*`, `img_mod.*`, `txt_mod.*`
- Input: `img_in.*`, Output: `proj_out.*`, `norm_out.*`
- Timestep: `time_text_embed.*`

#### Text Encoder (`Qwen2_5_VLForConditionalGeneration`)

- **LLM**: 28 hidden layers, 28 attention heads, 4 KV heads (GQA)
- Hidden size: 3584, intermediate: 18944, vocab: 152064
- **Vision encoder**: 32 blocks, 16 heads, hidden size 1280
- Keys: `model.layers.*`, `visual.blocks.*`, `model.embed_tokens.*`, `lm_head.*`

#### VAE (`AutoencoderKLQwenImage`)

- z_dim: 16, base_dim: 96
- Conv2d-based encoder/decoder (requires transposition for MLX)
- Temporal downsampling support: `[false, true, true]`

## Source Checkpoint Structure

Unlike other recipes (LTX, Mistral), Qwen-Image files are **already organized per-component** on HuggingFace:

```
Qwen/Qwen-Image-2512/
├── transformer/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors.index.json
│   └── diffusion_pytorch_model-{00001..00009}-of-00009.safetensors
├── text_encoder/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors.index.json
│   └── model-{00001..00004}-of-00004.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── tokenizer/
│   ├── vocab.json, merges.txt, tokenizer_config.json, ...
├── scheduler/
│   └── scheduler_config.json
└── model_index.json
```

No key classification step is needed — each component's weights are loaded from its own subdirectory.

## Quantization Strategy

Transformer and text_encoder Linear weights are quantized. VAE is skipped (conv-heavy).

**Deliberately excluded layers:**

| Layer | Reason |
|-------|--------|
| `img_in` (image input projection) | Sensitive input layer |
| `txt_in` (text input projection) | Sensitive input layer |
| `proj_out` (output projection) | Sensitive output layer |
| `norm_out` (output normalization) | Normalization layer |
| `time_text_embed` (timestep embedding) | Small, sensitive |
| `*_mod.*` (modulation layers) | Adaptive normalization |
| `embed_tokens` (LLM embedding) | Embedding layer |
| `lm_head` (LLM head) | Output projection |
| `patch_embed` (vision patch embedding) | Input layer |
| `merger` (vision merger) | Small projection |
| All norm layers | Too sensitive |
| Conv weights (VAE) | Not Linear — incompatible |

Recommended: `--bits 8 --group-size 64` (transformer: ~40.9 GB → ~20 GB, text_encoder: ~16.6 GB → ~8 GB).

## Output Files

```
output_dir/
├── transformer.safetensors        # ~40.9 GB (fp16) or ~20 GB (int8)
├── text_encoder.safetensors       # ~16.6 GB (fp16) or ~8 GB (int8)
├── vae.safetensors                # ~254 MB (always fp16)
├── split_model.json               # Split metadata
├── transformer_config.json        # Transformer architecture config
├── text_encoder_config.json       # Text encoder config
├── vae_config.json                # VAE config
├── model_index.json               # Pipeline config
├── scheduler_scheduler_config.json
├── tokenizer_*.json / .txt        # Tokenizer files
└── quantize_config.json           # Only if --quantize was used
```

## Validation Checks

The `validate` command verifies:

- All expected files exist (transformer, text_encoder, vae safetensors)
- Transformer blocks present with expected structural elements (img_in, proj_out, time_text_embed)
- 28 LLM layers in text_encoder
- 32 vision encoder blocks in text_encoder
- Embedding and lm_head present in text_encoder
- Conv2d weights present in VAE
- Quantization pairs (.scales/.biases) consistent if quantized
