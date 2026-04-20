# ERNIE-Image

[baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) is an 8B single-stream Diffusion Transformer for text-to-image generation, released by Baidu. Two variants share the same architecture; only the scheduler and guidance differ:

- **SFT** ([`baidu/ERNIE-Image`](https://huggingface.co/baidu/ERNIE-Image)) — 50-step, guidance ≈ 5
- **Turbo** ([`baidu/ERNIE-Image-Turbo`](https://huggingface.co/baidu/ERNIE-Image-Turbo)) — 8-step distilled, guidance = 1.0

## Quick Start

```bash
# Convert Turbo at fp16 (~31.6 GB source → ~23 GB output)
mlx-forge convert ernie-image --variant turbo

# Convert + quantize in one pass (12 GB int8, fits on 24-32 GB Macs)
mlx-forge convert ernie-image --variant turbo --quantize --bits 8

# int4 (6.4 GB, fits on 16 GB Macs)
mlx-forge convert ernie-image --variant sft --quantize --bits 4

# Preview conversion plan (no download, no writes)
mlx-forge convert ernie-image --variant turbo --dry-run

# Convert from a local checkpoint directory
mlx-forge convert ernie-image --variant turbo --checkpoint /path/to/ERNIE-Image-Turbo

# Validate a converted model
mlx-forge validate ernie-image models/ernie-image-turbo-mlx-q8
```

## CLI Options

### Convert

| Flag | Default | Description |
|------|---------|-------------|
| `--variant` | `turbo` | Which variant to convert: `sft` (50-step) or `turbo` (8-step) |
| `--checkpoint` | *(download)* | Path to local checkpoint directory |
| `--output` | `models/ernie-image-<variant>-mlx[-q<bits>]` | Output directory |
| `--quantize` | off | Quantize transformer Linear weights after conversion |
| `--bits` | `8` | Quantization bits (`4` or `8`) |
| `--group-size` | `64` | Quantization group size |
| `--dry-run` | off | Preview conversion plan without downloading or writing |

### Validate

| Flag | Default | Description |
|------|---------|-------------|
| `model_dir` | *(required)* | Path to converted model directory |

## Model Architecture

ERNIE-Image consists of three components, already split by directory on HuggingFace:

```
                ┌─────────────────────────────────────┐
                │       HuggingFace Repository        │
                │           (~31.6 GB total)          │
                └──────────────┬──────────────────────┘
                               │ per-directory download
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌────────────┐    ┌──────────────┐      ┌──────────┐
   │ Transformer │    │ Text Encoder │      │   VAE    │
   │    DiT      │    │   Mistral3   │      │ AutoKL   │
   │  36 layers  │    │  26 layers   │      │  Flux2   │
   │   32 heads  │    │  GQA 32/8    │      │  Conv2d  │
   └────────────┘    └──────────────┘      └──────────┘
    ~16.1 GB           ~7.7 GB              ~168 MB
    (2 shards)         (1 file)             (1 file)
```

### Component Details

#### Transformer (`ErnieImageTransformer2DModel`)

- 36 layers, 32 attention heads, head_dim 128
- Hidden size: 4096, FFN hidden: 12288 (GeGLU)
- `qk_layernorm=True` — RMSNorm on Q and K before RoPE
- **Triple-axis RoPE** `[32, 48, 48]` with `theta=256` — non-standard; covers the full 128-channel head via `(text_len, y, x)` position IDs
- Shared-AdaLN modulation: one `SiLU → Linear(H → 6H)` fed by the timestep embedding, broadcast to every block
- `text_in_dim=3072` matches Mistral3 text hidden_size — text embeds fed directly
- Keys: `layers.N.*`, `x_embedder.proj.*`, `final_linear.*`, `final_norm.*`, `adaLN_modulation.*`, `time_embedding.*`

#### Text Encoder (`Mistral3Model`)

- Multimodal checkpoint, but the port uses the **text path only**
- 26 layers, hidden size 3072, 32 attention heads / 8 KV heads (GQA)
- head_dim 128, YaRN RoPE (`factor=16`, `beta_fast=32`, `beta_slow=1`, `original_max_position_embeddings=16384`)
- `vision_tower.*` and `multi_modal_projector.*` keys are dropped at conversion time
- Keys: `language_model.layers.*`, `language_model.embed_tokens.*`, `language_model.norm.*`

#### VAE (`AutoencoderKLFlux2`)

- Flux 2 VAE — `block_out_channels=[128, 256, 512, 512]`, 4 down/up blocks
- Latent channels: 32; `patch_size=(2, 2)` → ×16 total downsample (×8 conv + ×2 patch)
- GroupNorm + SiLU; single-head mid-block self-attention
- Top-level `BatchNorm2d` for latent renormalization (kept as stored weights)
- `force_upcast=True` — the pipeline runs the VAE in fp32 even when the DiT is bf16
- Keys: `encoder.*`, `decoder.*`, `quant_conv.*`, `post_quant_conv.*`, `bn.*`

## Source Checkpoint Structure

ERNIE-Image files are organized per-component on HuggingFace:

```
baidu/ERNIE-Image[-Turbo]/
├── transformer/
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors.index.json
│   └── diffusion_pytorch_model-{00001..00002}-of-00002.safetensors
├── text_encoder/
│   ├── config.json
│   └── model.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── tokenizer/
│   └── tokenizer_config.json        # only the config; vocab comes from Pixtral
├── scheduler/
│   └── scheduler_config.json
└── model_index.json
```

Baidu publishes only `tokenizer_config.json` — the actual vocab is pulled from [`mistral-community/pixtral-12b`](https://huggingface.co/mistral-community/pixtral-12b) at conversion time and bundled into the output directory.

## Key Translations (PyTorch → MLX)

### Transformer

| PyTorch name | MLX name | Reason |
|---|---|---|
| `x_embedder.proj.weight` `(O, I, 1, 1)` | `x_embedder.proj.weight` `(O, I)` | `patch_size=1` Conv2d collapses to a Linear |
| `time_embedding.linear_1.*` | `time_embedding.linear1.*` | Match `mlx_arsenal.diffusion.TimestepEmbedding` naming |
| `time_embedding.linear_2.*` | `time_embedding.linear2.*` | Same as above |
| `adaLN_modulation.1.*` | `adaLN_modulation.linear.*` | PT `Sequential[1]` → flat MLX module |
| `layers.N.self_attention.to_out.0.weight` | `layers.N.self_attention.to_out_0.weight` | PT `ModuleList[0]` flattened |

### VAE

| PyTorch name | MLX name | Reason |
|---|---|---|
| Any `Conv2d.weight` `(O, I, H, W)` | `(O, H, W, I)` | MLX channels-last convolution layout |
| `bn.running_mean` / `bn.running_var` | *(unchanged)* | Latent-normalization stats kept as-is |

### Text encoder

No key renames — `mlx-lm`'s `mistral3.Model.sanitize` consumes the raw `language_model.*` keys directly. `vision_tower.*` and `multi_modal_projector.*` are dropped.

## Quantization Strategy

Only the transformer's block Linears (≥ 256 × 256) are quantized. VAE (conv-heavy) and text_encoder stay in fp16/bf16.

**Deliberately excluded layers:**

| Layer | Reason |
|-------|--------|
| `x_embedder.proj` | Sensitive input projection |
| `time_embedding.*` | Small, numerically fragile |
| `adaLN_modulation.linear` | Shared modulation — every block reads from it |
| `final_linear` | Sensitive output projection |
| `final_norm.linear` | AdaLN-continuous normalization |
| All `RMSNorm` weights | Not Linear — incompatible |
| Text encoder (all layers) | Kept in bf16 for text-path fidelity |
| VAE (all Conv2d) | Not Linear — incompatible |

Recommended: `--bits 8 --group-size 64` (transformer: ~16.1 GB → ~8 GB → total output ~12 GB).

## Output Files

```
output_dir/
├── transformer.safetensors             # ~16.1 GB (fp16) or ~8 GB (int8) or ~4 GB (int4)
├── text_encoder.safetensors            # ~7.7 GB (always bf16/fp16)
├── vae.safetensors                     # ~168 MB (always fp16)
├── split_model.json                    # Split metadata
├── transformer_config.json             # DiT architecture config
├── text_encoder_config.json            # Mistral3 config (text_config + vision_config)
├── vae_config.json                     # AutoencoderKLFlux2 config
├── model_index.json                    # Pipeline config
├── scheduler_scheduler_config.json     # FlowMatchEulerDiscreteScheduler params
├── tokenizer_tokenizer_config.json     # From baidu
├── tokenizer.json                      # From mistral-community/pixtral-12b (bundled)
├── special_tokens_map.json             # From pixtral-12b
└── quantize_config.json                # Only if --quantize was used
```

## Validation Checks

The `validate` command verifies:

- All three component safetensors exist (`transformer`, `text_encoder`, `vae`)
- 36 transformer blocks present with `layers.N.*` structural keys
- Patch embedder, time embedding, final linear, and AdaLN modulation present
- 26 Mistral3 LLM layers in `text_encoder`
- VAE Conv2d weights present with channels-last `(O, H, W, I)` layout
- Quantization pairs (`.scales` / `.biases`) consistent when `--quantize` was used
- No leftover PyTorch-style keys (no `.num_batches_tracked`, no `vision_tower.*`)

## Downstream inference

Converted checkpoints are consumed by [`ernie-image-mlx`](https://github.com/dgrauet/ernie-image-mlx) — a pure MLX implementation of the `ErnieImagePipeline` for Apple Silicon. End-to-end tested against diffusers (fp32 parity: DiT 3.1e-6, VAE encoder 1.7e-6, decoder 6.7e-6).
