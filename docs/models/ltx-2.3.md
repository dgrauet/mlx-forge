# LTX-2.3

[Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) is a 22B-parameter audio-video DiT (Diffusion Transformer).
This page documents the LTX-2.3 recipe: CLI usage, architecture, conversion details, and known gotchas.

## Quick Start

```bash
# Convert distilled variant (~46 GB download)
mlx-forge convert ltx-2.3

# Convert dev variant
mlx-forge convert ltx-2.3 --variant dev

# Convert + quantize in one pass
mlx-forge convert ltx-2.3 --quantize --bits 8

# Convert with upscalers (separate downloads from HF)
mlx-forge convert ltx-2.3 --spatial-upscaler x2 x1.5 --temporal-upscaler x2

# Convert from a local checkpoint
mlx-forge convert ltx-2.3 --checkpoint ./ltx-2.3-22b-distilled.safetensors

# Preview conversion plan (no download, no writes)
mlx-forge convert ltx-2.3 --quantize --bits 8 --spatial-upscaler x2 --dry-run

# Validate a converted model
mlx-forge validate ltx-2.3 models/ltx-2.3-mlx-distilled

# Split a legacy unified model
mlx-forge split ltx-2.3 /path/to/unified-model-dir
```

## CLI Options

### Convert

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(download)* | Path to a local `.safetensors` checkpoint |
| `--variant` | `distilled` | Model variant (`distilled` or `dev`) |
| `--output` | `models/ltx-2.3-mlx-<variant>` | Output directory |
| `--quantize` | off | Quantize transformer after conversion |
| `--bits` | `8` | Quantization bits (`4` or `8`) |
| `--group-size` | `64` | Quantization group size |
| `--dry-run` | off | Preview conversion plan without downloading or writing |
| `--spatial-upscaler` | *(none)* | Spatial upscaler scale(s): `x2`, `x1.5`, or both |
| `--spatial-upscaler-checkpoint` | *(download)* | Local path(s) to spatial upscaler checkpoints |
| `--temporal-upscaler` | *(none)* | Temporal upscaler scale(s): `x2` |
| `--temporal-upscaler-checkpoint` | *(download)* | Local path(s) to temporal upscaler checkpoints |

### Validate

| Flag | Default | Description |
|------|---------|-------------|
| `model_dir` | *(required)* | Path to converted model directory |
| `--source` | *(none)* | Source checkpoint for cross-reference validation |

## Model Architecture

LTX-2.3 consists of six core components from the main checkpoint, plus optional upscalers (separate downloads):

```
                ┌─────────────────────────────────────┐
                │          PyTorch Checkpoint          │
                │        (~46 GB, single file)         │
                └──────────────┬──────────────────────┘
                               │ classify_key()
          ┌────────┬───────────┼───────────┬──────────┬──────────┐
          ▼        ▼           ▼           ▼          ▼          ▼
   ┌────────┐ ┌─────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
   │ Trans- │ │Connector│ │  VAE   │ │  VAE   │ │ Audio  │ │Vocoder │
   │ former │ │         │ │Decoder │ │Encoder │ │  VAE   │ │        │
   │ 48 blk │ │ video+  │ │ Conv3d │ │ Conv3d │ │ Conv1d │ │ Conv1d │
   │ Linear │ │ audio+  │ │        │ │        │ │        │ │  +ups  │
   │  only  │ │  text   │ │        │ │        │ │        │ │        │
   └────────┘ └─────────┘ └────────┘ └────────┘ └────────┘ └────────┘
     ~44 GB     ~200 MB     ~300 MB    ~300 MB    ~50 MB     ~50 MB

   Optional upscalers (separate checkpoint files):
   ┌────────────┐ ┌────────────┐ ┌────────────┐
   │  Spatial   │ │  Spatial   │ │  Temporal  │
   │ Upscaler   │ │ Upscaler   │ │ Upscaler   │
   │    x2      │ │   x1.5     │ │    x2      │
   │  Conv3d    │ │ Conv3d +   │ │  Conv3d    │
   │ +PixShuf   │ │ Rational   │ │ +PixShuf   │
   └────────────┘ └────────────┘ └────────────┘
      ~950 MB       ~1.0 GB        ~250 MB
```

### Key Classification

| PyTorch prefix | Component |
|---------------|-----------|
| `model.diffusion_model.*` (general) | `transformer` |
| `model.diffusion_model.video_embeddings_connector.*` | `connector` |
| `model.diffusion_model.audio_embeddings_connector.*` | `connector` |
| `text_embedding_projection.*` | `connector` |
| `vae.decoder.*` | `vae_decoder` |
| `vae.encoder.*` | `vae_encoder` |
| `vae.per_channel_statistics.*` | `vae_shared_stats` |
| `audio_vae.*` | `audio_vae` |
| `vocoder.*` | `vocoder` |

## Key Sanitization

### Transformer

| PyTorch pattern | MLX pattern |
|----------------|-------------|
| `model.diffusion_model.` | *(removed)* |
| `.to_out.0.` | `.to_out.` |
| `.ff.net.0.proj.` | `.ff.proj_in.` |
| `.ff.net.2.` | `.ff.proj_out.` |
| `.audio_ff.net.0.proj.` | `.audio_ff.proj_in.` |
| `.audio_ff.net.2.` | `.audio_ff.proj_out.` |
| `.linear_1.` | `.linear1.` |
| `.linear_2.` | `.linear2.` |

### Other components

- **Connector**: strips `model.diffusion_model.` prefix; `text_embedding_projection.*` keys pass through.
- **VAE decoder/encoder**: strips `vae.decoder.` or `vae.encoder.` prefix.
- **Audio VAE**: strips `audio_vae.decoder.` prefix.
- **Vocoder**: strips `vocoder.` prefix.

## Conv Transposition per Component

| Component | Conv type | Needs transposition |
|-----------|----------|-------------------|
| Transformer | None (all-Linear) | No |
| VAE decoder/encoder | Conv3d | Yes |
| Audio VAE | Conv1d | Yes |
| Vocoder | Conv1d + ConvTranspose1d + buffers | Yes (ConvTranspose1d detected via `"ups"` in key) |
| Spatial upscaler x2 | Conv3d | Yes |
| Spatial upscaler x1.5 | Conv3d + Conv2d (rational resampler) + buffer | Yes |
| Temporal upscaler x2 | Conv3d | Yes |

### Conv-like Buffers

Some `register_buffer` tensors have conv-shaped layouts (3D+) and need the same transposition as conv weights. These are detected by suffix:

| Suffix | Component | Keys | Shape | Transpose |
|--------|-----------|------|-------|-----------|
| `.filter` | Vocoder | `*.upsample.filter`, `*.downsample.lowpass.filter` | (1,1,12) | Conv1d (0,2,1) |
| `.basis` | Vocoder | `mel_stft.stft_fn.forward_basis`, `mel_stft.stft_fn.inverse_basis` | (514,1,512) | Conv1d (0,2,1) |
| `.kernel` | Spatial upscaler x1.5 | `upsampler.blur_down.kernel` | (1,1,5,5) | Conv2d (0,2,3,1) |

## Quantization Strategy

Only `transformer_blocks` Linear weights are quantized (int8 recommended).

**Deliberately excluded layers:**

| Layer | Reason |
|-------|--------|
| `adaln_single` (timestep embedding) | Broken generation |
| `proj_out` (final projection) | Visual artifacts |
| `patchify_proj` (input patchification) | Visual artifacts |
| Conv weights (VAE, vocoder) | Not Linear — incompatible |
| Norm layers | Too sensitive |
| Embedding layers | Small tensors, negligible savings |

Recommended setting: `--bits 8 --group-size 64` (transformer: ~44 GB → ~22 GB).

## Upscalers

LTX-2.3 includes optional latent upscalers for multi-stage pipelines. These are **separate checkpoint files** on HuggingFace (not part of the main 46 GB checkpoint).

| Upscaler | Scale | HF file | Architecture | Size |
|----------|-------|---------|-------------|------|
| Spatial x2 | 2x resolution | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | Conv3d + PixelShuffle(2) | ~950 MB |
| Spatial x1.5 | 1.5x resolution | `ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors` | Conv3d + SpatialRationalResampler | ~1.0 GB |
| Temporal x2 | 2x frame rate | `ltx-2.3-temporal-upscaler-x2-1.0.safetensors` | Conv3d + PixelShuffle(1) | ~250 MB |

All upscalers share the `LatentUpsampler` architecture:
- `initial_conv` → `initial_norm` (GroupNorm) → SiLU
- 4 `res_blocks` (ResBlock = conv + norm + conv + norm + residual)
- `upsampler` (varies by type)
- 4 `post_upsample_res_blocks`
- `final_conv`

The spatial x1.5 upscaler uses a `SpatialRationalResampler` (Conv2d → PixelShuffle → BlurDownsample) instead of a simple PixelShuffle, which also adds a `blur_down.kernel` buffer (fixed binomial filter).

**Upscalers are never quantized** — they are entirely Conv-based (no Linear layers), and the quality loss would outweigh the negligible size savings.

## Shared VAE Statistics

The `vae.per_channel_statistics` keys (mean-of-means, std-of-means) exist once in the source checkpoint but are needed by both the VAE decoder and encoder.
During conversion, they are **duplicated** into both `vae_decoder.safetensors` and `vae_encoder.safetensors`.

## Critical Config Values

These are extracted from the safetensors file metadata and written to `config.json`:

| Config key | Value | Symptom if wrong |
|------------|-------|------------------|
| `connector_positional_embedding_max_pos` | `[4096]` | Model ignores all prompts, B&W vintage footage |
| `connector_rope_type` | `SPLIT` | Scrambled text embeddings |
| `num_layers` | `48` | Wrong number of transformer blocks |
| `apply_gated_attention` | `true` | Missing gating mechanism |
| `cross_attention_adaln` | `true` | Wrong cross-attention conditioning |
| `caption_channels` | `null` | V2 vs V1 mismatch |

## Output Files

```
output_dir/
├── config.json                       # Architectural parameters
├── embedded_config.json              # Original embedded config (if present)
├── split_model.json                  # Split metadata (source, variant, components)
├── transformer.safetensors           # ~44 GB (fp16) or ~22 GB (int8)
├── connector.safetensors             # ~200 MB
├── vae_decoder.safetensors           # ~300 MB
├── vae_encoder.safetensors           # ~300 MB
├── audio_vae.safetensors             # ~50 MB
├── vocoder.safetensors               # ~50 MB
├── quantize_config.json              # Only if --quantize was used
├── spatial_upscaler_x2_v1_1.safetensors   # ~950 MB (optional, --spatial-upscaler x2)
├── spatial_upscaler_x2_v1_1_config.json   # Upscaler config from safetensors metadata
├── spatial_upscaler_x1_5_v1_0.safetensors # ~1.0 GB (optional, --spatial-upscaler x1.5)
├── spatial_upscaler_x1_5_v1_0_config.json
├── temporal_upscaler_x2_v1_0.safetensors  # ~250 MB (optional, --temporal-upscaler)
└── temporal_upscaler_x2_v1_0_config.json
```

## Split Component Map

For the `split` command (legacy unified models):

| Key prefix | Output file |
|-----------|-------------|
| `transformer` | `transformer.safetensors` |
| `connector` | `connector.safetensors` |
| `text_embedding_projection` | `connector.safetensors` |
| `vae_decoder` | `vae_decoder.safetensors` |
| `vae_encoder` | `vae_encoder.safetensors` |
| `vocoder` | `vocoder.safetensors` |
| `audio_vae` | `audio_vae.safetensors` |

## Known Gotchas

### Conversion Pitfalls

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Lazy tensors saved without materialization | All-zero weights | Materialize before every save call |
| Conv weights not transposed | Garbled output, NaN activations | Transpose all conv weights to channels-last |
| ConvTranspose1d != Conv1d layout | Vocoder produces noise | `(I,O,K) → (O,K,I)` not `(O,I,K) → (O,K,I)` — detect via `"ups"` in key |
| `per_channel_statistics` shared | VAE decode fails with missing keys | Duplicate to both decoder and encoder |
| Conv-like buffers not transposed | Vocoder/upscaler produce incorrect output | Transpose 3D+ buffers ending in `.filter`, `.basis`, `.kernel` |
| `last_scale_shift_table` absent | Confusion during validation | Normal — initialized to zeros at load time |

### Quantization Pitfalls

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Quantizing adaln_single, proj_out, patchify_proj | Broken generation, artifacts | Only quantize `transformer_blocks` |
| Non-quantizable tensors not materialized first | Silently zeroed weights | Materialize ALL kept tensors before `mx.quantize()` |
| Accumulated lazy graph during quantization | OOM | Materialize each weight individually |
| Seed 42 + int8 quantization | Grayscale-only output | Use random seed or avoid seed 42 |

### Runtime Pitfalls (for model loaders)

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Connector `positional_embedding_max_pos` default `[1]` | Ignores prompts, B&W footage | Must be `[4096]` |
| Connector `rope_type` default `INTERLEAVED` | Scrambled text embeddings | Must be `SPLIT` |
| LTX-2.0 LoRAs loaded on 2.3 | Broken output | Different latent spaces — retrain LoRAs |
| Upsampler weights quantized | Quality loss / errors | Upsampler is Conv3d, skip quantization |
| VAE unpatchify H/W transposed | 4x4 block artifacts | Transpose order `(0,1,4,5,3,6,2)` not `(0,1,4,5,2,6,3)` |
| Audio latent reshape order | Garbled audio | `reshape(B,8,16,T).transpose(0,1,3,2)` NOT `reshape(B,8,T,16)` |

## Validation Checks

The `validate` command verifies:

- All expected files exist with non-zero sizes
- Config values match V2.3 requirements (gated attention, adaln, connector params)
- No PyTorch prefixes remain (`model.diffusion_model.`)
- No un-sanitized keys (`.ff.net.`, `.to_out.0.`)
- 48 transformer blocks present
- Gated attention keys and prompt_adaln_single keys present
- Conv weights in MLX channels-last layout
- Quantization pairs (.scales/.biases) are consistent
- Per-channel statistics present in VAE components
- Upscalers (if present): correct prefix, expected structure (initial_conv, res_blocks, post_upsample_res_blocks, final_conv), conv layout
- Optional: cross-reference tensor values against source checkpoint
