# Matrix-Game-3.0

[Skywork/Matrix-Game-3.0](https://huggingface.co/Skywork/Matrix-Game-3.0) — Memory-augmented interactive world model for real-time, streaming game generation with action conditioning (keyboard + mouse).

> MLX Forge converts all six components in a single pass: dit, dit_distilled, t5_encoder, vae, vae_lightvae, vae_lightvae_v2.

## Quick Start

```bash
# Convert everything (~56 GB download)
mlx-forge convert matrix-game-3.0

# Convert + quantize both DiT variants
mlx-forge convert matrix-game-3.0 --quantize --bits 8

# Preview what will happen
mlx-forge convert matrix-game-3.0 --dry-run

# Convert from local checkpoints
mlx-forge convert matrix-game-3.0 --dit-checkpoint ./base_model/diffusion_pytorch_model.safetensors

# Validate
mlx-forge validate matrix-game-3.0 models/matrix-game-3.0-mlx
```

## CLI Options

### Convert

| Flag | Default | Description |
|------|---------|-------------|
| `--dit-checkpoint` | *(download)* | Path to local base DiT `.safetensors` (skips download) |
| `--t5-checkpoint` | *(download)* | Path to local T5 `.pth` (skips download) |
| `--vae-checkpoint` | *(download)* | Path to local Wan2.2 VAE `.pth` (skips download) |
| `--output` | `models/matrix-game-3.0-mlx[-q<bits>]` | Output directory |
| `--quantize` | off | Quantize both DiT variants after conversion |
| `--bits` | `8` | Quantization bits (`4` or `8`) |
| `--group-size` | `64` | Quantization group size |
| `--dry-run` | off | Preview plan without downloading or writing |
| `--skip-tokenizer` | off | Skip copying tokenizer files |

### Validate

| Flag | Default | Description |
|------|---------|-------------|
| `model_dir` | *(required)* | Path to converted model directory |

## Architecture

```
 Skywork/Matrix-Game-3.0 (HuggingFace)
 ├── base_model/                    ─► dit.safetensors
 │   └── diffusion_pytorch_model.safetensors (12.9 GB, safetensors)
 ├── base_distilled_model/          ─► dit_distilled.safetensors
 │   └── diffusion_pytorch_model.safetensors (25.9 GB, safetensors)
 ├── models_t5_umt5-xxl-enc-bf16.pth (11.4 GB, .pth) ─► t5_encoder.safetensors
 ├── Wan2.2_VAE.pth                  (2.82 GB, .pth)  ─► vae.safetensors
 ├── MG-LightVAE.pth                (2.74 GB, .pth)   ─► vae_lightvae.safetensors
 ├── MG-LightVAE_v2.pth             (841 MB, .pth)    ─► vae_lightvae_v2.safetensors
 └── google/umt5-xxl/               (21.5 MB)         ─► google/umt5-xxl/ (tokenizer)
```

### Inference Modes

| Mode | DiT | VAE (decode) | VAE (encode) | Steps |
|------|-----|-------------|-------------|-------|
| Base (high quality) | `dit` | `vae` | `vae` | 50 |
| Distilled (real-time) | `dit_distilled` | `vae` | `vae` | 3 |
| Distilled + LightVAE v2 | `dit_distilled` | `vae_lightvae_v2` | `vae` (encoder) | 3 |

LightVAE variants use their own pruned decoder for fast decoding, but still require the Wan2.2 VAE encoder (full resolution) for encoding input images. Both are included in the converted output.

## Components

### dit / dit_distilled (WanModel DiT)

Both share the same architecture (30-layer WanModel diffusion transformer with action conditioning). The distilled variant is trained for few-step sampling (3 steps vs 50).

| Parameter | Value |
|-----------|-------|
| Layers | 30 |
| Heads | 24 |
| Hidden dim | 3072 |
| FFN dim | 14336 |
| Frequency dim | 256 |
| Input/output dim | 48 |
| Text length | 512 |
| Epsilon | 1e-6 |
| Sigma theta | 0.8 |
| Model type | `ti2v` (text-image-to-video) |

**Action model** (blocks 0-14):

| Parameter | Value |
|-----------|-------|
| Action blocks | 0-14 (first 15 of 30) |
| Keyboard input dim | 6 |
| Mouse input dim | 2 |
| Hidden size | 128 |
| Keyboard hidden dim | 1024 |
| Mouse hidden dim | 1024 |
| Window size | 3 |
| QK norm | Yes |
| QKV bias | No |
| RoPE theta | 256 |
| RoPE dim list | [8, 28, 28] |

**Per-block weight keys (MLX):**

```
blocks.{i}.self_attn.q.weight
blocks.{i}.self_attn.k.weight
blocks.{i}.self_attn.v.weight
blocks.{i}.self_attn.o.weight
blocks.{i}.cross_attn.q.weight
blocks.{i}.cross_attn.k.weight
blocks.{i}.cross_attn.v.weight
blocks.{i}.cross_attn.o.weight
blocks.{i}.ffn_linear1.weight
blocks.{i}.ffn_linear2.weight
blocks.{i}.norm1.weight          # RMSNorm
blocks.{i}.norm2.weight          # RMSNorm
blocks.{i}.norm3.weight          # RMSNorm
blocks.{i}.modulation.*          # Adaptive modulation
blocks.{i}.action_model.*        # Action conditioning (blocks 0-14 only)
```

**Root keys (MLX):**

```
patch_embedding.weight           # Conv3d flattened to Linear (in_dim*pt*ph*pw, dim)
text_embedding_linear1.weight
text_embedding_linear2.weight
time_embedding_linear1.weight
time_embedding_linear2.weight
time_projection_linear1.weight
head.*                           # Final output projection
cam_injector.*                   # Camera injection
```

### t5_encoder (UMT5-XXL)

Google's UMT5-XXL text encoder (encoder-only, 24 blocks).

| Parameter | Value |
|-----------|-------|
| Blocks | 24 |
| Architecture | T5 encoder-only |
| Source | `models_t5_umt5-xxl-enc-bf16.pth` |

**Key structure (no renaming needed):**

```
token_embedding.weight
blocks.{i}.attn.q.weight
blocks.{i}.attn.k.weight
blocks.{i}.attn.v.weight
blocks.{i}.attn.o.weight
blocks.{i}.attn.relative_attention_bias.weight
blocks.{i}.norm1.weight
blocks.{i}.norm2.weight
blocks.{i}.ffn.linear1.weight
blocks.{i}.ffn.linear2.weight
norm.weight                      # Final layer norm
```

### vae / vae_lightvae / vae_lightvae_v2 (WanVAE)

All three VAE variants share the same diffusers-style key format and use the same sanitizer. They differ in channel dimensions (pruning rate).

| Variant | Source file | Pruning rate | Size |
|---------|-----------|-------------|------|
| `vae` | `Wan2.2_VAE.pth` | 0.0 (full) | ~400 MB |
| `vae_lightvae` | `MG-LightVAE.pth` | 0.5 | ~2.8 GB |
| `vae_lightvae_v2` | `MG-LightVAE_v2.pth` | 0.75 | ~850 MB |

Architecture: Encoder3d + Decoder3d with ResNet blocks, mid-block attention, and Conv3d layers.

## Key Sanitization

### DiT (nn.Sequential -> named Linear)

| PyTorch pattern | MLX pattern |
|----------------|-------------|
| `text_embedding.0.` | `text_embedding_linear1.` |
| `text_embedding.2.` | `text_embedding_linear2.` |
| `time_embedding.0.` | `time_embedding_linear1.` |
| `time_embedding.2.` | `time_embedding_linear2.` |
| `time_projection.1.` | `time_projection_linear1.` |
| `.ffn.0.` | `.ffn_linear1.` |
| `.ffn.2.` | `.ffn_linear2.` |
| `.action_model.keyboard_embed.0.` | `.action_model.keyboard_embed_linear1.` |
| `.action_model.keyboard_embed.2.` | `.action_model.keyboard_embed_linear2.` |
| `.action_model.mouse_mlp.0.` | `.action_model.mouse_mlp_linear1.` |
| `.action_model.mouse_mlp.2.` | `.action_model.mouse_mlp_linear2.` |
| `.action_model.mouse_mlp.3.` | `.action_model.mouse_mlp_layernorm.` |

### T5

No key renaming needed. The `.pth` state_dict uses MLX-compatible names directly.

### VAE (diffusers -> WanVAE_)

| PyTorch pattern | MLX pattern |
|----------------|-------------|
| `quant_conv.` | `conv1.` |
| `post_quant_conv.` | `conv2.` |
| `encoder.conv_in.` | `encoder.conv1.` |
| `encoder.norm_out.` | `encoder.head.0.` |
| `encoder.conv_out.` | `encoder.head.2.` |
| `encoder.mid_block.resnets.{i}.` | `encoder.middle.{i*2}.` |
| `encoder.mid_block.attentions.0.` | `encoder.middle.1.` |
| `encoder.down_blocks.{b}.resnets.{r}.norm1.` | `encoder.downsamples.{b}.downsamples.{r}.residual.0.` |
| `encoder.down_blocks.{b}.resnets.{r}.conv1.` | `encoder.downsamples.{b}.downsamples.{r}.residual.2.` |
| `encoder.down_blocks.{b}.resnets.{r}.norm2.` | `encoder.downsamples.{b}.downsamples.{r}.residual.3.` |
| `encoder.down_blocks.{b}.resnets.{r}.conv2.` | `encoder.downsamples.{b}.downsamples.{r}.residual.6.` |
| `encoder.down_blocks.{b}.resnets.{r}.conv_shortcut.` | `encoder.downsamples.{b}.downsamples.{r}.shortcut.` |
| `encoder.down_blocks.{b}.downsampler.resample.` | `encoder.downsamples.{b}.downsamples.2.resample.` |
| `encoder.down_blocks.{b}.downsampler.time_conv.` | `encoder.downsamples.{b}.downsamples.2.time_conv.` |
| `decoder.*` | *(same pattern, with `upsamples` and index 3 for upsampler)* |

**Skipped keys:** `dynamic_feature_projection_heads.*` (training-only).

## Conv Transposition

| Component | Conv type | Transposition |
|-----------|----------|---------------|
| dit / dit_distilled | None (all Linear) | No |
| dit / dit_distilled `patch_embedding` | Conv3d | Special: (O,I,D,H,W) -> flatten (O,I\*D\*H\*W) -> transpose (I\*D\*H\*W,O) = Linear |
| t5_encoder | None (all Linear) | No |
| vae / vae_lightvae / vae_lightvae_v2 | Conv3d | Yes: channels-second (O,I,...) -> channels-last (O,...,I) |

## Quantization Strategy

Only DiT transformer block Linear weights are quantized. Both `dit` and `dit_distilled` are quantized when `--quantize` is used.

**Quantized** (blocks.N Linear .weight, 2D):
- `blocks.{i}.self_attn.{q,k,v,o}.weight`
- `blocks.{i}.cross_attn.{q,k,v,o}.weight`
- `blocks.{i}.ffn_linear1.weight`
- `blocks.{i}.ffn_linear2.weight`
- `blocks.{i}.action_model.*` (Linear weights)

**NOT quantized:**

| Layer | Reason |
|-------|--------|
| `time_embedding_*`, `time_projection_*` | Timestep conditioning -- sensitive |
| `patch_embedding` | Input projection |
| `head.*` | Final output projection |
| `*norm*` | RMSNorm / LayerNorm weights |
| `cam_injector`, `cam_scale`, `cam_shift` | Camera injection -- small, sensitive |
| `c2ws_hidden_states` | Camera hidden states |
| `modulation` | Adaptive modulation params |
| `text_embedding_*` | Text embedding projections -- small, sensitive |
| All VAE weights | Conv-based, not compatible with Linear quantization |
| All T5 weights | Not quantized (separate encoder) |

Recommended: `--bits 8 --group-size 64`.

## Checkpoint Format

| Source file | Format | Loader |
|-------------|--------|--------|
| `base_model/diffusion_pytorch_model.safetensors` | safetensors | `mx.load()` (lazy, memory-mapped) |
| `base_distilled_model/diffusion_pytorch_model.safetensors` | safetensors | `mx.load()` (lazy, memory-mapped) |
| `models_t5_umt5-xxl-enc-bf16.pth` | PyTorch | `torch.load(weights_only=True)` |
| `Wan2.2_VAE.pth` | PyTorch | `torch.load(weights_only=True)` |
| `MG-LightVAE.pth` | PyTorch (nested) | `torch.load()` + `_extract_state_dict()` |
| `MG-LightVAE_v2.pth` | PyTorch (nested) | `torch.load()` + `_extract_state_dict()` |

The `.pth` files require the optional `torch` dependency: `uv pip install torch`.

LightVAE checkpoints nest the state_dict under wrapper keys (`state_dict`, `gen_model`, or `generator`). The recipe recursively unwraps these to find the flat tensor dict.

## Output Files

```
matrix-game-3.0-mlx/
├── config.json                  # Architecture parameters (from base_model config)
├── split_model.json             # Component list + source metadata
├── dit.safetensors              # ~10.5 GB (fp16) or ~5.3 GB (int8)
├── dit_distilled.safetensors    # ~21 GB (fp16) or ~10.5 GB (int8)
├── t5_encoder.safetensors       # ~9 GB
├── vae.safetensors              # ~400 MB (Wan2.2)
├── vae_lightvae.safetensors     # ~2.8 GB (LightVAE v1)
├── vae_lightvae_v2.safetensors  # ~850 MB (LightVAE v2)
├── quantize_config.json         # Only if --quantize was used
└── google/umt5-xxl/
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── special_tokens_map.json
    └── spiece.model
```

## Critical Config Values

Extracted from `base_model/config.json`:

| Config key | Value | Notes |
|------------|-------|-------|
| `model_type` | `ti2v` | Text-image-to-video |
| `num_layers` | `30` | Transformer blocks |
| `num_heads` | `24` | Attention heads |
| `dim` | `3072` | Hidden dimension |
| `ffn_dim` | `14336` | FFN dimension |
| `in_dim` / `out_dim` | `48` | Latent channels |
| `text_len` | `512` | Max text tokens |
| `is_action_model` | `true` | Action conditioning enabled |
| `use_text_crossattn` | `true` | Cross-attention on text |

## Validation Checks

| Check | What it verifies |
|-------|-----------------|
| File structure | All 6 `.safetensors` + config + split_model exist |
| Config values | num_layers=30, num_heads=24, dim=3072, in_dim=48 |
| DiT prefix | All keys start with `dit.` / `dit_distilled.` |
| Sequential sanitized | No `.ffn.0.`, `.ffn.2.`, `text_embedding.0.`, `time_embedding.0.` |
| patch_embedding | Weight is 2D (Linear, not 5D Conv3d) |
| Block count | 30 transformer blocks |
| Action model | Action model keys present |
| Camera injection | `cam_injector`, `cam_scale`, `cam_shift` keys present |
| Modulation | Modulation parameter keys present |
| T5 prefix | All keys start with `t5_encoder.` |
| T5 structure | `token_embedding`, `norm`, 24 blocks |
| VAE prefix | All keys start with `vae.` / `vae_lightvae.` / `vae_lightvae_v2.` |
| VAE structure | Encoder and decoder keys present |
| Conv layout | Conv3d weights in channels-last (5D) |
| Quantization | Scales/biases pairs consistent (if quantized) |

## Known Gotchas

### Conversion Pitfalls

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| LightVAE .pth has nested state_dict | `AttributeError: 'OrderedDict' has no 'float'` | Use `_extract_state_dict()` to recursively unwrap |
| Lazy tensors saved without materialization | All-zero weights | `_materialize()` before every `mx.save_safetensors()` |
| Conv weights not transposed (VAE) | Garbled output | Transpose all Conv3d weights to channels-last |
| patch_embedding not flattened | Shape mismatch at inference | Flatten Conv3d (O,I,D,H,W) -> Linear (I\*D\*H\*W,O) |
| `.pth` loaded without `torch` | ImportError | Install: `uv pip install torch` |

### Runtime Pitfalls (for model loaders)

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Using LightVAE decoder with LightVAE encoder | Poor encoding quality | Use Wan2.2 (`vae`) encoder with LightVAE decoder |
| Wrong inference steps for distilled model | Bad quality or slow | Base: 50 steps, distilled: 3 steps |
| Missing action conditioning | Static video | Enable keyboard+mouse action model for blocks 0-14 |
