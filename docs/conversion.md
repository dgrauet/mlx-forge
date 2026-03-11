# Model Conversion

MLX Forge converts PyTorch `.safetensors` checkpoints into MLX-ready split models.
Each weight undergoes three transformations: **key sanitization**, **conv transposition**, and **materialization**.

## CLI Usage

```bash
mlx-forge convert <recipe> [options]
```

### LTX-2.3 Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(download)* | Path to a local `.safetensors` checkpoint |
| `--variant` | `distilled` | Model variant (`distilled` or `dev`) |
| `--output` | `~/.cache/huggingface/hub/ltx-2.3-mlx-<variant>` | Output directory |
| `--quantize` | off | Quantize transformer after conversion |
| `--bits` | `8` | Quantization bits (`4` or `8`) |
| `--group-size` | `64` | Quantization group size |

### Examples

```bash
# Download and convert (distilled, ~46 GB download)
mlx-forge convert ltx-2.3

# Convert a local checkpoint with quantization
mlx-forge convert ltx-2.3 --checkpoint ./ltx-2.3-22b-distilled.safetensors --quantize --bits 8

# Convert dev variant to a custom directory
mlx-forge convert ltx-2.3 --variant dev --output ~/models/ltx-2.3-dev
```

## The Conversion Pipeline

### Step 1: Checkpoint Acquisition

If `--checkpoint` is not provided, the checkpoint is downloaded from HuggingFace via `hf_hub_download`.
The filename is derived from the variant: `ltx-2.3-22b-distilled.safetensors` or `ltx-2.3-22b-dev.safetensors`.

### Step 2: Config Extraction

Model configuration is read from the safetensors file metadata (not a separate file).
The recipe extracts architectural parameters and writes `config.json` to the output directory.

For LTX-2.3, two config values are critical for correct inference:

| Config key | Required value | Symptom if wrong |
|------------|---------------|------------------|
| `connector_positional_embedding_max_pos` | `[4096]` | Model ignores all prompts |
| `connector_rope_type` | `SPLIT` | Scrambled text embeddings |

If the source metadata contains an embedded config JSON, it is also saved as `embedded_config.json`.

### Step 3: Lazy Weight Loading

```python
checkpoint_weights = mx.load(checkpoint_path)
```

`mx.load()` memory-maps the file, returning lazy tensor handles that consume nearly zero RAM.
Actual data is only read from disk when a tensor is materialized via `_materialize()` (which calls `mx.core.eval`).
This is essential for processing 46 GB checkpoints on machines with 32 GB of RAM.

### Step 4: Key Classification

Each weight key is classified into a component by matching its prefix:

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

Unclassified keys are skipped.

### Step 5: Per-Component Processing

Each component is processed independently:

1. **Sanitize keys** — rename PyTorch conventions to MLX conventions
2. **Transpose conv weights** — convert channels-second to channels-last layout
3. **Materialize** — force-evaluate each tensor to ensure it is not a dangling lazy reference
4. **Save** — write to `<component>.safetensors`
5. **Free memory** — `gc.collect()` + `mx.clear_cache()` before the next component

Processing one component at a time keeps peak RAM usage manageable.

### Step 6: Shared VAE Statistics

The `vae.per_channel_statistics` keys (mean-of-means, std-of-means) are shared between the VAE decoder and encoder.
Since the output is split into separate files, these statistics are **duplicated** into both `vae_decoder.safetensors` and `vae_encoder.safetensors`.

### Step 7: Optional Quantization

If `--quantize` is passed, the transformer weights are quantized in-place after conversion.
See [Quantization](quantization.md) for details.

## Key Sanitization

Sanitization renames PyTorch module paths to match MLX model implementations.
Each component has its own sanitizer function.

### Transformer key rewrites

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

- **Connector**: strips `model.diffusion_model.` prefix; `text_embedding_projection.*` keys pass through unchanged.
- **VAE decoder/encoder**: strips `vae.decoder.` or `vae.encoder.` prefix.
- **Audio VAE**: strips `audio_vae.decoder.` prefix.
- **Vocoder**: strips `vocoder.` prefix.

A sanitizer returning `None` means "skip this key".

## Conv Weight Transposition

PyTorch stores conv weights in channels-second layout. MLX expects channels-last.

| Layer | PyTorch layout | MLX layout | Transpose axes |
|-------|---------------|------------|---------------|
| Conv1d | `(O, I, K)` | `(O, K, I)` | `(0, 2, 1)` |
| Conv2d | `(O, I, H, W)` | `(O, H, W, I)` | `(0, 2, 3, 1)` |
| Conv3d | `(O, I, D, H, W)` | `(O, D, H, W, I)` | `(0, 2, 3, 4, 1)` |
| ConvTranspose1d | `(I, O, K)` | `(O, K, I)` | `(1, 2, 0)` |

**ConvTranspose1d gotcha**: the input and output channel axes are swapped compared to regular Conv1d.
For LTX-2.3, ConvTranspose1d weights are detected by the `"ups"` substring in vocoder weight keys.

**Which components need transposition**:
- Transformer: **no** — all-Linear, no conv layers.
- VAE decoder/encoder: **yes** — Conv3d weights.
- Audio VAE: **yes** — Conv1d weights.
- Vocoder: **yes** — Conv1d + ConvTranspose1d weights.

## Materialization and Memory Safety

The `_materialize()` helper forces tensor computation. It must be called before every `mx.save_safetensors()` call because **lazy tensors that have not been evaluated save as all zeros**.

The conversion pipeline materializes tensors at two points:

1. **Per weight** in `process_component()` — ensures cross-file lazy references are resolved before saving.
2. **Before quantization** — ensures non-quantizable tensors are not evicted by GPU work from `mx.quantize()`.

After each component is saved, `gc.collect()` and `mx.clear_cache()` free both Python objects and MLX GPU memory.

## Output Directory Structure

A successful LTX-2.3 conversion produces:

```
output_dir/
├── config.json                # Model configuration
├── embedded_config.json       # Original embedded config (if present)
├── split_model.json           # Split metadata (components, source, variant)
├── transformer.safetensors    # ~44 GB (fp16) or ~22 GB (int8)
├── connector.safetensors      # ~200 MB
├── vae_decoder.safetensors    # ~300 MB
├── vae_encoder.safetensors    # ~300 MB
├── audio_vae.safetensors      # ~50 MB
├── vocoder.safetensors        # ~50 MB
└── quantize_config.json       # Only if --quantize was used
```

## Writing a New Recipe

See the [Adding a New Model Recipe](../README.md#adding-a-new-model-recipe) section in the README for the required interface.

A recipe module must implement:

| Function | Purpose |
|----------|---------|
| `classify_key(key) -> str \| None` | Map PyTorch key to component name |
| Sanitizer functions per component | Rename keys to MLX conventions |
| `maybe_transpose(key, weight, component) -> array` | Transpose conv weights if needed |
| `convert(args)` | Main conversion entry point |
| `validate(args)` | Model-specific validation |
| `add_convert_args(parser)` | Register CLI arguments |
| `add_validate_args(parser)` | Register CLI arguments |
| `add_split_args(parser)` | Register CLI arguments |

Register the recipe in `recipes/__init__.py`:

```python
AVAILABLE_RECIPES = {
    "ltx-2.3": "mlx_forge.recipes.ltx23",
    "my-model": "mlx_forge.recipes.my_model",
}
```

The recipe name (dict key) is the user-facing CLI name. The module filename can differ.
