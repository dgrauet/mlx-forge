# MLX Forge

Convert, quantize, split, and validate ML models for [Apple MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

## Features

- **Convert** PyTorch checkpoints to MLX format (safetensors, channels-last conv layout)
- **Quantize** model weights to int4/int8 with selective layer targeting
- **Split** large unified model files into per-component files for memory-constrained machines
- **Validate** converted models: file structure, key naming, weight shapes, quantization integrity

## Supported Models

| Model | Recipe | Status |
|-------|--------|--------|
| [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (22B video DiT) | `ltx-2.3` | Stable |

## Installation

```bash
# From source
git clone https://github.com/dgrauet/mlx-forge.git
cd mlx-forge
pip install -e .

# Or with uv
uv pip install -e .
```

Requires macOS with Apple Silicon and Python 3.11+.

## Usage

### Convert LTX-2.3

```bash
# Convert distilled variant (downloads ~46 GB from HuggingFace)
mlx-forge convert ltx-2.3

# Convert dev variant
mlx-forge convert ltx-2.3 --variant dev

# Convert with int8 quantization
mlx-forge convert ltx-2.3 --quantize --bits 8

# Convert from local checkpoint
mlx-forge convert ltx-2.3 --checkpoint /path/to/ltx-2.3-22b-distilled.safetensors --quantize --bits 8

# Custom output directory
mlx-forge convert ltx-2.3 --output ~/models/ltx-2.3-mlx --quantize --bits 4
```

### Validate

```bash
# Validate converted model
mlx-forge validate ltx-2.3 ~/.cache/huggingface/hub/ltx-2.3-mlx-distilled

# Validate with cross-reference against source checkpoint
mlx-forge validate ltx-2.3 ~/.cache/huggingface/hub/ltx-2.3-mlx-distilled --source /path/to/original.safetensors
```

### Split (legacy unified models)

```bash
mlx-forge split ltx-2.3 /path/to/unified-model-dir
```

### Generic quantization

```bash
# Quantize any safetensors file
mlx-forge quantize model.safetensors --bits 8

# Only quantize keys with a specific prefix
mlx-forge quantize model.safetensors --key-prefix transformer. --bits 4
```

## Architecture

```
mlx_forge/
├── cli.py           # CLI entry point
├── transpose.py     # Conv weight layout transposition (generic)
├── quantize.py      # Quantization engine (generic)
├── split.py         # Model splitting (generic)
├── validate.py      # Validation framework (generic)
└── recipes/
    └── ltx23.py     # LTX-2.3: key mapping, config, validation
```

Generic tools live at the top level. Model-specific logic lives in **recipes**. Adding support for a new model means creating a new recipe file.

## Adding a New Model Recipe

Create `src/mlx_forge/recipes/my_model.py` with:

```python
def classify_key(key: str) -> str | None:
    """Map PyTorch key -> component name."""
    ...

def sanitize_key(key: str) -> str:
    """PyTorch key naming -> MLX key naming."""
    ...

def convert(args) -> None:
    """Main conversion entry point."""
    ...

def validate(args) -> None:
    """Model-specific validation."""
    ...

def add_convert_args(parser) -> None:
    """Register CLI arguments for convert."""
    ...

def add_validate_args(parser) -> None:
    """Register CLI arguments for validate."""
    ...
```

Then register it in `recipes/__init__.py`:

```python
AVAILABLE_RECIPES = {
    "ltx-2.3": "mlx_forge.recipes.ltx23",
    "my-model": "mlx_forge.recipes.my_model",
}
```

## Key Technical Notes

### Conv Weight Transposition

PyTorch stores conv weights as `(O, I, ...)` while MLX expects channels-last `(O, ..., I)`:

| Layer | PyTorch | MLX |
|-------|---------|-----|
| Conv1d | (O, I, K) | (O, K, I) |
| Conv2d | (O, I, H, W) | (O, H, W, I) |
| Conv3d | (O, I, D, H, W) | (O, D, H, W, I) |
| ConvTranspose1d | (I, O, K) | (O, K, I) |

### Quantization

- Only Linear `.weight` matrices are quantized (affine mode with scales + biases)
- Conv, norm, embedding, and other layers stay in original precision
- **Critical**: non-quantizable tensors must be materialized before `mx.quantize()` runs, or lazy tensor buffers get evicted

### Memory Safety

- Source checkpoints are loaded lazily via `mx.load()` (memory-mapped, ~0 GB initially)
- Components are processed one at a time to stay within 32 GB RAM
- Explicit `gc.collect()` + `mx.clear_cache()` between components
- Each weight tensor is individually materialized before quantization to prevent OOM from accumulated lazy computation graphs

## LTX-2.3 Known Gotchas

Hard-won lessons from building and debugging the LTX-2.3 MLX pipeline:

### Conversion Pitfalls

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Lazy tensors saved without materialization | All-zero weights in output safetensors | Materialize tensors (via `_materialize()`) before every save call |
| Conv weights not transposed | Garbled output, NaN activations | Transpose all conv weights to channels-last layout |
| ConvTranspose1d != Conv1d layout | Vocoder produces noise | `(I,O,K) -> (O,K,I)` not `(O,I,K) -> (O,K,I)` — detect via "ups" in key |
| `per_channel_statistics` shared | VAE decode fails with missing keys | Duplicate to both `vae_decoder.safetensors` and `vae_encoder.safetensors` |
| `last_scale_shift_table` absent | Confusion during validation | Normal — initialized to zeros at load time, not in checkpoint |

### Quantization Pitfalls

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Quantizing adaln_single, proj_out, patchify_proj | Broken generation, visual artifacts | Only quantize `transformer_blocks` Linear weights |
| Non-quantizable tensors not materialized first | Silently zeroed weights (data corruption) | Materialize ALL kept tensors before any `mx.quantize()` call |
| Accumulated lazy graph during quantization loop | OOM during quantization | Materialize each weight individually before quantizing |
| Seed 42 + int8 quantization | Grayscale-only output | Model/quantization artifact — use random seed or avoid seed 42 |

### Runtime Pitfalls (for model loaders)

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Connector `positional_embedding_max_pos` default `[1]` | Model ignores all prompts, produces B&W vintage footage | Must be `[4096]` for LTX-2.3 |
| Connector `rope_type` default `INTERLEAVED` | Scrambled text embeddings | Must be `SPLIT` for LTX-2.3 |
| LTX-2.0 LoRAs loaded on 2.3 | Broken output | Different latent spaces — LoRAs must be retrained for 2.3 |
| Upsampler weights quantized | Quality loss / errors | Upsampler is Conv3d, not Linear — skip quantization |
| VAE unpatchify H/W transposed | 4x4 block artifacts (looks like low resolution) | Transpose order `(0,1,4,5,3,6,2)` not `(0,1,4,5,2,6,3)` |
| Audio latent reshape order | Garbled audio | `reshape(B,8,16,T).transpose(0,1,3,2)` NOT `reshape(B,8,T,16)` |

## License

Apache 2.0
