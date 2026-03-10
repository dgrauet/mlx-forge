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
| [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (22B video DiT) | `ltx23` | Stable |

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
mlx-forge convert ltx23

# Convert with int8 quantization
mlx-forge convert ltx23 --quantize --bits 8

# Convert from local checkpoint
mlx-forge convert ltx23 --checkpoint /path/to/ltx-2.3-22b-distilled.safetensors --quantize --bits 8

# Custom output directory
mlx-forge convert ltx23 --output ~/models/ltx23-mlx --quantize --bits 4
```

### Validate

```bash
# Validate converted model
mlx-forge validate ltx23 ~/.cache/huggingface/hub/ltx23-mlx

# Validate with cross-reference against source checkpoint
mlx-forge validate ltx23 ~/.cache/huggingface/hub/ltx23-mlx --source /path/to/original.safetensors
```

### Split (legacy unified models)

```bash
mlx-forge split ltx23 ~/.cache/huggingface/hub/ltx2-mlx-av-int4
```

### Generic quantization

```bash
# Quantize any safetensors file
mlx-forge quantize model.safetensors --bits 8

# Only quantize keys with a specific prefix
mlx-forge quantize model.safetensors --prefix transformer. --bits 4
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
    "ltx23": "mlx_forge.recipes.ltx23",
    "my_model": "mlx_forge.recipes.my_model",
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

## License

Apache 2.0
