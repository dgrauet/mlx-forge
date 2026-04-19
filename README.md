# MLX Forge

Convert, quantize, split, validate, and upload ML models for [Apple MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

> **Tip:** if you use Claude Code for MLX ports, the [`porting-pytorch-to-mlx`](https://github.com/dgrauet/claude-skill-mlx-porting) skill wraps the end-to-end porting workflow (scaffolding, parity testing, attention patterns, pitfalls) and delegates weight conversion to `mlx-forge`.

## Features

- **Convert** PyTorch checkpoints to MLX format (safetensors, channels-last conv layout)
- **Quantize** model weights to int4/int8 with selective layer targeting
- **Split** large unified model files into per-component files for memory-constrained machines
- **Validate** converted models: file structure, key naming, weight shapes, quantization integrity
- **Upload** converted models to HuggingFace Hub with auto-derived repo naming, model cards, and collections

## Supported Models

| Model | Recipe | Status |
|-------|--------|--------|
| [LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) (22B video DiT) | `ltx-2.3` | Stable |
| [Fish S2 Pro](https://huggingface.co/fishaudio/s2-pro) (5B TTS) | `fish-s2-pro` | Stable |
| [Mistral Small 3.1](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) (24B VLM) | `mistral-small-3.1` | Stable |
| [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image-2512) (57B text-to-image DiT) | `qwen-image-2512` | Stable |

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

For recipes that load PyTorch `.pth` checkpoints (e.g. Fish S2 Pro codec):

```bash
pip install 'mlx-forge[torch]'
```

## Usage

### Convert

```bash
# Convert a model (downloads checkpoint from HuggingFace)
mlx-forge convert ltx-2.3
mlx-forge convert fish-s2-pro
mlx-forge convert mistral-small-3.1
mlx-forge convert qwen-image-2512

# Convert with int8 quantization
mlx-forge convert ltx-2.3 --quantize --bits 8

# Preview conversion plan (no download, no writes)
mlx-forge convert fish-s2-pro --dry-run

# Convert from a local checkpoint
mlx-forge convert ltx-2.3 --checkpoint /path/to/checkpoint.safetensors
```

See model-specific options in [docs/models/](docs/models/).

### Validate

```bash
mlx-forge validate ltx-2.3 models/ltx-2.3-mlx-distilled
mlx-forge validate fish-s2-pro models/fish-s2-pro-mlx
mlx-forge validate mistral-small-3.1 models/mistral-small-3.1-mlx
mlx-forge validate qwen-image-2512 models/qwen-image-2512-mlx
```

### Split (legacy unified models)

```bash
mlx-forge split ltx-2.3 /path/to/unified-model-dir
```

### Upload to HuggingFace Hub

```bash
# Upload with auto-derived repo name (reads split_model.json metadata)
mlx-forge upload models/ltx-2.3-mlx-distilled

# Upload to a specific repo or organization
mlx-forge upload models/fish-s2-pro-mlx --repo-id myuser/my-model
mlx-forge upload models/mistral-small-3.1-mlx --namespace my-org

# Upload and add to a collection
mlx-forge upload ./my-model --collection "MLX Forge Models"
```

Requires authentication: run `huggingface-cli login` or set the `HF_TOKEN` environment variable.

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
├── convert.py       # Shared conversion utilities (download, load, classify, process)
├── transpose.py     # Conv weight layout transposition (generic)
├── quantize.py      # Quantization engine (generic)
├── split.py         # Model splitting (generic)
├── validate.py      # Validation framework (generic)
├── upload.py        # HuggingFace Hub upload + model card (generic)
└── recipes/
    ├── ltx_23.py    # LTX-2.3: key mapping, config, validation
    ├── fish_s2.py   # Fish S2 Pro: Dual-AR TTS + DAC codec
    ├── mistral_small_31.py  # Mistral Small 3.1: 24B VLM (Pixtral + dense LLM)
    └── qwen_image_2512.py  # Qwen-Image: text-to-image MMDiT
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
    "ltx-2.3": "mlx_forge.recipes.ltx_23",
    "fish-s2-pro": "mlx_forge.recipes.fish_s2",
    "mistral-small-3.1": "mlx_forge.recipes.mistral_small_31",
    "qwen-image-2512": "mlx_forge.recipes.qwen_image",
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

## Model-Specific Documentation

Each recipe has its own detailed guide with architecture, key mapping, known gotchas, and validation details:

- [LTX-2.3](docs/models/ltx-2.3.md) — 22B video DiT (6 components, Conv3d/Conv1d transposition)
- [Fish S2 Pro](docs/models/fish-s2-pro.md) — 5B TTS (Dual-AR + DAC codec)
- [Mistral Small 3.1](docs/models/mistral-small-3.1.md) — 24B VLM (Pixtral vision + dense LLM)
- [Qwen-Image](docs/models/qwen-image-2512.md) — 57B text-to-image MMDiT (Flux-style)

## License

Apache 2.0
