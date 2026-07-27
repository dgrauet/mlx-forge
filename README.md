# MLX Forge

Convert, quantize, split, validate, and upload ML models for [Apple MLX](https://github.com/ml-explore/mlx) on Apple Silicon.

> **Tip:** if you use Claude Code for MLX ports, the [`mlx-porting`](https://github.com/dgrauet/claude-skill-mlx-porting) skill wraps the end-to-end porting workflow (scaffolding, parity testing, attention patterns, pitfalls) and delegates weight conversion to `mlx-forge`.

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
| [Ideogram 4](https://huggingface.co/ideogram-ai/ideogram-4-fp8) (FP8 text-to-image DiT) | `ideogram-4` | Stable |
| [Matrix-Game 3.0](https://huggingface.co/Skywork/Matrix-Game-3.0) (interactive world model: DiT ×2 + UMT5-XXL + 3 VAEs) | `matrix-game-3.0` | Stable |
| [CogVideoX-Fun 1.5](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP) (5B image-to-video inpainting) | `cogvideox-fun-v1.5-5b-inp` | Stable |
| [VOID](https://huggingface.co/netflix/void-model) (Netflix dual-pass CogVideoX transformer) | `void-model` | Stable |
| [Hunyuan3D 2.1](https://huggingface.co/tencent/Hunyuan3D-2.1) (3.3B shape DiT + 1.7B PBR paint UNet) | `hunyuan3d-2.1` | Stable |
| [ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) (8B text-to-image DiT) | `ernie-image` | Stable |
| [ERNIE-Image Prompt Enhancer](https://huggingface.co/baidu/ERNIE-Image-Turbo/tree/main/pe) (3B Ministral3 CausalLM) | `ernie-image-pe` | Stable |
| [V-JEPA 2.1 ViT-L](https://github.com/facebookresearch/vjepa2) (encoder + predictor, RoPE) | `vjepa-2.1-vitl` | Stable |
| [V-JEPA 2.0 ViT-L](https://github.com/facebookresearch/vjepa2) (encoder + predictor + SSv2/Diving48/EK100 probes) | `vjepa-2.0-vitl` | Stable |

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

For recipes that load PyTorch `.pth`/`.pt` checkpoints (e.g. Matrix-Game, V-JEPA):

```bash
pip install 'mlx-forge[torch]'
```

## Usage

### Convert

```bash
# Convert a model (downloads checkpoint from HuggingFace)
mlx-forge convert ltx-2.3
mlx-forge convert ernie-image

# Convert with int8 quantization
mlx-forge convert ltx-2.3 --quantize --bits 8

# Preview conversion plan (no download, no writes)
mlx-forge convert ltx-2.3 --dry-run

# Convert from a local checkpoint
mlx-forge convert ltx-2.3 --checkpoint /path/to/checkpoint.safetensors
```

**V-JEPA 2** recipes are different: the weights are Meta torch-hub `.pt` files
(not on HuggingFace), so they need an explicit local `--source` and the `[torch]`
extra. Output dirs are versioned (`models/vjepa-2.X-vitl-mlx`):

```bash
# V-JEPA 2.1 ViT-L (encoder + predictor) — single .pt
mlx-forge convert vjepa-2.1-vitl --source ~/weights/vjepa2_1_vitl_dist_vitG_384.pt

# V-JEPA 2.0 ViT-L (encoder + predictor + all three attentive probes)
mlx-forge convert vjepa-2.0-vitl \
    --source ~/weights/vitl.pt \
    --ssv2-source ~/weights/ssv2-vitl.pt \
    --diving48-source ~/weights/diving48-vitl-256.pt \
    --ek100-source ~/weights/ek100-vitl-256.pt
```

Converted MLX weights are published at
[`dgrauet/vjepa-2.1-vitl-mlx`](https://huggingface.co/dgrauet/vjepa-2.1-vitl-mlx)
and [`dgrauet/vjepa-2.0-vitl-mlx`](https://huggingface.co/dgrauet/vjepa-2.0-vitl-mlx).

See model-specific options in [docs/models/](docs/models/).

### Validate

```bash
mlx-forge validate ltx-2.3 models/ltx-2.3-mlx-distilled
mlx-forge validate ernie-image models/ernie-image-mlx
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
mlx-forge upload models/ernie-image-mlx --repo-id myuser/my-model
mlx-forge upload models/ernie-image-mlx --namespace my-org

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
    ├── __init__.py  # Registry (AVAILABLE_RECIPES) + the contract every recipe must satisfy
    ├── ltx_23.py    # LTX-2.3: key mapping, config, validation
    ├── ernie_image.py  # ERNIE-Image: 8B single-stream text-to-image DiT
    └── ...          # 10 recipes in total — see AVAILABLE_RECIPES
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

def split(args) -> None:
    """Split a unified checkpoint into components.

    Required even when the model needs no splitting — in that case print why
    and return. The CLI dispatches on this name; omitting it is a recipe bug
    that tests/test_recipe_contract.py will catch.
    """
    ...

def add_split_args(parser) -> None:
    """Register CLI arguments for split."""
    ...
```

`add_convert_args` should build its common block with
`mlx_forge.convert.add_common_convert_args()`, which registers
`--output/--quantize/--bits/--group-size/--dry-run` consistently across recipes.

Then register it in `recipes/__init__.py`:

```python
AVAILABLE_RECIPES = {
    "ltx-2.3": "mlx_forge.recipes.ltx_23",
    "ernie-image": "mlx_forge.recipes.ernie_image",
    "my-model": "mlx_forge.recipes.my_model",
}
```

The contract is enforced: `cli.py` refuses to dispatch a command a recipe does
not implement, and `tests/test_recipe_contract.py` runs every command against
every registered recipe.

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

## Understanding the Codebase

- [Recipe anatomy](docs/recipe-anatomy.md) — what varies between recipes and
  why: the six axes upstream forces, the shared layer everything else must use,
  and the metadata gaps that remain.

## Model-Specific Documentation

Each recipe has its own detailed guide with architecture, key mapping, known gotchas, and validation details:

- [LTX-2.3](docs/models/ltx-2.3.md) — 22B video DiT (6 components, Conv3d/Conv1d transposition)
- [ERNIE-Image](docs/models/ernie-image.md) — 8B single-stream text-to-image DiT (+ separate 3B `ernie-image-pe` Prompt Enhancer recipe)
- [V-JEPA 2](docs/models/vjepa-2.md) — Meta video world model: ViT-L encoder + predictor, RoPE (2.1 `vjepa-2.1-vitl` / 2.0 `vjepa-2.0-vitl` + attentive probes)
- [Matrix-Game 3.0](docs/models/matrix-game-3.0.md) — interactive world model: two DiT backbones, UMT5-XXL, three VAE variants

`ideogram-4`, `cogvideox-fun-v1.5-5b-inp`, `void-model` and `hunyuan3d-2.1` have no
dedicated guide yet — run `mlx-forge convert <recipe> --dry-run` for their conversion
plan, and read the module docstring in `src/mlx_forge/recipes/` for architecture notes.

## License

Apache 2.0
