# Model Conversion

MLX Forge converts PyTorch `.safetensors` checkpoints into MLX-ready split models.
Each weight undergoes three transformations: **key sanitization**, **conv transposition**, and **materialization**.

## CLI Usage

```bash
mlx-forge convert <recipe> [options]
```

Each recipe registers its own CLI flags via `add_convert_args()`. Run `mlx-forge convert <recipe> --help` for recipe-specific options.

## The Conversion Pipeline

```
 ┌──────────────────────┐
 │  PyTorch Checkpoint   │   .safetensors file (single or sharded)
 │  (on disk or HF Hub)  │
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  1. Acquire           │   Download from HF Hub or use local --checkpoint
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  2. Extract Config    │   Read metadata from safetensors header → config.json
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  3. Lazy Load         │   mx.load() — memory-mapped, ~0 GB RAM
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  4. Classify Keys     │   Route each key to a component via classify_key()
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  5. Per-Component     │   For each component:
 │     Processing        │     a. Sanitize keys
 │                       │     b. Transpose conv weights
 │                       │     c. Materialize tensors
 │                       │     d. Save to <component>.safetensors
 │                       │     e. Free memory
 └──────────┬───────────┘
            │
            ▼
 ┌──────────────────────┐
 │  6. Optional:         │   Quantize selected components
 │     Quantization      │   (see quantization.md)
 └──────────────────────┘
```

### Step 1: Checkpoint Acquisition

If `--checkpoint` is not provided, the checkpoint is downloaded from HuggingFace via `hf_hub_download`.
Each recipe defines the expected filename and repository.

### Step 2: Config Extraction

Model configuration is read from the safetensors file metadata (not a separate file).
The recipe extracts architectural parameters and writes `config.json` to the output directory.

If the source metadata contains an embedded config JSON, it is also saved as `embedded_config.json`.

### Step 3: Lazy Weight Loading

```python
checkpoint_weights = mx.load(checkpoint_path)
```

`mx.load()` memory-maps the file, returning lazy tensor handles that consume nearly zero RAM.
Actual data is only read from disk when a tensor is materialized via `mx.core.eval`.
This is essential for processing large checkpoints on machines with limited RAM.

### Step 4: Key Classification

Each weight key is classified into a component by matching its prefix.
The recipe's `classify_key()` function determines which component a key belongs to.
Unclassified keys are skipped.

```
 ┌────────────────────────────────┐
 │   All checkpoint keys          │
 └───────┬──────┬──────┬─────────┘
         │      │      │
    classify_key()  for each key
         │      │      │
         ▼      ▼      ▼
    ┌────────┐ ┌────┐ ┌────────┐
    │comp. A │ │ B  │ │comp. C │  ... grouped by component
    └────────┘ └────┘ └────────┘
```

### Step 5: Per-Component Processing

Each component is processed independently to keep peak memory usage manageable:

1. **Sanitize keys** — rename PyTorch conventions to MLX conventions
2. **Transpose conv weights** — convert channels-second to channels-last layout
3. **Materialize** — force-evaluate each tensor to ensure it is not a dangling lazy reference
4. **Save** — write to `<component>.safetensors`
5. **Free memory** — `gc.collect()` + `mx.clear_cache()` before the next component

### Step 6: Optional Quantization

If `--quantize` is passed, selected weights are quantized in-place after conversion.
See [Quantization](quantization.md) for details.

## Key Sanitization

Sanitization renames PyTorch module paths to match MLX model implementations.
Each component has its own sanitizer function defined by the recipe.

A sanitizer returning `None` means "skip this key" — it will not appear in the output.

## Conv Weight Transposition

PyTorch stores conv weights in channels-second layout. MLX expects channels-last.

```
 PyTorch layout              MLX layout
 ┌───────────────┐           ┌───────────────┐
 │ (O, I, ...)   │  ──────►  │ (O, ..., I)   │
 │ channels 2nd  │ transpose │ channels last  │
 └───────────────┘           └───────────────┘
```

| Layer | PyTorch layout | MLX layout | Transpose axes |
|-------|---------------|------------|---------------|
| Conv1d | `(O, I, K)` | `(O, K, I)` | `(0, 2, 1)` |
| Conv2d | `(O, I, H, W)` | `(O, H, W, I)` | `(0, 2, 3, 1)` |
| Conv3d | `(O, I, D, H, W)` | `(O, D, H, W, I)` | `(0, 2, 3, 4, 1)` |
| ConvTranspose1d | `(I, O, K)` | `(O, K, I)` | `(1, 2, 0)` |

**ConvTranspose1d gotcha**: the input and output channel axes are swapped compared to regular Conv1d.
Recipes must detect ConvTranspose layers (e.g., via key naming conventions) and apply the correct transpose.

**Linear weights do NOT need transposition** — only conv layers require it.

## Materialization and Memory Safety

```
  ┌─────────────────────────────────────────────────────┐
  │  CRITICAL: Lazy tensors save as ALL ZEROS           │
  │  if not materialized before mx.save_safetensors()   │
  └─────────────────────────────────────────────────────┘
```

The `_materialize()` helper forces tensor computation via `mx.core.eval`.
It must be called before every `mx.save_safetensors()` call.

The conversion pipeline materializes tensors at two points:

1. **Per weight** in `process_component()` — ensures cross-file lazy references are resolved before saving.
2. **Before quantization** — ensures non-quantizable tensors are not evicted by GPU work from `mx.quantize()`.

After each component is saved, `gc.collect()` and `mx.clear_cache()` free both Python objects and MLX GPU memory.

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
    "my-model": "mlx_forge.recipes.my_model",
}
```

The recipe name (dict key) is the user-facing CLI name. The module filename can differ.

## Model-Specific Guides

- [LTX-2.3](models/ltx-2.3.md) — 22B audio-video DiT
- [Fish S2 Pro](models/fish-s2-pro.md) — 5B TTS (Phase 1)
