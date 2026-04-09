---
name: mlx-recipe
description: Create or update MLX Forge conversion recipes for new models. Use this skill whenever the user asks to add a new model, create a recipe, convert a model, mirror a HuggingFace repo, or update an existing recipe to be more complete. Also trigger when discussing what's missing from a recipe or comparing a recipe against its source repository. This skill is essential because recipes that skip components (text encoders, tokenizers, schedulers) produce incomplete conversions that can't run end-to-end inference.
---

# MLX Forge Recipe Creation

This skill guides the creation and update of mlx-forge conversion recipes. The goal is to produce recipes that **fully mirror** the source repository — every weight component converted, every config/tokenizer/scheduler copied — so the output is a self-contained MLX model ready for end-to-end inference.

Recipes that skip components are incomplete and useless for inference. The most common failure mode is assuming only the "main" weights matter and ignoring text encoders, tokenizers, schedulers, and pipeline configs. This skill prevents that by enforcing a systematic audit-first approach.

## Phase 1: Audit the Source Repository

Before writing a single line of code, build a complete inventory of the source. This is the most important phase — skipping it is how components get missed.

### 1.1 Enumerate all files

Fetch the HuggingFace model API to get the complete file tree:

```
https://huggingface.co/api/models/{org}/{model}
```

List every file in the `siblings` array. Group them by directory. The result should look like:

```
root/          → model_index.json, configuration.json, README.md, ...
transformer/   → config.json, diffusion_pytorch_model.safetensors, ...
text_encoder/  → config.json, model-00001-of-00002.safetensors, ...
vae/           → config.json, diffusion_pytorch_model.safetensors, ...
tokenizer/     → tokenizer_config.json, spiece.model, ...
scheduler/     → scheduler_config.json
```

### 1.2 Classify every file

For EACH file, assign one of these categories:

| Category | Description | Action |
|----------|-------------|--------|
| **weight** | `.safetensors` or `.pth` files containing model weights | Convert (load, sanitize keys, transpose convs, materialize, save) |
| **weight-index** | `.safetensors.index.json` shard index files | Use to discover shards, don't copy to output |
| **component-config** | `config.json` inside a component subdir | Copy as `{component}_config.json` |
| **pipeline-config** | Top-level configs (`model_index.json`, `configuration.json`) | Copy to output root |
| **tokenizer** | Tokenizer files (`spiece.model`, `tokenizer.json`, `vocab.json`, `merges.txt`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`) | Copy with `tokenizer_` prefix |
| **scheduler** | Scheduler config (`scheduler_config.json`) | Copy with `scheduler_` prefix |
| **skip** | README, LICENSE, .gitattributes, preprocessor_config.json | Don't download |

Present this classification table to the user for confirmation before proceeding.

### 1.3 Verify sharding assumptions

For each weight component, determine:
- **Single file**: just one `.safetensors` — use `single_filename` parameter in `load_weights()`
- **Sharded**: has an `.index.json` + multiple numbered shards — use `index_filename` parameter

Check the ACTUAL files on HuggingFace. Do NOT assume sharding exists if no index file is present. Do NOT assume single-file if an index exists.

Read the index JSON if present to discover:
- Exact shard filenames (they vary: `model-00001-of-00002.safetensors` vs `diffusion_pytorch_model-00001-of-00009.safetensors`)
- Total size (from metadata)
- Weight distribution across shards

### 1.4 Identify component architecture

For each weight component, fetch its `config.json` and extract concrete values:
- `_class_name` — the model class (e.g., `T5EncoderModel`, `CogVideoXTransformer3DModel`)
- `num_layers` / `num_blocks` — exact layer count (used in validation)
- `in_channels` — exact input channels (critical for inpainting/specialized models)
- Hidden sizes, attention heads, patch sizes
- Whether it contains convolutions (need transposition) or is purely Linear

For pipeline models (`model_index.json`), note:
- Pipeline class (e.g., `CogVideoXPipeline`)
- Component-to-class mapping (what library each component comes from)

### 1.5 Cross-check architecture against actual weights

Do NOT trust assumptions about tensor shapes or layer types — verify them. For each weight component, inspect a few actual weight tensors from the source safetensors (or read the index) to confirm:

- **Tensor dimensionality**: Is `patch_embed.proj.weight` a Conv3d (5D) or a Linear (2D)? The class name in `config.json` might suggest Conv3d, but some architectures flatten the patch embedding into a Linear. The actual tensor shape is the ground truth.
- **Input dimensions**: Compute expected values from the config (e.g., `in_channels × patch_size_t × patch_size_h × patch_size_w` for a flattened Linear patch embed) and verify they match the actual weight shape.
- **Layer counts**: Verify the number of block keys matches `num_layers` from the config.

This phase exists because the most dangerous bugs come from assumptions that sound reasonable but don't match the actual weights. If the config says `in_channels: 33` but the validation hardcodes `48`, the validation will fail on a correctly converted model. Always derive validation constants from the config, never from guesses about the architecture.

## Phase 2: Naming

Naming is error-prone. Derive all names systematically from the model identity.

### 2.1 Recipe naming rules

Given a model like `alibaba-pai/CogVideoX-Fun-V1.5-5b-InP`:

| Name | Format | Example |
|------|--------|---------|
| **Recipe CLI name** | lowercase, hyphens, include version/variant | `cogvideo-fun-v1.5-5b-inp` |
| **Python filename** | lowercase, underscores, **must include version** | `cogvideox_fun_v1_5_5b_inp.py` |
| **Python module** | `mlx_forge.recipes.{stem}` | `mlx_forge.recipes.cogvideox_fun_v1_5_5b_inp` |
| **Output directory** | `models/{cli-name}-mlx[-q{bits}]` | `models/cogvideo-fun-v1.5-5b-inp-mlx` |
| **Source download dir** | `models/{cli-name}-src` | `models/cogvideo-fun-v1.5-5b-inp-src` |
| **HF output repo** | `{namespace}/{SourceModelName}-mlx[-q{bits}]` | `myns/CogVideoX-Fun-V1.5-5b-InP-mlx` |

Rules:
- The filename uses underscores as word separators (NEVER camelCase, NEVER hyphens in filenames)
- The CLI name uses hyphens (matching the convention users expect on the command line)
- **The Python filename MUST include the full model name and version** — different versions of the same model get separate recipe files. In the filename, each segment of the name (including each digit of the version) is separated by underscores. This avoids ambiguity: `v1_5` is clearly version 1.5, while `v15` could mean version 15. This matches the existing convention:

```
ltx_23.py                       ← LTX-2.3
matrix_game_3_0.py              ← Matrix-Game-3.0
mistral_small_31.py             ← Mistral-Small-3.1
qwen_image_2512.py              ← Qwen-Image-2512
cogvideox_fun_v1_5_5b_inp.py    ← CogVideoX-Fun-V1.5-5b-InP
```

- The full model name is preserved in the filename (CogVideoX → `cogvideox`, not `cogvideo`)
- Model size and variant identifiers (`5b`, `inp`, `pro`, `xxl`) are also part of the filename when they distinguish the recipe from other variants
- The CLI name preserves the original version format with dots (e.g., `v1.5`, `3.0`)
- Keep names concise but unambiguous — someone reading the recipe list should understand which model and version it targets
- **HuggingFace output repo** mirrors the source model name exactly (preserving original casing) with a `-mlx` suffix, and `-q{bits}` if quantized. This makes it easy to find the MLX version of any model by appending `-mlx` to the source name.

Present the proposed names to the user for confirmation.

### 2.2 Register in `__init__.py`

Add to `AVAILABLE_RECIPES` in `src/mlx_forge/recipes/__init__.py`:

```python
"recipe-cli-name": "mlx_forge.recipes.module_name",
```

## Phase 3: Implementation

Use the patterns from existing recipes. The codebase provides utilities in `convert.py`, `validate.py`, `transpose.py`, and `quantize.py` — use them instead of reimplementing.

### 3.1 Recipe file structure

Every recipe exports these functions:

```python
def convert(args) -> None          # Main conversion
def validate(args) -> None         # Validate converted model
def add_convert_args(parser)       # CLI arguments for convert
def add_validate_args(parser)      # CLI arguments for validate
def add_split_args(parser)         # CLI arguments for split
```

### 3.2 Constants section

Define these at module level:

```python
REPO_ID = "org/model-name"
COMPONENTS = ["transformer", "text_encoder", "vae"]  # weight components only

_COMPONENT_SIZE_MB = {
    "transformer": 9_500,
    "text_encoder": 9_500,
    "vae": 400,
}

# HF files to download — one list per component + one for configs
_HF_TRANSFORMER_FILES = [...]
_HF_TEXT_ENCODER_FILES = [...]
_HF_VAE_FILES = [...]
_HF_CONFIG_FILES = [...]  # tokenizer, scheduler, model_index, etc.

_ALL_HF_FILES = _HF_TRANSFORMER_FILES + _HF_TEXT_ENCODER_FILES + _HF_VAE_FILES + _HF_CONFIG_FILES

COMPONENT_PREFIX = {comp: comp for comp in COMPONENTS}

_SKIP_QUANTIZE_COMPONENTS = {"vae"}  # conv-heavy components
```

### 3.3 Key sanitization

For each component, write a sanitizer function. Most diffusers models have clean keys (identity sanitizer). Some need prefix stripping or key remapping.

Check by reading a few weight keys from the source to verify they match expectations.

### 3.4 Conv transposition

Determine which components have convolutions:
- **Conv3d** (5D weights): video models (transformer, VAE)
- **Conv2d** (4D weights): image models (VAE, vision encoders)
- **Conv1d** (3D weights): audio models (vocoders, codecs)
- **Pure Linear** (2D weights): text encoders, language models — NO transposition needed

Use `transpose_conv()` from `transpose.py`. Detection: `key.endswith(".weight") and value.ndim >= 3`.

### 3.5 Convert function

Follow this structure:

1. Parse output directory
2. Handle `--dry-run` early return
3. Download all HF files (or use local `--source`)
4. Convert each weight component:
   - Load with `load_weights()` (lazy/memory-mapped)
   - Sanitize keys, transpose convs, materialize each weight
   - Save as `{component}.safetensors` with `{component}.` prefix on all keys
   - Free memory: `del weights; gc.collect(); mx.clear_cache()`
5. Build and save `config.json` (aggregates per-component configs)
6. Copy per-component configs as `{component}_config.json`
7. Copy pipeline files (tokenizer, scheduler, model_index) with directory prefix
8. Save `split_model.json` manifest
9. Optional quantization (skip conv-heavy components)
10. Print summary with file sizes

### 3.6 Pipeline file copying pattern

```python
for config_file in _HF_CONFIG_FILES:
    src = source_dir / config_file
    if src.exists():
        if "/" in config_file:
            prefix = config_file.split("/")[0]
            dest = output_dir / f"{prefix}_{Path(config_file).name}"
        else:
            dest = output_dir / Path(config_file).name
        shutil.copy2(str(src), str(dest))
```

### 3.7 Quantization

For each quantizable component, write a `should_quantize_{component}()` predicate:

```python
def should_quantize_transformer(key: str, weight: mx.array) -> bool:
    if weight.ndim != 2 or not key.endswith(".weight"):
        return False
    # Skip: embeddings, norms, input/output projections, patch_embed, time_embed
    # Quantize: attention qkv/out, ffn weights in blocks
    ...
```

General rules for what to exclude from quantization:
- Embedding layers (`embed_tokens`, `shared.weight`, `patch_embed`)
- Normalization weights (`norm`, `layer_norm`, `rms_norm`)
- Input/output projections not inside blocks (`proj_out`, `proj_in`)
- Time/timestep embeddings (`time_embed`, `timestep`)
- Relative attention biases
- Any 1D tensor (biases, scales)
- Any conv weight (ndim >= 3)

### 3.8 Memory management (critical)

These rules prevent silent data corruption:

1. **Materialize before save**: Call `_materialize(weight)` on every tensor before `mx.save_safetensors()`. Lazy tensors save as zeros.
2. **Materialize before quantize**: Non-quantizable tensors must be materialized BEFORE calling `mx.quantize()`. GPU work from quantization can evict lazy tensor backing buffers.
3. **Free between components**: `del weights; gc.collect(); mx.clear_cache()` after each component.

## Phase 4: Validation

Write a `validate()` function that checks everything. Validation is the safety net — it catches mistakes the conversion missed.

### 4.1 File structure checks

```python
for comp in COMPONENTS:
    validate_file_exists(model_dir, f"{comp}.safetensors", result)
validate_file_exists(model_dir, "config.json", result)
validate_file_exists(model_dir, "split_model.json", result)
validate_file_exists(model_dir, "model_index.json", result)
```

### 4.2 Per-component weight checks

For each component:
1. Load weights
2. Verify all keys have `{component}.` prefix
3. Check expected layer count (from config: `num_layers`, `num_blocks`, etc.)
4. For conv components: `validate_conv_layout(weights, result, ndim=N)`
5. For quantized models: `validate_quantization(weights, result, block_key=...)`
6. Report parameter count
7. Free memory

### 4.3 Architecture-specific checks

Based on the model type, add targeted checks:
- **Inpainting models**: verify expanded input channels on patch_embed
- **Text encoders**: verify shared embedding, final layer norm
- **VAE**: verify encoder and decoder key groups present
- **Language models**: verify embed_tokens, lm_head

## Phase 5: Verification

Before declaring the recipe complete, run these checks:

### 5.1 Lint and format

```bash
uv run ruff check src/mlx_forge/recipes/{filename}.py
uv run ruff format src/mlx_forge/recipes/{filename}.py
```

### 5.2 Dry run

```bash
uv run python -m mlx_forge.cli convert {recipe-name} --dry-run
```

Verify:
- All components listed with correct sizes
- All pipeline files mentioned
- File count matches `_ALL_HF_FILES`
- Quantization plan is correct (if `--quantize` flag tested)

### 5.3 Completeness checklist

Before finishing, verify against this checklist:

- [ ] ALL weight components from source repo are in COMPONENTS list
- [ ] ALL config files are in `_HF_CONFIG_FILES` (tokenizer, scheduler, model_index)
- [ ] Sharding matches reality (index file exists iff we reference it)
- [ ] Conv transposition applied to correct components (not to pure-Linear ones)
- [ ] Quantization skips conv-heavy components
- [ ] Quantization predicates exclude embeddings, norms, and sensitive layers
- [ ] Validation checks every component's weights
- [ ] Validation checks pipeline files (model_index.json)
- [ ] `_materialize()` called on every weight before save
- [ ] Memory freed after each component (`gc.collect()` + `mx.clear_cache()`)
- [ ] Recipe registered in `__init__.py`
- [ ] Ruff passes (lint + format)
- [ ] Dry run succeeds and output looks correct
- [ ] Dry run with `--quantize` shows correct quantization plan

Present this checklist to the user with pass/fail status.
