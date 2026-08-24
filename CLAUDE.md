# CLAUDE.md — MLX Forge

## Project Overview

CLI tool to convert, quantize, split, and validate ML models for Apple MLX on Apple Silicon.
Generic framework with model-specific "recipes" (11 supported).

## Quick Start

```bash
uv sync                                        # install
mlx-forge convert <recipe>                     # convert (downloads from HF)
mlx-forge convert <recipe> --quantize --bits 8 # convert + quantize
mlx-forge --help                               # list recipes & commands
```

## Tech Stack

- Python 3.11+, `uv` package manager
- MLX (`mlx>=0.31.0`)
- safetensors, huggingface-hub
- CLI: argparse with subcommands + recipe dispatch
- Linter/formatter: ruff

## Architecture

```
src/mlx_forge/
├── cli.py           # CLI dispatcher (convert/validate/split/quantize/upload)
├── convert.py       # Shared conversion utilities (download, load, classify, process)
├── transpose.py     # Conv weight layout transposition (PyTorch -> MLX)
├── quantize.py      # Generic quantization engine
├── split.py         # Split unified safetensors into components
├── validate.py      # Validation framework (pass/fail/warn)
├── upload.py        # HuggingFace Hub upload + model card generation
├── templates/       # Model card Jinja templates
└── recipes/         # Model-specific conversion logic
    ├── __init__.py             # Registry (AVAILABLE_RECIPES) + recipe contract
    ├── ltx_23.py                # LTX-2.3: 22B video DiT
    ├── ltx_25.py                # LTX-2.5: pre-split video+audio DiT, VAEs, upscalers
    ├── ltx_25_text_encoder.py   # LTX-2.5's Gemma-4 text encoder converter
    ├── ideogram_4.py            # Ideogram 4: FP8 text-to-image DiT
    ├── matrix_game_3_0.py       # Matrix-Game 3.0: interactive world model
    ├── cogvideox_fun_v1_5_5b_inp.py  # CogVideoX-Fun 1.5 5B (image-to-video)
    ├── hunyuan3d_21.py          # Hunyuan3D 2.1 (shape + paint stages)
    ├── ernie_image.py           # ERNIE-Image (SFT + Turbo)
    ├── ernie_image_pe.py        # ERNIE-Image PE variant
    ├── vjepa_2_1_vitl.py        # V-JEPA 2.1 ViT-L (encoder + predictor)
    ├── vjepa_2_0_vitl.py        # V-JEPA 2.0 ViT-L (+ attentive probes)
    └── void_model.py            # Netflix VOID: dual-pass CogVideoX transformer
```

## Adding a New Recipe

1. Create `src/mlx_forge/recipes/<model>.py`
2. Implement `classify_key()`, sanitizer functions, and the six functions the
   CLI dispatches to — `convert`/`validate`/`split` plus their
   `add_*_args` counterparts. All six are **required**: `COMMAND_REQUIREMENTS`
   in `recipes/__init__.py` is the contract, `cli.py` enforces it, and
   `tests/test_recipe_contract.py` checks every registered recipe.
   A model that needs no splitting still implements `split` — as a no-op that
   prints why. Omitting it used to produce an AttributeError traceback.
3. Register in `recipes/__init__.py` AVAILABLE_RECIPES dict

Use the shared helpers rather than re-deriving them:

- `convert.add_common_convert_args()` — the
  `--output/--quantize/--bits/--group-size/--dry-run` block. Call it where
  `--output` belongs, then add recipe-specific arguments.
- `convert.load_torch_state_dict()` — `.pt`/`.pth`/`.ckpt` loading with the
  torch-missing hint. Defaults to `weights_only=True`; pass `False` only for a
  trusted checkpoint that stores pickled objects, and say why at the call site.
- `convert.copy_required_files()` — pipeline-file copies that fail loudly on a
  missing required file.

Before adding or changing a recipe, read [docs/recipe-anatomy.md](docs/recipe-anatomy.md):
it separates what a recipe must do differently (upstream-imposed) from what it
must not (the shared layer), and lists the metadata still passed by CLI flag.

## Critical Rules

### Memory Management
- Load source checkpoints lazily via `mx.load()` (memory-mapped)
- Process components one at a time, free between each
- `gc.collect()` + `mx.clear_cache()` after each component
- ALWAYS materialize tensors via `_materialize()` before `mx.save_safetensors()` — lazy tensors save as zeros

### Quantization Safety
- Materialize non-quantizable tensors BEFORE calling `mx.quantize()`
- GPU work from quantization can evict lazy tensor backing buffers
- Only quantize Linear .weight matrices — never conv, norm, embedding layers

### Conv Transposition
- PyTorch: channels-second (O, I, ...) -> MLX: channels-last (O, ..., I)
- Transformer Linear weights do NOT need transposition

## Conventions

- Type hints on all functions
- Google-style docstrings
- ruff for formatting/linting (line-length 100)

## Dev workflow

```bash
uv run pytest         # tests
uv run ruff check     # lint
uv run ruff format    # format
uv run ty check       # type check (strict)
```

`main` is branch-protected: changes go through PRs. CI runs lint, type, test (3.11/3.12/3.13), commitlint, and a smoke-test job.

### torch is an optional dependency

CI installs `uv sync --extra dev` — **without** torch. Consequences:

- Keep the `# ty: ignore[unresolved-import]` on every `import torch`. They look
  unused locally (`ty` reports `unused-ignore-comment` once torch is installed
  in your venv) but they are what keeps the blocking `type` job green in CI.
  Do not "clean them up".
- Conversely, `ty check` passing locally does not mean it passes in CI. To
  reproduce the CI environment:

  ```bash
  UV_PROJECT_ENVIRONMENT=/tmp/ci-venv uv sync --extra dev
  UV_PROJECT_ENVIRONMENT=/tmp/ci-venv uv run ty check
  ```

- Tests touching torch use `pytest.importorskip("torch")` so they skip rather
  than fail where it is absent.

## Delta workflow (adding a variant to an existing repo)

When upstream publishes a new transformer variant or LoRA for a model that's
already converted and uploaded, use the delta workflow instead of regenerating
the full model:

1. **Convert delta** — only the new transformer + LoRAs:
   ```bash
   mlx-forge convert <recipe> --variant <new> --skip-shared --output models/<name>-delta
   ```
   Skips connector, vae_*, audio_vae, vocoder, vae_shared_stats, and upscalers.
   Writes `split_model.json` with `"delta": true`.

2. **Validate** — auto-detects delta mode:
   ```bash
   mlx-forge validate <recipe> models/<name>-delta
   ```
   Logs `[INFO] Delta mode (skipping shared component checks)` and verifies
   only the components present.

3. **Upload delta** — skip files already on remote:
   ```bash
   mlx-forge upload models/<name>-delta --repo-id <user/repo> --add-only
   ```
   Refuses if the repo doesn't exist (use a normal upload first to create it).
   Each new file gets its own commit (more resilient against transient HF
   upload hangs we've observed).

4. **Refresh card** — derive variants from remote, regenerate README:
   ```bash
   mlx-forge upload models/<name>-delta --repo-id <user/repo> --card-only
   ```
   Idempotent. Re-running always produces a card matching the current remote
   state, regardless of what the local model_dir contains.

Currently the only recipes that support `--skip-shared` are `ltx-2.3` and
`ltx-2.5`. Other
recipes can opt in by mirroring the LTX-2.3 implementation pattern (see
`src/mlx_forge/recipes/ltx_23.py`, search for `skip_shared`).
