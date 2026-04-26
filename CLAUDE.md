# CLAUDE.md — MLX Forge

## Project Overview

CLI tool to convert, quantize, split, and validate ML models for Apple MLX on Apple Silicon.
Generic framework with model-specific "recipes". First recipe: LTX-2.3 (22B video DiT).

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
└── recipes/         # Model-specific conversion logic
    ├── __init__.py   # Recipe registry
    ├── ltx_23.py     # LTX-2.3 recipe
    ├── fish_s2.py    # Fish S2 Pro: Dual-AR TTS + DAC codec
    ├── mistral_small_31.py  # Mistral Small 3.1: 24B VLM (Pixtral + dense LLM)
    └── qwen_image_2512.py # Qwen-Image: text-to-image MMDiT (Flux-style)
```

## Adding a New Recipe

1. Create `src/mlx_forge/recipes/<model>.py`
2. Implement: `classify_key()`, sanitizer functions, `convert()`, `validate()`
3. Export `add_convert_args()`, `add_validate_args()`, `add_split_args()` for CLI
4. Register in `recipes/__init__.py` AVAILABLE_RECIPES dict

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

Currently the only recipe that supports `--skip-shared` is `ltx-2.3`. Other
recipes can opt in by mirroring the LTX-2.3 implementation pattern (see
`src/mlx_forge/recipes/ltx_23.py`, search for `skip_shared`).
