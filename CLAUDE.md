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
├── transpose.py     # Conv weight layout transposition (PyTorch -> MLX)
├── quantize.py      # Generic quantization engine
├── split.py         # Split unified safetensors into components
├── validate.py      # Validation framework (pass/fail/warn)
├── upload.py        # HuggingFace Hub upload + model card generation
└── recipes/         # Model-specific conversion logic
    ├── __init__.py   # Recipe registry
    ├── ltx_23.py     # LTX-2.3 recipe
    └── fish_s2.py    # Fish S2 Pro recipe (Phase 1)
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
