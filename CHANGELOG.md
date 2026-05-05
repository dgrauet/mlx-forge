# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-05

### Removed

- **Breaking:** `mistral-small-3.1` recipe and associated docs/tests
- **Breaking:** `qwen-image-2512` recipe and associated docs/tests

### Changed

- Refreshed `CLAUDE.md`: list current recipes, add Quick Start and Dev workflow sections

## [0.1.0] - 2026-05-05

### Added

- CLI with `convert`, `validate`, `split`, `quantize`, and `upload` subcommands
- Generic conversion framework (lazy loading, component-by-component processing, conv weight transposition, materialization-aware quantization)
- Recipes:
  - **LTX-2.3** — 22B video DiT (transformer + VAE + text encoders)
  - **Fish S2 Pro** — Dual-AR TTS + DAC codec
  - **Mistral Small 3.1** — 24B VLM (Pixtral vision + dense LLM)
  - **Qwen-Image** — text-to-image MMDiT (Flux-style)
- Delta workflow for adding transformer/LoRA variants to existing repos (`--skip-shared`, `--add-only`, `--card-only`)
- Validation framework (pass/fail/warn)
- HuggingFace Hub upload with auto-generated model cards

[0.2.0]: https://github.com/dgrauet/mlx-forge/releases/tag/v0.2.0
[0.1.0]: https://github.com/dgrauet/mlx-forge/releases/tag/v0.1.0
