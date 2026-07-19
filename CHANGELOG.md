# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3](https://github.com/dgrauet/mlx-forge/compare/v0.3.2...v0.3.3) (2026-07-19)


### Bug Fixes

* **matrix-game:** keep the resample segment in LightVAE resample keys ([#30](https://github.com/dgrauet/mlx-forge/issues/30)) ([bf7cd88](https://github.com/dgrauet/mlx-forge/commit/bf7cd88cd04fbda3083ef22d34b7941e945d9b57))

## [0.3.2](https://github.com/dgrauet/mlx-forge/compare/v0.3.1...v0.3.2) (2026-07-09)


### Bug Fixes

* **ltx-2.3:** only bundle distilled LoRAs when a dev variant is present ([#26](https://github.com/dgrauet/mlx-forge/issues/26)) ([71a177c](https://github.com/dgrauet/mlx-forge/commit/71a177c8337c98dfec934e5ce4909fb2dfd1348a))

## [0.3.1](https://github.com/dgrauet/mlx-forge/compare/v0.3.0...v0.3.1) (2026-06-19)


### Features

* **recipes:** add ideogram-4 FP8 → MLX conversion recipe ([#19](https://github.com/dgrauet/mlx-forge/issues/19)) ([ad95a11](https://github.com/dgrauet/mlx-forge/commit/ad95a1113e30d3fbdb2eb78212e1ca10ba6906a1))

## [0.3.0](https://github.com/dgrauet/mlx-forge/compare/v0.2.1...v0.3.0) (2026-05-27)


### ⚠ BREAKING CHANGES

* the convert/validate recipe names change from `vjepa2-vitl` / `vjepa2-vit-l-rope` to `vjepa-2.0-vitl` / `vjepa-2.1-vitl`.

### Documentation

* add docs/models/vjepa-2.md model guide ([#17](https://github.com/dgrauet/mlx-forge/issues/17)) ([e5a7bb3](https://github.com/dgrauet/mlx-forge/commit/e5a7bb31a8d0e81ffdbdfab43ddb79baf856bb3d))
* add the V-JEPA 2 recipes to the README ([#15](https://github.com/dgrauet/mlx-forge/issues/15)) ([f8cc8fc](https://github.com/dgrauet/mlx-forge/commit/f8cc8fcdbb90ee8213a8943938dde571833c7379))


### Code Refactoring

* rename V-JEPA 2 recipes to the versioned scheme ([#18](https://github.com/dgrauet/mlx-forge/issues/18)) ([2cd2bb5](https://github.com/dgrauet/mlx-forge/commit/2cd2bb50ddb11031173dfac5f74353c45f716b18))

## [0.2.1](https://github.com/dgrauet/mlx-forge/compare/v0.2.0...v0.2.1) (2026-05-27)


### Features

* **recipes:** add vjepa2-vit-l-rope conversion recipe ([#8](https://github.com/dgrauet/mlx-forge/issues/8)) ([5fdb51c](https://github.com/dgrauet/mlx-forge/commit/5fdb51c4b5df24b6853e501e9882e3348336e6a1))
* **recipes:** add vjepa2-vitl (V-JEPA 2.0 ViT-L + attentive probes) ([#9](https://github.com/dgrauet/mlx-forge/issues/9)) ([be2b5de](https://github.com/dgrauet/mlx-forge/commit/be2b5de3d25fbdaddd1f1aa3e9bcd78713529167))


### Bug Fixes

* **recipes:** version the V-JEPA 2 default output dirs to match HF naming ([#12](https://github.com/dgrauet/mlx-forge/issues/12)) ([40d8130](https://github.com/dgrauet/mlx-forge/commit/40d81300e6c0628364c9bc1f5a9074def0074d9e))
* **upload:** keep dir-name -q{bits} in derive_repo_id when split omits it ([#11](https://github.com/dgrauet/mlx-forge/issues/11)) ([0afd51c](https://github.com/dgrauet/mlx-forge/commit/0afd51cce3c5c26e767a2c815621eedf3e3bec68))
* **upload:** strip existing -mlx suffix in derive_repo_id ([#10](https://github.com/dgrauet/mlx-forge/issues/10)) ([b15ef28](https://github.com/dgrauet/mlx-forge/commit/b15ef2826191d820964a84c0a4270d0fc3b1c085))

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
