# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.9](https://github.com/dgrauet/mlx-forge/compare/v0.4.8...v0.4.9) (2026-08-15)


### Features

* **cards:** give a quantized build a card that describes it ([#79](https://github.com/dgrauet/mlx-forge/issues/79)) ([ca25f1d](https://github.com/dgrauet/mlx-forge/commit/ca25f1d5c75bd7ac5847b3c40b099e84bdf2b511))
* **recipes:** declare what each quantizer touches, and complete the card metadata ([#81](https://github.com/dgrauet/mlx-forge/issues/81)) ([11f18fe](https://github.com/dgrauet/mlx-forge/commit/11f18fefa7d603ac9b64a7f04d4dbce7b6ac6d0e))

## [0.4.8](https://github.com/dgrauet/mlx-forge/compare/v0.4.7...v0.4.8) (2026-08-14)


### Features

* **upload:** make --link add to the recipe's links instead of replacing them ([#78](https://github.com/dgrauet/mlx-forge/issues/78)) ([187e90e](https://github.com/dgrauet/mlx-forge/commit/187e90e3078737e8213da7a9146eaa3afb2bab65))


### Bug Fixes

* **cards:** ship the upstream licence with the weights it binds ([#74](https://github.com/dgrauet/mlx-forge/issues/74)) ([03e3a6f](https://github.com/dgrauet/mlx-forge/commit/03e3a6f7718548b74e211a4e55cc2ba97438b498))
* **upload:** let a corrected licence reach already-converted packs ([#76](https://github.com/dgrauet/mlx-forge/issues/76)) ([9209483](https://github.com/dgrauet/mlx-forge/commit/92094835543044c3cc608640a656a835329e4d78))
* **vjepa-2.1:** write the recipe declaration into the manifest ([#77](https://github.com/dgrauet/mlx-forge/issues/77)) ([08aa4b3](https://github.com/dgrauet/mlx-forge/commit/08aa4b3634c624fe8b5943a3d9c682fcd68174e0))

## [0.4.7](https://github.com/dgrauet/mlx-forge/compare/v0.4.6...v0.4.7) (2026-08-10)


### Features

* **cards:** generate the card from the recipe declaration, losslessly ([#68](https://github.com/dgrauet/mlx-forge/issues/68)) ([c46fd7e](https://github.com/dgrauet/mlx-forge/commit/c46fd7eb1874d2748cd4550c632c5f6f6efa8196))
* **upload:** add --dry-run, and let --card-only work without local weights ([#70](https://github.com/dgrauet/mlx-forge/issues/70)) ([948bd24](https://github.com/dgrauet/mlx-forge/commit/948bd247bc5d023d4cd64a24cf68d32b548df59b))


### Bug Fixes

* **cards:** keep the blank line between the metadata list and the next heading ([#71](https://github.com/dgrauet/mlx-forge/issues/71)) ([870cf6d](https://github.com/dgrauet/mlx-forge/commit/870cf6d2dba76a672288d3fb84afc8ad484b5cce))
* **scripts:** make the card checker mirror the CLI it is meant to predict ([#72](https://github.com/dgrauet/mlx-forge/issues/72)) ([a49cd8f](https://github.com/dgrauet/mlx-forge/commit/a49cd8f8b94f95e775f3d3a6f3b95db522824df0))
* **upload:** derive the repo name from the recipe, not from a path segment ([#73](https://github.com/dgrauet/mlx-forge/issues/73)) ([d639cd6](https://github.com/dgrauet/mlx-forge/commit/d639cd64ed96e5f6b943756ab07317252ae9ad57))

## [0.4.6](https://github.com/dgrauet/mlx-forge/compare/v0.4.5...v0.4.6) (2026-07-28)


### Features

* **metadata:** record the variant a directory holds, when there is one ([#67](https://github.com/dgrauet/mlx-forge/issues/67)) ([258adff](https://github.com/dgrauet/mlx-forge/commit/258adff8bb725c382144b09ba3b3b0a4911b3487))
* **upload:** bind a converted directory back to its recipe ([#66](https://github.com/dgrauet/mlx-forge/issues/66)) ([6f1d725](https://github.com/dgrauet/mlx-forge/commit/6f1d72530c9010093d00592629436e8c140b0b7e))


### Bug Fixes

* **upload:** declare base_model and license so a card refresh cannot degrade them ([#64](https://github.com/dgrauet/mlx-forge/issues/64)) ([37493c8](https://github.com/dgrauet/mlx-forge/commit/37493c878e31025c877f38ebecdd1f461d7d98fa))

## [0.4.5](https://github.com/dgrauet/mlx-forge/compare/v0.4.4...v0.4.5) (2026-07-27)


### Bug Fixes

* **vjepa-2.0:** nest the components map so the manifest is actually read ([#61](https://github.com/dgrauet/mlx-forge/issues/61)) ([b001614](https://github.com/dgrauet/mlx-forge/commit/b001614a1c9075e4c5eb72f9657848f9abdf7b14))

## [0.4.4](https://github.com/dgrauet/mlx-forge/compare/v0.4.3...v0.4.4) (2026-07-27)


### Features

* **recipes:** declare publication metadata instead of retyping it as flags ([#58](https://github.com/dgrauet/mlx-forge/issues/58)) ([ae01f35](https://github.com/dgrauet/mlx-forge/commit/ae01f359cafe506ae3f29c6ee0118045245c3c42))


### Bug Fixes

* **upload:** keep Hub plumbing out of the card, and warn about --cli-snippet ([#60](https://github.com/dgrauet/mlx-forge/issues/60)) ([e193938](https://github.com/dgrauet/mlx-forge/commit/e193938b2748868db7fa67e495420b4534db336d))


### Documentation

* map what varies between recipes, and why ([#57](https://github.com/dgrauet/mlx-forge/issues/57)) ([74663dc](https://github.com/dgrauet/mlx-forge/commit/74663dc49ab357d947576a2f383011b5ccf2fa9b))

## [0.4.3](https://github.com/dgrauet/mlx-forge/compare/v0.4.2...v0.4.3) (2026-07-26)


### Bug Fixes

* **upload:** list every published file in the model card ([#55](https://github.com/dgrauet/mlx-forge/issues/55)) ([8c71fa1](https://github.com/dgrauet/mlx-forge/commit/8c71fa1a123b6ab0a711b5eea96ccbd118d30a65))

## [0.4.2](https://github.com/dgrauet/mlx-forge/compare/v0.4.1...v0.4.2) (2026-07-26)


### Bug Fixes

* **matrix-game:** validate now checks quantization on quantized models ([#53](https://github.com/dgrauet/mlx-forge/issues/53)) ([54cebc5](https://github.com/dgrauet/mlx-forge/commit/54cebc5328ad0c895b4074036e25c6b99bd42760))

## [0.4.1](https://github.com/dgrauet/mlx-forge/compare/v0.4.0...v0.4.1) (2026-07-26)


### Bug Fixes

* **recipes:** exit non-zero on failed validation, abort on missing ideogram files ([#49](https://github.com/dgrauet/mlx-forge/issues/49)) ([8c9901e](https://github.com/dgrauet/mlx-forge/commit/8c9901e85c09786b318566e8cfbaa08e4370031f))


### Documentation

* list every supported recipe and pin the docs to the registry ([#47](https://github.com/dgrauet/mlx-forge/issues/47)) ([5c19cd8](https://github.com/dgrauet/mlx-forge/commit/5c19cd86c92effff7e84d3a309408d7e3e4804ee))

## [0.4.0](https://github.com/dgrauet/mlx-forge/compare/v0.3.6...v0.4.0) (2026-07-26)


### ⚠ BREAKING CHANGES

* **recipes:** `mlx-forge convert|validate|split fish-s2-pro` no longer exists. Use the MLX Community conversion instead.

### Features

* **recipes:** remove the fish-s2-pro recipe ([#45](https://github.com/dgrauet/mlx-forge/issues/45)) ([c5742ae](https://github.com/dgrauet/mlx-forge/commit/c5742aed24b3d57397f66d9981e31141633d6f92))

## [0.3.6](https://github.com/dgrauet/mlx-forge/compare/v0.3.5...v0.3.6) (2026-07-26)


### Bug Fixes

* **upload:** publish every converted file + materialize skipped quantize weights ([#41](https://github.com/dgrauet/mlx-forge/issues/41)) ([baf440d](https://github.com/dgrauet/mlx-forge/commit/baf440d9ae5aa6252a0ef8bd80ce4c68d9097671))

## [0.3.5](https://github.com/dgrauet/mlx-forge/compare/v0.3.4...v0.3.5) (2026-07-25)


### Bug Fixes

* **cli:** make the recipe command contract explicit and testable ([#36](https://github.com/dgrauet/mlx-forge/issues/36)) ([58b013f](https://github.com/dgrauet/mlx-forge/commit/58b013fe0338a242a35f675296dbb79a2ddd782e))


### Documentation

* refresh CLAUDE.md and README for the current recipe set and contract ([#38](https://github.com/dgrauet/mlx-forge/issues/38)) ([3d5532e](https://github.com/dgrauet/mlx-forge/commit/3d5532ef89c37078c3ff36d35f4d914e2f3bd540))

## [0.3.4](https://github.com/dgrauet/mlx-forge/compare/v0.3.3...v0.3.4) (2026-07-20)


### Bug Fixes

* **cogvideox:** fail loudly when required pipeline files are missing ([#32](https://github.com/dgrauet/mlx-forge/issues/32)) ([e061a6d](https://github.com/dgrauet/mlx-forge/commit/e061a6dafeeb7b2a22190d82cabff5bd1684c375))
* **recipes:** strict pipeline-file copies across fish_s2, matrix-game, ernie ([#32](https://github.com/dgrauet/mlx-forge/issues/32) audit) ([#34](https://github.com/dgrauet/mlx-forge/issues/34)) ([cd2e660](https://github.com/dgrauet/mlx-forge/commit/cd2e6600370634874dfd93030233dccfd27ebc52))

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
