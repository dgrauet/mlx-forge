# Matrix-Game-4.0

[Skywork/Matrix-Game-4.0](https://huggingface.co/Skywork/Matrix-Game-4.0) — scene-specialized causal video renderers. An executable game owns gameplay, physics and input and renders a low-poly control video; the model re-renders it as clean RGB (480×864). One checkpoint per scene: Medieval Castle, Ancient Egypt, City Racing.

> MLX Forge converts four components in a single pass: dit_castle, dit_ancient_egypt, dit_racing, taehv. T5, the Wan2.2 VAE and the tokenizer are **referenced** in [`dgrauet/matrix-game-3.0-mlx`](https://huggingface.co/dgrauet/matrix-game-3.0-mlx), not copied — see [Shared components](#shared-components).

Status: **Experimental** — no MLX inference port consumes this layout yet.

## Quick Start

```bash
# Convert everything (~30 GB of scene DiTs to download)
mlx-forge convert matrix-game-4.0

# Convert + quantize the three scene DiTs
mlx-forge convert matrix-game-4.0 --quantize --bits 8

# Preview what will happen
mlx-forge convert matrix-game-4.0 --dry-run

# Validate
mlx-forge validate matrix-game-4.0 models/matrix-game-4.0-mlx
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output` | `models/matrix-game-4.0-mlx[-q<bits>]` | Output directory |
| `--quantize` | off | Quantize the three scene DiTs after conversion |
| `--bits` | `8` | Quantization bits (`4` or `8`) |
| `--group-size` | `64` | Quantization group size |
| `--dry-run` | off | Preview plan without downloading or writing |

## Architecture

```
 Skywork/Matrix-Game-4.0
 ├── fps-castle/model.safetensors    (10.0 GB, BF16) ─► dit_castle.safetensors
 ├── fps-egypt/model.safetensors     (10.0 GB, BF16) ─► dit_ancient_egypt.safetensors
 ├── driving_game/model.safetensors  (10.0 GB, BF16) ─► dit_racing.safetensors
 ├── */inference.json, */sink.png                    ─► copied, Hub tree kept
 ├── base/taew2_2.pth                (23 MB, fp16)   ─► taehv.safetensors
 ├── LICENSE                                         ─► copied (Apache-2.0, the pack's licence)
 └── licenses/TAEHV-LICENSE                          ─► copied (MIT notice)

 alibaba-pai/Wan2.2-Fun-5B-Control @ b8bc1a65ab71 (pinned, as upstream's releases/models.json)
 └── config.json                                     ─► folded into config.json

 dgrauet/matrix-game-3.0-mlx @ ff0d4df16dde (referenced in split_model.json `shared`)
 ├── t5_encoder.safetensors          (10.8 GB, BF16)
 ├── vae.safetensors                 (2.7 GB, F32)
 └── google/umt5-xxl/                (tokenizer)
```

## Shared components

Upstream does not duplicate T5, the Wan2.2 VAE or the tokenizer: its `releases/models.json` fetches them from `alibaba-pai/Wan2.2-Fun-5B-Control` at a pinned revision. Those sources are byte-identical (sha256) to the ones Matrix-Game-3.0 ships, and re-converting them reproduced `matrix-game-3.0-mlx`'s `t5_encoder.safetensors` and `vae.safetensors` byte for byte. So this pack follows upstream and references them:

```json
"shared": {
  "repo_id": "dgrauet/matrix-game-3.0-mlx",
  "revision": "ff0d4df16dde5c86fca8cbd934a41fdff5eec00c",
  "components": {"t5_encoder": "t5_encoder.safetensors", "vae": "vae.safetensors"},
  "tokenizer": "google/umt5-xxl",
  "files": [{"path": "t5_encoder.safetensors", "sha256": "f8770d43…"}, …]
}
```

A consumer downloads `files` from `repo_id` at `revision` and checks each sha256. `convert` verifies the reference against the Hub before writing the manifest, and aborts if any file is missing or has changed. Pinning a revision means a later republish of matrix-game-3.0-mlx cannot silently change what this pack loads; bumping it is a deliberate recipe change.

Each scene DiT is a causal finetune of Wan2.2-Fun-5B-Control (30 blocks, dim 3072, 5.0B params). Its 148 input channels are noisy latent 48 | control latent 48 | zeros 52. There is no action module: control comes from the rendered video, not from keyboard/mouse embeddings as in Matrix-Game-3.0.

`config.json` carries the DiT architecture plus a `scenes` table mapping each runtime scene name (`castle`, `ancient-egypt`, `racing`) to its component, `inference.json` and sink image. The per-scene `inference.json` is the inference contract (sink policy, RoPE mode, steps, prompt) and must be honoured as-is: castle/egypt use an identity sink with fixed-relative RoPE, racing uses the supplied first frame with an absolute-position sink.

### Codecs

| Preset | Codec | Steps |
|--------|-------|-------|
| Quality | `vae` (Wan2.2 VAE) | 2 (1000, 500) |
| Realtime | `taehv` (TAEHV taew2_2) | 1 (1000) |

## Key mapping

- **DiT**: the whole-file `model.` wrapper prefix is stripped (as the reference loader does), `ref_conv.*` is dropped if present, then Matrix-Game-3.0's Sequential → named-Linear mapping applies (`ffn.0` → `ffn_linear1`, `text_embedding.2` → `text_embedding_linear2`, …). `patch_embedding` Conv3d `(3072, 148, 1, 2, 2)` is reshaped to Linear `(3072, 592)`.
- **TAEHV**: keys kept verbatim — the `encoder.N` / `decoder.N` indices encode where the parameter-free layers sit in the `nn.Sequential`. Every 4D tensor is a Conv2d and is transposed to channels-last.

## Licence

Apache-2.0, as upstream declares for the scene weights; the shared Wan/umT5 files are Apache-2.0 upstream too. The pack ships upstream's `LICENSE` and TAEHV's MIT notice under `licenses/`, both copied with the weights and checked by `validate`. Upstream's `licenses/MG4CF-LICENSE` is byte-identical to `LICENSE` and is not duplicated.

## Not converted

- `driving_game/vae_decoder/decoder.pth` — a TAEHV with a finetuned decoder (encoder identical to `taew2_2`), absent from `releases/models.json` and loaded nowhere in the reference source.
- `game/`, `racing/assets/` — engine-side geometry, maps and audio.

## Quantization

Only Linear weights inside `blocks.N` (self/cross-attention q/k/v/o and both FFN Linears) of the three scene DiTs. Patch/time/text embeddings, `time_projection`, the output head, norms and modulation stay bf16. TAEHV is never quantized; T5 and the VAE are not part of this pack.
