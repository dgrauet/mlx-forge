# V-JEPA 2

[facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) is Meta's
self-supervised video/image world model — a ViT encoder trained with a JEPA
objective plus a latent **predictor** (world model). Two recipes cover the ViT-L
slice of the two released lines:

- **`vjepa-2.1-vitl`** — V-JEPA **2.1** ViT-L (distilled from ViT-G @384). One
  checkpoint holds the **encoder + predictor**.
- **`vjepa-2.0-vitl`** — V-JEPA **2.0** ViT-L. One checkpoint holds the **encoder +
  predictor**; three separate checkpoints hold the **attentive-probe classifiers**
  (Something-Something v2, Diving-48, Epic-Kitchens-100).

Weights are MIT-licensed and distributed as Meta torch-hub `.pt` files (NOT on the
HuggingFace Hub), so both recipes require an explicit local `--source` and the
`[torch]` extra (`pip install 'mlx-forge[torch]'`). Converted MLX weights are
published at
[`dgrauet/vjepa-2.1-vitl-mlx`](https://huggingface.co/dgrauet/vjepa-2.1-vitl-mlx)
and
[`dgrauet/vjepa-2.0-vitl-mlx`](https://huggingface.co/dgrauet/vjepa-2.0-vitl-mlx).

Source checkpoint URLs (download manually):

| Line | File | URL |
|------|------|-----|
| 2.1 ViT-L | `vjepa2_1_vitl_dist_vitG_384.pt` | `https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt` |
| 2.0 ViT-L | `vitl.pt` | `https://dl.fbaipublicfiles.com/vjepa2/vitl.pt` |
| 2.0 SSv2 probe | `ssv2-vitl-16x2x3.pt` | `https://dl.fbaipublicfiles.com/vjepa2/evals/ssv2-vitl-16x2x3.pt` |
| 2.0 Diving48 probe | `diving48-vitl-256.pt` | `https://dl.fbaipublicfiles.com/vjepa2/evals/diving48-vitl-256.pt` |
| 2.0 EK100 probe | `ek100-vitl-256.pt` | `https://dl.fbaipublicfiles.com/vjepa2/evals/ek100-vitl-256.pt` |

> There are **no V-JEPA 2.1 probes** in the release; the eval probes above are
> 2.0 ViT-L (256px). The 2.1 line ships encoder + predictor only.

## Quick Start

```bash
pip install 'mlx-forge[torch]'   # torch is needed to read the .pt checkpoints

# --- V-JEPA 2.1 ViT-L (encoder + predictor) ---
mlx-forge convert vjepa-2.1-vitl --source ~/weights/vjepa2_1_vitl_dist_vitG_384.pt
mlx-forge validate vjepa-2.1-vitl models/vjepa-2.1-vitl-mlx

# --- V-JEPA 2.0 ViT-L (encoder + predictor + all three probes) ---
mlx-forge convert vjepa-2.0-vitl \
    --source          ~/weights/vitl.pt \
    --ssv2-source     ~/weights/ssv2-vitl-16x2x3.pt \
    --diving48-source ~/weights/diving48-vitl-256.pt \
    --ek100-source    ~/weights/ek100-vitl-256.pt
mlx-forge validate vjepa-2.0-vitl models/vjepa-2.0-vitl-mlx

# V-JEPA 2.0 encoder + predictor only (no probes)
mlx-forge convert vjepa-2.0-vitl --source ~/weights/vitl.pt

# Preview without writing
mlx-forge convert vjepa-2.1-vitl --source ~/weights/vjepa2_1_vitl_dist_vitG_384.pt --dry-run
```

## CLI Options

### Convert — `vjepa-2.1-vitl` (2.1)

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | *(required)* | Path to `vjepa2_1_vitl_dist_vitG_384.pt` |
| `--output` | `models/vjepa-2.1-vitl-mlx[-q<bits>]` | Output directory |
| `--quantize` | off | Quantize encoder + predictor block Linear weights |
| `--bits` | `8` | Quantization bits (`4` or `8`) |
| `--group-size` | `64` | Quantization group size |
| `--dry-run` | off | Preview conversion plan without writing |

### Convert — `vjepa-2.0-vitl` (2.0)

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | *(required)* | Path to `vitl.pt` (encoder + predictor) |
| `--ssv2-source` | *(optional)* | Path to the SSv2 AttentiveClassifier probe |
| `--diving48-source` | *(optional)* | Path to the Diving-48 probe |
| `--ek100-source` | *(optional)* | Path to the Epic-Kitchens-100 probe |
| `--output` | `models/vjepa-2.0-vitl-mlx[-q<bits>]` | Output directory |
| `--quantize` | off | Quantize encoder/predictor/probe block Linear weights |
| `--bits` | `8` | Quantization bits (`4` or `8`) |
| `--group-size` | `64` | Quantization group size |
| `--dry-run` | off | Preview conversion plan without writing |

### Validate (both recipes)

| Flag | Default | Description |
|------|---------|-------------|
| `model_dir` | *(required)* | Path to the converted model directory |

## Model Architecture

All four blocks (encoder, predictor, probe pooler) are RoPE transformers; only
`patch_embed` carries convolutions.

### Encoder (ViT-L, both lines)

- `embed_dim=1024`, `depth=24`, `num_heads=16`, `use_rope=True`.
- **2.0**: a single `PatchEmbed3D` Conv3d (tubelet `2×16×16`) and a single final
  `norm` LayerNorm — no per-block `norms_block`.
- **2.1**: dual patch embed — `patch_embed` (Conv3d, video) **and**
  `patch_embed_img` (Conv2d, single-frame), `img_temporal_dim_size=1`. Distilled
  head: `n_output_distillation=1` (the dense output is `(B, N, embed_dim)`, not a
  4-level concat).
- 3D RoPE uses an **interleaved** pair convention with unrotated tail dims (2.1)
  vs a tiled convention (2.0) — different between the lines, see the MLX port.

### Predictor (world model, both lines)

- `predictor_embed_dim=384`, `depth=12`, RoPE. `predictor_embed` projects
  `1024 → 384`; `predictor_proj` maps back to the target dim (`1024` for 2.0,
  `1664` = teacher_embed_dim for 2.1). Learnable `mask_tokens` of shape
  `(1, 1, 384)`. **No convolutions.**

### Probe (V-JEPA 2.0 only — `AttentiveClassifier`)

- `pooler`: 1 cross-attention block (`xattn.q` / `xattn.kv`) + 3 self-attention
  blocks, plus `query_tokens`.
- Classifier head(s), auto-detected from weight shapes:
  - SSv2: `linear` (174 classes)
  - Diving-48: `linear` (48 classes)
  - EK100: `verb_classifier` (97) + `noun_classifier` (289) + `action_classifier`
    (3568) — no single `linear`.

## Source Checkpoint Structure

```
vjepa2_1_vitl_dist_vitG_384.pt (2.1)   vitl.pt (2.0)
├── ema_encoder   (302 keys) ◄─ used    ├── target_encoder (292 keys) ◄─ used
├── encoder       (302 keys)            ├── encoder        (292 keys)
├── predictor     (162 keys) ◄─ used    ├── predictor      (160 keys) ◄─ used
├── opt / scaler / epoch / loss …       └── opt / scaler / epoch / loss …

ssv2-vitl-16x2x3.pt / diving48-*.pt / ek100-*.pt (2.0 probes)
└── classifiers[0]  (51–55 keys) ◄─ used
```

Encoder selection: the 2.1 recipe prefers the `ema_encoder` container (matching
upstream `checkpoint_key="ema_encoder"`), falling back through
`encoder / target_encoder / model / state_dict`; the 2.0 recipe reads
`target_encoder`. The predictor is read from `predictor` in the **same** `vitl.pt`
/ 2.1 checkpoint (loaded via mmap so the 5 GB file isn't materialized twice).

## Key Translations (PyTorch → MLX)

| Component | Prefix stripped | Transpose |
|-----------|-----------------|-----------|
| Encoder | `module.backbone.` / `module.` / `encoder.` / `backbone.` | `patch_embed.proj.weight` (and `patch_embed_img.proj.weight` for 2.1) |
| Predictor | `module.backbone.` | none — `mask_tokens` are `(1,1,384)` embeddings, **not** convs |
| Probe | `module.` | none |

Keys are otherwise byte-aligned 1:1 with the MLX modules.

## Conv Transposition

Only the encoder patch-embed weights are transposed from PyTorch
channels-second to MLX channels-last:

| Layer | PyTorch | MLX |
|-------|---------|-----|
| `patch_embed.proj.weight` (Conv3d, video) | `(O, I, D, H, W)` | `(O, D, H, W, I)` |
| `patch_embed_img.proj.weight` (Conv2d, 2.1 image path) | `(O, I, H, W)` | `(O, H, W, I)` |

The predictor and probes contain no convolutions; validation asserts no ≥4D
tensors leak into them (so `mask_tokens` keep their `(1,1,D)` shape).

## Quantization Strategy

`--quantize` targets only the block Linear `.weight` matrices:

- **Encoder**: `blocks.*.attn` / `mlp` Linears.
- **Predictor**: `predictor_blocks.*.attn` / `mlp` Linears.
- **Probe**: `pooler.*` Linears.

Kept full-precision (correctness-sensitive or non-Linear): all norms, all biases,
`patch_embed*`, `query_tokens`, classifier heads, `predictor_embed`,
`predictor_proj`, and `mask_tokens`.

> The world-model **predictor** is small and its response to quantization is not
> yet characterized — quantize the encoder for memory savings and keep an eye on
> the predictor if you quantize the full stack.

## Output Files

```
models/vjepa-2.1-vitl-mlx/              models/vjepa-2.0-vitl-mlx/
├── encoder.safetensors    (~1.16 GB)   ├── encoder.safetensors    (~1.16 GB)
├── predictor.safetensors  (~88 MB)     ├── predictor.safetensors  (~84 MB)
├── config.json                         ├── ssv2_probe.safetensors     (~190 MB)
└── split_model.json                    ├── diving48_probe.safetensors (~190 MB)
                                         ├── ek100_probe.safetensors    (~204 MB)
                                         ├── config.json
                                         └── split_model.json
```

`quantize_config.json` is added when `--quantize` is used. `split_model.json`
lists the components present (probes only appear if their `--*-source` was given).

## Critical Config Values

| Value | Encoder | Predictor |
|-------|---------|-----------|
| `embed_dim` | 1024 | 384 |
| `depth` | 24 | 12 |
| `num_heads` | 16 | (12) |
| base key count (2.1) | 302 | 162 |
| base key count (2.0) | 292 | 160 |
| `patch` / `tubelet` | 16 / 2 | — |

## Validation Checks

`mlx-forge validate` confirms, per component present:

- File structure (`config.json`, `split_model.json`, each `*.safetensors`).
- No residual `module.` / `backbone.` prefixes.
- Exact base key counts and block counts (encoder 24, predictor 12, probe 3
  self-attn blocks).
- Encoder `patch_embed.proj.weight` is channels-last (`…, I`) with `in_channels=3`.
- Predictor has no ≥4D tensors (no conv leaked in).
- Probe classifier head class counts match the detected heads.
- Quantization integrity (scales/biases) when the model is quantized.

## Known Gotchas

### Conversion

- **`ema_encoder` vs `encoder`.** Both exist in the checkpoint; inference uses the
  EMA target encoder. The recipe prefers `ema_encoder` (2.1) / `target_encoder`
  (2.0) — do not grab the bare `encoder`.
- **No conv transpose on the predictor.** Its only ≥3D tensors are `mask_tokens`
  `(1,1,384)` — transposing them would corrupt the world model.
- **`[torch]` extra + `--source` are mandatory.** The weights are not on the HF
  Hub; there is no auto-download path.

### Faithful all-zero weights (not a bug)

Several `predictor.mask_tokens.*` (e.g. indices 1–9 in 2.0) are **exactly zero in
the source** — they were initialized to zero and never trained (only
`mask_tokens.0` is used). The conversion copies them faithfully; validation does
not flag them.

### Runtime (for model loaders)

- Output dirs and the published repos are **versioned** (`vjepa-2.X-vitl-mlx`), so
  an auto-derived `mlx-forge upload` lands on `dgrauet/vjepa-2.X-vitl-mlx[-q{bits}]`.
- 3D RoPE differs between 2.0 and 2.1 (tiled vs interleaved) — load each line with
  its own encoder implementation.

## Downstream Inference

The MLX inference port lives at
[github.com/dgrauet/vjepa2-mlx](https://github.com/dgrauet/vjepa2-mlx) (encoder +
predictor + preprocessor + 2.0 probes + full JEPA training, parity-locked against
the PyTorch reference). Its `from_pretrained()` auto-downloads from
`dgrauet/vjepa-2.1-vitl-mlx` by default:

```python
from vjepa2_core_mlx.utils.weights import from_pretrained, from_pretrained_predictor

encoder = from_pretrained()                 # downloads dgrauet/vjepa-2.1-vitl-mlx
predictor = from_pretrained_predictor()     # same repo, predictor.safetensors
```
