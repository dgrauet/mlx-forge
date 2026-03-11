# Model Splitting

MLX Forge can split a unified `.safetensors` file into per-component files,
allowing each component to be loaded independently without pulling the entire model into memory.

## When to Use Split

The `convert` command already produces split output — each component gets its own file.
**Splitting is only needed for legacy or externally-produced unified models** where all weights are in a single `model.safetensors` file.

If your model still has PyTorch key names, use `convert` instead — it handles key sanitization and conv transposition in addition to splitting.

## CLI Usage

```bash
mlx-forge split <recipe> <model_dir>
```

The `model_dir` must contain a `model.safetensors` file.

### Example

```bash
mlx-forge split ltx-2.3 /path/to/unified-model-dir
```

## How It Works

The generic `split_model()` function performs these steps:

1. **Load** the unified file via `mx.load()` (lazy, memory-mapped).
2. **Group keys by prefix** — the first dot-separated segment of each key determines which component it belongs to.
3. **Match prefixes** against the recipe's component map.
4. **Route unmatched keys** to a fallback file (default: `transformer.safetensors`) or skip them.
5. **Save** each group to its own `.safetensors` file.
6. **Write** a `split_model.json` marker file.

## Component Map

Each recipe defines a mapping from key prefix to output filename.

### LTX-2.3

| Key prefix | Output file |
|-----------|-------------|
| `transformer` | `transformer.safetensors` |
| `connector` | `connector.safetensors` |
| `text_embedding_projection` | `connector.safetensors` |
| `vae_decoder` | `vae_decoder.safetensors` |
| `vae_encoder` | `vae_encoder.safetensors` |
| `vocoder` | `vocoder.safetensors` |
| `audio_vae` | `audio_vae.safetensors` |

Note that `text_embedding_projection` and `connector` keys both map to `connector.safetensors`.

## The split_model.json Marker

Written after every successful split:

```json
{
  "split": true,
  "files": {
    "transformer.safetensors": 1250,
    "connector.safetensors": 45,
    "vae_decoder.safetensors": 120,
    "vae_encoder.safetensors": 118,
    "audio_vae.safetensors": 30,
    "vocoder.safetensors": 25
  }
}
```

When produced by `convert` (rather than `split`), the marker includes additional metadata:

```json
{
  "format": "split",
  "model_version": "2.3.2",
  "components": ["transformer", "connector", "vae_decoder", "vae_encoder", "audio_vae", "vocoder"],
  "source": "Lightricks/LTX-2.3",
  "variant": "distilled",
  "quantized": true,
  "quantization_bits": 8
}
```

This metadata is used by the `upload` command to auto-derive repository names.

## Split vs. Convert

| | `convert` | `split` |
|---|---|---|
| Input | PyTorch checkpoint | Already-converted unified model |
| Key sanitization | Yes | No |
| Conv transposition | Yes | No |
| Config extraction | Yes | No |
| Component splitting | Yes | Yes |
| Quantization (optional) | Yes | No |

Use `convert` when starting from a PyTorch checkpoint. Use `split` only when you already have an MLX-format unified file that needs to be broken into components.

## Disk Space

After splitting, the original unified file is no longer needed.
The split files together have approximately the same total size as the original.

The tool prints a reminder after completion:

```
Split complete. Original model.safetensors can be removed to save disk space.
To remove: rm '/path/to/model.safetensors'
```
