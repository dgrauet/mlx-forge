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

## How It Works

```
  ┌─────────────────────────────────┐
  │   model.safetensors (unified)   │
  │   All components in one file    │
  └───────────────┬─────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────┐
  │  1. Lazy load via mx.load()     │  Memory-mapped, ~0 GB RAM
  └───────────────┬─────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────┐
  │  2. Group keys by prefix        │  First dot-separated segment
  │                                 │  determines the component
  │  "transformer.blocks.0.weight"  │  → transformer
  │  "vae_decoder.conv.weight"      │  → vae_decoder
  │  "vocoder.ups.0.weight"         │  → vocoder
  └───────────────┬─────────────────┘
                  │
                  ▼
  ┌─────────────────────────────────┐
  │  3. Match against component map │  Recipe defines prefix → filename
  └───────────────┬─────────────────┘
                  │
          ┌───────┼───────┐
          ▼       ▼       ▼
     ┌────────┐ ┌────┐ ┌────────┐
     │comp_a  │ │ b  │ │comp_c  │   Each saved to its own
     │.safe   │ │.sf │ │.safe   │   .safetensors file
     └────────┘ └────┘ └────────┘
                  │
                  ▼
  ┌─────────────────────────────────┐
  │  4. Write split_model.json      │  Marker file with metadata
  └─────────────────────────────────┘
```

### Key Grouping

The first dot-separated segment of each key determines its component.
Multiple prefixes can map to the same output file (e.g., both `connector.*` and `text_embedding_projection.*` keys can go to `connector.safetensors`).

### Fallback Routing

Keys whose prefix doesn't match any entry in the component map are routed to a fallback file (default: `transformer.safetensors`) or skipped, depending on the recipe configuration.

## Component Map

Each recipe defines a mapping from key prefix to output filename.
See model-specific guides for the exact mappings.

## The split_model.json Marker

Written after every successful split:

```json
{
  "split": true,
  "files": {
    "component_a.safetensors": 1250,
    "component_b.safetensors": 45,
    "component_c.safetensors": 120
  }
}
```

When produced by `convert` (rather than `split`), the marker includes additional metadata:

```json
{
  "format": "split",
  "model_version": "...",
  "components": ["component_a", "component_b", "component_c"],
  "source": "org/model-name",
  "variant": "default",
  "quantized": false
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

## Model-Specific Guides

- [LTX-2.3](models/ltx-2.3.md#split-component-map) — component map and output files
