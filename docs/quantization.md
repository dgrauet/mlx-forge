# Quantization

MLX Forge quantizes model weights from float16/float32 to int4 or int8 using MLX's affine quantization.
Each quantized weight is stored as three tensors: the packed integer data, per-group scales, and per-group biases.

## CLI Usage

### Generic Mode

Quantize any safetensors file directly:

```bash
mlx-forge quantize model.safetensors --bits 8
mlx-forge quantize model.safetensors --bits 4 --group-size 128
mlx-forge quantize model.safetensors --key-prefix transformer. --bits 8
mlx-forge quantize model.safetensors --output quantized.safetensors
```

| Flag | Default | Description |
|------|---------|-------------|
| `--bits` | `8` | Quantization bits (`4` or `8`) |
| `--group-size` | `64` | Number of elements sharing one scale/bias pair |
| `--key-prefix` | *(all)* | Only quantize weight keys starting with this prefix |
| `--output` | *(overwrite input)* | Output file path |

### Recipe-Integrated Mode

Quantize during conversion:

```bash
mlx-forge convert <recipe> --quantize --bits 8 --group-size 64
```

In this mode, the recipe controls which layers are quantized via its own predicate function.

## How Affine Quantization Works

```
  Original weight (float16)
  ┌──────────────────────────────────────────────┐
  │ 0.12  -0.45  0.78  0.33  -0.91  0.56  ...   │
  └──────────────────┬───────────────────────────┘
                     │
           split into groups of group_size
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ group 1 │ │ group 2 │ │ group N │
    └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │
    For each group:
    1. Find min, max
    2. Compute scale = (max - min) / (2^bits - 1)
    3. Compute bias  = min
    4. Pack integers = round((value - bias) / scale)
         │           │           │
         ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ packed  │ │ packed  │ │ packed  │   uint32 (multiple int values per uint32)
    │ + scale │ │ + scale │ │ + scale │   float16 per group
    │ + bias  │ │ + bias  │ │ + bias  │   float16 per group
    └─────────┘ └─────────┘ └─────────┘
```

At inference time, the weight is reconstructed: `W_float = W_int * scale + bias`.

The packed integer data is stored as `uint32` arrays (multiple int4/int8 values packed per uint32).

## What Gets Quantized

### Default Predicate

The generic `default_should_quantize` function selects a weight for quantization when:

- Key ends with `.weight`
- Tensor has 2 or more dimensions
- Tensor has at least 256 elements
- Neither dimension is 1 (excludes degenerate shapes)

This covers standard Linear layer weights while excluding biases, 1D norms, and embeddings.

### Recipe Predicates

Recipes can override the default predicate to apply stricter filtering.
Common exclusions include:

| Layer type | Why exclude |
|------------|------------|
| Timestep/conditioning embeddings | Too sensitive — causes broken generation |
| Final projection layers | Causes visual/audio artifacts |
| Input patchification layers | Causes quality degradation |
| Conv weights | Not Linear — incompatible with affine quantization |
| Norm layers | Too sensitive to precision loss |
| Embedding layers | Small tensors, negligible size savings |

See [model-specific guides](#model-specific-guides) for each recipe's exclusion list.

## The Materialization Safety Rule

This is the most critical implementation detail in the quantization pipeline.

```
  ┌─────────────────────────────────────────────────────────┐
  │  CRITICAL: mx.quantize() GPU work can EVICT             │
  │  lazy tensor backing buffers, zeroing them permanently   │
  └─────────────────────────────────────────────────────────┘
```

### The Problem

`mx.quantize()` triggers GPU computation. GPU work can **evict the memory-mapped backing buffers** of lazy tensors that have not yet been evaluated. Those tensors silently become all-zeros.

### The Consequence

If you load a safetensors file (lazy), then quantize one weight, other weights in the same dict that you haven't touched may be corrupted — permanently zeroed out.

### The Solution

The `quantize_weights()` function follows a strict ordering:

```
  ┌────────────────────────────────────────────┐
  │ 1. Partition into to_quantize and to_keep  │
  └─────────────────┬──────────────────────────┘
                    │
                    ▼
  ┌────────────────────────────────────────────┐
  │ 2. Materialize ALL to_keep tensors         │ ◄── BEFORE any GPU work
  └─────────────────┬──────────────────────────┘
                    │
                    ▼
  ┌────────────────────────────────────────────┐
  │ 3. For each weight in to_quantize:         │
  │    a. Materialize the weight               │
  │    b. mx.quantize() → GPU work             │
  │    c. Materialize quantized outputs        │
  │    d. Store q_weight, scales, biases       │
  │    e. Delete original                      │
  └────────────────────────────────────────────┘
```

Step 2 is the critical one: all tensors that will **not** be quantized must be materialized first, because the GPU work in step 3 may evict their backing buffers.

Step 3a materializes each weight individually rather than all at once, to avoid accumulating a large lazy computation graph that would cause OOM.

## Group Size Compatibility

If a weight's last dimension is not divisible by the group size, it is **silently skipped** and kept in original precision.
This is by design — `mx.quantize()` requires exact divisibility.

To check if any weights were skipped, compare the number of `.scales` keys in the output with the number of `.weight` keys that matched the quantization predicate.

## Output Artifacts

### Quantized Safetensors

For a quantized weight originally at key `layer.weight`:

| Key | Content |
|-----|---------|
| `layer.weight` | Packed `uint32` data (int4 or int8 values) |
| `layer.scales` | Per-group scale factors (`float16`) |
| `layer.biases` | Per-group bias/zero-point values (`float16`) |

### quantize_config.json

Written alongside the output file. The exact content depends on the recipe, but typically includes:

```json
{
  "quantization": {
    "bits": 8,
    "group_size": 64
  }
}
```

Recipes may add extra fields to indicate which layers were targeted.

## Choosing Bits and Group Size

| Setting | Compression | Quality | Use case |
|---------|------------|---------|----------|
| int8, group_size=64 | ~2x | Minimal loss | **Recommended default** |
| int8, group_size=128 | ~2x | Slightly lower | When memory is very tight |
| int4, group_size=64 | ~4x | Noticeable loss | Extreme memory constraints |
| int4, group_size=32 | ~4x | Better than g64 | Better int4 quality, larger scales overhead |

## Model-Specific Guides

- [LTX-2.3](models/ltx-2.3.md#quantization-strategy) — quantization strategy and exclusion list
- [Fish S2 Pro](models/fish-s2-pro.md#quantization-strategy) — quantization strategy
- [Mistral Small 3.1](models/mistral-small-3.1.md#quantization-strategy) — quantization strategy
