"""Conv weight transposition: PyTorch layout to MLX channels-last layout.

PyTorch stores conv weights as:
  Conv1d:          (O, I, K)
  Conv2d:          (O, I, H, W)
  Conv3d:          (O, I, D, H, W)
  ConvTranspose1d: (I, O, K)

MLX expects channels-last:
  Conv1d:          (O, K, I)
  Conv2d:          (O, H, W, I)
  Conv3d:          (O, D, H, W, I)
  ConvTranspose1d: (O, K, I)
"""

from __future__ import annotations

import mlx.core as mx


def transpose_conv(value: mx.array, *, is_conv_transpose: bool = False) -> mx.array:
    """Transpose a conv weight tensor from PyTorch to MLX layout.

    Args:
        value: Weight tensor to transpose.
        is_conv_transpose: If True, treat as ConvTranspose (I, O, K) -> (O, K, I).

    Returns:
        Transposed tensor.
    """
    if value.ndim == 5:
        # Conv3d: (O, I, D, H, W) -> (O, D, H, W, I)
        return mx.transpose(value, (0, 2, 3, 4, 1))
    if value.ndim == 4:
        # Conv2d: (O, I, H, W) -> (O, H, W, I)
        return mx.transpose(value, (0, 2, 3, 1))
    if value.ndim == 3:
        if is_conv_transpose:
            # ConvTranspose1d: (I, O, K) -> (O, K, I)
            return mx.transpose(value, (1, 2, 0))
        # Conv1d: (O, I, K) -> (O, K, I)
        return mx.transpose(value, (0, 2, 1))
    return value


def needs_transpose(key: str, value: mx.array) -> bool:
    """Heuristic: does this weight key look like a conv weight that needs transposition?

    Args:
        key: Weight key name.
        value: Weight tensor.

    Returns:
        True if this looks like a conv weight.
    """
    return "conv" in key.lower() and "weight" in key and value.ndim >= 3
