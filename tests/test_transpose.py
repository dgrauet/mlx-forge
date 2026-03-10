"""Tests for conv weight transposition."""

import mlx.core as mx
import pytest

from mlx_forge.transpose import needs_transpose, transpose_conv


class TestTransposeConv:
    def test_conv3d(self):
        # PyTorch Conv3d: (O=16, I=8, D=3, H=3, W=3)
        w = mx.zeros((16, 8, 3, 3, 3))
        result = transpose_conv(w)
        # MLX: (O=16, D=3, H=3, W=3, I=8)
        assert result.shape == (16, 3, 3, 3, 8)

    def test_conv2d(self):
        # PyTorch Conv2d: (O=32, I=16, H=3, W=3)
        w = mx.zeros((32, 16, 3, 3))
        result = transpose_conv(w)
        # MLX: (O=32, H=3, W=3, I=16)
        assert result.shape == (32, 3, 3, 16)

    def test_conv1d(self):
        # PyTorch Conv1d: (O=64, I=32, K=7)
        w = mx.zeros((64, 32, 7))
        result = transpose_conv(w)
        # MLX: (O=64, K=7, I=32)
        assert result.shape == (64, 7, 32)

    def test_conv_transpose_1d(self):
        # PyTorch ConvTranspose1d: (I=32, O=64, K=7)
        w = mx.zeros((32, 64, 7))
        result = transpose_conv(w, is_conv_transpose=True)
        # MLX: (O=64, K=7, I=32)
        assert result.shape == (64, 7, 32)

    def test_2d_no_transpose(self):
        # Linear weight: (out=128, in=64) — should not be transposed
        w = mx.zeros((128, 64))
        result = transpose_conv(w)
        assert result.shape == (128, 64)

    def test_1d_no_transpose(self):
        # Bias: (128,) — should not be transposed
        w = mx.zeros((128,))
        result = transpose_conv(w)
        assert result.shape == (128,)


class TestNeedsTranspose:
    def test_conv_weight(self):
        w = mx.zeros((16, 8, 3, 3, 3))
        assert needs_transpose("layer.conv.weight", w) is True

    def test_conv_bias(self):
        w = mx.zeros((16,))
        assert needs_transpose("layer.conv.bias", w) is False

    def test_linear_weight(self):
        w = mx.zeros((128, 64))
        assert needs_transpose("layer.linear.weight", w) is False

    def test_non_conv_3d(self):
        w = mx.zeros((16, 8, 3))
        assert needs_transpose("layer.linear.weight", w) is False
