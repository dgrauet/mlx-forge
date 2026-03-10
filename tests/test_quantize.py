"""Tests for quantization utilities."""

import mlx.core as mx

from mlx_forge.quantize import default_should_quantize, format_bytes


class TestDefaultShouldQuantize:
    def test_linear_weight(self):
        w = mx.zeros((512, 256))
        assert default_should_quantize("layer.weight", w) is True

    def test_bias_rejected(self):
        w = mx.zeros((512,))
        assert default_should_quantize("layer.bias", w) is False

    def test_non_weight_key(self):
        w = mx.zeros((512, 256))
        assert default_should_quantize("layer.scale", w) is False

    def test_small_tensor_rejected(self):
        w = mx.zeros((8, 8))
        assert default_should_quantize("layer.weight", w) is False

    def test_1d_rejected(self):
        w = mx.zeros((512,))
        assert default_should_quantize("layer.weight", w) is False

    def test_degenerate_2d_rejected(self):
        w = mx.zeros((512, 1))
        assert default_should_quantize("layer.weight", w) is False


class TestFormatBytes:
    def test_bytes(self):
        assert "100.00 B" == format_bytes(100)

    def test_megabytes(self):
        assert "1.00 MB" == format_bytes(1024 * 1024)

    def test_gigabytes(self):
        result = format_bytes(2.5 * 1024**3)
        assert "GB" in result
