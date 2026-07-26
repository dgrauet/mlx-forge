"""Tests for quantization utilities."""

import mlx.core as mx

from mlx_forge import quantize as quantize_mod
from mlx_forge.quantize import default_should_quantize, format_bytes, quantize_weights


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


class TestQuantizeWeightsMaterialization:
    """Every tensor leaving quantize_weights must be materialized.

    The module's own rule: mx.quantize() triggers GPU work that can evict a
    lazy tensor's memory-mapped backing buffer, and an unmaterialized tensor
    saves as zeros. Weights skipped for a non-divisible last dimension used to
    be copied into the result without ever being materialized, while later
    iterations kept issuing GPU work.
    """

    def _spy(self, monkeypatch):
        seen: list[int] = []
        real = quantize_mod._materialize

        def spy(*tensors):
            seen.extend(id(t) for t in tensors)
            real(*tensors)

        monkeypatch.setattr(quantize_mod, "_materialize", spy)
        return seen

    def test_skipped_weight_is_materialized(self, monkeypatch):
        seen = self._spy(monkeypatch)
        # last dim 30 is not divisible by group_size 64 -> skipped
        skipped = mx.ones((64, 30))
        weights = {"a.weight": skipped, "b.weight": mx.ones((64, 128))}

        result = quantize_weights(weights, bits=8, group_size=64, should_quantize=lambda k, w: True)

        assert id(result["a.weight"]) in seen, "skipped weight left unmaterialized"

    def test_skipped_weight_keeps_its_values(self, monkeypatch):
        weights = {"a.weight": mx.full((64, 30), 3.0), "b.weight": mx.ones((64, 128))}
        result = quantize_weights(weights, bits=8, group_size=64, should_quantize=lambda k, w: True)
        assert result["a.weight"].sum().item() == 64 * 30 * 3.0

    def test_every_returned_tensor_is_materialized(self, monkeypatch):
        seen = self._spy(monkeypatch)
        weights = {
            "kept.bias": mx.ones((16,)),
            "skipped.weight": mx.ones((64, 30)),
            "quantized.weight": mx.ones((64, 128)),
        }

        result = quantize_weights(
            weights,
            bits=8,
            group_size=64,
            should_quantize=lambda k, w: k.endswith(".weight"),
        )

        unmaterialized = [k for k, v in result.items() if id(v) not in seen]
        assert unmaterialized == []


class TestFormatBytes:
    def test_bytes(self):
        assert "100.00 B" == format_bytes(100)

    def test_megabytes(self):
        assert "1.00 MB" == format_bytes(1024 * 1024)

    def test_gigabytes(self):
        result = format_bytes(2.5 * 1024**3)
        assert "GB" in result
