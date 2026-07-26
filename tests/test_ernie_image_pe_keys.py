"""Tests for the ERNIE-Image PE recipe: key handling, quantization scope, output naming.

PE is the single-component variant. Its quantization policy deliberately
differs from the sibling recipe: `embed_tokens` IS quantized, because MLX's
QuantizedEmbedding supports `as_linear` and the table is ~768 MB.
"""

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_forge.recipes.ernie_image_pe import (
    _default_output_dir,
    _sanitize_pe_key,
    ernie_image_pe_should_quantize,
)


class TestSanitizePeKey:
    @pytest.mark.parametrize(
        "key",
        [
            "layers.0.self_attn.q_proj.weight",
            "embed_tokens.weight",
            "norm.weight",
        ],
    )
    def test_keys_pass_through_unchanged(self, key):
        assert _sanitize_pe_key(key) == key


class TestQuantizationScope:
    def test_block_linear_quantized(self):
        assert ernie_image_pe_should_quantize(
            "layers.0.self_attn.q_proj.weight", mx.zeros((512, 512))
        )

    def test_embed_tokens_is_quantized(self):
        """Unlike ernie-image: QuantizedEmbedding.as_linear keeps the tied head working."""
        assert ernie_image_pe_should_quantize("embed_tokens.weight", mx.zeros((131072, 3072)))

    def test_1d_rejected(self):
        assert ernie_image_pe_should_quantize("norm.weight", mx.zeros((512,))) is False

    def test_norm_rejected(self):
        assert (
            ernie_image_pe_should_quantize("layers.0.input_layernorm.weight", mx.zeros((512, 512)))
            is False
        )

    def test_bias_rejected(self):
        assert (
            ernie_image_pe_should_quantize("layers.0.self_attn.q_proj.bias", mx.zeros((512, 512)))
            is False
        )

    def test_small_matrix_rejected(self):
        assert ernie_image_pe_should_quantize("layers.0.q.weight", mx.zeros((128, 512))) is False
        assert ernie_image_pe_should_quantize("layers.0.q.weight", mx.zeros((512, 128))) is False

    def test_threshold_is_inclusive(self):
        assert ernie_image_pe_should_quantize("layers.0.q.weight", mx.zeros((256, 256)))


class TestDefaultOutputDir:
    def test_unquantized(self):
        assert _default_output_dir(False, 8) == Path("models/ernie-image-pe-mlx")

    def test_quantized_encodes_bits(self):
        assert _default_output_dir(True, 4) == Path("models/ernie-image-pe-mlx-q4")
        assert _default_output_dir(True, 8) == Path("models/ernie-image-pe-mlx-q8")

    def test_bits_ignored_when_not_quantizing(self):
        assert _default_output_dir(False, 4) == Path("models/ernie-image-pe-mlx")


class TestRecipeDefaults:
    def test_default_bits_is_four(self):
        """PE is the one recipe defaulting to int4 — pinned so it can't drift."""
        import argparse

        from mlx_forge.recipes.ernie_image_pe import add_convert_args

        parser = argparse.ArgumentParser()
        add_convert_args(parser)
        assert parser.parse_args([]).bits == 4
