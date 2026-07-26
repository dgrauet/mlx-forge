"""Tests for Ideogram-4 key sanitization, FP8 dequantization and quantization scope.

The recipe converts an FP8 checkpoint: weights arrive as raw uint8 bits paired
with a per-row float32 scale, and the text encoder ships a vision tower that
must not reach the output.
"""

import mlx.core as mx
import pytest

from mlx_forge.recipes.ideogram_4 import (
    _dequantize_fp8,
    _should_quantize_text_encoder,
    _should_quantize_transformer,
    maybe_transpose,
    sanitize_text_encoder_key,
    sanitize_transformer_key,
    sanitize_vae_key,
)


class TestSanitizeTransformerKey:
    @pytest.mark.parametrize(
        "key",
        [
            "blocks.0.attn.to_q.weight",
            "blocks.23.ff.net.0.proj.weight",
            "norm_out.linear.weight",
            "embed_image_indicator.weight",
        ],
    )
    def test_keys_pass_through_unchanged(self, key):
        assert sanitize_transformer_key(key) == key


class TestSanitizeTextEncoderKey:
    def test_strips_language_model_prefix(self):
        assert sanitize_text_encoder_key("language_model.layers.0.mlp.up_proj.weight") == (
            "layers.0.mlp.up_proj.weight"
        )

    def test_keeps_embed_tokens(self):
        assert sanitize_text_encoder_key("language_model.embed_tokens.weight") == (
            "embed_tokens.weight"
        )

    def test_keeps_final_norm(self):
        assert sanitize_text_encoder_key("language_model.norm.weight") == "norm.weight"

    @pytest.mark.parametrize(
        "key",
        [
            "vision_model.encoder.layers.0.self_attn.q_proj.weight",
            "vision_tower.embeddings.patch_embedding.weight",
            "multi_modal_projector.linear_1.weight",
        ],
    )
    def test_drops_non_language_model_towers(self, key):
        assert sanitize_text_encoder_key(key) is None

    def test_drops_lm_head(self):
        """lm_head sits under language_model. but is not one of the kept subtrees."""
        assert sanitize_text_encoder_key("language_model.lm_head.weight") is None


class TestSanitizeVaeKey:
    def test_drops_batchnorm_counter(self):
        assert sanitize_vae_key("encoder.down.0.norm.num_batches_tracked") is None

    def test_unwraps_attention_output_sequential(self):
        assert sanitize_vae_key("mid.attn_1.to_out.0.weight") == "mid.attn_1.to_out.weight"
        assert sanitize_vae_key("mid.attn_1.to_out.0.bias") == "mid.attn_1.to_out.bias"

    def test_other_keys_unchanged(self):
        assert sanitize_vae_key("encoder.conv_in.weight") == "encoder.conv_in.weight"

    def test_does_not_touch_unrelated_zero_index(self):
        key = "encoder.down.0.block.0.conv1.weight"
        assert sanitize_vae_key(key) == key


class TestMaybeTranspose:
    def test_vae_conv2d_is_transposed(self):
        w = mx.zeros((8, 4, 3, 3))  # (O, I, H, W)
        assert maybe_transpose("encoder.conv_in.weight", w, "vae").shape == (8, 3, 3, 4)

    def test_vae_linear_untouched(self):
        w = mx.zeros((8, 4))
        assert maybe_transpose("mid.attn_1.to_q.weight", w, "vae").shape == (8, 4)

    def test_vae_bias_untouched(self):
        w = mx.zeros((8,))
        assert maybe_transpose("encoder.conv_in.bias", w, "vae").shape == (8,)

    def test_transformer_conv_shaped_weight_untouched(self):
        """Only the VAE has convs — a 4D transformer tensor must not be permuted."""
        w = mx.zeros((8, 4, 3, 3))
        assert maybe_transpose("blocks.0.x.weight", w, "conditional_transformer").shape == (
            8,
            4,
            3,
            3,
        )


class TestQuantizationScope:
    def test_transformer_linear_quantized(self):
        assert _should_quantize_transformer("blocks.0.attn.to_q.weight", mx.zeros((64, 64))) is True

    def test_transformer_norm_excluded_by_ndim(self):
        assert _should_quantize_transformer("norm_out.weight", mx.zeros((64,))) is False

    def test_transformer_bias_excluded(self):
        assert _should_quantize_transformer("blocks.0.attn.to_q.bias", mx.zeros((64, 64))) is False

    def test_image_indicator_embedding_excluded(self):
        assert (
            _should_quantize_transformer("embed_image_indicator.weight", mx.zeros((2, 64))) is False
        )

    def test_text_encoder_linear_quantized(self):
        assert _should_quantize_text_encoder("layers.0.mlp.up_proj.weight", mx.zeros((64, 64)))

    def test_text_encoder_embedding_excluded(self):
        assert _should_quantize_text_encoder("embed_tokens.weight", mx.zeros((1000, 64))) is False


class TestDequantizeFp8:
    def test_weight_and_scale_merge_into_one_key(self):
        raw = {
            "layers.0.q.weight": mx.zeros((4, 8), dtype=mx.uint8),
            "layers.0.q.weight_scale": mx.ones((4,)),
        }
        out = _dequantize_fp8(raw)
        assert set(out) == {"layers.0.q.weight"}, "the scale must be consumed, not emitted"
        assert out["layers.0.q.weight"].dtype == mx.bfloat16

    def test_scale_is_applied_per_row(self):
        # fp8_e4m3 bit pattern 0x38 == 1.0
        raw = {
            "w.weight": mx.full((2, 4), 0x38, dtype=mx.uint8),
            "w.weight_scale": mx.array([2.0, 3.0]),
        }
        out = _dequantize_fp8(raw)["w.weight"]
        assert out[0, 0].item() == pytest.approx(2.0)
        assert out[1, 0].item() == pytest.approx(3.0)

    def test_uint8_weight_without_scale_passes_through(self):
        raw = {"w.weight": mx.zeros((2, 2), dtype=mx.uint8)}
        assert _dequantize_fp8(raw)["w.weight"].dtype == mx.uint8

    def test_non_fp8_tensors_untouched(self):
        raw = {"norm.weight": mx.ones((4,), dtype=mx.float32)}
        out = _dequantize_fp8(raw)
        assert out["norm.weight"].dtype == mx.float32
