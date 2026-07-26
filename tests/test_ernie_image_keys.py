"""Tests for ERNIE-Image key sanitization, weight transforms and quantization scope."""

import mlx.core as mx
import pytest

from mlx_forge.recipes.ernie_image import (
    _filter_text_encoder_keys,
    _sanitize_identity,
    _sanitize_transformer_key,
    _transformer_transform,
    _vae_transform,
    ernie_image_should_quantize,
)


class TestSanitizeTransformerKey:
    def test_timestep_embedding_linear_renamed(self):
        assert _sanitize_transformer_key("time_embedding.linear_1.weight") == (
            "time_embedding.linear1.weight"
        )
        assert _sanitize_transformer_key("time_embedding.linear_2.bias") == (
            "time_embedding.linear2.bias"
        )

    def test_adaln_sequential_index_becomes_named_linear(self):
        assert _sanitize_transformer_key("blocks.0.adaLN_modulation.1.weight") == (
            "blocks.0.adaLN_modulation.linear.weight"
        )

    def test_attention_output_modulelist_flattened(self):
        assert _sanitize_transformer_key("blocks.3.self_attention.to_out.0.weight") == (
            "blocks.3.self_attention.to_out_0.weight"
        )

    def test_untouched_key_passes_through(self):
        key = "blocks.0.self_attention.to_q.weight"
        assert _sanitize_transformer_key(key) == key

    def test_adaln_index_zero_not_rewritten(self):
        """Only the [1] entry of the Sequential is the Linear."""
        key = "blocks.0.adaLN_modulation.0.weight"
        assert _sanitize_transformer_key(key) == key

    def test_identity_sanitizer_is_a_no_op(self):
        assert _sanitize_identity("encoder.conv_in.weight") == "encoder.conv_in.weight"


class TestTransformerTransform:
    def test_patch_embed_conv_squeezed_to_linear(self):
        w = mx.zeros((128, 16, 1, 1))
        out = _transformer_transform("x_embedder.proj.weight", w, "transformer")
        assert out.shape == (128, 16)

    def test_non_pointwise_patch_embed_left_alone(self):
        """A real 2x2 kernel must not be squeezed — that would corrupt it silently."""
        w = mx.zeros((128, 16, 2, 2))
        out = _transformer_transform("x_embedder.proj.weight", w, "transformer")
        assert out.shape == (128, 16, 2, 2)

    def test_other_keys_untouched(self):
        w = mx.zeros((128, 128))
        assert _transformer_transform("blocks.0.to_q.weight", w, "transformer").shape == (128, 128)


class TestVaeTransform:
    def test_conv2d_transposed(self):
        w = mx.zeros((8, 4, 3, 3))
        assert _vae_transform("encoder.conv_in.weight", w, "vae").shape == (8, 3, 3, 4)

    def test_linear_untouched(self):
        w = mx.zeros((8, 4))
        assert _vae_transform("mid.attn.to_q.weight", w, "vae").shape == (8, 4)

    def test_bias_untouched(self):
        w = mx.zeros((8,))
        assert _vae_transform("encoder.conv_in.bias", w, "vae").shape == (8,)


class TestTextEncoderFilter:
    def test_vision_tower_dropped(self):
        weights = {
            "vision_tower.transformer.layers.0.attention.q_proj.weight": 1,
            "multi_modal_projector.linear_1.weight": 2,
            "layers.0.self_attn.q_proj.weight": 3,
            "embed_tokens.weight": 4,
        }
        kept = _filter_text_encoder_keys(weights)
        assert set(kept) == {"layers.0.self_attn.q_proj.weight", "embed_tokens.weight"}

    def test_empty_input(self):
        assert _filter_text_encoder_keys({}) == {}

    def test_substring_match_does_not_drop_a_language_key(self):
        """The filter is prefix-based: a key merely containing the word stays."""
        weights = {"layers.0.vision_tower_proj.weight": 1}
        assert set(_filter_text_encoder_keys(weights)) == {"layers.0.vision_tower_proj.weight"}


class TestQuantizationScope:
    def test_block_linear_quantized(self):
        assert ernie_image_should_quantize("blocks.0.to_q.weight", mx.zeros((512, 512))) is True

    def test_1d_rejected(self):
        assert ernie_image_should_quantize("blocks.0.norm.weight", mx.zeros((512,))) is False

    def test_bias_rejected(self):
        assert ernie_image_should_quantize("blocks.0.to_q.bias", mx.zeros((512, 512))) is False

    @pytest.mark.parametrize(
        "key",
        [
            "x_embedder.proj.weight",
            "text_proj.weight",
            "time_embedding.linear1.weight",
            "blocks.0.adaLN_modulation.linear.weight",
            "final_norm.weight",
            "final_linear.weight",
            "pos_embed.weight",
        ],
    )
    def test_sensitive_projections_excluded(self, key):
        assert ernie_image_should_quantize(key, mx.zeros((512, 512))) is False

    def test_small_matrix_excluded(self):
        assert ernie_image_should_quantize("blocks.0.to_q.weight", mx.zeros((128, 512))) is False
        assert ernie_image_should_quantize("blocks.0.to_q.weight", mx.zeros((512, 128))) is False

    def test_threshold_is_inclusive(self):
        assert ernie_image_should_quantize("blocks.0.to_q.weight", mx.zeros((256, 256))) is True
