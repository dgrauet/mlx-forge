"""Tests for Mistral Small 3.1 key classification, sanitization, and quantization predicate."""

import mlx.core as mx

from mlx_forge.recipes.mistral_small import (
    classify_key,
    mistral_should_quantize,
    sanitize_language_model_key,
    sanitize_multimodal_projector_key,
    sanitize_vision_tower_key,
)


class TestClassifyKey:
    def test_language_model_layer(self):
        assert (
            classify_key("language_model.model.layers.0.self_attn.q_proj.weight")
            == "language_model"
        )

    def test_language_model_embedding(self):
        assert classify_key("language_model.model.embed_tokens.weight") == "language_model"

    def test_language_model_norm(self):
        assert classify_key("language_model.model.norm.weight") == "language_model"

    def test_language_model_lm_head(self):
        assert classify_key("language_model.lm_head.weight") == "language_model"

    def test_vision_tower(self):
        assert (
            classify_key("vision_tower.transformer.layers.0.attention.wq.weight") == "vision_tower"
        )

    def test_vision_tower_deep(self):
        assert (
            classify_key("vision_tower.transformer.layers.23.feed_forward.w1.weight")
            == "vision_tower"
        )

    def test_multimodal_projector(self):
        assert classify_key("multimodal_projector.linear_1.weight") == "multimodal_projector"

    def test_multimodal_projector_bias(self):
        assert classify_key("multimodal_projector.linear_1.bias") == "multimodal_projector"

    def test_unknown(self):
        assert classify_key("some_other_module.weight") is None

    def test_empty_key(self):
        assert classify_key("weight") is None


class TestSanitizeLanguageModelKey:
    def test_strips_model_prefix(self):
        key = "language_model.model.layers.0.self_attn.q_proj.weight"
        assert sanitize_language_model_key(key) == "layers.0.self_attn.q_proj.weight"

    def test_strips_embedding(self):
        key = "language_model.model.embed_tokens.weight"
        assert sanitize_language_model_key(key) == "embed_tokens.weight"

    def test_strips_norm(self):
        key = "language_model.model.norm.weight"
        assert sanitize_language_model_key(key) == "norm.weight"

    def test_lm_head_special_case(self):
        key = "language_model.lm_head.weight"
        assert sanitize_language_model_key(key) == "lm_head.weight"

    def test_ffn_keys(self):
        key = "language_model.model.layers.5.mlp.gate_proj.weight"
        assert sanitize_language_model_key(key) == "layers.5.mlp.gate_proj.weight"


class TestSanitizeVisionTowerKey:
    def test_strips_prefix(self):
        key = "vision_tower.transformer.layers.0.attention.wq.weight"
        assert sanitize_vision_tower_key(key) == "transformer.layers.0.attention.wq.weight"

    def test_strips_feed_forward(self):
        key = "vision_tower.transformer.layers.10.feed_forward.w1.weight"
        assert sanitize_vision_tower_key(key) == "transformer.layers.10.feed_forward.w1.weight"

    def test_strips_norm(self):
        key = "vision_tower.transformer.layers.0.attention_norm.weight"
        assert sanitize_vision_tower_key(key) == "transformer.layers.0.attention_norm.weight"


class TestSanitizeMultimodalProjectorKey:
    def test_strips_prefix(self):
        key = "multimodal_projector.linear_1.weight"
        assert sanitize_multimodal_projector_key(key) == "linear_1.weight"

    def test_strips_bias(self):
        key = "multimodal_projector.linear_1.bias"
        assert sanitize_multimodal_projector_key(key) == "linear_1.bias"

    def test_strips_linear_2(self):
        key = "multimodal_projector.linear_2.weight"
        assert sanitize_multimodal_projector_key(key) == "linear_2.weight"


class TestShouldQuantize:
    def test_linear_weight_quantized(self):
        assert mistral_should_quantize("layers.0.self_attn.q_proj.weight", mx.zeros((5120, 5120)))

    def test_ffn_gate_proj_quantized(self):
        assert mistral_should_quantize("layers.0.mlp.gate_proj.weight", mx.zeros((32768, 5120)))

    def test_ffn_down_proj_quantized(self):
        assert mistral_should_quantize("layers.0.mlp.down_proj.weight", mx.zeros((5120, 32768)))

    def test_ffn_up_proj_quantized(self):
        assert mistral_should_quantize("layers.0.mlp.up_proj.weight", mx.zeros((32768, 5120)))

    def test_kv_proj_quantized(self):
        assert mistral_should_quantize("layers.0.self_attn.k_proj.weight", mx.zeros((1024, 5120)))

    def test_embedding_not_quantized(self):
        assert not mistral_should_quantize("embed_tokens.weight", mx.zeros((131072, 5120)))

    def test_lm_head_not_quantized(self):
        assert not mistral_should_quantize("lm_head.weight", mx.zeros((131072, 5120)))

    def test_rms_norm_not_quantized(self):
        assert not mistral_should_quantize("layers.0.input_layernorm.weight", mx.zeros((5120,)))

    def test_post_attention_norm_not_quantized(self):
        assert not mistral_should_quantize(
            "layers.0.post_attention_layernorm.weight", mx.zeros((5120,))
        )

    def test_final_norm_not_quantized(self):
        assert not mistral_should_quantize("norm.weight", mx.zeros((5120,)))

    def test_small_tensor_not_quantized(self):
        assert not mistral_should_quantize("layers.0.self_attn.q_proj.weight", mx.zeros((8, 8)))

    def test_bias_not_quantized(self):
        """Bias vectors (1D) should not be quantized."""
        assert not mistral_should_quantize("layers.0.self_attn.q_proj.bias", mx.zeros((5120,)))
