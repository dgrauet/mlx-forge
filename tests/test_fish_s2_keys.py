"""Tests for Fish S2 Pro key classification, sanitization, and quantization predicate."""

import mlx.core as mx

from mlx_forge.recipes.fish_s2 import (
    classify_key,
    fish_s2_should_quantize,
    sanitize_audio_decoder_key,
    sanitize_text_model_key,
)


class TestClassifyKey:
    def test_text_model(self):
        assert classify_key("text_model.model.layers.0.attention.wqkv.weight") == "text_model"

    def test_text_model_embedding(self):
        assert classify_key("text_model.model.embeddings.weight") == "text_model"

    def test_audio_decoder(self):
        assert classify_key("audio_decoder.layers.0.attention.wqkv.weight") == "audio_decoder"

    def test_audio_decoder_codebook(self):
        assert classify_key("audio_decoder.codebook_embeddings.weight") == "audio_decoder"

    def test_unknown(self):
        assert classify_key("some_other_module.weight") is None


class TestSanitizeTextModelKey:
    def test_strips_prefix(self):
        key = "text_model.model.layers.0.attention.wqkv.weight"
        assert sanitize_text_model_key(key) == "layers.0.attention.wqkv.weight"

    def test_strips_embedding(self):
        key = "text_model.model.embeddings.weight"
        assert sanitize_text_model_key(key) == "embeddings.weight"

    def test_strips_norm(self):
        key = "text_model.model.norm.weight"
        assert sanitize_text_model_key(key) == "norm.weight"


class TestSanitizeAudioDecoderKey:
    def test_strips_prefix(self):
        key = "audio_decoder.layers.0.feed_forward.w1.weight"
        assert sanitize_audio_decoder_key(key) == "layers.0.feed_forward.w1.weight"

    def test_strips_codebook(self):
        key = "audio_decoder.codebook_embeddings.weight"
        assert sanitize_audio_decoder_key(key) == "codebook_embeddings.weight"

    def test_strips_output(self):
        key = "audio_decoder.output.weight"
        assert sanitize_audio_decoder_key(key) == "output.weight"


class TestShouldQuantize:
    def test_linear_weight_quantized(self):
        assert fish_s2_should_quantize("layers.0.attention.wqkv.weight", mx.zeros((256, 256)))

    def test_ffn_weight_quantized(self):
        assert fish_s2_should_quantize("layers.0.feed_forward.w1.weight", mx.zeros((512, 256)))

    def test_output_weight_quantized(self):
        assert fish_s2_should_quantize("output.weight", mx.zeros((4096, 2560)))

    def test_embedding_not_quantized(self):
        assert not fish_s2_should_quantize("embeddings.weight", mx.zeros((155776, 2560)))

    def test_codebook_embedding_not_quantized(self):
        assert not fish_s2_should_quantize("codebook_embeddings.weight", mx.zeros((10, 4096)))

    def test_norm_not_quantized(self):
        assert not fish_s2_should_quantize("layers.0.attention_norm.weight", mx.zeros((2560,)))

    def test_q_norm_not_quantized(self):
        assert not fish_s2_should_quantize("layers.0.attention.q_norm.weight", mx.zeros((128,)))

    def test_small_tensor_not_quantized(self):
        assert not fish_s2_should_quantize("layers.0.attention.wo.weight", mx.zeros((8, 8)))
