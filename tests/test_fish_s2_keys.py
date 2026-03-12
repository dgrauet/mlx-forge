"""Tests for Fish S2 Pro key classification, sanitization, and quantization predicate."""

import mlx.core as mx

from mlx_forge.recipes.fish_s2 import (
    classify_key,
    codec_transform,
    fish_s2_should_quantize,
    sanitize_audio_decoder_key,
    sanitize_codec_key,
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

    def test_codec_encoder(self):
        assert classify_key("generator.encoder.block.0.conv.weight_g") == "codec"

    def test_codec_decoder(self):
        assert classify_key("generator.decoder.model.0.conv.weight_v") == "codec"

    def test_codec_quantizer(self):
        assert classify_key("generator.quantizer.semantic_quantizer.codebook.weight") == "codec"

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


class TestSanitizeCodecKey:
    def test_strips_generator_prefix(self):
        key = "generator.encoder.block.0.conv.weight_g"
        assert sanitize_codec_key(key) == "encoder.block.0.conv.weight_g"

    def test_strips_decoder(self):
        key = "generator.decoder.model.1.block.1.conv.weight_v"
        assert sanitize_codec_key(key) == "decoder.model.1.block.1.conv.weight_v"

    def test_strips_quantizer(self):
        key = "generator.quantizer.semantic_quantizer.codebook.weight"
        assert sanitize_codec_key(key) == "quantizer.semantic_quantizer.codebook.weight"

    def test_strips_only_first_generator(self):
        """Ensure only the leading generator. is stripped, not nested ones."""
        key = "generator.encoder.generator.block.0.weight"
        assert sanitize_codec_key(key) == "encoder.generator.block.0.weight"

    def test_strips_upsample(self):
        key = "generator.quantizer.upsample.0.0.conv.weight"
        assert sanitize_codec_key(key) == "quantizer.upsample.0.0.conv.weight"

    def test_strips_downsample(self):
        key = "generator.quantizer.downsample.0.0.conv.weight"
        assert sanitize_codec_key(key) == "quantizer.downsample.0.0.conv.weight"


class TestCodecTransform:
    def test_conv1d_transposed(self):
        """Conv1d: PyTorch (O, I, K) -> MLX (O, K, I)."""
        w = mx.zeros((64, 32, 7))  # (O=64, I=32, K=7)
        result = codec_transform("encoder.block.0.conv.weight", w, "codec")
        assert result.shape == (64, 7, 32)  # (O=64, K=7, I=32)

    def test_conv_transpose1d_transposed(self):
        """ConvTranspose1d in upsample: PyTorch (I, O, K) -> MLX (O, K, I)."""
        w = mx.zeros((32, 64, 7))  # (I=32, O=64, K=7)
        result = codec_transform("quantizer.upsample.0.0.conv.weight", w, "codec")
        assert result.shape == (64, 7, 32)  # (O=64, K=7, I=32)

    def test_2d_weight_unchanged(self):
        """Linear weights (2D) should pass through unchanged."""
        w = mx.zeros((256, 128))
        key = "encoder.block.1.block.5.layers.0.attention.wqkv.weight"
        result = codec_transform(key, w, "codec")
        assert result.shape == (256, 128)

    def test_1d_weight_unchanged(self):
        """Bias or norm weights (1D) should pass through unchanged."""
        w = mx.zeros((64,))
        result = codec_transform("encoder.block.1.block.3.bias", w, "codec")
        assert result.shape == (64,)

    def test_non_upsample_conv_not_transposed_as_conv_transpose(self):
        """Conv1d not in upsample should use regular transpose, not ConvTranspose."""
        w = mx.zeros((64, 32, 7))  # (O=64, I=32, K=7)
        result = codec_transform("encoder.block.0.conv.weight", w, "codec")
        # Regular Conv1d: (O, I, K) -> (O, K, I)
        assert result.shape == (64, 7, 32)

    def test_decoder_upsample_conv(self):
        """Decoder model upsampling ConvTranspose1d."""
        w = mx.zeros((128, 64, 16))  # (I=128, O=64, K=16)
        result = codec_transform("decoder.model.1.block.0.upsample.conv.weight", w, "codec")
        # In upsample path: ConvTranspose1d (I, O, K) -> (O, K, I)
        assert result.shape == (64, 16, 128)


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

    def test_conv1d_weight_not_quantized(self):
        """Conv1d weights are 3D (after transpose) and should not be quantized."""
        assert not fish_s2_should_quantize("encoder.block.0.conv.weight", mx.zeros((64, 7, 32)))
