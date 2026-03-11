"""Tests for LTX-2.3 key classification and sanitization."""

from mlx_forge.recipes.ltx23 import (
    classify_key,
    sanitize_audio_vae_key,
    sanitize_connector_key,
    sanitize_transformer_key,
    sanitize_vae_decoder_key,
    sanitize_vae_encoder_key,
    sanitize_vocoder_key,
)


class TestClassifyKey:
    def test_transformer(self):
        key = "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight"
        assert classify_key(key) == "transformer"

    def test_connector_video(self):
        key = "model.diffusion_model.video_embeddings_connector.weight"
        assert classify_key(key) == "connector"

    def test_connector_audio(self):
        key = "model.diffusion_model.audio_embeddings_connector.weight"
        assert classify_key(key) == "connector"

    def test_text_projection(self):
        assert classify_key("text_embedding_projection.aggregate_embed.weight") == "connector"

    def test_vae_decoder(self):
        assert classify_key("vae.decoder.conv_in.weight") == "vae_decoder"

    def test_vae_encoder(self):
        assert classify_key("vae.encoder.conv_in.weight") == "vae_encoder"

    def test_vae_shared_stats(self):
        assert classify_key("vae.per_channel_statistics.mean-of-means") == "vae_shared_stats"

    def test_audio_vae(self):
        assert classify_key("audio_vae.decoder.conv_in.weight") == "audio_vae"

    def test_vocoder(self):
        assert classify_key("vocoder.ups.0.weight") == "vocoder"

    def test_unknown(self):
        assert classify_key("some.random.key") is None


class TestSanitizeTransformerKey:
    def test_removes_prefix(self):
        key = "model.diffusion_model.adaln_single.linear.weight"
        assert sanitize_transformer_key(key) == "adaln_single.linear.weight"

    def test_to_out(self):
        key = "model.diffusion_model.transformer_blocks.0.attn1.to_out.0.weight"
        assert ".to_out.weight" in sanitize_transformer_key(key)

    def test_ff_net(self):
        key = "model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight"
        assert ".ff.proj_in.weight" in sanitize_transformer_key(key)

    def test_ff_out(self):
        key = "model.diffusion_model.transformer_blocks.0.ff.net.2.weight"
        assert ".ff.proj_out.weight" in sanitize_transformer_key(key)

    def test_adaln_linear(self):
        key = "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_1.weight"
        result = sanitize_transformer_key(key)
        assert ".linear1." in result


class TestSanitizeVaeDecoderKey:
    def test_decoder_prefix(self):
        assert sanitize_vae_decoder_key("vae.decoder.conv_in.weight") == "conv_in.weight"

    def test_stats_mean(self):
        result = sanitize_vae_decoder_key("vae.per_channel_statistics.mean-of-means")
        assert result == "per_channel_statistics.mean"

    def test_stats_std(self):
        result = sanitize_vae_decoder_key("vae.per_channel_statistics.std-of-means")
        assert result == "per_channel_statistics.std"

    def test_unknown(self):
        assert sanitize_vae_decoder_key("other.key") is None


class TestSanitizeVaeEncoderKey:
    def test_encoder_prefix(self):
        assert sanitize_vae_encoder_key("vae.encoder.conv_in.weight") == "conv_in.weight"

    def test_stats_mean(self):
        result = sanitize_vae_encoder_key("vae.per_channel_statistics.mean-of-means")
        assert result == "per_channel_statistics._mean_of_means"


class TestSanitizeAudioVaeKey:
    def test_decoder_prefix(self):
        assert sanitize_audio_vae_key("audio_vae.decoder.conv_in.weight") == "conv_in.weight"

    def test_stats(self):
        result = sanitize_audio_vae_key("audio_vae.per_channel_statistics.mean-of-means")
        assert result == "per_channel_statistics._mean_of_means"

    def test_unknown(self):
        assert sanitize_audio_vae_key("other.key") is None


class TestSanitizeVocoderKey:
    def test_prefix(self):
        assert sanitize_vocoder_key("vocoder.ups.0.weight") == "ups.0.weight"

    def test_unknown(self):
        assert sanitize_vocoder_key("other.key") is None


class TestSanitizeConnectorKey:
    def test_diffusion_model_prefix(self):
        key = "model.diffusion_model.video_embeddings_connector.weight"
        assert sanitize_connector_key(key) == "video_embeddings_connector.weight"

    def test_text_projection(self):
        key = "text_embedding_projection.aggregate_embed.weight"
        assert sanitize_connector_key(key) == key
