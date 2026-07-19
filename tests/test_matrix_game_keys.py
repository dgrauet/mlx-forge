"""Tests for Matrix-Game-3.0 VAE key sanitization.

The LightVAE checkpoints (MG-LightVAE*.pth) use diffusers-style keys that
must map onto WanVAE_ attribute names exactly as the PyTorch reference's
_map_lightvae_key_to_wanvae does — including the literal ``resample``
segment of the Resample module (``self.resample = nn.Sequential(...)``).
"""

from mlx_forge.recipes.matrix_game_3_0 import sanitize_vae_key


class TestLightVaeResampleKeys:
    def test_decoder_upsampler_resample_keeps_resample_segment(self):
        assert (
            sanitize_vae_key("decoder.up_blocks.0.upsampler.resample.1.weight")
            == "decoder.upsamples.0.upsamples.3.resample.1.weight"
        )
        assert (
            sanitize_vae_key("decoder.up_blocks.2.upsampler.resample.1.bias")
            == "decoder.upsamples.2.upsamples.3.resample.1.bias"
        )

    def test_encoder_downsampler_resample_keeps_resample_segment(self):
        assert (
            sanitize_vae_key("encoder.down_blocks.0.downsampler.resample.1.weight")
            == "encoder.downsamples.0.downsamples.2.resample.1.weight"
        )

    def test_resnet_and_time_conv_mappings_unchanged(self):
        assert (
            sanitize_vae_key("decoder.up_blocks.1.resnets.0.conv1.weight")
            == "decoder.upsamples.1.upsamples.0.residual.2.weight"
        )
        assert (
            sanitize_vae_key("decoder.up_blocks.0.upsampler.time_conv.weight")
            == "decoder.upsamples.0.upsamples.3.time_conv.weight"
        )
        assert (
            sanitize_vae_key("encoder.down_blocks.1.downsampler.time_conv.bias")
            == "encoder.downsamples.1.downsamples.2.time_conv.bias"
        )
