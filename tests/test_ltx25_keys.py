"""LTX-2.5 key classification and sanitisation, against real upstream names."""

import json
from functools import partial
from pathlib import Path

import pytest

from mlx_forge.recipes.ltx_25 import (
    classify_audio_key,
    classify_video_vae_key,
    sanitize_audio_vae_key,
    sanitize_vae_decoder_key,
    sanitize_vae_encoder_key,
    sanitize_vocoder_key,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ltx_25_keys.json"

DIT = "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
TEXT_ENCODER = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VAE_AV = "vae/ltx-2.5-video-vae-bf16.safetensors"
VAE_CONV = "vae/ltx-2.5-video-vae-conv-bf16.safetensors"
AUDIO = "vae/ltx-2.5-audio-vae-bf16.safetensors"
DURATION_HEAD = "model_patches/ltx-2.5-duration-head-bf16.safetensors"


@pytest.fixture(scope="session")
def ltx25_keys() -> dict:
    """Real upstream keys, harvested by scripts/harvest_ltx25_keys.py."""
    with open(FIXTURE) as handle:
        return json.load(handle)


class TestFixture:
    def test_covers_every_upstream_file(self, ltx25_keys):
        assert set(ltx25_keys) >= {DIT, TEXT_ENCODER, VAE_AV, VAE_CONV, AUDIO, DURATION_HEAD}

    def test_tensor_counts_match_upstream(self, ltx25_keys):
        # If upstream republishes with different counts, every downstream
        # assumption in this recipe needs rechecking — fail loudly, here.
        assert ltx25_keys[DIT]["tensor_count"] == 4349
        assert ltx25_keys[TEXT_ENCODER]["tensor_count"] == 686
        assert ltx25_keys[VAE_AV]["tensor_count"] == 396
        assert ltx25_keys[VAE_CONV]["tensor_count"] == 170
        assert ltx25_keys[AUDIO]["tensor_count"] == 1329
        assert ltx25_keys[DURATION_HEAD]["tensor_count"] == 15

    def test_the_two_video_vaes_collide_on_prefix(self, ltx25_keys):
        # This is why classification is per-file and not global. If it ever
        # stops being true, the architecture of this recipe can be simplified.
        av = {k.split(".")[0] for k in ltx25_keys[VAE_AV]["keys"]}
        conv = {k.split(".")[0] for k in ltx25_keys[VAE_CONV]["keys"]}
        assert av == conv == {"decoder", "encoder", "per_channel_statistics"}

    def test_checkpoints_carry_the_licence(self, ltx25_keys):
        assert "license" in ltx25_keys[VAE_CONV]["metadata_keys"]
        assert "config" in ltx25_keys[DIT]["metadata_keys"]


class TestVideoVaeClassification:
    def test_every_conv_key_is_classified(self, ltx25_keys):
        classify = partial(classify_video_vae_key, suffix="_conv")
        unclassified = [k for k in ltx25_keys[VAE_CONV]["keys"] if classify(k) is None]
        assert unclassified == []

    def test_components_carry_the_suffix(self, ltx25_keys):
        classify = partial(classify_video_vae_key, suffix="_conv")
        assert classify("decoder.conv_in.conv.weight") == "vae_decoder_conv"
        assert classify("encoder.conv_in.conv.weight") == "vae_encoder_conv"

    def test_the_av_file_uses_the_other_suffix(self, ltx25_keys):
        classify = partial(classify_video_vae_key, suffix="_av")
        assert classify("decoder.conv_in.weight") == "vae_decoder_av"
        unclassified = [k for k in ltx25_keys[VAE_AV]["keys"] if classify(k) is None]
        assert unclassified == []

    def test_stats_go_to_both_halves(self, ltx25_keys):
        # 2.3 ships no vae_shared_stats file: the statistics are folded into
        # the encoder and the decoder under different names. Same here.
        key = "per_channel_statistics.mean-of-means"
        assert sanitize_vae_decoder_key(key) == "per_channel_statistics.mean"
        assert sanitize_vae_encoder_key(key) == "per_channel_statistics._mean_of_means"


class TestAudioClassification:
    def test_splits_one_file_into_two_components(self, ltx25_keys):
        assert classify_audio_key("audio_vae.decoder.conv_in.conv.weight") == "audio_vae"
        assert classify_audio_key("vocoder.ups.0.weight") == "vocoder"
        unclassified = [k for k in ltx25_keys[AUDIO]["keys"] if classify_audio_key(k) is None]
        assert unclassified == []

    def test_sanitizers_strip_their_own_prefix_only(self):
        assert sanitize_audio_vae_key("audio_vae.decoder.conv_in.conv.bias") == (
            "decoder.conv_in.conv.bias"
        )
        assert sanitize_vocoder_key("vocoder.ups.0.weight") == "ups.0.weight"
        assert sanitize_vocoder_key("audio_vae.decoder.conv_in.conv.bias") is None
