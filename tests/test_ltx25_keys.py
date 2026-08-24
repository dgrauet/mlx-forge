"""LTX-2.5 key classification and sanitisation, against real upstream names."""

import json
from functools import partial
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_forge.recipes.ltx_25 import (
    EXPECTED_TENSOR_COUNTS,
    SOURCE_FILES,
    UPSCALER_FILES,
    _share_video_vae_statistics,
    classify_audio_key,
    classify_dit_key,
    classify_duration_head_key,
    classify_video_vae_key,
    maybe_transpose,
    sanitize_audio_vae_key,
    sanitize_connector_key,
    sanitize_duration_head_key,
    sanitize_transformer_key,
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
        # C1 (see .superpowers/sdd/2026-08-22-ltx-2.5-recipe/final-review.md):
        # the old version of this test only called the two sanitizers and
        # never asked classify_video_vae_key where a stats key actually goes
        # — it asserted a property the code did not have, and survived
        # fifteen task reviews doing it. classify_video_vae_key can only route
        # a key to one component, so it hands per_channel_statistics.* to the
        # decoder alone; _share_video_vae_statistics is what gives the
        # encoder its own copy afterwards. This drives the real pipeline —
        # classification, then that duplication step, then both sanitizers —
        # against real fixture keys, so it fails if either half regresses.
        classify = partial(classify_video_vae_key, suffix="_conv")
        keys_by_component: dict[str, list[str]] = {}
        for key in ltx25_keys[VAE_CONV]["keys"]:
            comp = classify(key)
            if comp:
                keys_by_component.setdefault(comp, []).append(key)

        stats_key = "per_channel_statistics.mean-of-means"
        assert stats_key in keys_by_component["vae_decoder_conv"]
        assert stats_key not in keys_by_component.get("vae_encoder_conv", [])

        _share_video_vae_statistics(keys_by_component, ("vae_encoder_conv", "vae_decoder_conv"))

        # Duplicated into the encoder, not moved out of the decoder.
        assert stats_key in keys_by_component["vae_encoder_conv"]
        assert stats_key in keys_by_component["vae_decoder_conv"]

        assert sanitize_vae_decoder_key(stats_key) == "per_channel_statistics.mean"
        assert sanitize_vae_encoder_key(stats_key) == "per_channel_statistics._mean_of_means"

    def test_share_video_vae_statistics_is_a_noop_off_video_vae_components(self):
        # Any other source's components — e.g. the DiT's ("transformer",
        # "connector") — must pass through untouched: neither name starts
        # with "vae_encoder"/"vae_decoder", so there is nothing to duplicate.
        keys_by_component = {"transformer": ["a.weight"], "connector": ["b.weight"]}
        before = {k: list(v) for k, v in keys_by_component.items()}
        _share_video_vae_statistics(keys_by_component, ("transformer", "connector"))
        assert keys_by_component == before


class TestExpectedTensorCounts:
    """EXPECTED_TENSOR_COUNTS reconciled against the harvested upstream headers.

    The validate() test suite (tests/test_ltx25_sources.py) builds its fixture
    packs *from* EXPECTED_TENSOR_COUNTS, so those tests can never catch the
    constants disagreeing with what convert() actually writes — that is
    exactly how C1a survived. This test is the other half: it checks the
    constants against ltx_25_keys.json's independently harvested
    `tensor_count`, which comes from upstream's own safetensors headers, not
    from this recipe's code.
    """

    def test_video_vae_counts_reconcile_with_the_fixture(self, ltx25_keys):
        # _share_video_vae_statistics duplicates the 2 per_channel_statistics
        # tensors into the encoder, so each pair is written with 2 more
        # tensors, combined, than the upstream file actually carries.
        for source in SOURCE_FILES:
            if source.path not in (VAE_CONV, VAE_AV):
                continue
            total = sum(EXPECTED_TENSOR_COUNTS[c] for c in source.components)
            assert total == ltx25_keys[source.path]["tensor_count"] + 2, source.path


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
        # The main generator, itself named "vocoder", flattens to the root
        # (the published 2.3 layout); its siblings keep their one level.
        assert sanitize_vocoder_key("vocoder.vocoder.act_post.act.alpha") == "act_post.act.alpha"
        assert (
            sanitize_vocoder_key("vocoder.bwe_generator.act_post.act.alpha")
            == "bwe_generator.act_post.act.alpha"
        )
        assert sanitize_vocoder_key("vocoder.mel_stft.mel_basis") == "mel_stft.mel_basis"
        assert sanitize_vocoder_key("audio_vae.decoder.conv_in.conv.bias") is None


class TestDitClassification:
    def test_every_dit_key_is_classified(self, ltx25_keys):
        unclassified = [k for k in ltx25_keys[DIT]["keys"] if classify_dit_key(k) is None]
        assert unclassified == []

    def test_connectors_are_separated_from_the_transformer(self):
        assert (
            classify_dit_key("model.diffusion_model.video_embeddings_connector.0.weight")
            == "connector"
        )
        assert (
            classify_dit_key("model.diffusion_model.audio_embeddings_connector.0.weight")
            == "connector"
        )
        assert (
            classify_dit_key("model.diffusion_model.transformer_blocks.0.attn1.to_q.weight")
            == "transformer"
        )

    def test_per_block_prompt_tables_stay_in_the_transformer(self, ltx25_keys):
        # 366 keys contain "prompt" or "connector" as a substring; only the two
        # *_embeddings_connector stacks are actually the connector.
        key = "model.diffusion_model.transformer_blocks.0.prompt_scale_shift_table"
        assert key in ltx25_keys[DIT]["keys"]
        assert classify_dit_key(key) == "transformer"

    def test_text_projection_is_not_the_dit_s_business(self):
        # In 2.5 it lives in the text encoder file, unlike 2.3.
        assert classify_dit_key("text_embedding_projection.video_aggregate_embed.weight") is None


class TestDitSanitisation:
    def test_no_pytorch_idiom_survives(self, ltx25_keys):
        for key in ltx25_keys[DIT]["keys"]:
            if classify_dit_key(key) != "transformer":
                continue
            out = sanitize_transformer_key(key)
            assert not out.startswith("model.diffusion_model."), key
            assert ".net." not in out, key
            assert ".to_out.0." not in out, key
            assert ".linear_1." not in out and ".linear_2." not in out, key

    def test_sanitisation_is_injective_over_the_real_keys(self, ltx25_keys):
        # Two upstream keys collapsing to one output silently drops a tensor.
        transformer = [k for k in ltx25_keys[DIT]["keys"] if classify_dit_key(k) == "transformer"]
        assert len({sanitize_transformer_key(k) for k in transformer}) == len(transformer)

    def test_known_renames(self):
        base = "model.diffusion_model.transformer_blocks.0"
        assert sanitize_transformer_key(f"{base}.attn1.to_out.0.weight") == (
            "transformer_blocks.0.attn1.to_out.weight"
        )
        assert sanitize_transformer_key(f"{base}.ff.net.0.proj.weight") == (
            "transformer_blocks.0.ff.proj_in.weight"
        )
        assert sanitize_transformer_key(f"{base}.ff.net.2.weight") == (
            "transformer_blocks.0.ff.proj_out.weight"
        )

    def test_connector_only_loses_the_container_prefix(self):
        assert (
            sanitize_connector_key(
                "model.diffusion_model.video_embeddings_connector.0.attn.to_q.weight"
            )
            == "video_embeddings_connector.0.attn.to_q.weight"
        )


class TestTransposition:
    def test_the_transformer_is_never_transposed(self):
        w = mx.zeros((4, 8))
        assert maybe_transpose(
            "transformer_blocks.0.attn1.to_q.weight", w, "transformer"
        ).shape == (
            4,
            8,
        )

    def test_a_vae_conv_weight_goes_channels_last(self):
        w = mx.zeros((16, 8, 3, 3, 3))  # (O, I, D, H, W)
        out = maybe_transpose("decoder.conv_in.conv.weight", w, "vae_decoder_conv")
        assert out.shape == (16, 3, 3, 3, 8)

    def test_a_vocoder_upsample_is_a_conv_transpose(self):
        w = mx.zeros((8, 16, 4))  # (I, O, K)
        out = maybe_transpose("ups.0.weight", w, "vocoder")
        assert out.shape == (16, 4, 8)


class TestDurationHead:
    def test_every_key_is_classified(self, ltx25_keys):
        keys = ltx25_keys[DURATION_HEAD]["keys"]
        assert keys, "fixture must carry the duration head"
        assert [k for k in keys if classify_duration_head_key(k) is None] == []

    def test_the_prefix_is_stripped(self):
        assert (
            sanitize_duration_head_key("duration_head.attention_pooler.cross_attn.out_proj.weight")
            == "attention_pooler.cross_attn.out_proj.weight"
        )

    def test_a_foreign_key_is_skipped(self):
        assert sanitize_duration_head_key("vocoder.ups.0.weight") is None


class TestUpscalers:
    def test_names_carry_scale_and_version(self):
        # LTX-2.3's convention: a bare "spatial_upscaler" cannot say which of
        # several published upscalers a file holds.
        assert set(UPSCALER_FILES) == {
            "spatial_upscaler_x2_v1_0",
            "temporal_upscaler_x2_v1_0",
        }

    def test_paths_point_into_the_upstream_folder(self):
        assert all(p.startswith("latent_upscale_models/") for p in UPSCALER_FILES.values())
