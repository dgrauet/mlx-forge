"""Tests for CogVideoX-Fun key handling, conv transposition and quantization scope.

The pipeline-file copy is covered separately in test_cogvideox_config_copy.py.
"""

import json

import mlx.core as mx
import pytest

from mlx_forge.recipes.cogvideox_fun_v1_5_5b_inp import (
    COMPONENTS,
    _build_config,
    maybe_transpose,
    sanitize_text_encoder_key,
    sanitize_transformer_key,
    sanitize_vae_key,
    should_quantize_transformer,
)


class TestSanitizers:
    @pytest.mark.parametrize(
        "sanitize,key",
        [
            (sanitize_transformer_key, "transformer_blocks.0.attn1.to_q.weight"),
            (sanitize_transformer_key, "patch_embed.proj.weight"),
            (sanitize_text_encoder_key, "encoder.block.0.layer.0.SelfAttention.q.weight"),
            (sanitize_text_encoder_key, "shared.weight"),
            (sanitize_vae_key, "encoder.down_blocks.0.resnets.0.conv1.conv.weight"),
        ],
    )
    def test_keys_pass_through_unchanged(self, sanitize, key):
        assert sanitize(key) == key


class TestMaybeTranspose:
    def test_conv3d_transposed(self):
        w = mx.zeros((16, 32, 3, 3, 3))  # (O, I, D, H, W)
        out = maybe_transpose("encoder.conv_in.weight", w, "vae")
        assert out.shape == (16, 3, 3, 3, 32)

    def test_conv2d_transposed(self):
        w = mx.zeros((16, 32, 3, 3))
        assert maybe_transpose("encoder.conv_in.weight", w, "vae").shape == (16, 3, 3, 32)

    def test_conv1d_transposed(self):
        w = mx.zeros((16, 32, 3))
        assert maybe_transpose("encoder.conv_in.weight", w, "vae").shape == (16, 3, 32)

    def test_linear_untouched(self):
        w = mx.zeros((1920, 1920))
        assert maybe_transpose(
            "transformer_blocks.0.attn1.to_q.weight", w, "transformer"
        ).shape == (
            1920,
            1920,
        )

    def test_norm_1d_untouched(self):
        w = mx.zeros((1920,))
        assert maybe_transpose("norm1.weight", w, "transformer").shape == (1920,)

    def test_non_weight_key_untouched(self):
        """A 3D+ tensor that isn't a .weight must not be permuted."""
        w = mx.zeros((16, 32, 3, 3))
        assert maybe_transpose("encoder.conv_in.bias", w, "vae").shape == (16, 32, 3, 3)


class TestQuantizationScope:
    def test_block_linear_quantized(self):
        assert should_quantize_transformer(
            "transformer_blocks.0.attn1.to_q.weight", mx.zeros((1920, 1920))
        )

    def test_conv_rejected_by_ndim(self):
        assert (
            should_quantize_transformer("patch_embed.proj.weight", mx.zeros((1920, 32, 2, 2)))
            is False
        )

    def test_bias_rejected(self):
        assert (
            should_quantize_transformer("transformer_blocks.0.attn1.to_q.bias", mx.zeros((1920, 1)))
            is False
        )

    @pytest.mark.parametrize(
        "key",
        [
            "patch_embed.proj.weight",
            "time_embedding.linear_1.weight",
            "timestep_embedder.weight",
            "norm_final.weight",
            "pos_embedding.weight",
        ],
    )
    def test_sensitive_layers_excluded(self, key):
        assert should_quantize_transformer(key, mx.zeros((1920, 1920))) is False

    def test_final_proj_out_excluded(self):
        assert should_quantize_transformer("proj_out.weight", mx.zeros((64, 1920))) is False

    def test_proj_out_inside_a_block_is_quantized(self):
        """The exclusion targets the unpatchify projection, not block internals."""
        assert should_quantize_transformer(
            "transformer_blocks.0.attn1.proj_out.weight", mx.zeros((1920, 1920))
        )

    def test_transformer_prefix_is_stripped_before_matching(self):
        """Keys already carrying the component prefix must match the same rules."""
        assert (
            should_quantize_transformer(
                "transformer.patch_embed.proj.weight", mx.zeros((1920, 1920))
            )
            is False
        )


class TestBuildConfig:
    def _write_source(self, tmp_path):
        for comp in COMPONENTS:
            d = tmp_path / comp
            d.mkdir(parents=True)
            (d / "config.json").write_text(json.dumps({"_class_name": comp.title()}))
        return tmp_path

    def test_embeds_every_component_config(self, tmp_path):
        config = _build_config(self._write_source(tmp_path))
        assert config["model_type"] == "cogvideox-fun-inpaint"
        for comp in COMPONENTS:
            assert config[comp]["_class_name"] == comp.title()

    def test_local_source_takes_precedence(self, tmp_path):
        download = tmp_path / "download"
        download.mkdir()
        local = self._write_source(tmp_path / "local")
        config = _build_config(download, local_source=local)
        assert "transformer" in config

    def test_absent_component_config_is_omitted(self, tmp_path):
        src = self._write_source(tmp_path)
        (src / "vae" / "config.json").unlink()
        config = _build_config(src)
        assert "vae" not in config
        assert "transformer" in config
