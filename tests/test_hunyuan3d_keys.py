"""Tests for Hunyuan3D-2.1 key classification and sanitization."""

import mlx.core as mx

from mlx_forge.recipes.hunyuan3d_21 import (
    SHAPE_CKPT_SECTIONS as _SECTIONS,
)
from mlx_forge.recipes.hunyuan3d_21 import (
    sanitize_dit_key,
)
from mlx_forge.recipes.hunyuan3d_21 import (
    sanitize_shape_image_encoder_key as sanitize_image_encoder_key,
)
from mlx_forge.recipes.hunyuan3d_21 import (
    sanitize_shape_vae_key as sanitize_vae_key,
)
from mlx_forge.recipes.hunyuan3d_21 import (
    shape_should_quantize as should_quantize,
)


def classify_key(section: str, key: str):
    return _SECTIONS.get(section)


class TestClassifyKey:
    """Test key classification from ckpt top-level dict keys."""

    # DiT keys (from ckpt["model"])
    def test_dit_x_embedder(self):
        assert classify_key("model", "x_embedder.weight") == "dit"

    def test_dit_t_embedder(self):
        assert classify_key("model", "t_embedder.mlp.0.weight") == "dit"

    def test_dit_block(self):
        assert classify_key("model", "blocks.0.attn1.to_q.weight") == "dit"

    def test_dit_moe_gate(self):
        assert classify_key("model", "blocks.15.moe.gate.weight") == "dit"

    def test_dit_moe_expert(self):
        assert classify_key("model", "blocks.15.moe.experts.0.net.0.linear.weight") == "dit"

    def test_dit_final_layer(self):
        assert classify_key("model", "final_layer.linear.weight") == "dit"

    # VAE keys (from ckpt["vae"])
    def test_vae_fourier(self):
        assert classify_key("vae", "fourier_embedder.frequencies") == "vae"

    def test_vae_encoder(self):
        assert classify_key("vae", "encoder.cross_attn.to_q.weight") == "vae"

    def test_vae_transformer(self):
        assert classify_key("vae", "transformer.resblocks.0.attn.in_proj_weight") == "vae"

    def test_vae_geo_decoder(self):
        assert classify_key("vae", "geo_decoder.output_proj.weight") == "vae"

    # DINOv2 keys (from ckpt["conditioner"])
    def test_dino_embeddings(self):
        assert (
            classify_key(
                "conditioner",
                "main_image_encoder.model.embeddings.patch_embeddings.projection.weight",
            )
            == "image_encoder"
        )

    def test_dino_encoder_layer(self):
        assert (
            classify_key(
                "conditioner",
                "main_image_encoder.model.encoder.layer.0.attention.attention.query.weight",
            )
            == "image_encoder"
        )

    # Unknown section returns None
    def test_unknown_section(self):
        assert classify_key("unknown_section", "some.key") is None


class TestSanitizeDitKey:
    def test_x_embedder(self):
        assert sanitize_dit_key("x_embedder.weight") == "x_embedder.weight"

    def test_block_self_attn(self):
        assert sanitize_dit_key("blocks.0.attn1.to_q.weight") == "blocks.0.attn1.to_q.weight"

    def test_block_cross_attn(self):
        assert sanitize_dit_key("blocks.0.attn2.to_k.weight") == "blocks.0.attn2.to_k.weight"

    def test_block_mlp(self):
        assert sanitize_dit_key("blocks.5.mlp.fc1.weight") == "blocks.5.mlp.fc1.weight"

    def test_block_moe_gate(self):
        assert sanitize_dit_key("blocks.15.moe.gate.weight") == "blocks.15.moe.gate.weight"

    def test_block_moe_expert(self):
        assert (
            sanitize_dit_key("blocks.15.moe.experts.0.net.0.linear.weight")
            == "blocks.15.moe.experts.0.net.0.linear.weight"
        )

    def test_block_moe_shared(self):
        assert (
            sanitize_dit_key("blocks.15.moe.shared_experts.net.0.linear.weight")
            == "blocks.15.moe.shared_experts.net.0.linear.weight"
        )

    def test_skip_linear(self):
        assert sanitize_dit_key("blocks.11.skip_linear.weight") == "blocks.11.skip_linear.weight"

    def test_final_layer(self):
        assert sanitize_dit_key("final_layer.linear.weight") == "final_layer.linear.weight"

    def test_timestep_embedder(self):
        assert sanitize_dit_key("t_embedder.mlp.0.weight") == "t_embedder.mlp.0.weight"


class TestSanitizeVaeKey:
    def test_fourier_frequencies(self):
        assert sanitize_vae_key("fourier_embedder.frequencies") == "fourier_embedder.frequencies"

    def test_transformer_block(self):
        assert (
            sanitize_vae_key("transformer.resblocks.0.attn.in_proj_weight")
            == "transformer.resblocks.0.attn.in_proj_weight"
        )

    def test_geo_decoder(self):
        result = sanitize_vae_key("geo_decoder.output_proj.weight")
        assert result == "geo_decoder.output_proj.weight"

    def test_post_kl(self):
        assert sanitize_vae_key("post_kl.weight") == "post_kl.weight"


class TestSanitizeImageEncoderKey:
    def test_strip_prefix(self):
        """Strips main_image_encoder.model. prefix."""
        key = "main_image_encoder.model.embeddings.patch_embeddings.projection.weight"
        assert sanitize_image_encoder_key(key) == "embeddings.patch_embeddings.projection.weight"

    def test_encoder_layer(self):
        key = "main_image_encoder.model.encoder.layer.0.attention.attention.query.weight"
        assert sanitize_image_encoder_key(key) == "encoder.layer.0.attention.attention.query.weight"

    def test_layernorm(self):
        key = "main_image_encoder.model.layernorm.weight"
        assert sanitize_image_encoder_key(key) == "layernorm.weight"

    def test_non_matching_returns_none(self):
        """Keys not starting with expected prefix are skipped."""
        assert sanitize_image_encoder_key("some.random.key") is None


class TestShouldQuantize:
    """Test the quantization predicate."""

    def test_block_attn_weight_quantized(self):
        w = mx.zeros((128, 128))
        assert should_quantize("dit.blocks.0.attn1.to_q.weight", w) is True

    def test_block_mlp_weight_quantized(self):
        w = mx.zeros((128, 128))
        assert should_quantize("dit.blocks.5.mlp.fc1.weight", w) is True

    def test_moe_expert_weight_quantized(self):
        w = mx.zeros((128, 128))
        assert should_quantize("dit.blocks.15.moe.experts.0.net.0.linear.weight", w) is True

    def test_moe_gate_excluded(self):
        w = mx.zeros((8, 2048))
        assert should_quantize("dit.blocks.15.moe.gate.weight", w) is False

    def test_norm_excluded(self):
        w = mx.zeros((128, 128))
        assert should_quantize("dit.blocks.0.norm1.weight", w) is False

    def test_embedder_excluded(self):
        w = mx.zeros((128, 128))
        assert should_quantize("dit.x_embedder.weight", w) is False

    def test_final_layer_excluded(self):
        w = mx.zeros((128, 128))
        assert should_quantize("dit.final_layer.linear.weight", w) is False

    def test_1d_excluded(self):
        w = mx.zeros((128,))
        assert should_quantize("dit.blocks.0.attn1.to_q.weight", w) is False

    def test_bias_excluded(self):
        w = mx.zeros((128,))
        assert should_quantize("dit.blocks.0.attn1.to_q.bias", w) is False

    def test_scales_excluded(self):
        w = mx.zeros((128, 128))
        assert should_quantize("dit.blocks.0.attn1.to_q.scales", w) is False
