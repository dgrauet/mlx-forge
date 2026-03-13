"""Tests for Qwen-Image key sanitization and quantization predicate."""

import mlx.core as mx

from mlx_forge.recipes.qwen_image_2512 import (
    qwen_image_should_quantize,
    sanitize_key,
    vae_transform,
)


class TestSanitizeKey:
    def test_identity(self):
        assert sanitize_key("transformer_blocks.0.attn.to_q.weight") == (
            "transformer_blocks.0.attn.to_q.weight"
        )

    def test_identity_text_encoder(self):
        assert sanitize_key("model.layers.0.self_attn.q_proj.weight") == (
            "model.layers.0.self_attn.q_proj.weight"
        )

    def test_identity_vae(self):
        assert sanitize_key("encoder.down_blocks.0.resnets.0.conv1.weight") == (
            "encoder.down_blocks.0.resnets.0.conv1.weight"
        )


class TestShouldQuantize:
    def test_transformer_attn_quantized(self):
        assert qwen_image_should_quantize(
            "transformer_blocks.0.attn.to_q.weight", mx.zeros((3072, 3072))
        )

    def test_transformer_attn_kv_quantized(self):
        assert qwen_image_should_quantize(
            "transformer_blocks.0.attn.to_k.weight", mx.zeros((3072, 3072))
        )

    def test_transformer_cross_attn_quantized(self):
        assert qwen_image_should_quantize(
            "transformer_blocks.0.attn.add_q_proj.weight", mx.zeros((3072, 3584))
        )

    def test_transformer_img_mlp_quantized(self):
        assert qwen_image_should_quantize(
            "transformer_blocks.0.img_mlp.0.weight", mx.zeros((12288, 3072))
        )

    def test_transformer_txt_mlp_quantized(self):
        assert qwen_image_should_quantize(
            "transformer_blocks.0.txt_mlp.0.weight", mx.zeros((12288, 3584))
        )

    def test_text_encoder_linear_quantized(self):
        assert qwen_image_should_quantize(
            "model.layers.0.self_attn.q_proj.weight", mx.zeros((3584, 3584))
        )

    def test_text_encoder_mlp_quantized(self):
        assert qwen_image_should_quantize(
            "model.layers.0.mlp.gate_proj.weight", mx.zeros((18944, 3584))
        )

    def test_img_in_not_quantized(self):
        assert not qwen_image_should_quantize("img_in.weight", mx.zeros((3072, 3072)))

    def test_proj_out_not_quantized(self):
        assert not qwen_image_should_quantize("proj_out.weight", mx.zeros((256, 3072)))

    def test_norm_out_not_quantized(self):
        assert not qwen_image_should_quantize("norm_out.linear.weight", mx.zeros((3072, 3072)))

    def test_time_text_embed_not_quantized(self):
        assert not qwen_image_should_quantize(
            "time_text_embed.timestep_embedder.linear_1.weight", mx.zeros((768, 256))
        )

    def test_modulation_not_quantized(self):
        assert not qwen_image_should_quantize(
            "transformer_blocks.0.img_mod.linear.weight", mx.zeros((9216, 3072))
        )

    def test_embed_tokens_not_quantized(self):
        assert not qwen_image_should_quantize("model.embed_tokens.weight", mx.zeros((152064, 3584)))

    def test_lm_head_not_quantized(self):
        assert not qwen_image_should_quantize("lm_head.weight", mx.zeros((152064, 3584)))

    def test_patch_embed_not_quantized(self):
        assert not qwen_image_should_quantize(
            "visual.patch_embed.proj.weight", mx.zeros((1280, 1176))
        )

    def test_merger_not_quantized(self):
        assert not qwen_image_should_quantize("visual.merger.mlp.0.weight", mx.zeros((3584, 5120)))

    def test_norm_not_quantized(self):
        assert not qwen_image_should_quantize(
            "model.layers.0.input_layernorm.weight", mx.zeros((3584,))
        )

    def test_bias_not_quantized(self):
        assert not qwen_image_should_quantize(
            "transformer_blocks.0.attn.to_q.bias", mx.zeros((3072,))
        )

    def test_small_tensor_not_quantized(self):
        assert not qwen_image_should_quantize(
            "transformer_blocks.0.attn.to_q.weight", mx.zeros((8, 8))
        )


class TestVaeTransform:
    def test_conv2d_transposed(self):
        # PyTorch Conv2d: (O, I, H, W) -> MLX: (O, H, W, I)
        weight = mx.zeros((64, 32, 3, 3))
        result = vae_transform("encoder.conv_in.weight", weight, "vae")
        assert result.shape == (64, 3, 3, 32)

    def test_non_conv_unchanged(self):
        weight = mx.zeros((64, 32))
        result = vae_transform("encoder.linear.weight", weight, "vae")
        assert result.shape == (64, 32)

    def test_norm_unchanged(self):
        weight = mx.zeros((64,))
        result = vae_transform("encoder.norm.weight", weight, "vae")
        assert result.shape == (64,)
