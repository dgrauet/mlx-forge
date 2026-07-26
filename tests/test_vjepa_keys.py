"""Tests for the V-JEPA 2.0 and 2.1 recipes' key handling and quantization scope.

Both recipes read Meta `.pt` checkpoints whose weights sit under container keys
with shared prefixes, and both must transpose exactly the patch-embed convs and
nothing else — an attention `proj.weight` is a Linear with a colliding name.
"""

import mlx.core as mx
import pytest

from mlx_forge.recipes import vjepa_2_0_vitl as v20
from mlx_forge.recipes import vjepa_2_1_vitl as v21

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------- #
# V-JEPA 2.0
# --------------------------------------------------------------------------- #


class TestV20Sanitizers:
    def test_encoder_prefix_stripped(self):
        assert v20._sanitize_encoder_key("module.backbone.blocks.0.attn.qkv.weight") == (
            "blocks.0.attn.qkv.weight"
        )

    def test_encoder_key_without_prefix_unchanged(self):
        assert v20._sanitize_encoder_key("blocks.0.attn.qkv.weight") == "blocks.0.attn.qkv.weight"

    def test_probe_prefix_stripped(self):
        assert v20._sanitize_probe_key("module.pooler.blocks.0.attn.qkv.weight") == (
            "pooler.blocks.0.attn.qkv.weight"
        )

    def test_probe_strips_only_the_leading_occurrence(self):
        assert v20._sanitize_probe_key("module.module.linear.weight") == "module.linear.weight"


class TestV20EncoderTransform:
    def test_patch_embed_conv3d_transposed(self):
        w = mx.zeros((1024, 3, 2, 16, 16))  # (O, I, D, H, W)
        out = v20._encoder_transform("patch_embed.proj.weight", w)
        assert out.shape == (1024, 2, 16, 16, 3)

    def test_attention_proj_linear_untouched(self):
        """`attn.proj.weight` is a Linear despite the name — must not be permuted."""
        w = mx.zeros((1024, 1024))
        assert v20._encoder_transform("blocks.0.attn.proj.weight", w).shape == (1024, 1024)

    def test_bias_untouched(self):
        w = mx.zeros((1024,))
        assert v20._encoder_transform("patch_embed.proj.bias", w).shape == (1024,)


class TestV20QuantizationScope:
    def test_encoder_block_linear_quantized(self):
        assert v20._encoder_should_quantize("blocks.0.attn.qkv.weight", mx.zeros((3072, 1024)))

    def test_encoder_norm_excluded(self):
        assert (
            v20._encoder_should_quantize("blocks.0.norm1.weight", mx.zeros((1024, 1024))) is False
        )

    def test_encoder_outside_blocks_excluded(self):
        assert v20._encoder_should_quantize("pos_embed.weight", mx.zeros((1024, 1024))) is False

    def test_predictor_block_linear_quantized(self):
        assert v20._predictor_should_quantize(
            "predictor_blocks.0.attn.qkv.weight", mx.zeros((1152, 384))
        )

    @pytest.mark.parametrize(
        "key",
        [
            "predictor_embed.weight",
            "predictor_proj.weight",
            "predictor_norm.weight",
            "predictor_blocks.0.norm1.weight",
        ],
    )
    def test_predictor_sensitive_keys_excluded(self, key):
        assert v20._predictor_should_quantize(key, mx.zeros((384, 384))) is False

    def test_probe_pooler_linear_quantized(self):
        assert v20._probe_should_quantize("pooler.blocks.0.attn.qkv.weight", mx.zeros((768, 256)))

    def test_probe_classifier_head_excluded(self):
        assert v20._probe_should_quantize("linear.weight", mx.zeros((768, 1024))) is False

    def test_probe_query_tokens_excluded(self):
        assert v20._probe_should_quantize("pooler.query_tokens", mx.zeros((1, 1024))) is False


class TestV20ProbeHeadDetection:
    def test_single_linear_head(self):
        raw = {"module.linear.weight": mx.zeros((174, 1024)), "module.linear.bias": mx.zeros(174)}
        assert v20._detect_probe_heads(raw) == {"linear": 174}

    def test_multi_head_epic_kitchens(self):
        raw = {
            "module.verb_classifier.weight": mx.zeros((97, 1024)),
            "module.noun_classifier.weight": mx.zeros((300, 1024)),
            "module.action_classifier.weight": mx.zeros((3806, 1024)),
        }
        assert v20._detect_probe_heads(raw) == {
            "verb_classifier": 97,
            "noun_classifier": 300,
            "action_classifier": 3806,
        }

    def test_pooler_weights_are_not_heads(self):
        raw = {"module.pooler.blocks.0.attn.qkv.weight": mx.zeros((768, 256))}
        assert v20._detect_probe_heads(raw) == {}


# --------------------------------------------------------------------------- #
# V-JEPA 2.1
# --------------------------------------------------------------------------- #


class TestV21Sanitizer:
    def test_keys_pass_through(self):
        assert v21.sanitize_key("blocks.0.attn.qkv.weight") == "blocks.0.attn.qkv.weight"

    def test_no_key_is_dropped(self):
        assert v21.sanitize_key("norms_block.0.bias") is not None


class TestV21TransformWeight:
    def test_tubelet_patch_embed_transposed(self):
        w = mx.zeros((1024, 3, 2, 16, 16))
        assert v21.transform_weight("patch_embed.proj.weight", w).shape == (1024, 2, 16, 16, 3)

    def test_image_patch_embed_transposed(self):
        """The 2.1 encoder carries a second, single-frame patch embed."""
        w = mx.zeros((1024, 3, 1, 16, 16))
        assert v21.transform_weight("patch_embed_img.proj.weight", w).shape == (1024, 1, 16, 16, 3)

    def test_conv2d_rank_also_handled(self):
        w = mx.zeros((1024, 3, 16, 16))
        assert v21.transform_weight("patch_embed.proj.weight", w).shape == (1024, 16, 16, 3)

    def test_attention_proj_linear_untouched(self):
        w = mx.zeros((1024, 1024))
        assert v21.transform_weight("blocks.0.attn.proj.weight", w).shape == (1024, 1024)


class TestV21ToMx:
    def test_torch_tensor_converted(self):
        out = v21._to_mx(torch.ones(2, 3))
        assert isinstance(out, mx.array)
        assert out.shape == (2, 3)

    def test_mx_array_passes_through(self):
        w = mx.zeros((2, 3))
        assert v21._to_mx(w).shape == (2, 3)

    def test_bfloat16_is_upcast(self):
        out = v21._to_mx(torch.ones(2, 2, dtype=torch.bfloat16))
        assert out.dtype == mx.float32


class TestV21Unwrapping:
    def test_encoder_taken_from_ema_container(self):
        raw = {
            "ema_encoder": {"blocks.0.attn.qkv.weight": torch.ones(4, 4)},
            "encoder": {"wrong.weight": torch.ones(1)},
            "epoch": 42,
        }
        out = v21._unwrap_encoder(raw)
        assert set(out) == {"blocks.0.attn.qkv.weight"}, "ema_encoder must win over encoder"

    def test_shared_prefix_stripped(self):
        raw = {"ema_encoder": {"module.blocks.0.attn.qkv.weight": torch.ones(4, 4)}}
        assert set(v21._unwrap_encoder(raw)) == {"blocks.0.attn.qkv.weight"}

    def test_flat_tensor_dict_left_alone(self):
        raw = {"blocks.0.attn.qkv.weight": torch.ones(4, 4)}
        assert set(v21._unwrap_encoder(raw)) == {"blocks.0.attn.qkv.weight"}

    def test_predictor_prefix_stripped(self):
        raw = {"predictor": {"module.backbone.predictor_blocks.0.attn.qkv.weight": torch.ones(2)}}
        assert set(v21._unwrap_predictor(raw)) == {"predictor_blocks.0.attn.qkv.weight"}

    def test_predictor_missing_raises(self):
        with pytest.raises(KeyError):
            v21._unwrap_predictor({"ema_encoder": {}})


class TestV21QuantizationScope:
    def test_encoder_block_linear_quantized(self):
        assert v21.should_quantize_encoder("blocks.0.attn.qkv.weight", mx.zeros((3072, 1024)))

    def test_encoder_patch_embed_excluded(self):
        assert (
            v21.should_quantize_encoder("patch_embed.proj.weight", mx.zeros((1024, 768))) is False
        )

    def test_encoder_norm_excluded(self):
        assert v21.should_quantize_encoder("blocks.0.norm1.weight", mx.zeros((1024, 1024))) is False

    def test_encoder_bias_excluded(self):
        assert v21.should_quantize_encoder("blocks.0.attn.qkv.bias", mx.zeros((3072, 1))) is False

    def test_predictor_block_linear_quantized(self):
        assert v21.should_quantize_predictor(
            "predictor_blocks.0.mlp.fc1.weight", mx.zeros((1536, 384))
        )

    @pytest.mark.parametrize(
        "key",
        [
            "predictor_embed.weight",
            "predictor_proj.weight",
            "predictor_proj_context.weight",
            "predictor_norm.weight",
            "predictor_blocks.0.norm2.weight",
        ],
    )
    def test_predictor_sensitive_keys_excluded(self, key):
        assert v21.should_quantize_predictor(key, mx.zeros((384, 384))) is False

    def test_encoder_predicate_ignores_predictor_blocks(self):
        """The two components are quantized by separate predicates; no crossover."""
        assert v21.should_quantize_predictor(
            "blocks.0.attn.qkv.weight", mx.zeros((3072, 1024))
        ) is (False)
