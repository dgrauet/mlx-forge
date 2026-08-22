"""The Gemma-4 text encoder: a model, five files hidden in U8 tensors, and a
quantisation policy that is not mlx-lm's.
"""

import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_forge.recipes.ltx_25_text_encoder import (
    ASSET_FILENAMES,
    classify_text_encoder_key,
    extract_assets,
    sanitize_text_encoder_key,
    should_quantize_gemma,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ltx_25_keys.json"
TEXT_ENCODER = "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"


@pytest.fixture(scope="session")
def te_keys() -> list[str]:
    with open(FIXTURE) as handle:
        return json.load(handle)[TEXT_ENCODER]["keys"]


class TestClassification:
    def test_every_key_is_classified(self, te_keys):
        assert [k for k in te_keys if classify_text_encoder_key(k) is None] == []

    def test_assets_are_separated_from_weights(self):
        assert classify_text_encoder_key("tokenizer_json") == "text_encoder_asset"
        assert classify_text_encoder_key("hf_asset__tokenizer_config.json") == "text_encoder_asset"
        assert classify_text_encoder_key("model.layers.0.mlp.up_proj.weight") == "text_encoder"

    def test_the_projections_stay_with_the_text_encoder(self):
        # Upstream packages them here; 2.3 put them in the connector. Following
        # upstream is what keeps the conversion loop free of cross-file state.
        assert (
            classify_text_encoder_key("text_embedding_projection.video_aggregate_embed.weight")
            == "text_encoder"
        )
        assert (
            classify_text_encoder_key("audio_projector.embedding_projection.weight")
            == "text_encoder"
        )

    def test_asset_keys_are_never_sanitised_as_weights(self):
        assert sanitize_text_encoder_key("tokenizer_json") is None


class TestAssetExtraction:
    def test_writes_five_real_files(self, tmp_path):
        payload = b'{"hello": "world"}'
        weights = {key: mx.array(list(payload), dtype=mx.uint8) for key in ASSET_FILENAMES}
        written = extract_assets(weights, tmp_path)

        assert sorted(p.name for p in written) == sorted(ASSET_FILENAMES.values())
        for path in written:
            assert path.read_bytes() == payload

    def test_the_tokenizer_lands_under_its_conventional_name(self):
        assert ASSET_FILENAMES["tokenizer_json"] == "tokenizer.json"

    def test_the_hf_asset_prefix_becomes_the_filename(self):
        assert ASSET_FILENAMES["hf_asset__chat_template.jinja"] == "chat_template.jinja"
        assert ASSET_FILENAMES["hf_asset__generation_config.json"] == "generation_config.json"

    def test_a_missing_asset_is_reported_not_skipped(self, tmp_path):
        # A pack whose tokenizer silently failed to extract loads and then
        # produces nonsense. Better to stop here.
        with pytest.raises(SystemExit, match="tokenizer.json"):
            extract_assets({}, tmp_path)


class TestQuantisationPolicy:
    """Driven over `text_encoder.`-prefixed keys, not bare upstream names.

    I1 (final-review.md): process_component writes text-encoder keys under
    component_prefix="text_encoder", and quantize_component reloads that file
    and passes should_quantize_gemma the on-disk key straight through. A test
    over bare keys — what this class used to do — cannot see that: it would
    keep passing even if the prefix strip were removed entirely, which is
    exactly how the exclusions went unexercised in production for as long as
    they did.
    """

    def test_attention_and_mlp_projections_are_quantised(self):
        w = mx.zeros((3840, 3840))
        for key in (
            "text_encoder.model.layers.0.self_attn.q_proj.weight",
            "text_encoder.model.layers.0.self_attn.o_proj.weight",
            "text_encoder.model.layers.0.mlp.gate_proj.weight",
            "text_encoder.model.layers.0.mlp.down_proj.weight",
        ):
            assert should_quantize_gemma(key, w), key

    def test_the_embedding_table_is_not(self):
        # Gemma is a feature extractor here, not a generator: its hidden states
        # feed the DiT directly, so a quantised embedding table moves the
        # starting point of all 48 layers. mlx-lm would quantise it; we do not.
        assert not should_quantize_gemma(
            "text_encoder.model.embed_tokens.weight", mx.zeros((262144, 3840))
        )

    def test_the_projectors_are_not(self):
        # Six tensors carrying the whole of the conditioning, for no weight.
        for key in (
            "text_encoder.text_embedding_projection.video_aggregate_embed.weight",
            "text_encoder.audio_projector.embedding_projection.weight",
            "text_encoder.multi_modal_projector.embedding_projection.weight",
        ):
            assert not should_quantize_gemma(key, mx.zeros((3840, 3840))), key

    def test_the_vision_branch_is_not(self):
        assert not should_quantize_gemma(
            "text_encoder.vision_model.encoder.layer.0.weight", mx.zeros((64, 64))
        )

    def test_norms_and_scalars_are_not(self):
        assert not should_quantize_gemma(
            "text_encoder.model.layers.0.input_layernorm.weight", mx.zeros((3840,))
        )
        assert not should_quantize_gemma("text_encoder.model.layers.0.layer_scalar", mx.zeros((1,)))

    def test_already_quantised_artefacts_are_not(self):
        assert not should_quantize_gemma(
            "text_encoder.model.layers.0.mlp.up_proj.scales", mx.zeros((16, 16))
        )
        assert not should_quantize_gemma(
            "text_encoder.model.layers.0.mlp.up_proj.biases", mx.zeros((16, 16))
        )

    def test_a_future_vit_attention_tensor_is_still_excluded(self):
        # This is what I1 actually broke: an unprefixed check would let a
        # vision-branch tensor that happens to end in one of the quantised
        # suffixes (exactly what a ViT's own self-attention normally has)
        # sail through, because startswith(_UNQUANTISED_PREFIXES) was being
        # tested against a key that already carried "text_encoder." in
        # production. With the prefix stripped first, this must stay
        # excluded even though its suffix alone would otherwise qualify.
        assert not should_quantize_gemma(
            "text_encoder.vision_model.encoder.layer.0.self_attn.q_proj.weight",
            mx.zeros((64, 64)),
        )
