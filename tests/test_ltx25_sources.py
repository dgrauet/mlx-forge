"""SOURCE_FILES invariants, output-name guards, and the quantisation record."""

import json

import mlx.core as mx
import pytest

from mlx_forge.recipes.ltx_25 import (
    QUANTIZED_COMPONENTS,
    ltx25_should_quantize,
    write_ltx25_quantize_config,
)


class TestDitQuantisation:
    def test_block_linear_weights_are_quantised(self):
        assert ltx25_should_quantize(
            "transformer_blocks.0.attn1.to_q.weight", mx.zeros((4096, 4096))
        )

    def test_tables_and_heads_outside_the_blocks_are_not(self):
        for key in ("scale_shift_table", "patchify_proj.weight", "proj_out.weight"):
            assert not ltx25_should_quantize(key, mx.zeros((4096, 4096))), key

    def test_one_dimensional_tensors_are_not(self):
        assert not ltx25_should_quantize(
            "transformer_blocks.0.prompt_scale_shift_table", mx.zeros((4096,))
        )

    def test_already_quantised_artefacts_are_not(self):
        assert not ltx25_should_quantize(
            "transformer_blocks.0.attn1.to_q.scales", mx.zeros((64, 64))
        )


class TestQuantizeConfig:
    def test_records_a_scope_per_component(self, tmp_path):
        # 2.3 wrote a single only_transformer_blocks flag. Two components with
        # different rules cannot be described by one boolean.
        path = write_ltx25_quantize_config(tmp_path, bits=8, group_size=64)
        record = json.loads(path.read_text())["quantization"]
        assert record["bits"] == 8
        assert record["group_size"] == 64
        assert set(record["components"]) == set(QUANTIZED_COMPONENTS)

    def test_names_only_what_is_actually_quantised(self):
        assert set(QUANTIZED_COMPONENTS) == {"transformer", "text_encoder"}

    @pytest.mark.parametrize("component", ["vae_decoder_conv", "vocoder", "duration_head"])
    def test_bf16_components_are_absent(self, component):
        assert component not in QUANTIZED_COMPONENTS
