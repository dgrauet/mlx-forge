"""SOURCE_FILES invariants, output-name guards, and the quantisation record."""

import json

import mlx.core as mx
import pytest

from mlx_forge.recipes.ltx_25 import (
    QUANTIZED_COMPONENTS,
    SOURCE_FILES,
    UPSTREAM_TRANSFORMERS,
    download_size_mb,
    ltx25_should_quantize,
    output_size_mb,
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


class TestSourceTable:
    def test_covers_every_upstream_file_we_convert(self):
        paths = {s.path for s in SOURCE_FILES}
        assert set(UPSTREAM_TRANSFORMERS.values()) <= paths
        assert "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" in paths
        assert "vae/ltx-2.5-video-vae-bf16.safetensors" in paths
        assert "vae/ltx-2.5-video-vae-conv-bf16.safetensors" in paths
        assert "vae/ltx-2.5-audio-vae-bf16.safetensors" in paths
        assert "model_patches/ltx-2.5-duration-head-bf16.safetensors" in paths

    def test_excludes_the_upstream_prequantised_variants(self):
        # comfy-int8-convrot and nvfp4 are ComfyUI formats; we quantise from bf16.
        paths = {s.path for s in SOURCE_FILES}
        assert not any("convrot" in p or "nvfp4" in p for p in paths)

    def test_the_small_components_come_first(self):
        # Ordering exists so a wrong classifier fails in minutes, not hours.
        first_big = next(i for i, s in enumerate(SOURCE_FILES) if s.size_mb > 10_000)
        assert all(s.size_mb < 10_000 for s in SOURCE_FILES[:first_big])
        assert first_big >= 5

    def test_every_entry_can_convert_itself(self):
        for source in SOURCE_FILES:
            assert (source.classify is not None) or (source.converter is not None), source.path

    def test_no_component_is_named_plainly_vae(self):
        # The whole point of the _conv/_av suffixes.
        names = {c for s in SOURCE_FILES for c in s.components}
        assert "vae_decoder" not in names
        assert "vae_encoder" not in names
        assert {"vae_decoder_conv", "vae_decoder_av"} <= names


class TestFootprint:
    def test_a_full_run_reports_both_numbers(self):
        download = download_size_mb(["dev", "distilled"], skip_shared=False)
        output = output_size_mb(["dev", "distilled"], skip_shared=False)
        assert 115_000 < download < 135_000  # ~124 GB
        assert 110_000 < output < 125_000  # ~118 GB

    def test_skip_shared_only_counts_the_transformers(self):
        assert download_size_mb(["dev"], skip_shared=True) < 45_000

    def test_one_variant_is_cheaper_than_two(self):
        assert download_size_mb(["dev"], skip_shared=False) < download_size_mb(
            ["dev", "distilled"], skip_shared=False
        )
