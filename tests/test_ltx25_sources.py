"""SOURCE_FILES invariants, output-name guards, and the quantisation record."""

import argparse
import hashlib
import json
import json as _json

import mlx.core as mx
import pytest

from mlx_forge.recipes import ltx_25
from mlx_forge.recipes.ltx_25 import (
    LORA_FILES,
    PASSTHROUGH_FILES,
    QUANTIZED_COMPONENTS,
    SOURCE_FILES,
    UPSTREAM_TRANSFORMERS,
    _is_upscaler_conv_weight,
    _require_license_verified,
    _selected_loras,
    _verify_license_if_carried,
    connector_fingerprint,
    download_size_mb,
    ltx25_should_quantize,
    output_size_mb,
    verify_embedded_license,
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
        path = write_ltx25_quantize_config(tmp_path, bits=8, group_size=64, skip_shared=False)
        record = json.loads(path.read_text())["quantization"]
        assert record["bits"] == 8
        assert record["group_size"] == 64
        assert set(record["components"]) == set(QUANTIZED_COMPONENTS)

    def test_a_delta_pack_names_only_the_transformer(self, tmp_path):
        # I3 (final-review.md): a --skip-shared --quantize pack contains no
        # text encoder, but write_ltx25_quantize_config used to write both
        # QUANTIZED_COMPONENTS entries unconditionally — the manifest would
        # claim a component the pack does not ship, defeating the whole
        # point of recording scope per component (a runtime is meant to
        # rebuild each without guessing).
        path = write_ltx25_quantize_config(tmp_path, bits=8, group_size=64, skip_shared=True)
        record = json.loads(path.read_text())["quantization"]
        assert set(record["components"]) == {"transformer"}

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

    def test_a_distilled_only_run_excludes_the_dead_lora(self):
        # The estimate must derive from the same selection convert()'s copy
        # step uses (_selected_loras), or the two can disagree — exactly the
        # drift SOURCE_FILES exists to prevent.
        with_dev = download_size_mb(["dev"], skip_shared=False)
        distilled_only = download_size_mb(["distilled"], skip_shared=False)
        assert with_dev - distilled_only == int(PASSTHROUGH_FILES["size_mb"])

    def test_an_explicit_lora_choice_restores_it_for_a_distilled_only_run(self):
        distilled_only = download_size_mb(["distilled"], skip_shared=False)
        forced = download_size_mb(["distilled"], skip_shared=False, lora=["distilled-450"])
        assert forced - distilled_only == int(PASSTHROUGH_FILES["size_mb"])


class TestUpscalerConvWeight:
    # Real shapes from the live spatial/temporal upscaler safetensors headers.
    def test_conv2d_upsampler_weight_is_caught(self):
        assert _is_upscaler_conv_weight("upsampler.0.weight", mx.zeros((4096, 1024, 3, 3)))

    def test_conv3d_upsampler_weight_is_caught(self):
        assert _is_upscaler_conv_weight("upsampler.0.weight", mx.zeros((1024, 512, 3, 3, 3)))

    def test_2d_weight_is_not_a_conv(self):
        assert not _is_upscaler_conv_weight("initial_norm.weight", mx.zeros((1024, 1024)))

    def test_rank3_non_weight_tensor_is_not_a_conv(self):
        assert not _is_upscaler_conv_weight("some.bias", mx.zeros((3, 3, 3)))


AUGUST = "  LTX-2.x Community License Agreement   \n  License date: August 11, 2026  \n"


class TestConnectorFingerprint:
    _KEY = "model.diffusion_model.video_embeddings_connector.proj.weight"

    def test_differs_when_a_connector_tensor_differs(self):
        base = {self._KEY: mx.zeros((4, 4))}
        changed = {self._KEY: mx.ones((4, 4))}
        assert connector_fingerprint(base) != connector_fingerprint(changed)

    def test_ignores_non_connector_tensors(self):
        base = {self._KEY: mx.zeros((4, 4))}
        with_extra = dict(base)
        with_extra["model.diffusion_model.transformer_blocks.0.attn.weight"] = mx.ones((8, 8))
        assert connector_fingerprint(base) == connector_fingerprint(with_extra)

    def test_matches_a_hand_computed_digest_for_a_single_tensor(self):
        weights = {self._KEY: mx.zeros((2,))}
        expected = hashlib.sha256()
        expected.update(self._KEY.encode())
        tensor = mx.zeros((2,)).astype(mx.float32)
        # mx.array implements the buffer protocol at runtime but its stubs
        # don't declare __buffer__, so ty can't see it satisfies Buffer.
        expected.update(bytes(memoryview(tensor)))  # ty: ignore[invalid-argument-type]
        assert connector_fingerprint(weights) == expected.hexdigest()


class TestEmbeddedLicenceCheck:
    def test_accepts_a_text_differing_only_in_trailing_space(self, tmp_path):
        # What we ship is GitHub's bytes (34 441); what the weights carry is
        # 34 562. The 580 lines are identical once trailing space is removed.
        # Leading indentation matches AUGUST's; only the trailing space differs.
        shipped = tmp_path / "LICENSE"
        shipped.write_text(
            "  LTX-2.x Community License Agreement\n  License date: August 11, 2026\n"
        )
        verify_embedded_license({"license": AUGUST}, shipped)  # must not raise

    def test_rejects_a_text_whose_words_differ(self, tmp_path):
        shipped = tmp_path / "LICENSE"
        shipped.write_text("LTX-2 Community License Agreement\nLicense date: January 5, 2026\n")
        with pytest.raises(SystemExit, match="does not match the agreement"):
            verify_embedded_license({"license": AUGUST}, shipped)

    def test_a_checkpoint_carrying_no_licence_is_not_a_pass(self, tmp_path):
        # Silence must not read as agreement.
        shipped = tmp_path / "LICENSE"
        shipped.write_text("anything\n")
        with pytest.raises(SystemExit, match="carries no licence"):
            verify_embedded_license({}, shipped)


class TestVerifyLicenseIfCarried:
    """Not every LTX-2.5 checkpoint embeds the licence (the temporal upscaler
    and the text encoder do not), so convert() must only check the files that
    carry one, and abort at the end if none did."""

    def test_a_file_carrying_a_matching_licence_verifies(self, tmp_path):
        shipped = tmp_path / "LICENSE"
        shipped.write_text(
            "  LTX-2.x Community License Agreement\n  License date: August 11, 2026\n"
        )
        assert _verify_license_if_carried({"license": AUGUST}, shipped) is True

    def test_a_file_carrying_a_mismatched_licence_still_aborts(self, tmp_path):
        shipped = tmp_path / "LICENSE"
        shipped.write_text("LTX-2 Community License Agreement\nLicense date: January 5, 2026\n")
        with pytest.raises(SystemExit, match="does not match the agreement"):
            _verify_license_if_carried({"license": AUGUST}, shipped)

    def test_a_file_carrying_no_licence_is_skipped_not_failed(self, tmp_path):
        # e.g. the temporal upscaler, or the text encoder's ['format', 'gemma_config'].
        shipped = tmp_path / "LICENSE"
        shipped.write_text("anything\n")
        assert _verify_license_if_carried({"format": "pt", "gemma_config": "{}"}, shipped) is False


class TestRequireLicenseVerified:
    def test_does_not_raise_once_something_verified(self, tmp_path):
        _require_license_verified(True, tmp_path / "LICENSE")  # must not raise

    def test_a_run_where_nothing_verified_aborts_naming_the_problem(self, tmp_path):
        with pytest.raises(SystemExit, match="no file"):
            _require_license_verified(False, tmp_path / "LICENSE")


class TestLicenseCheckAcrossAPack:
    """The four run-level scenarios from convert()'s own loop, exercised on
    plain metadata dicts standing in for SOURCE_FILES entries — no real
    checkpoint needed, mirroring how verify_embedded_license is already
    tested."""

    _MATCHING_SHIPPED = "  LTX-2.x Community License Agreement\n  License date: August 11, 2026\n"
    _NO_LICENSE = {"format": "pt", "gemma_config": "{}"}  # e.g. the text encoder
    _WITH_LICENSE = {"license": AUGUST}

    def _run(self, header_metadata_list, license_path):
        verified_any = False
        for header_metadata in header_metadata_list:
            if _verify_license_if_carried(header_metadata, license_path):
                verified_any = True
        _require_license_verified(verified_any, license_path)

    def test_a_run_where_every_file_carries_it_completes(self, tmp_path):
        shipped = tmp_path / "LICENSE"
        shipped.write_text(self._MATCHING_SHIPPED)
        self._run([self._WITH_LICENSE, self._WITH_LICENSE], shipped)  # must not raise

    def test_a_run_where_some_files_carry_none_still_completes(self, tmp_path):
        shipped = tmp_path / "LICENSE"
        shipped.write_text(self._MATCHING_SHIPPED)
        # The conv video VAE (carries it) first, then two files that do not
        # (temporal upscaler, text encoder) — the real SOURCE_FILES ordering.
        self._run([self._WITH_LICENSE, self._NO_LICENSE, self._NO_LICENSE], shipped)

    def test_a_run_where_no_file_carries_it_aborts_naming_the_problem(self, tmp_path):
        shipped = tmp_path / "LICENSE"
        shipped.write_text(self._MATCHING_SHIPPED)
        with pytest.raises(SystemExit, match="no file"):
            self._run([self._NO_LICENSE, self._NO_LICENSE], shipped)

    def test_a_mismatch_on_any_carrying_file_still_aborts(self, tmp_path):
        shipped = tmp_path / "LICENSE"
        shipped.write_text("this text matches nothing\n")
        with pytest.raises(SystemExit, match="does not match the agreement"):
            self._run([self._NO_LICENSE, self._WITH_LICENSE, self._NO_LICENSE], shipped)


class TestLoraSelection:
    def test_defaults_to_every_declared_lora_when_dev_is_present(self):
        assert _selected_loras(["dev"], skip_shared=False, lora=None) == sorted(LORA_FILES)
        assert _selected_loras(["dev", "distilled"], skip_shared=False, lora=None) == sorted(
            LORA_FILES
        )

    def test_defaults_to_none_for_a_distilled_only_pack(self, capsys):
        # distilled-* LoRAs are meant to run on the dev transformer; a
        # distilled checkpoint has the distillation baked in and never loads
        # one, so bundling it by default would be dead weight.
        assert _selected_loras(["distilled"], skip_shared=False, lora=None) == []
        assert "no 'dev' variant" in capsys.readouterr().out

    def test_an_explicit_choice_is_honoured_even_for_a_distilled_only_pack(self):
        assert _selected_loras(["distilled"], skip_shared=False, lora=["distilled-450"]) == [
            "distilled-450"
        ]

    def test_skip_shared_ships_no_lora_even_if_named_explicitly(self):
        assert _selected_loras(["dev"], skip_shared=True, lora=["distilled-450"]) == []


def _pack(tmp_path, **files):
    import mlx.core as mx

    for name, weights in files.items():
        mx.save_safetensors(str(tmp_path / f"{name}.safetensors"), weights)
    (tmp_path / "split_model.json").write_text(_json.dumps({"recipe": "ltx-2.5"}))
    return tmp_path


def _full_pack(tmp_path, *, omit=()):
    """A pack that satisfies every check `validate()` currently runs, so any
    test built on it isolates exactly the gap it targets: if the pack is
    otherwise complete and only `omit` is missing, a still-passing run means
    the omitted component is not actually gated on.
    """
    import mlx.core as mx

    # The seven components EXPECTED_TENSOR_COUNTS covered before this fix,
    # each with the right tensor count and no accidental "<component>."
    # literal prefix (that would trip validate_no_pytorch_prefix).
    counts = dict(ltx_25.EXPECTED_TENSOR_COUNTS)
    for component, count in counts.items():
        if component in omit:
            continue
        weights = {f"w{i}.weight": mx.zeros((2, 2)) for i in range(count)}
        mx.save_safetensors(str(tmp_path / f"{component}.safetensors"), weights)

    # Assets are always written, even when `omit` drops the text_encoder
    # weight file itself — otherwise a missing-weights test would fail for
    # the wrong reason (missing assets, already covered by
    # test_a_missing_text_encoder_asset_fails) rather than the one it targets.
    for filename in ltx_25.TEXT_ENCODER_ASSET_FILES:
        content = "{}" if filename.endswith(".json") else "not json"
        (tmp_path / filename).write_text(content)

    if "transformer" not in omit:
        mx.save_safetensors(
            str(tmp_path / ltx_25.VARIANT_FILENAMES["dev"]),
            {"transformer_blocks.0.attn1.to_q.weight": mx.zeros((4, 4))},
        )

    (tmp_path / "split_model.json").write_text(
        _json.dumps({"recipe": "ltx-2.5", "transformer_variants": ["dev"]})
    )
    return tmp_path


class TestValidate:
    def test_a_missing_text_encoder_asset_fails(self, tmp_path, capsys):
        import mlx.core as mx

        _pack(tmp_path, text_encoder={"text_encoder.model.embed_tokens.weight": mx.zeros((4, 4))})
        with pytest.raises(SystemExit):
            ltx_25.validate(argparse.Namespace(model_dir=str(tmp_path)))
        assert "tokenizer.json" in capsys.readouterr().out

    def test_the_two_vaes_are_told_apart_by_tensor_count(self, tmp_path):
        # 86 against 312: the two video VAE decoders differ by 226 tensors,
        # so swapping the files is detectable without loading a single weight.
        # These restate the constants rather than deriving them — that is
        # exactly the gap test_video_vae_counts_reconcile_with_the_fixture
        # (tests/test_ltx25_keys.py) closes, by checking them against the
        # harvested upstream tensor_count instead.
        assert ltx_25.EXPECTED_TENSOR_COUNTS["vae_decoder_conv"] == 86
        assert ltx_25.EXPECTED_TENSOR_COUNTS["vae_decoder_av"] == 312
        assert ltx_25.EXPECTED_TENSOR_COUNTS["duration_head"] == 15


class TestSharedComponentsAreAllGated:
    """Every entry in SHARED_COMPONENTS must be checked by validate(), not just
    the seven video/audio/duration_head files. A pack that is otherwise
    complete but missing the text encoder weights, or either upscaler, must
    fail — not silently exit 0."""

    def test_covers_every_shared_component(self):
        assert set(ltx_25.SHARED_COMPONENTS) <= set(ltx_25.EXPECTED_TENSOR_COUNTS)

    def test_text_encoder_weight_count_is_bf16_only(self):
        # The checkpoint holds 686 tensors total (681 BF16 + 5 U8 assets);
        # sanitize_text_encoder_key refuses the 5 U8 asset keys, so the
        # written text_encoder.safetensors holds 681.
        assert ltx_25.EXPECTED_TENSOR_COUNTS["text_encoder"] == 681

    def test_upscaler_weight_counts(self):
        assert ltx_25.EXPECTED_TENSOR_COUNTS["spatial_upscaler_x2_v1_0"] == 72
        assert ltx_25.EXPECTED_TENSOR_COUNTS["temporal_upscaler_x2_v1_0"] == 72

    def test_a_pack_missing_the_text_encoder_weights_fails(self, tmp_path):
        _full_pack(tmp_path, omit=("text_encoder",))
        with pytest.raises(SystemExit):
            ltx_25.validate(argparse.Namespace(model_dir=str(tmp_path)))

    def test_a_pack_missing_the_spatial_upscaler_fails(self, tmp_path):
        _full_pack(tmp_path, omit=("spatial_upscaler_x2_v1_0",))
        with pytest.raises(SystemExit):
            ltx_25.validate(argparse.Namespace(model_dir=str(tmp_path)))

    def test_a_pack_missing_the_temporal_upscaler_fails(self, tmp_path):
        _full_pack(tmp_path, omit=("temporal_upscaler_x2_v1_0",))
        with pytest.raises(SystemExit):
            ltx_25.validate(argparse.Namespace(model_dir=str(tmp_path)))

    def test_a_genuinely_complete_pack_passes(self, tmp_path, capsys):
        _full_pack(tmp_path)
        ltx_25.validate(argparse.Namespace(model_dir=str(tmp_path)))  # must not raise
        assert "All checks passed" in capsys.readouterr().out


class TestSplit:
    def test_is_a_no_op_that_explains_itself(self, capsys):
        ltx_25.split(argparse.Namespace(model_dir="."))
        out = capsys.readouterr().out
        assert "already" in out.lower()
