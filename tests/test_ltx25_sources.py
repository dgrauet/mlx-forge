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
    _source_download_dir,
    _verify_license_if_carried,
    connector_fingerprint,
    download_size_mb,
    ltx25_should_quantize,
    output_size_mb,
    verify_embedded_license,
    write_ltx25_quantize_config,
)
from mlx_forge.upload import iter_model_files


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

    def test_accepts_a_text_differing_only_in_indentation(self, tmp_path):
        # August 2026: upstream moved the agreement to LICENSE-2_x and
        # re-rendered it flush-left; the embedded text keeps the centred
        # indentation. Word-for-word identical — must pass.
        shipped = tmp_path / "LICENSE"
        shipped.write_text("LTX-2.x Community License Agreement\nLicense date: August 11, 2026\n")
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


#: Default embedded DiT config for `_write_transformer_checkpoint` fixtures —
#: a stand-in for the real upstream `{"scheduler": {...}, "transformer": {...}}`
#: object, shaped enough (a JSON string, non-empty) to exercise convert()'s
#: real "raw_config missing/differs" checks without needing the actual 2199
#: byte upstream payload.
_DIT_CONFIG = _json.dumps({"scheduler": {"num_steps": 20}, "transformer": {"num_layers": 48}})


def _write_transformer_checkpoint(
    path, *, connector_value: float, config: str | None = _DIT_CONFIG
) -> None:
    """A minimal stand-in for one of UPSTREAM_TRANSFORMERS' real 42 GB files.

    Carries one transformer key and both connector stacks, real license
    metadata (so convert() doesn't need network to fetch or check one), and
    nothing else — enough for `classify_dit_key`/the sanitizers/`maybe_transpose`
    to run their real logic, without the weight itself needing a realistic shape.

    The embedded `license` text is AUGUST (defined above, for
    TestEmbeddedLicenceCheck) — the same fixture text already proven to
    match, once normalised, the LICENSE `_prepare` writes to `output_dir`.

    `config` mirrors upstream's `__metadata__["config"]` — a JSON string, not
    a dict — since that is the shape `convert()`'s embedded-config check reads
    (see TestEmbeddedConfigEmission). Defaults to `_DIT_CONFIG` so every
    caller that predates that check keeps working without passing one
    explicitly; pass `None` to omit it (a checkpoint with no config at all).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {
        "model.diffusion_model.transformer_blocks.0.attn1.to_q.weight": mx.zeros((4, 4)),
        "model.diffusion_model.video_embeddings_connector.proj.weight": mx.full(
            (2, 2), connector_value
        ),
        "model.diffusion_model.audio_embeddings_connector.proj.weight": mx.full(
            (2, 2), connector_value
        ),
    }
    metadata = {"license": AUGUST}
    if config is not None:
        metadata["config"] = config
    mx.save_safetensors(str(path), weights, metadata=metadata)


def _convert_args(
    tmp_path, *, variants, skip_shared, dry_run=False, config_only=False
) -> argparse.Namespace:
    return argparse.Namespace(
        output=str(tmp_path / "out"),
        quantize=False,
        bits=8,
        group_size=64,
        dry_run=dry_run,
        variant=variants,
        skip_shared=skip_shared,
        lora=[],
        config_only=config_only,
    )


class TestSkipSharedConnector:
    """`--skip-shared` must omit the connector from a delta pack (it is now a
    member of SHARED_COMPONENTS), while still computing and comparing the
    dev/distilled connector fingerprint — only the *write* is conditional.

    `download_hf_files` is stubbed to a no-op and the "download" directory is
    pre-populated directly, so these exercise convert()'s real selection and
    write-loop logic without touching the network or downloading anything.
    """

    def _stub_download(self, monkeypatch, tmp_path):
        def fake_download(repo, paths, download_dir):
            for rel in paths:
                local = download_dir / rel
                assert local.exists(), f"test fixture did not pre-populate {local}"

        monkeypatch.setattr(ltx_25, "download_hf_files", fake_download)

    def _prepare(self, tmp_path, *, connector_values):
        args = _convert_args(tmp_path, variants=["dev", "distilled"], skip_shared=True)
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True)
        # Pre-create LICENSE so `convert()` finds it already in place and
        # never calls ensure_license_file — which would otherwise reach out
        # to GitHub for real. Its text matches _TRANSFORMER_LICENSE once
        # normalised, so `_verify_license_if_carried` accepts it exactly like
        # a real run's fetched copy would.
        (output_dir / "LICENSE").write_text(
            "  LTX-2.x Community License Agreement\n  License date: August 11, 2026\n"
        )
        download_dir = ltx_25._source_download_dir(output_dir)
        _write_transformer_checkpoint(
            download_dir / UPSTREAM_TRANSFORMERS["dev"],
            connector_value=connector_values[0],
        )
        _write_transformer_checkpoint(
            download_dir / UPSTREAM_TRANSFORMERS["distilled"],
            connector_value=connector_values[1],
        )
        return args, output_dir

    def test_a_delta_run_writes_no_connector(self, tmp_path, monkeypatch):
        self._stub_download(monkeypatch, tmp_path)
        args, output_dir = self._prepare(tmp_path, connector_values=(1.0, 1.0))

        ltx_25.convert(args)

        assert not (output_dir / "connector.safetensors").exists()
        assert (output_dir / "transformer-dev.safetensors").exists()
        assert (output_dir / "transformer-distilled.safetensors").exists()

    def test_a_full_run_writes_the_connector_once(self, tmp_path, monkeypatch):
        self._stub_download(monkeypatch, tmp_path)
        args = _convert_args(tmp_path, variants=["dev", "distilled"], skip_shared=False)
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True)
        (output_dir / "LICENSE").write_text(
            "  LTX-2.x Community License Agreement\n  License date: August 11, 2026\n"
        )
        download_dir = ltx_25._source_download_dir(output_dir)
        _write_transformer_checkpoint(
            download_dir / UPSTREAM_TRANSFORMERS["dev"], connector_value=1.0
        )
        _write_transformer_checkpoint(
            download_dir / UPSTREAM_TRANSFORMERS["distilled"], connector_value=1.0
        )
        # This test's question is only whether convert() writes the connector
        # on a non-delta run; the other shared sources (vae_*, audio_vae,
        # duration_head, upscalers, text_encoder) are irrelevant to it and
        # would need a real ~5 GB fixture set to exercise honestly. Narrowing
        # `_selected_sources` to just the two transformer files keeps this
        # test about the connector specifically, the same way `--skip-shared`
        # narrows it in the code under test.
        transformer_only = [
            s for s in ltx_25.SOURCE_FILES if s.path in UPSTREAM_TRANSFORMERS.values()
        ]
        monkeypatch.setattr(
            ltx_25, "_selected_sources", lambda variants, *, skip_shared: transformer_only
        )

        ltx_25.convert(args)

        assert (output_dir / "connector.safetensors").exists()
        connector = ltx_25.load_safetensors(output_dir / "connector.safetensors")
        assert all(k.startswith("connector.") for k in connector)

    def test_fingerprint_mismatch_still_aborts_under_skip_shared(self, tmp_path, monkeypatch):
        self._stub_download(monkeypatch, tmp_path)
        args, output_dir = self._prepare(tmp_path, connector_values=(1.0, 2.0))

        with pytest.raises(SystemExit, match="different"):
            ltx_25.convert(args)

        # The abort happens while processing the second (distilled) file, so
        # the first file's non-shared output (its transformer) is already on
        # disk — proving the fingerprint check really ran against real
        # written state, not that convert() failed before doing anything.
        assert (output_dir / "transformer-dev.safetensors").exists()
        assert not (output_dir / "connector.safetensors").exists()

    def test_shared_components_constant_is_actually_consulted(self, tmp_path, monkeypatch):
        # If the write loop stopped reading SHARED_COMPONENTS (or someone
        # emptied the constant), this test — not just an assertion on the
        # constant's contents — is what would catch it: emptying it here
        # must flip the delta run's connector-writing behaviour.
        self._stub_download(monkeypatch, tmp_path)
        args, output_dir = self._prepare(tmp_path, connector_values=(1.0, 1.0))
        monkeypatch.setattr(ltx_25, "SHARED_COMPONENTS", frozenset())

        ltx_25.convert(args)

        assert (output_dir / "connector.safetensors").exists()


class TestEmbeddedConfigEmission:
    """`convert()` must write embedded_config.json from the DiT's own
    __metadata__["config"] — the same key `_verify_license_if_carried` reads
    "license" from — verbatim, and abort loudly if dev and distilled ever
    disagree rather than silently writing one variant's config for both.

    Uses the same download stub and pre-populated source directory as
    TestSkipSharedConnector, narrowed to the two transformer files so this
    doesn't need the ~5 GB of other shared fixtures a real run would touch.
    """

    def _stub_download(self, monkeypatch):
        def fake_download(repo, paths, download_dir):
            for rel in paths:
                local = download_dir / rel
                assert local.exists(), f"test fixture did not pre-populate {local}"

        monkeypatch.setattr(ltx_25, "download_hf_files", fake_download)

    def _prepare(self, tmp_path, *, configs):
        args = _convert_args(tmp_path, variants=["dev", "distilled"], skip_shared=False)
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True)
        (output_dir / "LICENSE").write_text(
            "  LTX-2.x Community License Agreement\n  License date: August 11, 2026\n"
        )
        download_dir = ltx_25._source_download_dir(output_dir)
        _write_transformer_checkpoint(
            download_dir / UPSTREAM_TRANSFORMERS["dev"], connector_value=1.0, config=configs[0]
        )
        _write_transformer_checkpoint(
            download_dir / UPSTREAM_TRANSFORMERS["distilled"],
            connector_value=1.0,
            config=configs[1],
        )
        transformer_only = [
            s for s in ltx_25.SOURCE_FILES if s.path in UPSTREAM_TRANSFORMERS.values()
        ]
        return args, output_dir, transformer_only

    def test_a_normal_conversion_writes_the_upstream_object_unchanged(self, tmp_path, monkeypatch):
        self._stub_download(monkeypatch)
        args, output_dir, transformer_only = self._prepare(
            tmp_path, configs=(_DIT_CONFIG, _DIT_CONFIG)
        )
        monkeypatch.setattr(
            ltx_25, "_selected_sources", lambda variants, *, skip_shared: transformer_only
        )

        ltx_25.convert(args)

        config_path = output_dir / "embedded_config.json"
        assert config_path.exists()
        assert json.loads(config_path.read_text()) == json.loads(_DIT_CONFIG)

    def test_differing_configs_abort_naming_the_disagreement(self, tmp_path, monkeypatch):
        self._stub_download(monkeypatch)
        other_config = _json.dumps({"scheduler": {"num_steps": 30}, "transformer": {}})
        args, output_dir, transformer_only = self._prepare(
            tmp_path, configs=(_DIT_CONFIG, other_config)
        )
        monkeypatch.setattr(
            ltx_25, "_selected_sources", lambda variants, *, skip_shared: transformer_only
        )

        with pytest.raises(SystemExit, match="dev and distilled transformers carry different"):
            ltx_25.convert(args)

        assert not (output_dir / "embedded_config.json").exists()

    def test_identical_configs_succeed_and_write_one_file(self, tmp_path, monkeypatch):
        self._stub_download(monkeypatch)
        args, output_dir, transformer_only = self._prepare(
            tmp_path, configs=(_DIT_CONFIG, _DIT_CONFIG)
        )
        monkeypatch.setattr(
            ltx_25, "_selected_sources", lambda variants, *, skip_shared: transformer_only
        )

        ltx_25.convert(args)

        assert (output_dir / "embedded_config.json").exists()

    def test_a_transformer_carrying_no_config_aborts(self, tmp_path, monkeypatch):
        self._stub_download(monkeypatch)
        args, output_dir, transformer_only = self._prepare(tmp_path, configs=(None, None))
        monkeypatch.setattr(
            ltx_25, "_selected_sources", lambda variants, *, skip_shared: transformer_only
        )

        with pytest.raises(SystemExit, match="carries no 'config'"):
            ltx_25.convert(args)


class TestConfigOnlyBackfill:
    """`--config-only` backfills embedded_config.json into an existing pack
    without downloading or reading whole checkpoints, and never touches a
    `.safetensors` file — the acceptance criterion for the two real packs
    this is meant to run against."""

    def _pack_with_local_sources(self, tmp_path, *, configs=(_DIT_CONFIG, _DIT_CONFIG)):
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True)
        # Create split_model.json to make this a valid pack
        (output_dir / "split_model.json").write_text(_json.dumps({"recipe": "ltx-2.5"}))
        download_dir = ltx_25._source_download_dir(output_dir)
        _write_transformer_checkpoint(
            download_dir / UPSTREAM_TRANSFORMERS["dev"], connector_value=1.0, config=configs[0]
        )
        _write_transformer_checkpoint(
            download_dir / UPSTREAM_TRANSFORMERS["distilled"],
            connector_value=1.0,
            config=configs[1],
        )
        return output_dir

    def test_writes_into_a_pack_that_has_none(self, tmp_path):
        output_dir = self._pack_with_local_sources(tmp_path)
        args = _convert_args(
            tmp_path, variants=["dev", "distilled"], skip_shared=False, config_only=True
        )

        ltx_25.convert(args)

        config_path = output_dir / "embedded_config.json"
        assert config_path.exists()
        assert json.loads(config_path.read_text()) == json.loads(_DIT_CONFIG)

    def test_touches_no_safetensors_file(self, tmp_path):
        output_dir = self._pack_with_local_sources(tmp_path)
        # A pre-existing weight file convert() must not read or rewrite.
        weight_path = output_dir / "transformer-dev.safetensors"
        mx.save_safetensors(str(weight_path), {"w": mx.zeros((2, 2))})
        before = weight_path.stat().st_mtime_ns

        args = _convert_args(
            tmp_path, variants=["dev", "distilled"], skip_shared=False, config_only=True
        )
        ltx_25.convert(args)

        assert weight_path.stat().st_mtime_ns == before

    def test_respects_dry_run(self, tmp_path):
        output_dir = self._pack_with_local_sources(tmp_path)
        args = _convert_args(
            tmp_path,
            variants=["dev", "distilled"],
            skip_shared=False,
            config_only=True,
            dry_run=True,
        )

        ltx_25.convert(args)

        assert not (output_dir / "embedded_config.json").exists()

    def test_fetches_the_header_over_http_when_no_local_source_is_present(
        self, tmp_path, monkeypatch
    ):
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True)
        # Create split_model.json to make this a valid pack
        (output_dir / "split_model.json").write_text(_json.dumps({"recipe": "ltx-2.5"}))
        # No _source_download_dir(output_dir) populated at all — every
        # variant must fall back to _remote_header_metadata.
        monkeypatch.setattr(
            ltx_25,
            "_remote_header_metadata",
            lambda repo, filename: {"config": _DIT_CONFIG},
        )
        args = _convert_args(
            tmp_path, variants=["dev", "distilled"], skip_shared=False, config_only=True
        )

        ltx_25.convert(args)

        config_path = output_dir / "embedded_config.json"
        assert config_path.exists()
        assert json.loads(config_path.read_text()) == json.loads(_DIT_CONFIG)

    def test_differing_configs_abort_naming_the_disagreement(self, tmp_path):
        other_config = _json.dumps({"scheduler": {"num_steps": 30}, "transformer": {}})
        output_dir = self._pack_with_local_sources(tmp_path, configs=(_DIT_CONFIG, other_config))
        args = _convert_args(
            tmp_path, variants=["dev", "distilled"], skip_shared=False, config_only=True
        )

        with pytest.raises(SystemExit, match="dev and distilled transformers carry different"):
            ltx_25.convert(args)

        assert not (output_dir / "embedded_config.json").exists()

    def test_refuses_when_target_directory_does_not_exist(self, tmp_path):
        """--config-only must refuse when the target directory does not exist."""
        output_dir = tmp_path / "out"
        # Ensure output directory doesn't exist
        assert not output_dir.exists()

        args = _convert_args(
            tmp_path, variants=["dev", "distilled"], skip_shared=False, config_only=True
        )

        with pytest.raises(SystemExit, match="split_model.json"):
            ltx_25.convert(args)

        # Guard must run before mkdir: directory should not be created
        assert not output_dir.exists()

    def test_refuses_when_target_directory_has_no_split_model(self, tmp_path):
        """--config-only must refuse when split_model.json is missing."""
        output_dir = tmp_path / "out"
        output_dir.mkdir(parents=True)
        # Directory exists but has no split_model.json
        assert not (output_dir / "split_model.json").exists()

        args = _convert_args(
            tmp_path, variants=["dev", "distilled"], skip_shared=False, config_only=True
        )

        with pytest.raises(SystemExit, match="split_model.json"):
            ltx_25.convert(args)

        # Directory should still exist but remain unchanged
        assert output_dir.exists()
        assert not (output_dir / "split_model.json").exists()
        assert not (output_dir / "embedded_config.json").exists()

    def test_succeeds_when_pack_exists_with_split_model(self, tmp_path, monkeypatch):
        """--config-only must work when split_model.json exists."""
        output_dir = self._pack_with_local_sources(tmp_path)
        # Pre-create split_model.json to make this a real pack
        (output_dir / "split_model.json").write_text(_json.dumps({"recipe": "ltx-2.5"}))

        args = _convert_args(
            tmp_path, variants=["dev", "distilled"], skip_shared=False, config_only=True
        )

        ltx_25.convert(args)

        config_path = output_dir / "embedded_config.json"
        assert config_path.exists()
        assert json.loads(config_path.read_text()) == json.loads(_DIT_CONFIG)

    def test_dry_run_config_only_refuses_without_split_model(self, tmp_path):
        """--dry-run --config-only must refuse the same way as normal --config-only."""
        output_dir = tmp_path / "out"
        assert not output_dir.exists()

        args = _convert_args(
            tmp_path,
            variants=["dev", "distilled"],
            skip_shared=False,
            config_only=True,
            dry_run=True,
        )

        with pytest.raises(SystemExit, match="split_model.json"):
            ltx_25.convert(args)

        # Guard must run before anything is created
        assert not output_dir.exists()


def _pack(tmp_path, **files):
    import mlx.core as mx

    for name, weights in files.items():
        mx.save_safetensors(str(tmp_path / f"{name}.safetensors"), weights)
    (tmp_path / "split_model.json").write_text(_json.dumps({"recipe": "ltx-2.5"}))
    return tmp_path


def _full_pack(tmp_path, *, omit=(), quantized_counts=None):
    """A pack that satisfies every check `validate()` currently runs, so any
    test built on it isolates exactly the gap it targets: if the pack is
    otherwise complete and only `omit` is missing, a still-passing run means
    the omitted component is not actually gated on.

    Keys are built the way `process_component` actually writes them —
    `f"{component}.{sanitized}"` — rather than a shape that dodges the real
    condition. A fixture with keys like "w0.weight" never carries the
    component's own name, so it can never trip (or validate a fix for)
    `_validate_no_leaked_pytorch_prefix`, which looks specifically for the
    component prefix appearing twice.

    `quantized_counts` extends this to the quantized shape a real q8/q4 pack
    actually has: `{"text_encoder": 328}` replaces 328 of that component's
    plain `w{i}.weight` tensors with real quantized triplets (`.weight` +
    `.scales` + `.biases`), and writes `quantize_config.json` recording the
    component as quantized — the same two facts (file shape, and the config
    that tells `validate()` to expect it) a real `--quantize` conversion
    produces together. Earlier fixtures here only ever built unquantized
    packs, which is exactly why the bf16-only `EXPECTED_TENSOR_COUNTS` check
    never saw a quantized pack until a real conversion did.
    """
    import mlx.core as mx

    from mlx_forge.quantize import write_quantize_config

    quantized_counts = quantized_counts or {}

    # The seven components EXPECTED_TENSOR_COUNTS covered before this fix,
    # each with the right tensor count and a realistic single component
    # prefix (not doubled — that would trip
    # `_validate_no_leaked_pytorch_prefix` for audio_vae/duration_head).
    counts = dict(ltx_25.EXPECTED_TENSOR_COUNTS)
    for component, count in counts.items():
        if component in omit:
            continue
        n_quantized = quantized_counts.get(component, 0)
        n_plain = count - n_quantized
        weights = {f"{component}.w{i}.weight": mx.zeros((2, 2)) for i in range(n_plain)}
        for i in range(n_quantized):
            # The real q8 text encoder's quantized tensors all live under
            # model.layers.* — validate() checks exactly that.
            if component == "text_encoder":
                key = f"{component}.model.layers.{i}.q.weight"
            else:
                key = f"{component}.q{i}.weight"
            weights[key] = mx.zeros((2, 2), dtype=mx.int8)
            weights[f"{key}.scales"] = mx.zeros((2,))
            weights[f"{key}.biases"] = mx.zeros((2,))
        mx.save_safetensors(str(tmp_path / f"{component}.safetensors"), weights)

    if quantized_counts:
        # A quantized pack always quantizes the transformer alongside
        # whatever shared components were quantized (see the transformer
        # write below) — record it here too, not just the components
        # `quantized_counts` names explicitly.
        components = {name: ltx_25.QUANTIZED_COMPONENTS[name] for name in quantized_counts}
        components.setdefault("transformer", ltx_25.QUANTIZED_COMPONENTS["transformer"])
        write_quantize_config(tmp_path, bits=8, group_size=64, components=components)

    # Assets are always written, even when `omit` drops the text_encoder
    # weight file itself — otherwise a missing-weights test would fail for
    # the wrong reason (missing assets, already covered by
    # test_a_missing_text_encoder_asset_fails) rather than the one it targets.
    for filename in ltx_25.TEXT_ENCODER_ASSET_FILES:
        content = "{}" if filename.endswith(".json") else "not json"
        (tmp_path / filename).write_text(content)

    if "transformer" not in omit:
        # A quantized pack always quantizes the transformer alongside
        # whatever else is quantized (write_ltx25_quantize_config records
        # both unless --skip-shared drops text_encoder) — the "Transformer
        # Variants" section below assumes exactly that once
        # quantize_config.json exists, so this must mirror it or the
        # unrelated .scales/.biases check fails for the wrong reason.
        # All 48 blocks, like the real packs: validate() counts distinct
        # transformer_blocks indices, so a one-block fixture would fail the
        # structural check for the wrong reason.
        if quantized_counts:
            transformer_weights = {}
            for i in range(ltx_25.TRANSFORMER_BLOCK_COUNT):
                key = f"transformer_blocks.{i}.attn1.to_q.weight"
                transformer_weights[key] = mx.zeros((4, 4), dtype=mx.int8)
                transformer_weights[f"{key}.scales"] = mx.zeros((4,))
                transformer_weights[f"{key}.biases"] = mx.zeros((4,))
        else:
            transformer_weights = {
                f"transformer_blocks.{i}.attn1.to_q.weight": mx.zeros((4, 4))
                for i in range(ltx_25.TRANSFORMER_BLOCK_COUNT)
            }
        # Real packs carry the upstream model_version in the safetensors
        # header (validate checks it); mirror that.
        mx.save_safetensors(
            str(tmp_path / ltx_25.VARIANT_FILENAMES["dev"]),
            transformer_weights,
            metadata={"model_version": "2.5.0"},
        )

    (tmp_path / "split_model.json").write_text(
        _json.dumps({"recipe": "ltx-2.5", "transformer_variants": ["dev"]})
    )
    # Real packs carry embedded_config.json with model_version at the root
    # (validate checks both); the transformer section mirrors the lowercased
    # norm-type casing the recipe emits.
    (tmp_path / "embedded_config.json").write_text(
        _json.dumps(
            {
                "transformer": {"text_encoder_norm_type": "per_token_rms"},
                "scheduler": {},
                "model_version": "2.5.0",
            }
        )
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


class TestLeakedPrefixDetection:
    """RED/GREEN coverage for `_validate_no_leaked_pytorch_prefix`.

    Every key `process_component` writes legitimately starts with its own
    component prefix once (`f"{component}.{sanitized}"`), so a fixture built
    that way — like `_full_pack` — must pass. Only a *second* occurrence of
    the prefix, left behind by a sanitizer that failed to strip the upstream
    PyTorch prefix, is a real defect: this class pins that a key like
    `audio_vae.audio_vae.decoder...` is caught, while the correct single-
    prefix form is not.
    """

    def test_audio_vae_key_with_a_leaked_prefix_fails(self, tmp_path):
        pack = _full_pack(tmp_path)
        count = ltx_25.EXPECTED_TENSOR_COUNTS["audio_vae"]
        weights = {f"audio_vae.w{i}.weight": mx.zeros((2, 2)) for i in range(count - 1)}
        weights["audio_vae.audio_vae.decoder.conv_in.conv.bias"] = mx.zeros((2, 2))
        mx.save_safetensors(str(pack / "audio_vae.safetensors"), weights)
        with pytest.raises(SystemExit):
            ltx_25.validate(argparse.Namespace(model_dir=str(pack)))

    def test_duration_head_key_with_a_leaked_prefix_fails(self, tmp_path):
        pack = _full_pack(tmp_path)
        count = ltx_25.EXPECTED_TENSOR_COUNTS["duration_head"]
        weights = {f"duration_head.w{i}.weight": mx.zeros((2, 2)) for i in range(count - 1)}
        weights["duration_head.duration_head.attention_pooler.cross_attn.in_proj_bias"] = mx.zeros(
            (2, 2)
        )
        mx.save_safetensors(str(pack / "duration_head.safetensors"), weights)
        with pytest.raises(SystemExit):
            ltx_25.validate(argparse.Namespace(model_dir=str(pack)))

    def test_a_vocoder_key_with_a_leaked_prefix_fails(self, tmp_path):
        # vocoder.vocoder.ups.5.weight is exactly the defect that shipped:
        # the sanitizer failed to flatten the main generator (itself named
        # "vocoder") to the component root, and the loader crashed on 667
        # unknown parameters. validate() must catch it.
        pack = _full_pack(tmp_path)
        count = ltx_25.EXPECTED_TENSOR_COUNTS["vocoder"]
        weights = {f"vocoder.w{i}.weight": mx.zeros((2, 2)) for i in range(count - 1)}
        weights["vocoder.vocoder.ups.5.weight"] = mx.zeros((2, 2))
        mx.save_safetensors(str(pack / "vocoder.safetensors"), weights)
        with pytest.raises(SystemExit):
            ltx_25.validate(argparse.Namespace(model_dir=str(pack)))


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


class TestQuantizedComponentCounts:
    """EXPECTED_TENSOR_COUNTS holds bf16 counts. Quantization turns each
    quantized tensor into three entries (weight + .scales + .biases), so a
    quantized pack legitimately holds more tensors than the bf16 count — the
    defect a real q8 text_encoder pack (681 -> 1337 tensors) uncovered.
    """

    def test_a_quantized_text_encoder_with_the_derived_count_passes(self, tmp_path, capsys):
        _full_pack(tmp_path, quantized_counts={"text_encoder": 328})
        ltx_25.validate(argparse.Namespace(model_dir=str(tmp_path)))  # must not raise
        assert "All checks passed" in capsys.readouterr().out

    def test_a_quantized_pack_still_holding_the_bf16_count_fails(self, tmp_path):
        # quantize_config.json claims text_encoder was quantized, but the
        # file itself holds exactly the bf16 count — i.e. nothing was
        # actually quantized. The plain bf16 check must not let this pass.
        from mlx_forge.quantize import write_quantize_config

        _full_pack(tmp_path)
        write_quantize_config(
            tmp_path,
            bits=8,
            group_size=64,
            components={"text_encoder": ltx_25.QUANTIZED_COMPONENTS["text_encoder"]},
        )
        with pytest.raises(SystemExit):
            ltx_25.validate(argparse.Namespace(model_dir=str(tmp_path)))

    def test_a_quantized_pack_with_mismatched_scales_and_biases_fails(self, tmp_path):
        _full_pack(tmp_path, quantized_counts={"text_encoder": 328})
        weights = ltx_25.load_safetensors(tmp_path / "text_encoder.safetensors")
        # Drop one .biases key so scales (328) and biases (327) disagree —
        # the total count alone would hide this.
        del weights["text_encoder.model.layers.0.q.weight.biases"]
        mx.save_safetensors(str(tmp_path / "text_encoder.safetensors"), weights)
        with pytest.raises(SystemExit):
            ltx_25.validate(argparse.Namespace(model_dir=str(tmp_path)))

    def test_an_unquantized_pack_still_passes_with_the_plain_count(self, tmp_path, capsys):
        # Regression guard: the unquantized path must stay exactly as it was.
        _full_pack(tmp_path)
        ltx_25.validate(argparse.Namespace(model_dir=str(tmp_path)))  # must not raise
        assert "All checks passed" in capsys.readouterr().out


class TestSplit:
    def test_is_a_no_op_that_explains_itself(self, capsys):
        ltx_25.split(argparse.Namespace(model_dir="."))
        out = capsys.readouterr().out
        assert "already" in out.lower()


class TestSourceDownloadDir:
    """`convert()` must not download upstream checkpoints into output_dir.

    `upload.iter_model_files` is the single source of truth for "what the
    model is" and walks the output directory recursively with no exclusion
    for a source-checkpoint cache, so anything convert() writes under
    output_dir gets uploaded. The gated LTX-2.5 checkpoints (76+ GB) must
    live somewhere iter_model_files never looks.
    """

    def test_download_dir_is_not_inside_output_dir(self, tmp_path):
        output_dir = tmp_path / "ltx-2.5-mlx"
        download_dir = _source_download_dir(output_dir)
        # Assert the relationship directly — not by string-matching a
        # hardcoded name like ".source" — so the test still catches a
        # regression even if the sibling's naming scheme changes.
        assert download_dir != output_dir
        assert output_dir not in download_dir.parents
        assert download_dir.parent == output_dir.parent

    def test_download_dir_is_a_sibling_for_an_arbitrary_output_path(self, tmp_path):
        # --output can point anywhere; the sibling must be derived from
        # whatever output_dir is, not from a hardcoded "models/" prefix.
        output_dir = tmp_path / "some" / "nested" / "custom-output"
        download_dir = _source_download_dir(output_dir)
        assert output_dir not in download_dir.parents
        assert download_dir.parent == output_dir.parent


class TestConvertedPackHasNoSourceCheckpoints:
    """Regression test for the source-in-output-dir defect.

    A fixture that never had a source-checkpoint directory in the first
    place could pass this test even with the defect still present — see
    `_full_pack`'s docstring for the same lesson applied to leaked-prefix
    detection. So this fixture reproduces the real shape `convert()`
    produces: a pack directory plus its sibling `-src` download directory
    sitting next to it, both populated, mirroring what a real conversion
    leaves on disk.
    """

    def test_iter_model_files_excludes_the_sibling_source_dir(self, tmp_path):
        output_dir = tmp_path / "ltx-2.5-mlx"
        output_dir.mkdir()
        _full_pack(output_dir)

        download_dir = _source_download_dir(output_dir)
        download_dir.mkdir()
        diffusion_models = download_dir / "diffusion_models"
        diffusion_models.mkdir()
        checkpoint = diffusion_models / "ltx-2.5-22b-dev-transformer-bf16.safetensors"
        checkpoint.write_bytes(b"not a real checkpoint, just large enough to matter if counted")

        files = iter_model_files(output_dir)
        relative_paths = {p.relative_to(output_dir).as_posix() for p in files}
        assert not any("-src" in part for path in relative_paths for part in path.split("/"))
        assert all(download_dir not in p.parents for p in files)
