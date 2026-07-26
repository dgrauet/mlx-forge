"""Tests for the VOID recipe.

VOID reads plain safetensors from a local --source, so unlike the other
recipes it can be driven end to end here: convert, quantize and validate all
run against synthetic tensors, no network and no torch.
"""

import argparse
import json

import mlx.core as mx
import pytest

from mlx_forge.convert import load_safetensors
from mlx_forge.recipes import void_model
from mlx_forge.recipes.void_model import (
    PASS_FILES,
    convert,
    sanitize_key,
    should_quantize_transformer,
    validate,
)


def _source_dir(tmp_path):
    """A source tree shaped like the released VOID checkpoint, in miniature."""
    src = tmp_path / "src"
    src.mkdir()
    for pass_file in PASS_FILES:
        mx.save_safetensors(
            str(src / pass_file),
            {
                "transformer_blocks.0.attn1.to_q.weight": mx.ones((128, 128)),
                "transformer_blocks.0.attn1.to_q.bias": mx.zeros((128,)),
                "transformer_blocks.0.norm1.weight": mx.ones((128,)),
                "patch_embed.proj.weight": mx.ones((128, 384)),
                "time_embed.linear_1.weight": mx.ones((128, 128)),
                "proj_out.weight": mx.ones((64, 128)),
            },
        )
    return src


def _args(source, output, **kwargs):
    return argparse.Namespace(
        source=str(source),
        output=str(output),
        quantize=kwargs.get("quantize", False),
        bits=kwargs.get("bits", 8),
        group_size=kwargs.get("group_size", 64),
        dry_run=kwargs.get("dry_run", False),
    )


class TestSanitizeKey:
    @pytest.mark.parametrize(
        "key",
        [
            "transformer_blocks.0.attn1.to_q.weight",
            "patch_embed.proj.weight",
            "norm_out.linear.bias",
        ],
    )
    def test_keys_pass_through_unchanged(self, key):
        assert sanitize_key(key) == key


class TestQuantizationScope:
    def test_block_linear_quantized(self):
        assert should_quantize_transformer(
            "transformer_blocks.0.attn1.to_q.weight", mx.zeros((128, 128))
        )

    def test_bias_rejected(self):
        assert (
            should_quantize_transformer("transformer_blocks.0.attn1.to_q.bias", mx.zeros((128,)))
            is False
        )

    @pytest.mark.parametrize(
        "key",
        [
            "patch_embed.proj.weight",
            "time_embed.linear_1.weight",
            "timestep_embedder.weight",
            "norm_out.linear.weight",
            "pos_embed.weight",
        ],
    )
    def test_sensitive_layers_excluded(self, key):
        assert should_quantize_transformer(key, mx.zeros((128, 128))) is False

    def test_final_proj_out_excluded(self):
        assert should_quantize_transformer("proj_out.weight", mx.zeros((64, 128))) is False

    def test_proj_out_inside_a_block_is_quantized(self):
        """The exclusion targets the final unpatchify projection, not block internals."""
        assert should_quantize_transformer(
            "transformer_blocks.0.attn1.proj_out.weight", mx.zeros((128, 128))
        )


class TestConvertEndToEnd:
    def test_writes_every_pass_and_config(self, tmp_path):
        out = tmp_path / "out"
        convert(_args(_source_dir(tmp_path), out))

        for pass_file in PASS_FILES:
            assert (out / pass_file).exists()
        config = json.loads((out / "config.json").read_text())
        assert config["passes"] == ["void_pass1", "void_pass2"]

    def test_values_round_trip(self, tmp_path):
        out = tmp_path / "out"
        convert(_args(_source_dir(tmp_path), out))

        saved = load_safetensors(out / PASS_FILES[0])
        assert saved["transformer_blocks.0.attn1.to_q.weight"].sum().item() == 128 * 128
        assert saved["patch_embed.proj.weight"].sum().item() == 128 * 384

    def test_every_weight_is_materialized_before_save(self, tmp_path, monkeypatch):
        """A lazy tensor written to safetensors saves as zeros under memory pressure.

        Checked by spying on _materialize rather than by inspecting the saved
        values: at this fixture's size nothing evicts, so a round-trip assertion
        would pass even with the materialize call removed (verified).
        """
        seen: list[int] = []
        real = void_model._materialize

        def spy(*tensors):
            seen.extend(id(t) for t in tensors)
            real(*tensors)

        monkeypatch.setattr(void_model, "_materialize", spy)
        loaded: dict[str, mx.array] = {}
        real_load = void_model.load_safetensors

        def capture(path):
            weights = real_load(path)
            loaded.update(weights)
            return weights

        monkeypatch.setattr(void_model, "load_safetensors", capture)

        convert(_args(_source_dir(tmp_path), tmp_path / "out"))

        assert loaded, "fixture did not load any weight"
        assert [k for k, v in loaded.items() if id(v) not in seen] == []

    def test_keys_are_preserved(self, tmp_path):
        out = tmp_path / "out"
        convert(_args(_source_dir(tmp_path), out))

        saved = load_safetensors(out / PASS_FILES[0])
        assert "transformer_blocks.0.attn1.to_q.weight" in saved
        assert "patch_embed.proj.weight" in saved

    def test_dry_run_without_source_does_not_download(self, tmp_path, monkeypatch):
        """--dry-run must not touch the network: convert() downloaded first and
        checked args.dry_run afterwards, so previewing a plan started a
        multi-GB fetch."""
        called = []
        monkeypatch.setattr(void_model, "download_hf_files", lambda *a, **k: called.append(a))

        args = _args(tmp_path / "unused", tmp_path / "out", dry_run=True)
        args.source = None
        void_model.convert(args)

        assert called == [], "dry run hit the network"
        assert not (tmp_path / "out").exists()

    def test_dry_run_writes_nothing(self, tmp_path, capsys):
        out = tmp_path / "out"
        convert(_args(_source_dir(tmp_path), out, dry_run=True))

        assert not out.exists()
        assert "DRY RUN" in capsys.readouterr().out

    def test_missing_source_dir_exits(self, tmp_path):
        args = _args(tmp_path / "nope", tmp_path / "out")
        with pytest.raises(SystemExit):
            convert(args)


class TestConvertQuantized:
    def test_quantized_output_carries_scales_and_config(self, tmp_path):
        out = tmp_path / "out"
        convert(_args(_source_dir(tmp_path), out, quantize=True, bits=8))

        qconfig = json.loads((out / "quantize_config.json").read_text())
        assert qconfig["quantization"]["bits"] == 8

        saved = load_safetensors(out / PASS_FILES[0])
        assert "transformer_blocks.0.attn1.to_q.scales" in saved
        assert "transformer_blocks.0.attn1.to_q.biases" in saved

    def test_excluded_layers_stay_unquantized(self, tmp_path):
        out = tmp_path / "out"
        convert(_args(_source_dir(tmp_path), out, quantize=True, bits=8))

        saved = load_safetensors(out / PASS_FILES[0])
        assert "patch_embed.proj.scales" not in saved
        assert "time_embed.linear_1.scales" not in saved
        # ...and their values survive the quantization pass intact
        assert saved["patch_embed.proj.weight"].sum().item() == 128 * 384

    def test_norms_and_biases_survive(self, tmp_path):
        out = tmp_path / "out"
        convert(_args(_source_dir(tmp_path), out, quantize=True, bits=4))

        saved = load_safetensors(out / PASS_FILES[0])
        assert saved["transformer_blocks.0.norm1.weight"].sum().item() == 128


class TestValidate:
    def test_structural_checks_pass_on_a_converted_model(self, tmp_path, capsys):
        """File layout, patch-embed width and the no-conv rule hold on real output.

        The overall run still fails: validate pins the released model's 1024
        base keys per pass, which a miniature fixture cannot have. That check
        is doing its job, so this asserts the per-check outcomes instead of a
        clean exit.
        """
        out = tmp_path / "out"
        convert(_args(_source_dir(tmp_path), out))

        with pytest.raises(SystemExit):
            validate(argparse.Namespace(model_dir=str(out)))

        report = capsys.readouterr().out
        assert "void_pass1.safetensors exists" in report
        assert "patch_embed input dim == 384" in report
        assert "all weights are 1D/2D" in report
        # the only failures are the key-count checks, one per pass
        assert "2 checks failed" in report

    def test_detects_a_missing_pass_file(self, tmp_path, capsys):
        out = tmp_path / "out"
        convert(_args(_source_dir(tmp_path), out))
        (out / PASS_FILES[1]).unlink()

        with pytest.raises(SystemExit):
            validate(argparse.Namespace(model_dir=str(out)))
        assert PASS_FILES[1] in capsys.readouterr().out

    def test_rejects_a_missing_directory(self, tmp_path):
        with pytest.raises(SystemExit):
            validate(argparse.Namespace(model_dir=str(tmp_path / "nope")))
