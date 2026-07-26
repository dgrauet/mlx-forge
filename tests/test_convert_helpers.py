"""Tests for the argument and checkpoint-loading helpers shared by recipes."""

from __future__ import annotations

import argparse
import importlib

import pytest

from mlx_forge.convert import add_common_convert_args, load_torch_state_dict
from mlx_forge.recipes import AVAILABLE_RECIPES

torch = pytest.importorskip("torch")

RECIPE_NAMES = sorted(AVAILABLE_RECIPES)

#: Recipes whose --bits default deliberately differs from the 8-bit norm.
BITS_DEFAULT_OVERRIDES = {"ernie-image-pe": 4}


class TestAddCommonConvertArgs:
    def _parse(self, **kwargs):
        parser = argparse.ArgumentParser()
        add_common_convert_args(
            parser, output_default="models/x-mlx[-q<bits>]", quantize_help="Quantize x", **kwargs
        )
        return parser

    def test_defaults(self):
        ns = self._parse().parse_args([])
        assert ns.output is None
        assert ns.quantize is False
        assert ns.bits == 8
        assert ns.group_size == 64
        assert ns.dry_run is False

    def test_values_parse(self):
        ns = self._parse().parse_args(
            ["--output", "out", "--quantize", "--bits", "4", "--group-size", "32", "--dry-run"]
        )
        assert (ns.output, ns.quantize, ns.bits, ns.group_size, ns.dry_run) == (
            "out",
            True,
            4,
            32,
            True,
        )

    def test_bits_default_override_is_reflected_in_help(self):
        parser = self._parse(bits_default=4)
        assert parser.parse_args([]).bits == 4
        bits = next(a for a in parser._actions if a.dest == "bits")
        assert "default: 4" in bits.help

    def test_rejects_unsupported_bit_width(self):
        with pytest.raises(SystemExit):
            self._parse().parse_args(["--bits", "6"])

    def test_output_default_appears_in_help(self):
        output = next(a for a in self._parse()._actions if a.dest == "output")
        assert "models/x-mlx[-q<bits>]" in output.help


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_every_recipe_exposes_the_common_block(recipe_name: str):
    """The shared block must stay uniform across recipes (defaults, not wording)."""
    module = importlib.import_module(AVAILABLE_RECIPES[recipe_name])
    parser = argparse.ArgumentParser()
    module.add_convert_args(parser)
    ns = parser.parse_args([])

    assert ns.output is None
    assert ns.quantize is False
    assert ns.group_size == 64
    assert ns.dry_run is False
    assert ns.bits == BITS_DEFAULT_OVERRIDES.get(recipe_name, 8)

    bits = next(a for a in parser._actions if a.dest == "bits")
    assert bits.choices == [4, 8]


class TestLoadTorchStateDict:
    def _write(self, tmp_path, obj, name="ckpt.pt"):
        path = tmp_path / name
        torch.save(obj, str(path))
        return path

    def test_round_trip(self, tmp_path):
        path = self._write(tmp_path, {"a": torch.ones(2, 3)})
        state = load_torch_state_dict(path)
        assert list(state) == ["a"]
        assert state["a"].shape == (2, 3)

    def test_mmap_round_trip(self, tmp_path):
        path = self._write(tmp_path, {"a": torch.ones(4)})
        state = load_torch_state_dict(path, mmap=True)
        assert state["a"].sum().item() == 4.0

    def test_nested_containers_survive(self, tmp_path):
        """Recipes unwrap containers themselves — the loader must not flatten."""
        path = self._write(tmp_path, {"target_encoder": {"w": torch.zeros(2)}, "epoch": 3})
        state = load_torch_state_dict(path)
        assert state["epoch"] == 3
        assert "w" in state["target_encoder"]

    def test_weights_only_true_rejects_pickled_objects(self, tmp_path):
        """The safe default must refuse a checkpoint carrying arbitrary objects."""
        path = self._write(tmp_path, {"cfg": argparse.Namespace(x=1)})
        with pytest.raises(Exception):  # noqa: B017 - torch raises its own UnpicklingError
            load_torch_state_dict(path)

    def test_weights_only_false_accepts_them(self, tmp_path):
        """The vjepa-2.1 path: opt in explicitly for a trusted checkpoint."""
        path = self._write(tmp_path, {"cfg": argparse.Namespace(x=1)})
        state = load_torch_state_dict(path, weights_only=False)
        assert state["cfg"].x == 1

    def test_label_is_printed(self, tmp_path, capsys):
        path = self._write(tmp_path, {"a": torch.ones(1)})
        load_torch_state_dict(path, label="encoder from vitl.pt")
        assert "encoder from vitl.pt" in capsys.readouterr().out

    def test_missing_torch_exits_with_install_hint(self, tmp_path, monkeypatch):
        path = self._write(tmp_path, {"a": torch.ones(1)})
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

        def fake_import(name, *a, **kw):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *a, **kw)

        monkeypatch.setattr("builtins.__import__", fake_import)
        with pytest.raises(SystemExit) as exc_info:
            load_torch_state_dict(path)
        assert "PyTorch is required" in str(exc_info.value)


class TestShardFilenames:
    """No recipe calls this since fish-s2-pro was removed; tested so it can't rot."""

    def test_single_shard(self):
        from mlx_forge.convert import shard_filenames

        assert shard_filenames(1) == [
            "model-00001-of-00001.safetensors",
            "model.safetensors.index.json",
        ]

    def test_multi_shard_is_zero_padded(self):
        from mlx_forge.convert import shard_filenames

        names = shard_filenames(3)
        assert names[:3] == [
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ]

    def test_index_file_is_last(self):
        from mlx_forge.convert import shard_filenames

        assert shard_filenames(2)[-1] == "model.safetensors.index.json"

    def test_custom_prefix(self):
        from mlx_forge.convert import shard_filenames

        names = shard_filenames(2, prefix="diffusion_pytorch_model")
        assert names[0] == "diffusion_pytorch_model-00001-of-00002.safetensors"
        assert names[-1] == "diffusion_pytorch_model.safetensors.index.json"

    def test_matches_what_load_weights_expects(self, tmp_path):
        """The generated index name is the one load_weights() looks for."""
        import json

        import mlx.core as mx

        from mlx_forge.convert import load_weights, shard_filenames

        names = shard_filenames(2)
        for shard in names[:-1]:
            mx.save_safetensors(str(tmp_path / shard), {f"w_{shard[6:11]}": mx.ones((2, 2))})
        (tmp_path / names[-1]).write_text(
            json.dumps({"weight_map": {f"w_{s[6:11]}": s for s in names[:-1]}})
        )

        weights = load_weights(tmp_path)
        assert set(weights) == {"w_00001", "w_00002"}


class TestQuantizeComponentFilename:
    """`filename=` exists for files not named after their component (LTX variants)."""

    def _model(self, tmp_path, name):
        import mlx.core as mx

        mx.save_safetensors(
            str(tmp_path / name),
            {"blocks.0.attn.weight": mx.ones((64, 128)), "blocks.0.norm.weight": mx.ones((64,))},
        )

    def test_override_targets_the_named_file(self, tmp_path):
        from mlx_forge.convert import load_safetensors, quantize_component
        from mlx_forge.quantize import default_should_quantize

        self._model(tmp_path, "transformer-distilled.safetensors")

        quantize_component(
            tmp_path,
            "transformer (distilled)",
            bits=8,
            should_quantize=default_should_quantize,
            filename="transformer-distilled.safetensors",
        )

        out = load_safetensors(tmp_path / "transformer-distilled.safetensors")
        assert "blocks.0.attn.scales" in out, "the override file was not quantized"

    def test_without_override_the_component_name_is_used(self, tmp_path):
        from mlx_forge.convert import load_safetensors, quantize_component
        from mlx_forge.quantize import default_should_quantize

        self._model(tmp_path, "encoder.safetensors")

        quantize_component(tmp_path, "encoder", bits=8, should_quantize=default_should_quantize)

        assert "blocks.0.attn.scales" in load_safetensors(tmp_path / "encoder.safetensors")

    def test_missing_file_warns_instead_of_crashing(self, tmp_path, capsys):
        """vjepa-2.0's local copy lacked this guard and raised instead."""
        from mlx_forge.convert import quantize_component
        from mlx_forge.quantize import default_should_quantize

        quantize_component(tmp_path, "absent", bits=8, should_quantize=default_should_quantize)

        assert "not found" in capsys.readouterr().out


class TestDefaultOutputDir:
    def test_unquantized(self):
        from mlx_forge.convert import default_output_dir

        assert str(default_output_dir("ltx-2.3", quantize=False, bits=8)) == "models/ltx-2.3-mlx"

    def test_quantized_encodes_bits(self):
        from mlx_forge.convert import default_output_dir

        assert str(default_output_dir("ltx-2.3", quantize=True, bits=4)) == "models/ltx-2.3-mlx-q4"

    def test_bits_ignored_when_not_quantizing(self):
        from pathlib import Path

        from mlx_forge.convert import default_output_dir

        assert default_output_dir("x", quantize=False, bits=4) == Path("models/x-mlx")

    def test_upload_can_parse_the_bits_back_out(self):
        """derive_repo_id() recovers the bit width from this suffix."""
        from unittest.mock import MagicMock

        from mlx_forge.convert import default_output_dir
        from mlx_forge.upload import derive_repo_id

        api = MagicMock()
        api.whoami.return_value = {"name": "u"}
        out = default_output_dir("vjepa-2.0-vitl", quantize=True, bits=8)
        assert derive_repo_id({}, out, api=api) == "u/vjepa-2.0-vitl-mlx-q8"


class TestWriteSplitModel:
    def test_writes_the_payload_verbatim(self, tmp_path):
        import json

        from mlx_forge.convert import write_split_model

        info = {"format": "split", "source": "Org/Model", "components": ["a", "b"]}
        path = write_split_model(tmp_path, info)

        assert path.name == "split_model.json"
        assert json.loads(path.read_text()) == info

    def test_accepts_the_other_recipe_schemas(self, tmp_path):
        """Content is per-recipe on purpose — the helper must not impose one."""
        import json

        from mlx_forge.convert import write_split_model

        flat = {"encoder": "encoder.safetensors", "predictor": "predictor.safetensors"}
        assert json.loads(write_split_model(tmp_path, flat).read_text()) == flat


class TestPrintOutputSummary:
    def test_lists_nested_files(self, tmp_path, capsys):
        from mlx_forge.convert import print_output_summary

        (tmp_path / "a.safetensors").write_bytes(b"x" * 2048)
        nested = tmp_path / "tokenizer"
        nested.mkdir()
        (nested / "tokenizer.json").write_bytes(b"y")

        print_output_summary(tmp_path)

        out = capsys.readouterr().out
        assert "a.safetensors" in out
        assert "tokenizer/tokenizer.json" in out, "files in subdirectories must be listed"


class TestCopyRequiredFilesKeepTree:
    def test_keep_tree_preserves_that_directory_only(self, tmp_path):
        from mlx_forge.convert import copy_required_files

        files = [
            "tokenizer/tokenizer.json",
            "scheduler/scheduler_config.json",
            "model_index.json",
        ]
        src = tmp_path / "src"
        for f in files:
            p = src / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        out = tmp_path / "out"
        out.mkdir()

        copy_required_files(src, out, files, flatten=True, keep_tree={"tokenizer"})

        assert (out / "tokenizer" / "tokenizer.json").exists()
        assert (out / "scheduler_scheduler_config.json").exists()
        assert (out / "model_index.json").exists()
