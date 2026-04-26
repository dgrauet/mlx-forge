"""Tests for LTX-2.3 delta-mode convert/validate."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from mlx_forge.recipes.ltx_23 import add_convert_args


class TestSkipSharedFlag:
    def test_default_is_false(self):
        parser = argparse.ArgumentParser()
        add_convert_args(parser)
        args = parser.parse_args([])
        assert args.skip_shared is False

    def test_flag_sets_true(self):
        parser = argparse.ArgumentParser()
        add_convert_args(parser)
        args = parser.parse_args(["--skip-shared"])
        assert args.skip_shared is True


def _make_convert_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_convert_args(parser)
    args = parser.parse_args([])
    args.checkpoint = None
    args.variant = ["distilled-1.1"]
    args.output = str(tmp_path)
    args.spatial_upscaler = []
    args.temporal_upscaler = []
    args.spatial_upscaler_checkpoint = []
    args.temporal_upscaler_checkpoint = []
    args.lora = []
    args.quantize = False
    args.bits = 8
    args.group_size = 64
    args.dry_run = False
    args.skip_shared = False
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class TestSkipSharedConvertEffect:
    def test_skip_shared_forces_is_first_false(self, tmp_path):
        """When skip_shared is True, _convert_variant must be called with is_first=False."""
        from mlx_forge.recipes import ltx_23

        args = _make_convert_args(tmp_path, skip_shared=True)
        captured: list[bool] = []

        def fake_convert_variant(args, variant, output_dir, *, is_first):
            captured.append(is_first)
            return ({"model_version": "2.3.0", "embedded_config": {}}, True, 0)

        with patch.object(ltx_23, "_convert_variant", side_effect=fake_convert_variant):
            ltx_23.convert(args)

        assert captured == [False]  # is_first must be forced to False with skip_shared

    def test_split_model_json_has_delta_flag(self, tmp_path):
        from mlx_forge.recipes import ltx_23

        args = _make_convert_args(tmp_path, skip_shared=True)

        def fake_convert_variant(args, variant, output_dir, *, is_first):
            return ({"model_version": "2.3.0", "embedded_config": {}}, True, 0)

        with patch.object(ltx_23, "_convert_variant", side_effect=fake_convert_variant):
            ltx_23.convert(args)

        split = json.loads((tmp_path / "split_model.json").read_text())
        assert split["delta"] is True
        assert split["components"] == ["transformer-distilled-1.1"]
        assert split["transformer_variants"] == ["distilled-1.1"]

    def test_normal_mode_no_delta_flag(self, tmp_path):
        from mlx_forge.recipes import ltx_23

        args = _make_convert_args(tmp_path, skip_shared=False)

        def fake_convert_variant(args, variant, output_dir, *, is_first):
            return ({"model_version": "2.3.0", "embedded_config": {}}, True, 0)

        with patch.object(ltx_23, "_convert_variant", side_effect=fake_convert_variant):
            ltx_23.convert(args)

        split = json.loads((tmp_path / "split_model.json").read_text())
        assert "delta" not in split or split["delta"] is False


class TestValidateDeltaMode:
    def _write_minimal_split(self, dir: Path, *, delta: bool, variants: list[str]) -> None:
        split = {
            "format": "split",
            "model_version": "2.3.0",
            "components": [],
            "transformer_variants": variants,
            "lora": [],
            "source": "Lightricks/LTX-2.3",
        }
        if delta:
            split["delta"] = True
        (dir / "split_model.json").write_text(json.dumps(split))

    def _write_minimal_config(self, dir: Path) -> None:
        cfg = {
            "model_version": "2.3.0",
            "is_v2": True,
            "apply_gated_attention": True,
            "caption_channels": None,
            "num_layers": 48,
            "num_attention_heads": 32,
            "attention_head_dim": 128,
            "connector_positional_embedding_max_pos": [4096],
            "connector_rope_type": "SPLIT",
            "variants": {"distilled-1.1": {"cross_attention_adaln": True}},
        }
        (dir / "config.json").write_text(json.dumps(cfg))

    def test_delta_skips_shared_checks(self, tmp_path, capsys):
        """validate with delta:true must not run shared-component file checks."""
        from mlx_forge.recipes import ltx_23

        self._write_minimal_split(tmp_path, delta=True, variants=["distilled-1.1"])
        self._write_minimal_config(tmp_path)

        # No shared component files. No transformer file either; just check
        # that the FAIL is about the transformer, not about connector/vae/etc.
        ns = argparse.Namespace(model_dir=str(tmp_path), source=None)
        try:
            ltx_23.validate(ns)
        except SystemExit:
            pass
        out = capsys.readouterr().out
        assert "Delta mode" in out
        # Shared component file-existence checks must be skipped:
        assert "connector.safetensors exists" not in out
        assert "vae_decoder.safetensors exists" not in out

    def test_normal_mode_still_strict(self, tmp_path, capsys):
        """validate without delta key must still run shared-component checks."""
        from mlx_forge.recipes import ltx_23

        self._write_minimal_split(tmp_path, delta=False, variants=["distilled-1.1"])
        self._write_minimal_config(tmp_path)

        ns = argparse.Namespace(model_dir=str(tmp_path), source=None)
        try:
            ltx_23.validate(ns)
        except SystemExit:
            pass
        out = capsys.readouterr().out
        assert "Delta mode" not in out
        # Strict mode: connector check ran (file is missing → check FAIL appears in output)
        assert "connector.safetensors" in out

    def test_delta_mode_fails_when_transformer_missing(self, tmp_path, capsys):
        """Delta mode + declared transformer variant but file missing → FAIL with message."""
        from mlx_forge.recipes import ltx_23

        self._write_minimal_split(tmp_path, delta=True, variants=["distilled-1.1"])
        self._write_minimal_config(tmp_path)
        # No transformer file — should FAIL with a message naming the missing file

        ns = argparse.Namespace(model_dir=str(tmp_path), source=None)
        try:
            ltx_23.validate(ns)
        except SystemExit:
            pass
        out = capsys.readouterr().out
        # The validation must mention the specific missing transformer file
        assert "Delta mode" in out  # delta mode entered
        assert "transformer-distilled-1.1.safetensors" in out  # missing file mentioned somewhere
