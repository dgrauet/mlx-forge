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
        assert split["components"] == []  # no shared components in delta mode
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
