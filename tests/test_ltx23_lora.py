"""Tests for LTX-2.3 LoRA bundling.

The shared distilled LoRAs run at inference on the *dev* transformer. Distilled
variants are pre-distilled checkpoints that never load them, so bundling the
LoRAs into a distilled-only package is multiple GB of dead weight. convert()
must only sync them when a dev transformer is present.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from mlx_forge.recipes import ltx_23
from mlx_forge.recipes.ltx_23 import LORA_FILES, add_convert_args

_ALL_LORAS = list(LORA_FILES)
_ALL_LORAS_FILENAMES = [LORA_FILES[name] for name in _ALL_LORAS]


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
    args.lora = list(_ALL_LORAS)
    args.quantize = False
    args.bits = 8
    args.group_size = 64
    args.dry_run = False
    args.skip_shared = False
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _fake_convert_variant(args, variant, output_dir, *, is_first):
    return ({"model_version": "2.3.0", "embedded_config": {}}, True, 0)


def _run_convert(args) -> dict:
    with patch.object(ltx_23, "_convert_variant", side_effect=_fake_convert_variant):
        ltx_23.convert(args)
    return json.loads((Path(args.output) / "split_model.json").read_text())


class TestLoraBundling:
    def test_distilled_only_skips_loras(self, tmp_path):
        """A distilled-only package must not bundle the LoRAs (default --lora=all)."""
        args = _make_convert_args(tmp_path, variant=["distilled-1.1"])
        # download_hf_files must never be reached, since the gate empties the list.
        with patch.object(
            ltx_23, "download_hf_files", side_effect=AssertionError("downloaded LoRA")
        ):
            split = _run_convert(args)

        assert split["lora"] == []
        for filename in _ALL_LORAS_FILENAMES:
            assert not (tmp_path / filename).exists()

    def test_dev_variant_bundles_loras(self, tmp_path):
        """A package with a dev transformer keeps the LoRAs."""
        # Pre-create the LoRA files so the sync takes the "already exists" path
        # (no network) but still records them in the manifest.
        for filename in _ALL_LORAS_FILENAMES:
            (tmp_path / filename).write_bytes(b"stub")

        args = _make_convert_args(tmp_path, variant=["dev"])
        split = _run_convert(args)

        assert split["lora"] == _ALL_LORAS_FILENAMES

    def test_distilled_plus_dev_bundles_loras(self, tmp_path):
        """Mixed package (dev present) still bundles the LoRAs."""
        for filename in _ALL_LORAS_FILENAMES:
            (tmp_path / filename).write_bytes(b"stub")

        args = _make_convert_args(tmp_path, variant=["distilled", "dev"])
        split = _run_convert(args)

        assert split["lora"] == _ALL_LORAS_FILENAMES

    def test_explicit_empty_lora_stays_empty(self, tmp_path):
        """Passing no LoRAs is honored even with a dev variant."""
        args = _make_convert_args(tmp_path, variant=["dev"], lora=[])
        split = _run_convert(args)

        assert split["lora"] == []
