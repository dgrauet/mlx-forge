"""`upload --dry-run` must not write into the model directory.

The project audit found that `backfill_from_recipe`, `backfill_quantization`
and `persist_card_metadata` all wrote `split_model.json` before `_run_upload`
ever reached its dry-run branch, despite `--dry-run` promising "without
writing or uploading anything". The helpers now take `dry_run` and only
report; these tests pin that the manifest is untouched while the merged
metadata is still computed in memory for the card diff.
"""

from __future__ import annotations

import json

from mlx_forge.quantize import write_quantize_config
from mlx_forge.upload import (
    backfill_from_recipe,
    backfill_quantization,
    persist_card_metadata,
)


def _manifest(tmp_path) -> dict:
    info = {"recipe": "ltx-2.3"}
    (tmp_path / "split_model.json").write_text(json.dumps(info))
    return info


def test_backfill_from_recipe_dry_run_reports_without_writing(tmp_path, capsys):
    info = _manifest(tmp_path)
    before = (tmp_path / "split_model.json").read_text()

    merged = backfill_from_recipe(tmp_path, dict(info), dry_run=True)

    assert (tmp_path / "split_model.json").read_text() == before
    assert len(merged) > len(info)  # still computed in memory for the card
    assert "[dry-run]" in capsys.readouterr().out


def test_backfill_quantization_dry_run_reports_without_writing(tmp_path, capsys):
    info = _manifest(tmp_path)
    write_quantize_config(tmp_path, bits=8, group_size=64)
    before = (tmp_path / "split_model.json").read_text()

    merged = backfill_quantization(tmp_path, dict(info), dry_run=True)

    assert (tmp_path / "split_model.json").read_text() == before
    assert merged["quantized"] is True
    assert "[dry-run]" in capsys.readouterr().out


def test_persist_card_metadata_dry_run_reports_without_writing(tmp_path, capsys):
    info = _manifest(tmp_path)
    before = (tmp_path / "split_model.json").read_text()

    merged = persist_card_metadata(
        tmp_path,
        dict(info),
        usage_url="https://example.org/infer",
        links=None,
        cli_snippet=None,
        note=None,
        dry_run=True,
    )

    assert (tmp_path / "split_model.json").read_text() == before
    assert merged["usage_url"] == "https://example.org/infer"
    assert "[dry-run]" in capsys.readouterr().out
