"""Hunyuan3D-2.1 records quantization in quantize_config.json (shared contract).

Regression test for the audit finding: the recipe supported --quantize but
recorded it only in split_model.json, so upload's card recovery and validate's
scales/biases gate — both keyed on quantize_config.json — never saw it.
"""

import json
from types import SimpleNamespace

import mlx_forge.convert as convert
from mlx_forge.quantize import read_quantize_config
from mlx_forge.recipes.hunyuan3d_21 import _quantization_recorded, _write_config_files


def _args(**over):
    base = dict(stage="shape", quantize=True, bits=8, group_size=64)
    base.update(over)
    return SimpleNamespace(**base)


def test_write_config_files_records_quantize_config(tmp_path, monkeypatch):
    # Licence fetching is network-bound and orthogonal here.
    monkeypatch.setattr(convert, "ensure_license_file", lambda *a, **k: [])

    _write_config_files(tmp_path, {"components": ["dit"]}, ["dit"], _args(), "dit")

    qconfig = read_quantize_config(tmp_path)
    assert qconfig is not None
    assert qconfig["bits"] == 8
    assert qconfig["group_size"] == 64
    assert qconfig["quantized_components"] == ["dit"]
    # split_model.json records the same flat quantization shape every recipe
    # uses; quantize_config.json above stays the authority for bits/group_size.
    split_model = json.loads((tmp_path / "split_model.json").read_text())
    assert split_model["quantized"] is True
    assert split_model["quantization_bits"] == 8
    assert split_model["quantization_group_size"] == 64


def test_write_config_files_unquantized_writes_no_config(tmp_path, monkeypatch):
    monkeypatch.setattr(convert, "ensure_license_file", lambda *a, **k: [])

    _write_config_files(tmp_path, {"components": ["dit"]}, ["dit"], _args(quantize=False), "dit")

    assert read_quantize_config(tmp_path) is None
    assert not _quantization_recorded(tmp_path)


def test_quantization_recorded_accepts_legacy_split_model_record(tmp_path):
    (tmp_path / "split_model.json").write_text(
        json.dumps({"quantization": {"bits": 8, "group_size": 64}})
    )
    assert _quantization_recorded(tmp_path)


def test_quantization_recorded_false_on_empty_dir(tmp_path):
    assert not _quantization_recorded(tmp_path)
