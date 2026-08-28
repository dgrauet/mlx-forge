"""compare_pack.py: values are compared on the data section, never the header."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import mlx.core as mx

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "compare_pack.py"


def _load():
    spec = importlib.util.spec_from_file_location("compare_pack", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metadata_only_difference_is_equal(tmp_path):
    m = _load()
    w = {"a.weight": mx.arange(16).reshape(4, 4).astype(mx.float32)}
    mx.save_safetensors(str(tmp_path / "plain.safetensors"), w)
    mx.save_safetensors(str(tmp_path / "meta.safetensors"), w, metadata={"model_version": "1"})
    assert m.data_sha256(tmp_path / "plain.safetensors") == m.data_sha256(
        tmp_path / "meta.safetensors"
    )


def test_value_difference_is_a_mismatch(tmp_path):
    m = _load()
    local = tmp_path / "local"
    remote = tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    mx.save_safetensors(str(local / "x.safetensors"), {"a": mx.zeros((2, 2))})
    mx.save_safetensors(str(remote / "x.safetensors"), {"a": mx.ones((2, 2))})
    mx.save_safetensors(str(local / "y.safetensors"), {"b": mx.zeros((2,))})
    mx.save_safetensors(str(remote / "y.safetensors"), {"b": mx.zeros((2,))})

    mismatches = m.compare(
        local,
        {
            "x.safetensors": remote / "x.safetensors",
            "y.safetensors": remote / "y.safetensors",
        },
    )

    assert [name for name, _, _ in mismatches] == ["x.safetensors"]
