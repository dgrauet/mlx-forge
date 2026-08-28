"""harvest_keys.py: the reduction rule and the fixture shape, network stubbed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "harvest_keys.py"


def _load():
    spec = importlib.util.spec_from_file_location("harvest_keys", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_keep_reduces_to_indices_zero_and_one():
    m = _load()
    assert m.keep("blocks.0.attn.weight")
    assert m.keep("blocks.1.attn.weight")
    assert not m.keep("blocks.2.attn.weight")
    assert m.keep("norm.weight")


def test_harvest_produces_the_fixture_shape(monkeypatch):
    m = _load()
    header = {
        "__metadata__": {"model_version": "1.0"},
        "blocks.0.w": {"dtype": "BF16"},
        "blocks.7.w": {"dtype": "BF16"},
        "norm.w": {"dtype": "F32"},
    }
    monkeypatch.setattr(m, "read_header", lambda repo, filename: dict(header))

    out = m.harvest("acme/model", ["a.safetensors"])

    assert out == {
        "a.safetensors": {
            "metadata_keys": ["model_version"],
            "tensor_count": 3,
            "dtypes": {"BF16": 2, "F32": 1},
            "keys": ["blocks.0.w", "norm.w"],
        }
    }


def test_cli_requires_repo_file_and_out():
    m = _load()
    import pytest

    with pytest.raises(SystemExit):
        m.parse_args([])
    args = m.parse_args(["--repo", "a/b", "--file", "x.safetensors", "--out", "o.json"])
    assert args.files == ["x.safetensors"]
