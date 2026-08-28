"""harvest_keys.py: the reduction rule and the fixture shape, network stubbed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

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


def test_keep_only_treats_all_digit_segments_as_indices():
    m = _load()
    assert m.keep("t5_encoder.block.0.w")
    assert m.keep("vae_lightvae_v2.decoder.1.w")
    assert not m.keep("blocks.2.w")
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


def test_summarise_is_the_shared_reduction():
    m = _load()
    header = {
        "blocks.0.w": {"dtype": "BF16", "shape": [4, 4]},
        "blocks.7.w": {"dtype": "BF16", "shape": [4, 4]},
        "norm.w": {"dtype": "F32", "shape": [4]},
    }
    assert m.summarise(header) == {
        "metadata_keys": [],
        "tensor_count": 3,
        "dtypes": {"BF16": 2, "F32": 1},
        "keys": ["blocks.0.w", "norm.w"],
    }


def test_torch_dtype_names_are_safetensors_spelling():
    m = _load()
    assert m.torch_dtype_name("torch.float32") == "F32"
    assert m.torch_dtype_name("torch.bfloat16") == "BF16"
    assert m.torch_dtype_name("torch.float16") == "F16"
    assert m.torch_dtype_name("torch.int64") == "I64"


def test_harvest_torch_reads_a_state_dict_and_a_section(tmp_path):
    torch = pytest.importorskip("torch")
    m = _load()
    state = {"blocks.0.w": torch.zeros(2, 2), "blocks.3.w": torch.zeros(2, 2, dtype=torch.bfloat16)}
    torch.save({"target_encoder": state, "epoch": 3}, tmp_path / "ckpt.pt")

    out = m.harvest_torch(tmp_path / "ckpt.pt", section="target_encoder")

    assert out == {
        "ckpt.pt": {
            "metadata_keys": [],
            "tensor_count": 2,
            "dtypes": {"F32": 1, "BF16": 1},
            "keys": ["blocks.0.w"],
        }
    }


def test_cli_torch_and_repo_are_exclusive():
    m = _load()
    import pytest as _pytest

    with _pytest.raises(SystemExit):
        m.parse_args(["--torch", "a.pt", "--repo", "a/b", "--out", "o.json"])
    args = m.parse_args(["--torch", "a.pt", "--section", "state_dict", "--out", "o.json"])
    assert args.torch == Path("a.pt") and args.section == "state_dict"
