"""CogVideoX-Fun recipe: pipeline config copy must be strict (completeness).

The published dgrauet/CogVideoX-Fun-V1.5-5b-InP-mlx-q8 shipped without
spiece.model: the copy loop silently skipped any missing _HF_CONFIG_FILES
(`if src.exists()`), so a local --source lacking tokenizer/spiece.model
produced an artifact whose tokenizer cannot load at all (VOID dogfood,
2026-07-20). A recipe's output must be complete or the conversion must
fail loudly.
"""

import pytest

from mlx_forge.recipes.cogvideox_fun_v1_5_5b_inp import (
    _HF_CONFIG_FILES,
    copy_pipeline_configs,
)


def _make_source(tmp_path, files):
    src = tmp_path / "src"
    for f in files:
        p = src / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return src


def test_copies_all_with_flattened_names(tmp_path):
    src = _make_source(tmp_path, _HF_CONFIG_FILES)
    out = tmp_path / "out"
    out.mkdir()
    copy_pipeline_configs(src, out)
    assert (out / "tokenizer_spiece.model").exists()
    assert (out / "scheduler_scheduler_config.json").exists()
    assert (out / "model_index.json").exists()


def test_missing_file_fails_loudly(tmp_path):
    files = [f for f in _HF_CONFIG_FILES if f != "tokenizer/spiece.model"]
    src = _make_source(tmp_path, files)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit) as e:
        copy_pipeline_configs(src, out)
    assert "tokenizer/spiece.model" in str(e.value), "the error must name the missing file"
