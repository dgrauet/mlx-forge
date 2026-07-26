"""Ideogram-4 recipe: pipeline file copy must be strict (completeness).

The `if not src.exists(): print WARNING; continue` pattern shipped incomplete
models twice (cogvideox q8 without spiece.model, matrix-game without
google/umt5-xxl/spiece.model) and was removed from every other recipe in #32
and #34 — ideogram-4 was in neither audit and kept it. A warning scrolling past
in the middle of a multi-GB conversion is not a failure signal: the output must
be complete or the conversion must abort naming what is missing.
"""

import pytest

from mlx_forge.recipes.ideogram_4 import _HF_CONFIG_FILES, _copy_pipeline_files


def _make_source(tmp_path, files):
    src = tmp_path / "src"
    for f in files:
        p = src / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return src


def _out(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    return out


def test_copies_every_pipeline_file(tmp_path):
    src = _make_source(tmp_path, _HF_CONFIG_FILES)
    out = _out(tmp_path)

    _copy_pipeline_files(src, out)

    assert (out / "model_index.json").exists()
    # scheduler/ is flattened with a prefix...
    assert (out / "scheduler_scheduler_config.json").exists()
    # ...while tokenizer/ keeps its directory, as the runtime expects
    assert (out / "tokenizer" / "tokenizer.json").exists()
    assert (out / "tokenizer" / "chat_template.jinja").exists()


@pytest.mark.parametrize("missing", _HF_CONFIG_FILES)
def test_any_missing_file_fails_loudly(tmp_path, missing):
    src = _make_source(tmp_path, [f for f in _HF_CONFIG_FILES if f != missing])
    out = _out(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        _copy_pipeline_files(src, out)

    assert missing in str(exc_info.value), "the error must name the missing file"


def test_error_lists_every_missing_file_at_once(tmp_path):
    """One run should tell you everything to fetch, not just the first gap."""
    src = _make_source(tmp_path, ["model_index.json"])
    out = _out(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        _copy_pipeline_files(src, out)

    message = str(exc_info.value)
    for f in _HF_CONFIG_FILES:
        if f != "model_index.json":
            assert f in message


def test_nothing_is_written_when_a_file_is_missing(tmp_path):
    """Abort before copying, so a partial output dir is never left behind."""
    src = _make_source(tmp_path, [f for f in _HF_CONFIG_FILES if f != "tokenizer/tokenizer.json"])
    out = _out(tmp_path)

    with pytest.raises(SystemExit):
        _copy_pipeline_files(src, out)

    assert list(out.iterdir()) == []
