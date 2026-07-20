"""Shared strict pipeline-file copy (audit follow-up of #32).

The cogvideox silent-skip bug (`if src.exists()`) had already shipped a
second incomplete artifact: dgrauet/matrix-game-3.0-mlx lacks
google/umt5-xxl/spiece.model (present upstream in Skywork/Matrix-Game-3.0).
Centralize the strict copy in convert.copy_required_files and use it in
fish_s2, matrix_game_3_0 and ernie_image; genuinely optional files
(ernie_image_pe's chat_template.jinja, bypassed at runtime by the port)
warn instead of aborting.
"""

import pytest

from mlx_forge.convert import copy_required_files


def _mk(tmp_path, files):
    src = tmp_path / "src"
    for f in files:
        p = src / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return src


def test_copies_flattened(tmp_path):
    src = _mk(tmp_path, ["a/b.json", "c.json"])
    out = tmp_path / "out"
    out.mkdir()
    copy_required_files(src, out, ["a/b.json", "c.json"], flatten=True)
    assert (out / "a_b.json").exists()
    assert (out / "c.json").exists()


def test_copies_preserving_tree(tmp_path):
    src = _mk(tmp_path, ["google/umt5-xxl/spiece.model"])
    out = tmp_path / "out"
    out.mkdir()
    copy_required_files(src, out, ["google/umt5-xxl/spiece.model"], flatten=False)
    assert (out / "google/umt5-xxl/spiece.model").exists()


def test_missing_required_aborts_with_full_list(tmp_path):
    src = _mk(tmp_path, ["c.json"])
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit) as e:
        copy_required_files(src, out, ["a/b.json", "c.json", "d.json"], flatten=True)
    msg = str(e.value)
    assert "a/b.json" in msg and "d.json" in msg


def test_missing_optional_warns_but_copies_rest(tmp_path, capsys):
    src = _mk(tmp_path, ["c.json"])
    out = tmp_path / "out"
    out.mkdir()
    copy_required_files(
        src,
        out,
        ["c.json", "chat_template.jinja"],
        flatten=True,
        optional={"chat_template.jinja"},
    )
    assert (out / "c.json").exists()
    assert "WARNING" in capsys.readouterr().out
    assert not (out / "chat_template.jinja").exists()


def test_recipes_use_the_strict_helper():
    """Structural pin: the audited recipes must not keep a silent
    `if src.exists()` copy for their required pipeline files."""
    import inspect

    from mlx_forge.recipes import ernie_image, fish_s2, matrix_game_3_0

    for mod in (fish_s2, matrix_game_3_0, ernie_image):
        src = inspect.getsource(mod)
        assert "copy_required_files" in src, mod.__name__
