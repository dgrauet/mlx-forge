"""The uploaded repo must contain everything the conversion produced.

Two shipped incidents (#32, #34) came from a converted model missing its
tokenizer files; both were fixed on the *copy* side (`copy_required_files`).
The upload side re-filtered on suffix — `*.safetensors`, `*.json`, `README.md` —
so `spiece.model` and `chat_template.jinja` were dropped again on the way out,
and `--add-only` only ever looked at the top level. These tests pin the
invariant on the upload side too.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlx_forge.upload import upload_model


def _api(remote_files: list[str] | None = None) -> MagicMock:
    api = MagicMock()
    info = MagicMock()
    info.siblings = [MagicMock(rfilename=f) for f in (remote_files or [])]
    api.model_info.return_value = info
    api.create_repo.return_value = "https://huggingface.co/test/repo"
    return api


def _model_dir(tmp_path):
    """A converted model shaped like the real recipes' output."""
    (tmp_path / "transformer.safetensors").write_bytes(b"x" * 10)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "split_model.json").write_text("{}")
    (tmp_path / "README.md").write_text("# Card")
    # cogvideox / fish flatten tokenizer files next to the weights
    (tmp_path / "tokenizer_spiece.model").write_bytes(b"sp")
    # ernie-image ships a chat template
    (tmp_path / "chat_template.jinja").write_text("{{ x }}")
    # matrix-game copies with flatten=False -> real subdirectory
    nested = tmp_path / "google" / "umt5-xxl"
    nested.mkdir(parents=True)
    (nested / "spiece.model").write_bytes(b"sp")
    (nested / "tokenizer_config.json").write_text("{}")
    return tmp_path


class TestFullUpload:
    def test_tokenizer_files_are_not_filtered_out(self, tmp_path):
        api = _api()
        upload_model(_model_dir(tmp_path), api=api, repo_id="test/repo")

        kwargs = api.upload_folder.call_args.kwargs
        patterns = kwargs.get("allow_patterns")
        assert patterns is None, (
            "an allow-list of suffixes silently drops tokenizer files; "
            f"got allow_patterns={patterns}"
        )

    def test_lists_what_it_will_push(self, tmp_path, capsys):
        """The deny-list means a stray file goes up — make that visible first."""
        model_dir = _model_dir(tmp_path)
        (model_dir / "leftover.pth").write_bytes(b"z" * 4096)

        upload_model(model_dir, api=_api(), repo_id="test/repo")

        out = capsys.readouterr().out
        assert "leftover.pth" in out
        assert "google/umt5-xxl/spiece.model" in out

    def test_junk_is_still_excluded(self, tmp_path):
        api = _api()
        upload_model(_model_dir(tmp_path), api=api, repo_id="test/repo")

        ignored = api.upload_folder.call_args.kwargs.get("ignore_patterns") or []
        assert any(".DS_Store" in p for p in ignored)
        assert any("__pycache__" in p for p in ignored)


class TestAddOnlyCompleteness:
    def _uploaded(self, api) -> list[str]:
        return [c.kwargs["path_in_repo"] for c in api.upload_file.call_args_list]

    def test_uploads_tokenizer_files(self, tmp_path):
        api = _api(remote_files=["transformer.safetensors"])
        upload_model(_model_dir(tmp_path), api=api, repo_id="test/repo", add_only=True)

        uploaded = self._uploaded(api)
        assert "tokenizer_spiece.model" in uploaded
        assert "chat_template.jinja" in uploaded

    def test_uploads_nested_files_at_their_relative_path(self, tmp_path):
        api = _api(remote_files=["transformer.safetensors"])
        upload_model(_model_dir(tmp_path), api=api, repo_id="test/repo", add_only=True)

        uploaded = self._uploaded(api)
        assert "google/umt5-xxl/spiece.model" in uploaded, (
            "nested files must keep their path in the repo, not be flattened or skipped"
        )
        assert "google/umt5-xxl/tokenizer_config.json" in uploaded

    def test_remote_comparison_uses_the_repo_relative_path(self, tmp_path):
        """A nested file already on the remote must not be re-uploaded."""
        api = _api(remote_files=["google/umt5-xxl/spiece.model"])
        upload_model(_model_dir(tmp_path), api=api, repo_id="test/repo", add_only=True)

        assert "google/umt5-xxl/spiece.model" not in self._uploaded(api)

    def test_basename_collision_does_not_hide_a_new_file(self, tmp_path):
        """`spiece.model` on the remote root must not mask the nested one."""
        api = _api(remote_files=["spiece.model"])
        upload_model(_model_dir(tmp_path), api=api, repo_id="test/repo", add_only=True)

        assert "google/umt5-xxl/spiece.model" in self._uploaded(api)

    def test_junk_is_not_uploaded(self, tmp_path):
        model_dir = _model_dir(tmp_path)
        (model_dir / ".DS_Store").write_bytes(b"junk")
        cache = model_dir / "__pycache__"
        cache.mkdir()
        (cache / "x.pyc").write_bytes(b"junk")

        api = _api()
        upload_model(model_dir, api=api, repo_id="test/repo", add_only=True)

        uploaded = self._uploaded(api)
        assert not any(".DS_Store" in u or "__pycache__" in u for u in uploaded)

    def test_nothing_to_upload_counts_every_file(self, tmp_path, capsys):
        model_dir = _model_dir(tmp_path)
        remote = [
            "transformer.safetensors",
            "config.json",
            "split_model.json",
            "README.md",
            "tokenizer_spiece.model",
            "chat_template.jinja",
            "google/umt5-xxl/spiece.model",
            "google/umt5-xxl/tokenizer_config.json",
        ]
        api = _api(remote_files=remote)
        upload_model(model_dir, api=api, repo_id="test/repo", add_only=True)

        api.upload_file.assert_not_called()
        assert "Nothing to upload" in capsys.readouterr().out


class TestCardOnlyUnaffected:
    def test_card_only_still_pushes_just_the_readme(self, tmp_path):
        api = _api()
        upload_model(_model_dir(tmp_path), api=api, repo_id="test/repo", card_only=True)

        api.upload_folder.assert_not_called()
        assert api.upload_file.call_args.kwargs["path_in_repo"] == "README.md"


@pytest.mark.parametrize("mode", ["full", "add_only"])
def test_no_upload_path_drops_a_produced_file(tmp_path, mode):
    """Whatever the mode, the set of files considered must cover the directory."""
    from mlx_forge.upload import iter_model_files

    model_dir = _model_dir(tmp_path)
    on_disk = {p.relative_to(model_dir).as_posix() for p in model_dir.rglob("*") if p.is_file()}
    considered = {p.relative_to(model_dir).as_posix() for p in iter_model_files(model_dir)}
    assert considered == on_disk
