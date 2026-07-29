"""The uploaded repo must contain everything the conversion produced.

Two shipped incidents (#32, #34) came from a converted model missing its
tokenizer files; both were fixed on the *copy* side (`copy_required_files`).
The upload side re-filtered on suffix — `*.safetensors`, `*.json`, `README.md` —
so `spiece.model` and `chat_template.jinja` were dropped again on the way out,
and `--add-only` only ever looked at the top level. These tests pin the
invariant on the upload side too.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

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
    # cogvideox flattens tokenizer files next to the weights
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
    def test_card_only_never_repushes_the_weights(self, tmp_path):
        """The point of --card-only: refresh metadata without re-hashing GBs.

        It pushes README.md and split_model.json (the card's metadata carrier),
        and nothing else.
        """
        api = _api()
        upload_model(_model_dir(tmp_path), api=api, repo_id="test/repo", card_only=True)

        api.upload_folder.assert_not_called()
        pushed = [c.kwargs["path_in_repo"] for c in api.upload_file.call_args_list]
        assert not any(p.endswith(".safetensors") for p in pushed)
        assert "README.md" in pushed


@pytest.mark.parametrize("mode", ["full", "add_only"])
def test_no_upload_path_drops_a_produced_file(tmp_path, mode):
    """Whatever the mode, the set of files considered must cover the directory."""
    from mlx_forge.upload import iter_model_files

    model_dir = _model_dir(tmp_path)
    on_disk = {p.relative_to(model_dir).as_posix() for p in model_dir.rglob("*") if p.is_file()}
    considered = {p.relative_to(model_dir).as_posix() for p in iter_model_files(model_dir)}
    assert considered == on_disk


class TestModelCardFileListing:
    """The card's Files section must match what the upload actually publishes.

    It listed only *.safetensors and *.json from the top level, so the tokenizer
    files that #41 started uploading were absent from the card, and nested ones
    were invisible entirely.
    """

    def _card(self, model_dir):
        from mlx_forge.upload import generate_model_card

        return generate_model_card(model_dir, split_info={}, config={}, repo_id="user/model-mlx")

    def test_tokenizer_files_are_listed(self, tmp_path):
        card = self._card(_model_dir(tmp_path))
        assert "tokenizer_spiece.model" in card
        assert "chat_template.jinja" in card

    def test_nested_files_are_listed_with_their_path(self, tmp_path):
        card = self._card(_model_dir(tmp_path))
        assert "google/umt5-xxl/spiece.model" in card, (
            "a bare basename does not tell the reader where the file lives in the repo"
        )

    def test_weights_are_still_listed(self, tmp_path):
        card = self._card(_model_dir(tmp_path))
        assert "transformer.safetensors" in card
        assert "config.json" in card

    def test_junk_is_not_listed(self, tmp_path):
        model_dir = _model_dir(tmp_path)
        (model_dir / ".DS_Store").write_bytes(b"junk")

        assert ".DS_Store" not in self._card(model_dir)

    def test_the_card_does_not_list_itself(self, tmp_path):
        """README.md is written after generation; listing it would make the
        card differ between a first run and a --card-only refresh."""
        model_dir = _model_dir(tmp_path)
        first = self._card(model_dir)
        (model_dir / "README.md").write_text(first)

        assert first == self._card(model_dir), "card is not idempotent across runs"
        assert "`README.md`" not in first


class TestCardMetadataPersistence:
    """Operator-supplied card metadata must survive a --card-only refresh.

    `links` and `usage_url` already fell back to split_model.json; `cli_snippet`
    did not, so refreshing a card without re-typing --cli-snippet republished it
    without its Usage section. Measured on the Hub: matrix-game-3.0-mlx has no
    Usage section at all.
    """

    def test_snippet_is_persisted_for_later_refreshes(self, tmp_path):
        import json

        from mlx_forge.upload import persist_card_metadata

        (tmp_path / "split_model.json").write_text(json.dumps({"source": "Org/M"}))

        persist_card_metadata(
            tmp_path,
            {"source": "Org/M"},
            usage_url=None,
            links=None,
            cli_snippet="pip install thing\nthing run",
        )

        stored = json.loads((tmp_path / "split_model.json").read_text())
        assert stored["cli_snippet"] == "pip install thing\nthing run"
        assert stored["source"] == "Org/M", "existing fields must survive"

    def test_refresh_without_the_flag_keeps_the_usage_section(self, tmp_path):
        import json

        from mlx_forge.upload import generate_model_card, persist_card_metadata

        (tmp_path / "split_model.json").write_text(json.dumps({"source": "Org/M"}))
        info = persist_card_metadata(
            tmp_path,
            {"source": "Org/M"},
            usage_url=None,
            links=None,
            cli_snippet="pip install thing",
        )
        first = generate_model_card(
            tmp_path,
            split_info=info,
            config={},
            repo_id="u/m",
            cli_snippet=info.get("cli_snippet"),
        )

        # later: mlx-forge upload --card-only, no flags
        reloaded = json.loads((tmp_path / "split_model.json").read_text())
        refreshed = generate_model_card(
            tmp_path,
            split_info=reloaded,
            config={},
            repo_id="u/m",
            cli_snippet=reloaded.get("cli_snippet"),
        )

        assert "pip install thing" in first
        assert "pip install thing" in refreshed

    def test_nothing_written_when_no_metadata_supplied(self, tmp_path):
        from mlx_forge.upload import persist_card_metadata

        info = persist_card_metadata(
            tmp_path, {"source": "Org/M"}, usage_url=None, links=None, cli_snippet=None
        )
        assert info == {"source": "Org/M"}
        assert not (tmp_path / "split_model.json").exists()


class TestCardFileListingProvenance:
    """Every section of a card must describe the same repo.

    transformer_variants is derived from the remote, model_files was derived
    from the local directory. After a delta upload the two disagree: the
    published ltx-2.3-mlx-q8 card never mentions the 1.1 transformer or LoRA
    that --add-only put in the repo.
    """

    def test_remote_only_files_are_listed(self, tmp_path):
        from mlx_forge.upload import generate_model_card

        (tmp_path / "transformer-dev.safetensors").write_bytes(b"x" * 10)

        card = generate_model_card(
            tmp_path,
            split_info={},
            config={},
            repo_id="u/ltx",
            file_listing={"transformer-distilled-1.1.safetensors": 2048},
        )

        assert "transformer-distilled-1.1.safetensors" in card, (
            "a file added by a delta upload must appear in the card"
        )

    def test_local_files_still_listed(self, tmp_path):
        from mlx_forge.upload import generate_model_card

        (tmp_path / "transformer-dev.safetensors").write_bytes(b"x" * 10)
        card = generate_model_card(
            tmp_path,
            split_info={},
            config={},
            repo_id="u/ltx",
            file_listing={"transformer-dev.safetensors": 10},
        )
        assert "transformer-dev.safetensors" in card

    def test_falls_back_to_the_local_directory(self, tmp_path):
        """No listing supplied (fresh repo): behave as before."""
        from mlx_forge.upload import generate_model_card

        (tmp_path / "a.safetensors").write_bytes(b"x" * 10)
        assert "a.safetensors" in generate_model_card(
            tmp_path, split_info={}, config={}, repo_id="u/m"
        )


class TestCardOnlyCarriesMetadata:
    def test_split_model_is_pushed_alongside_the_readme(self, tmp_path):
        model_dir = _model_dir(tmp_path)
        api = _api()

        upload_model(model_dir, api=api, repo_id="test/repo", card_only=True)

        pushed = [c.kwargs["path_in_repo"] for c in api.upload_file.call_args_list]
        assert pushed == ["README.md", "split_model.json"], (
            "metadata recorded for the next refresh must reach the remote"
        )

    def test_absent_split_model_is_not_invented(self, tmp_path):
        model_dir = _model_dir(tmp_path)
        (model_dir / "split_model.json").unlink()
        api = _api()

        upload_model(model_dir, api=api, repo_id="test/repo", card_only=True)

        assert [c.kwargs["path_in_repo"] for c in api.upload_file.call_args_list] == ["README.md"]


class TestCardExcludesHubPlumbing:
    """The remote listing contains files the local one never did."""

    def test_gitattributes_is_not_listed(self, tmp_path):
        from mlx_forge.upload import generate_model_card

        card = generate_model_card(
            tmp_path,
            split_info={},
            config={},
            repo_id="u/m",
            file_listing={".gitattributes": 1519, "model.safetensors": 100},
        )
        assert ".gitattributes" not in card
        assert "model.safetensors" in card

    def test_readme_is_not_listed(self, tmp_path):
        from mlx_forge.upload import generate_model_card

        card = generate_model_card(
            tmp_path,
            split_info={},
            config={},
            repo_id="u/m",
            file_listing={"README.md": 900, "model.safetensors": 100},
        )
        assert "`README.md`" not in card

    def test_nested_dotfiles_are_not_listed(self, tmp_path):
        from mlx_forge.upload import generate_model_card

        card = generate_model_card(
            tmp_path,
            split_info={},
            config={},
            repo_id="u/m",
            file_listing={"tokenizer/.DS_Store": 6148, "tokenizer/spiece.model": 100},
        )
        assert ".DS_Store" not in card
        assert "tokenizer/spiece.model" in card


class TestCardOnlyListingProvenance:
    """In --card-only nothing local is uploaded, so the remote is the truth."""

    def test_remote_sizes_win_over_a_divergent_local_build(self, tmp_path):
        from mlx_forge.cli import _card_file_listing

        model_dir = _model_dir(tmp_path)  # local transformer.safetensors is 10 bytes
        api = _api()
        info = api.model_info.return_value
        info.siblings = [MagicMock(rfilename="transformer.safetensors", size=999)]

        listing = _card_file_listing(api, "test/repo", model_dir, card_only=True)

        assert listing["transformer.safetensors"] == 999, (
            "a local directory holding another build must not set the published size"
        )

    def test_a_full_upload_still_prefers_the_local_build(self, tmp_path):
        from mlx_forge.cli import _card_file_listing

        model_dir = _model_dir(tmp_path)
        api = _api()
        api.model_info.return_value.siblings = [
            MagicMock(rfilename="transformer.safetensors", size=999)
        ]

        listing = _card_file_listing(api, "test/repo", model_dir, card_only=False)

        assert listing["transformer.safetensors"] == 10, "the local files are what goes up"


class TestDryRunShowsWhatIsPushed:
    """--dry-run is only useful if it renders the same bytes a real run pushes.

    It did not: the CLI generated the card, then upload_model regenerated it
    from the manifest alone. The dry run showed the good card and the push
    published the poor one — which is how a published card lost a section.
    """

    def _dir(self, tmp_path):
        import json

        (tmp_path / "split_model.json").write_text(
            json.dumps({"source": "Lightricks/LTX-2.3", "recipe": "ltx-2.3"})
        )
        (tmp_path / "config.json").write_text("{}")
        return tmp_path

    def _api(self):
        api = MagicMock()
        info = MagicMock()
        info.siblings = [MagicMock(rfilename="transformer-dev.safetensors", size=4096)]
        api.model_info.return_value = info
        api.create_repo.return_value = "https://huggingface.co/test/repo"
        return api

    def _run(self, model_dir, api, *extra):
        from mlx_forge.cli import main

        argv = ["mlx-forge", "upload", str(model_dir), "--repo-id", "test/repo", "--card-only"]
        with (
            patch("sys.argv", argv + list(extra)),
            patch("huggingface_hub.HfApi", return_value=api),
        ):
            main()

    def test_dry_run_output_matches_the_pushed_bytes(self, tmp_path, capsys):
        model_dir = self._dir(tmp_path)

        # dry run: nothing written, nothing uploaded
        api = self._api()
        self._run(model_dir, api, "--dry-run")
        api.upload_file.assert_not_called()
        assert not (model_dir / "README.md").exists()

        # real run
        api2 = self._api()
        self._run(model_dir, api2)
        pousse = Path(
            next(
                c
                for c in api2.upload_file.call_args_list
                if c.kwargs["path_in_repo"] == "README.md"
            ).kwargs["path_or_fileobj"]
        ).read_text()

        # every non-blank line the dry run announced must be in what went up
        annonce = [
            line[1:]
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("+") and not line.startswith("+++") and line[1:].strip()
        ]
        assert annonce, "the dry run announced nothing"
        for line in annonce:
            assert line in pousse, f"dry run showed {line!r} but it is not in the pushed card"

    def test_dry_run_writes_no_readme(self, tmp_path):
        model_dir = self._dir(tmp_path)
        self._run(model_dir, self._api(), "--dry-run")
        assert not (model_dir / "README.md").exists()
