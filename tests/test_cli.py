"""Tests for CLI argument parsing and command dispatch."""

from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import errors as hf_errors

from mlx_forge.cli import _get_recipe, main


class TestGetRecipe:
    def test_valid_recipe(self):
        mod = _get_recipe("ltx-2.3")
        assert hasattr(mod, "convert")

    def test_unknown_recipe_exits(self):
        with pytest.raises(SystemExit):
            _get_recipe("nonexistent-recipe")


class TestMainNoArgs:
    def test_no_command_exits_zero(self):
        with patch("sys.argv", ["mlx-forge"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0


class TestMainVersion:
    def test_version_flag(self, capsys):
        with patch("sys.argv", ["mlx-forge", "--version"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "mlx-forge" in captured.out


class TestConvertDispatch:
    def test_convert_calls_recipe(self):
        mock_recipe = MagicMock()
        with (
            patch("sys.argv", ["mlx-forge", "convert", "ltx-2.3"]),
            patch("mlx_forge.cli._get_recipe", return_value=mock_recipe),
        ):
            main()
            mock_recipe.add_convert_args.assert_called_once()
            mock_recipe.convert.assert_called_once()

    def test_convert_dry_run_creates_no_files(self, tmp_path, capsys):
        with patch(
            "sys.argv",
            ["mlx-forge", "convert", "ltx-2.3", "--dry-run", "--output", str(tmp_path / "out")],
        ):
            main()
        assert not (tmp_path / "out").exists()
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
        assert "transformer-distilled.safetensors" in captured.out

    def test_convert_invalid_recipe(self):
        with patch("sys.argv", ["mlx-forge", "convert", "bad-recipe"]):
            with pytest.raises(SystemExit):
                main()


class TestValidateDispatch:
    def test_validate_calls_recipe(self):
        mock_recipe = MagicMock()
        with (
            patch("sys.argv", ["mlx-forge", "validate", "ltx-2.3"]),
            patch("mlx_forge.cli._get_recipe", return_value=mock_recipe),
        ):
            main()
            mock_recipe.add_validate_args.assert_called_once()
            mock_recipe.validate.assert_called_once()


class TestSplitDispatch:
    def test_split_calls_recipe(self):
        mock_recipe = MagicMock()
        with (
            patch("sys.argv", ["mlx-forge", "split", "ltx-2.3"]),
            patch("mlx_forge.cli._get_recipe", return_value=mock_recipe),
        ):
            main()
            mock_recipe.add_split_args.assert_called_once()
            mock_recipe.split.assert_called_once()


class TestQuantizeCommand:
    def test_quantize_missing_file_exits(self, tmp_path):
        with patch("sys.argv", ["mlx-forge", "quantize", str(tmp_path / "missing.safetensors")]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_quantize_calls_quantize_file(self, tmp_path):
        fake_input = tmp_path / "model.safetensors"
        fake_input.touch()
        with (
            patch("sys.argv", ["mlx-forge", "quantize", str(fake_input), "--bits", "4"]),
            patch("mlx_forge.quantize.quantize_file") as mock_qf,
        ):
            main()
            mock_qf.assert_called_once()
            call_kwargs = mock_qf.call_args
            assert call_kwargs[1]["bits"] == 4


class TestUploadCommand:
    def test_upload_missing_dir_exits(self, tmp_path):
        with patch("sys.argv", ["mlx-forge", "upload", str(tmp_path / "missing_dir")]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_upload_no_safetensors_exits(self, tmp_path):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        with patch("sys.argv", ["mlx-forge", "upload", str(model_dir)]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_upload_no_split_info_no_repo_id_exits(self, tmp_path):
        import mlx.core as mx

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        mx.save_safetensors(str(model_dir / "model.safetensors"), {"w": mx.zeros((2, 2))})
        with (
            patch("sys.argv", ["mlx-forge", "upload", str(model_dir)]),
            patch("huggingface_hub.HfApi"),
        ):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_upload_with_explicit_repo_id(self, tmp_path):
        import mlx.core as mx

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        mx.save_safetensors(str(model_dir / "model.safetensors"), {"w": mx.zeros((2, 2))})
        with (
            patch(
                "sys.argv",
                ["mlx-forge", "upload", str(model_dir), "--repo-id", "user/my-model"],
            ),
            patch("huggingface_hub.HfApi"),
            patch(
                "mlx_forge.upload.generate_model_card",
                return_value="# card",
            ),
            patch(
                "mlx_forge.upload.upload_model",
                return_value="https://hf.co/user/my-model",
            ),
        ):
            main()


class TestRecipeHelp:
    """`mlx-forge convert <recipe> --help` must show the recipe's own flags.

    Two-pass parsing used to swallow --help before the recipe parser existed,
    so every recipe's flags were undiscoverable from the CLI.
    """

    def test_convert_recipe_help_shows_recipe_flags(self, capsys):
        with patch("sys.argv", ["mlx-forge", "convert", "ltx-2.5", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--skip-shared" in out
        assert "--config-only" in out

    def test_validate_recipe_help_shows_positional(self, capsys):
        with patch("sys.argv", ["mlx-forge", "validate", "ltx-2.3", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        assert "model_dir" in capsys.readouterr().out


class TestShowCardDiffErrors:
    """A dry run must not report a network or auth failure as a brand-new repo."""

    def _call(self, monkeypatch, exc):
        import mlx_forge.cli as cli_mod

        def boom(repo_id, filename):
            raise exc

        monkeypatch.setattr("huggingface_hub.hf_hub_download", boom)
        return cli_mod._show_card_diff(
            api=MagicMock(), repo_id="acme/demo", card="# new", model_dir=None, card_only=False
        )

    def test_a_missing_card_is_reported_as_new(self, monkeypatch, capsys):
        self._call(monkeypatch, hf_errors.EntryNotFoundError("no README"))
        assert "has no card yet" in capsys.readouterr().out

    def test_a_missing_repo_is_reported_as_new(self, monkeypatch, capsys):
        self._call(monkeypatch, hf_errors.RepositoryNotFoundError("no repo", response=MagicMock()))
        assert "has no card yet" in capsys.readouterr().out

    def test_a_hub_failure_aborts_instead_of_pretending(self, monkeypatch, capsys):
        with pytest.raises(SystemExit):
            self._call(monkeypatch, hf_errors.HfHubHTTPError("503", response=MagicMock()))
        assert "has no card yet" not in capsys.readouterr().out

    def test_a_gated_repo_aborts_instead_of_pretending(self, monkeypatch, capsys):
        # GatedRepoError subclasses RepositoryNotFoundError, so it must be
        # caught before the "genuinely new" clause.
        with pytest.raises(SystemExit):
            self._call(monkeypatch, hf_errors.GatedRepoError("gated", response=MagicMock()))
        assert "has no card yet" not in capsys.readouterr().out

    def test_an_offline_failure_aborts_instead_of_pretending(self, monkeypatch, capsys):
        # hf_hub_download wraps an offline failure in LocalEntryNotFoundError,
        # which subclasses EntryNotFoundError, so it must be caught before
        # the "genuinely new" clause.
        with pytest.raises(SystemExit):
            self._call(monkeypatch, hf_errors.LocalEntryNotFoundError("offline"))
        assert "has no card yet" not in capsys.readouterr().out


class TestCardOnlyListing:
    """In --card-only mode the remote is the whole truth; offline must not
    silently publish local sizes."""

    def test_offline_card_only_aborts(self, tmp_path):
        from huggingface_hub.errors import HfHubHTTPError

        import mlx_forge.cli as cli_mod

        api = MagicMock()
        api.model_info.side_effect = HfHubHTTPError("offline", response=MagicMock())
        (tmp_path / "split_model.json").write_text("{}")
        with pytest.raises(SystemExit):
            cli_mod._card_file_listing(api, "acme/demo", tmp_path, card_only=True)

    def test_offline_full_upload_falls_back_to_local(self, tmp_path):
        from huggingface_hub.errors import HfHubHTTPError

        import mlx_forge.cli as cli_mod

        api = MagicMock()
        api.model_info.side_effect = HfHubHTTPError("offline", response=MagicMock())
        (tmp_path / "a.safetensors").write_bytes(b"x" * 4)
        listing = cli_mod._card_file_listing(api, "acme/demo", tmp_path, card_only=False)
        assert listing == {"a.safetensors": 4}

    def test_gated_card_only_aborts(self, tmp_path):
        # GatedRepoError subclasses RepositoryNotFoundError, whose handler
        # swallows it — --card-only must not fall through to inventing a
        # listing from the local directory of a repo it cannot see.
        import mlx_forge.cli as cli_mod

        api = MagicMock()
        api.model_info.side_effect = hf_errors.GatedRepoError("gated", response=MagicMock())
        (tmp_path / "split_model.json").write_text("{}")
        with pytest.raises(SystemExit):
            cli_mod._card_file_listing(api, "acme/demo", tmp_path, card_only=True)
