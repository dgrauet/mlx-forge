"""Tests for CLI argument parsing and command dispatch."""

from unittest.mock import MagicMock, patch

import pytest

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
        assert "transformer.safetensors" in captured.out

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
