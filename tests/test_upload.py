"""Tests for upload utilities: repo ID derivation, model card generation, metadata loading."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest
from huggingface_hub.errors import RepositoryNotFoundError

from mlx_forge.upload import derive_repo_id, generate_model_card, load_model_metadata


class TestLoadModelMetadata:
    def test_both_files_present(self, tmp_path):
        split = {"source": "Org/Model", "split": True}
        config = {"model_version": "2.3"}
        (tmp_path / "split_model.json").write_text(json.dumps(split))
        (tmp_path / "config.json").write_text(json.dumps(config))

        s, c = load_model_metadata(tmp_path)
        assert s["source"] == "Org/Model"
        assert c["model_version"] == "2.3"

    def test_missing_files(self, tmp_path):
        s, c = load_model_metadata(tmp_path)
        assert s == {}
        assert c == {}

    def test_only_split(self, tmp_path):
        (tmp_path / "split_model.json").write_text(json.dumps({"split": True}))
        s, c = load_model_metadata(tmp_path)
        assert s["split"] is True
        assert c == {}


class TestDeriveRepoId:
    def _make_api(self, username="testuser"):
        api = MagicMock()
        api.whoami.return_value = {"name": username}
        return api

    def test_basic(self):
        split_info = {"source": "Lightricks/LTX-2.3"}
        repo_id = derive_repo_id(split_info, Path("/tmp/model"), api=self._make_api())
        assert repo_id == "testuser/ltx-2.3-mlx"

    def test_with_quantization(self):
        split_info = {
            "source": "Lightricks/LTX-2.3",
            "quantized": True,
            "quantization_bits": 8,
        }
        repo_id = derive_repo_id(split_info, Path("/tmp/model"), api=self._make_api())
        assert repo_id == "testuser/ltx-2.3-mlx-q8"

    def test_explicit_namespace(self):
        split_info = {"source": "Lightricks/LTX-2.3"}
        api = self._make_api()
        repo_id = derive_repo_id(split_info, Path("/tmp/model"), api=api, namespace="myorg")
        assert repo_id == "myorg/ltx-2.3-mlx"
        api.whoami.assert_not_called()

    def test_no_source_uses_dir_name(self):
        split_info = {"source": ""}
        repo_id = derive_repo_id(split_info, Path("/tmp/my-model"), api=self._make_api())
        assert repo_id == "testuser/my-model-mlx"

    def test_dir_name_already_has_mlx_suffix_not_doubled(self):
        # Converted dirs end in "-mlx"; the derived repo must not become "-mlx-mlx".
        split_info = {"source": ""}
        repo_id = derive_repo_id(split_info, Path("/tmp/vjepa-2.0-vitl-mlx"), api=self._make_api())
        assert repo_id == "testuser/vjepa-2.0-vitl-mlx"

    def test_dir_name_q_suffix_kept_when_split_has_quant_flag(self):
        split_info = {"source": "", "quantized": True, "quantization_bits": 8}
        repo_id = derive_repo_id(
            split_info, Path("/tmp/vjepa-2.0-vitl-mlx-q8"), api=self._make_api()
        )
        assert repo_id == "testuser/vjepa-2.0-vitl-mlx-q8"

    def test_dir_name_q_suffix_kept_when_split_omits_quant_flag(self):
        # Realistic vjepa2 case: quantization is recorded in quantize_config.json,
        # NOT split_model.json, so the dir-name -q8 must survive (no q8 loss, and
        # the quantized repo must not collide with the non-quantized one).
        split_info = {"source": ""}
        repo_id = derive_repo_id(
            split_info, Path("/tmp/vjepa-2.0-vitl-mlx-q8"), api=self._make_api()
        )
        assert repo_id == "testuser/vjepa-2.0-vitl-mlx-q8"

    def test_whoami_failure_raises(self):
        api = MagicMock()
        api.whoami.side_effect = Exception("not logged in")
        import pytest

        with pytest.raises(SystemExit):
            derive_repo_id({"source": "Org/Model"}, Path("/tmp/m"), api=api)

    def test_quantized_false_no_q_suffix(self):
        split_info = {
            "source": "Org/Model",
            "quantized": False,
            "quantization_bits": 8,
        }
        repo_id = derive_repo_id(split_info, Path("/tmp/m"), api=self._make_api())
        assert "q8" not in repo_id


class TestGenerateModelCard:
    def test_basic_card(self, tmp_path):
        # Create a dummy safetensors file
        mx.save_safetensors(str(tmp_path / "transformer.safetensors"), {"w": mx.zeros((2, 2))})
        (tmp_path / "config.json").write_text("{}")

        card = generate_model_card(
            tmp_path,
            split_info={
                "source": "Org/Model",
                "transformer_variants": ["distilled", "dev"],
            },
            config={"model_version": "2.3"},
            repo_id="user/model-mlx",
        )
        assert "---" in card
        assert "library_name: mlx" in card
        assert "base_model: Org/Model" in card
        assert "user/model-mlx" in card
        assert "distilled" in card
        assert "2.3" in card
        assert "mlx-forge" in card

    def test_quantized_card(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        card = generate_model_card(
            tmp_path,
            split_info={"source": "Org/Model", "quantized": True, "quantization_bits": 8},
            config={},
            repo_id="user/model-mlx-q8",
        )
        assert "int8" in card

    def test_no_base_model(self, tmp_path):
        card = generate_model_card(
            tmp_path,
            split_info={},
            config={},
            repo_id="user/model-mlx",
        )
        assert "MLX format model." in card
        assert "base_model" not in card

    def test_custom_base_model(self, tmp_path):
        card = generate_model_card(
            tmp_path,
            split_info={"source": "Org/Model"},
            config={},
            repo_id="user/model-mlx",
            base_model="Custom/Base",
        )
        assert "base_model: Custom/Base" in card

    def test_file_listing(self, tmp_path):
        mx.save_safetensors(str(tmp_path / "model.safetensors"), {"w": mx.zeros((4, 4))})
        (tmp_path / "config.json").write_text('{"key": "val"}')

        card = generate_model_card(
            tmp_path,
            split_info={},
            config={},
            repo_id="user/m",
        )
        assert "model.safetensors" in card
        assert "config.json" in card

    def test_license_param(self, tmp_path):
        card = generate_model_card(
            tmp_path,
            split_info={},
            config={},
            repo_id="user/m",
            license_id="mit",
        )
        assert "license: mit" in card

    def test_cli_snippet_emits_bash_block(self, tmp_path):
        snippet = "pip install tool\ntool generate -p 'hello'"
        card = generate_model_card(
            tmp_path,
            split_info={},
            config={},
            repo_id="user/m",
            cli_snippet=snippet,
        )
        assert "## Usage" in card
        assert "```bash" in card
        assert "pip install tool" in card
        assert "tool generate -p 'hello'" in card
        # The fence must close so downstream markdown renderers don't swallow the rest.
        assert card.count("```") >= 2

    def test_cli_snippet_and_usage_url_both_render(self, tmp_path):
        card = generate_model_card(
            tmp_path,
            split_info={},
            config={},
            repo_id="user/m",
            usage_url="https://github.com/org/proj",
            cli_snippet="proj run",
        )
        # Both elements must appear under the same Usage heading — one project link,
        # one bash example. Without this the card silently drops one of the two.
        assert "[proj](https://github.com/org/proj)" in card
        assert "```bash\nproj run\n```" in card

    def test_no_usage_section_when_neither_present(self, tmp_path):
        card = generate_model_card(
            tmp_path,
            split_info={},
            config={},
            repo_id="user/m",
        )
        assert "## Usage" not in card


class TestModelCardTemplate:
    def test_renders_minimal_card(self):
        from mlx_forge.upload import generate_model_card

        card = generate_model_card(
            Path("/tmp/dummy"),
            split_info={"source": "Org/Model", "transformer_variants": ["dev"]},
            config={"model_version": "2.3.0"},
            repo_id="user/model-mlx",
        )
        assert "library_name: mlx" in card
        assert "user/model-mlx" in card
        assert "[Org/Model](https://huggingface.co/Org/Model)" in card
        assert "Transformer variants:" in card
        assert "dev" in card

    def test_renders_quantized(self):
        from mlx_forge.upload import generate_model_card

        card = generate_model_card(
            Path("/tmp/dummy"),
            split_info={
                "source": "Org/Model",
                "transformer_variants": ["dev"],
                "quantized": True,
                "quantization_bits": 8,
            },
            config={"model_version": "2.3.0"},
            repo_id="user/model-mlx-q8",
        )
        assert "Quantization:" in card
        assert "int8" in card

    def test_omits_optional_sections(self):
        from mlx_forge.upload import generate_model_card

        card = generate_model_card(
            Path("/tmp/dummy"),
            split_info={"source": "Org/Model"},
            config={},
            repo_id="user/model-mlx",
        )
        # No transformer_variants → no Transformer variants line
        assert "Transformer variants:" not in card
        # No usage_url and no cli_snippet → no Usage section
        assert "## Usage" not in card
        # No links → no Related Projects section
        assert "## Related Projects" not in card


class TestAddOnlyArgparse:
    def test_default_is_false(self):
        from mlx_forge.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["upload", "models/foo"])
        assert args.add_only is False

    def test_flag_sets_true(self):
        from mlx_forge.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["upload", "models/foo", "--add-only"])
        assert args.add_only is True


class TestAddOnlyBehavior:
    def _setup_dir(self, tmp_path):
        # Three safetensors files locally
        (tmp_path / "transformer-distilled-1.1.safetensors").write_bytes(b"x" * 100)
        (tmp_path / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors").write_bytes(b"y" * 50)
        (tmp_path / "vae_decoder.safetensors").write_bytes(b"z" * 200)
        (tmp_path / "split_model.json").write_text("{}")

    def _make_api(self, remote_files: list[str], repo_exists: bool = True) -> MagicMock:
        api = MagicMock()
        if repo_exists:
            info = MagicMock()
            info.siblings = [MagicMock(rfilename=f) for f in remote_files]
            api.model_info.return_value = info
        else:
            api.model_info.side_effect = RepositoryNotFoundError("not found", response=MagicMock())
        api.create_repo.return_value = "https://huggingface.co/test/repo"
        return api

    def test_uploads_only_new_files(self, tmp_path):
        from mlx_forge.upload import upload_model

        self._setup_dir(tmp_path)
        # Remote already has vae_decoder; transformer + lora are new
        api = self._make_api(
            remote_files=["vae_decoder.safetensors", "config.json"],
            repo_exists=True,
        )

        upload_model(tmp_path, api=api, repo_id="test/repo", add_only=True)

        uploaded = [c.kwargs["path_in_repo"] for c in api.upload_file.call_args_list]
        assert "transformer-distilled-1.1.safetensors" in uploaded
        assert "ltx-2.3-22b-distilled-lora-384-1.1.safetensors" in uploaded
        assert "vae_decoder.safetensors" not in uploaded

    def test_refuses_when_repo_not_found(self, tmp_path, capsys):
        from mlx_forge.upload import upload_model

        self._setup_dir(tmp_path)
        api = self._make_api(remote_files=[], repo_exists=False)

        with pytest.raises(SystemExit):
            upload_model(tmp_path, api=api, repo_id="test/repo", add_only=True)
        api.upload_file.assert_not_called()

    def test_nothing_to_upload_exits_cleanly(self, tmp_path, capsys):
        from mlx_forge.upload import upload_model

        self._setup_dir(tmp_path)
        api = self._make_api(
            remote_files=[
                "transformer-distilled-1.1.safetensors",
                "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
                "vae_decoder.safetensors",
                "split_model.json",
            ],
            repo_exists=True,
        )

        upload_model(tmp_path, api=api, repo_id="test/repo", add_only=True)

        api.upload_file.assert_not_called()
        out = capsys.readouterr().out
        assert "Nothing to upload" in out

    def test_refuses_when_model_dir_missing(self, tmp_path, capsys):
        from mlx_forge.upload import upload_model

        api = self._make_api(remote_files=[], repo_exists=True)
        missing = tmp_path / "does-not-exist"

        with pytest.raises(SystemExit):
            upload_model(missing, api=api, repo_id="test/repo", add_only=True)
        api.upload_file.assert_not_called()
        out = capsys.readouterr().out
        assert "does not exist" in out

    def test_includes_readme_when_absent_on_remote(self, tmp_path):
        from mlx_forge.upload import upload_model

        # Local has only a README.md (and split_model.json)
        (tmp_path / "README.md").write_text("# Card")
        (tmp_path / "split_model.json").write_text("{}")

        api = self._make_api(remote_files=["split_model.json"], repo_exists=True)
        upload_model(tmp_path, api=api, repo_id="test/repo", add_only=True)

        uploaded = [c.kwargs["path_in_repo"] for c in api.upload_file.call_args_list]
        assert "README.md" in uploaded


class TestUploadModeMutex:
    def test_card_only_and_add_only_are_mutually_exclusive(self):
        from mlx_forge.cli import build_parser

        parser = build_parser()
        # argparse calls SystemExit on conflicting mutex group args
        with pytest.raises(SystemExit):
            parser.parse_args(["upload", "models/foo", "--card-only", "--add-only"])


class TestCardOnlyRemoteRefresh:
    """Remote-derived variants reach the card — assembled by the CLI, not here.

    upload_model used to regenerate the card itself, from the manifest alone and
    without the file listing, links or license the caller had assembled. The
    card that went up was therefore not the one --dry-run had shown: refreshing
    dgrauet/matrix-game-3.0-mlx dropped its Related Projects section. It now
    pushes the file on disk, and these tests drive the CLI.
    """

    def _remote(self, *filenames):
        api = MagicMock()
        info = MagicMock()
        info.siblings = [MagicMock(rfilename=f, size=1) for f in filenames]
        api.model_info.return_value = info
        api.create_repo.return_value = "https://huggingface.co/test/repo"
        return api

    def _run(self, model_dir, api):
        from mlx_forge.cli import main

        with (
            patch(
                "sys.argv",
                ["mlx-forge", "upload", str(model_dir), "--repo-id", "test/repo", "--card-only"],
            ),
            patch("huggingface_hub.HfApi", return_value=api),
        ):
            main()
        return (model_dir / "README.md").read_text()

    def test_card_only_uses_remote_variants(self, tmp_path):
        (tmp_path / "transformer-distilled-1.1.safetensors").write_bytes(b"x")
        (tmp_path / "split_model.json").write_text(
            json.dumps({"source": "Lightricks/LTX-2.3", "transformer_variants": ["distilled-1.1"]})
        )
        (tmp_path / "config.json").write_text(json.dumps({"model_version": "2.3.0"}))

        card = self._run(
            tmp_path,
            self._remote(
                "transformer-distilled.safetensors",
                "transformer-dev.safetensors",
                "transformer-distilled-1.1.safetensors",
                "ltx-2.3-22b-distilled-lora-384.safetensors",
                "config.json",
            ),
        )

        for variant in ("distilled", "dev", "distilled-1.1"):
            assert variant in card

    def test_the_card_pushed_is_the_card_generated(self, tmp_path):
        """upload_model must not rebuild it: that is how sections went missing."""
        from mlx_forge.upload import upload_model

        (tmp_path / "split_model.json").write_text(json.dumps({"source": "Org/M"}))
        (tmp_path / "README.md").write_text("# sentinel\n\n## Related Projects\n\n- **x:** y\n")
        api = self._remote("model.safetensors")

        upload_model(tmp_path, api=api, repo_id="test/repo", card_only=True)

        pushed = next(
            c for c in api.upload_file.call_args_list if c.kwargs["path_in_repo"] == "README.md"
        )
        assert Path(pushed.kwargs["path_or_fileobj"]).read_text() == (
            "# sentinel\n\n## Related Projects\n\n- **x:** y\n"
        )

    def test_missing_card_is_refused(self, tmp_path):
        from mlx_forge.upload import upload_model

        (tmp_path / "split_model.json").write_text(json.dumps({"source": "Org/M"}))
        with pytest.raises(SystemExit):
            upload_model(tmp_path, api=self._remote(), repo_id="test/repo", card_only=True)


def test_add_only_warns_about_flags_it_ignores(tmp_path, capsys):
    from unittest.mock import MagicMock

    from mlx_forge.upload import upload_model

    api = MagicMock()
    info = MagicMock()
    info.siblings = []
    api.model_info.return_value = info
    (tmp_path / "a.safetensors").write_bytes(b"x")

    upload_model(
        tmp_path,
        api=api,
        repo_id="acme/demo",
        commit_message="m",
        private=True,
        collection_title="My collection",
        add_only=True,
    )

    out = capsys.readouterr().out
    assert "--private" in out and "ignored" in out
    assert "--collection" in out
