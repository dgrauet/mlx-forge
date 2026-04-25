"""Tests for upload utilities: repo ID derivation, model card generation, metadata loading."""

import json
from pathlib import Path
from unittest.mock import MagicMock

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


class TestCardOnlyRemoteRefresh:
    def test_card_only_uses_remote_variants(self, tmp_path):
        from mlx_forge.upload import upload_model

        # Local dir has only one variant (delta convert leftover)
        (tmp_path / "transformer-distilled-1.1.safetensors").write_bytes(b"x")
        (tmp_path / "split_model.json").write_text(
            json.dumps({"source": "Lightricks/LTX-2.3", "transformer_variants": ["distilled-1.1"]})
        )
        (tmp_path / "config.json").write_text(json.dumps({"model_version": "2.3.0"}))

        # Remote has all three transformer variants
        api = MagicMock()
        info = MagicMock()
        info.siblings = [
            MagicMock(rfilename="transformer-distilled.safetensors"),
            MagicMock(rfilename="transformer-dev.safetensors"),
            MagicMock(rfilename="transformer-distilled-1.1.safetensors"),
            MagicMock(rfilename="ltx-2.3-22b-distilled-lora-384.safetensors"),
            MagicMock(rfilename="ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
            MagicMock(rfilename="config.json"),
        ]
        api.model_info.return_value = info
        api.create_repo.return_value = "https://huggingface.co/test/repo"

        upload_model(tmp_path, api=api, repo_id="test/repo", card_only=True)

        readme_call = next(
            c for c in api.upload_file.call_args_list if c.kwargs["path_in_repo"] == "README.md"
        )
        readme_path = readme_call.kwargs["path_or_fileobj"]
        readme_text = Path(readme_path).read_text()
        # All three transformer variants must appear in the card
        assert "distilled" in readme_text
        assert "dev" in readme_text
        assert "distilled-1.1" in readme_text

    def test_card_only_falls_back_on_network_error(self, tmp_path):
        """When api.model_info raises a network error, fall back to local split_info."""
        from huggingface_hub.errors import HfHubHTTPError

        from mlx_forge.upload import upload_model

        # Local has TWO variants; remote will fail to respond
        (tmp_path / "transformer-distilled.safetensors").write_bytes(b"x")
        (tmp_path / "transformer-dev.safetensors").write_bytes(b"y")
        (tmp_path / "split_model.json").write_text(
            json.dumps(
                {
                    "source": "Lightricks/LTX-2.3",
                    "transformer_variants": ["distilled", "dev"],
                }
            )
        )
        (tmp_path / "config.json").write_text(json.dumps({"model_version": "2.3.0"}))

        api = MagicMock()
        api.model_info.side_effect = HfHubHTTPError("503 Service Unavailable", response=MagicMock())
        api.create_repo.return_value = "https://huggingface.co/test/repo"

        # Should NOT raise; falls back to local split_info
        upload_model(tmp_path, api=api, repo_id="test/repo", card_only=True)

        readme_call = next(
            c for c in api.upload_file.call_args_list if c.kwargs["path_in_repo"] == "README.md"
        )
        readme_path = readme_call.kwargs["path_or_fileobj"]
        readme_text = Path(readme_path).read_text()
        # Local variants are present in the card
        assert "distilled" in readme_text
        assert "dev" in readme_text
