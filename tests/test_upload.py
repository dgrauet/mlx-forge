"""Tests for upload utilities: repo ID derivation, model card generation, metadata loading."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx

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
