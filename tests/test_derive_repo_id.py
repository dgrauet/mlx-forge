"""The auto-derived repo name must be the repo the model belongs to.

`mlx-forge upload models/ernie-image-pe-mlx` derived `dgrauet/pe-mlx`: the name
came from the last path segment of `source`, which is
"baidu/ERNIE-Image-Turbo/pe" — a subfolder. A real run would have created a junk
repo. Caught by --dry-run before it happened.

The declaration is the reliable signal: `recipe` (+ `variant`) identifies the
build, and is exactly what the published repo names encode.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from mlx_forge.upload import derive_repo_id


def _api(user: str = "dgrauet"):
    api = MagicMock()
    api.whoami.return_value = {"name": user}
    return api


#: (recipe, variant, quantization bits) -> the repo published on the Hub today.
PUBLISHED = [
    ("ltx-2.3", None, None, "dgrauet/ltx-2.3-mlx"),
    ("ltx-2.3", None, 8, "dgrauet/ltx-2.3-mlx-q8"),
    ("matrix-game-3.0", None, None, "dgrauet/matrix-game-3.0-mlx"),
    ("void-model", None, 4, "dgrauet/void-model-mlx-q4"),
    ("hunyuan3d-2.1", None, None, "dgrauet/hunyuan3d-2.1-mlx"),
    ("ernie-image-pe", None, None, "dgrauet/ernie-image-pe-mlx"),
    ("ernie-image-pe", None, 4, "dgrauet/ernie-image-pe-mlx-q4"),
    ("ernie-image", "sft", None, "dgrauet/ernie-image-sft-mlx"),
    ("ernie-image", "turbo", 8, "dgrauet/ernie-image-turbo-mlx-q8"),
    ("vjepa-2.0-vitl", None, None, "dgrauet/vjepa-2.0-vitl-mlx"),
    ("vjepa-2.1-vitl", None, None, "dgrauet/vjepa-2.1-vitl-mlx"),
]


@pytest.mark.parametrize("recipe,variant,bits,expected", PUBLISHED)
def test_matches_the_published_repo(recipe, variant, bits, expected, tmp_path):
    split_info: dict = {"recipe": recipe, "source": "irrelevant/when-recipe-is-known"}
    if variant:
        split_info["variant"] = variant
    if bits:
        split_info["quantized"] = True
        split_info["quantization_bits"] = bits

    assert derive_repo_id(split_info, tmp_path, api=_api()) == expected


class TestSubfolderSource:
    def test_a_subfolder_never_becomes_the_repo_name(self, tmp_path):
        """The reported bug: "baidu/ERNIE-Image-Turbo/pe" derived "pe-mlx"."""
        model_dir = tmp_path / "ernie-image-pe-mlx"
        model_dir.mkdir()

        repo = derive_repo_id({"source": "baidu/ERNIE-Image-Turbo/pe"}, model_dir, api=_api())

        assert repo != "dgrauet/pe-mlx"
        assert repo == "dgrauet/ernie-image-pe-mlx"

    def test_a_subfolder_does_not_resolve_to_its_parent_repo(self, tmp_path):
        """Worse than a junk name: targeting the sibling model's real repo."""
        model_dir = tmp_path / "ernie-image-pe-mlx"
        model_dir.mkdir()

        repo = derive_repo_id({"source": "baidu/ERNIE-Image-Turbo/pe"}, model_dir, api=_api())

        assert repo != "dgrauet/ernie-image-turbo-mlx"


class TestFallbacks:
    def test_directory_name_is_used_when_the_manifest_is_bare(self, tmp_path):
        model_dir = tmp_path / "vjepa-2.0-vitl-mlx-q8"
        model_dir.mkdir()
        assert derive_repo_id({}, model_dir, api=_api()) == "dgrauet/vjepa-2.0-vitl-mlx-q8"

    def test_plain_source_still_works(self, tmp_path):
        assert (
            derive_repo_id({"source": "Lightricks/LTX-2.3"}, tmp_path, api=_api())
            == "dgrauet/ltx-2.3-mlx"
        )

    def test_explicit_namespace(self, tmp_path):
        assert (
            derive_repo_id({"recipe": "ltx-2.3"}, tmp_path, api=_api(), namespace="org")
            == "org/ltx-2.3-mlx"
        )

    def test_bits_from_the_directory_name_survive(self, tmp_path):
        """Some recipes record quantization only in the dir name."""
        model_dir = tmp_path / "vjepa-2.1-vitl-mlx-q8"
        model_dir.mkdir()
        assert (
            derive_repo_id({"recipe": "vjepa-2.1-vitl"}, model_dir, api=_api())
            == "dgrauet/vjepa-2.1-vitl-mlx-q8"
        )


def test_no_double_mlx_suffix(tmp_path):
    model_dir = tmp_path / "ltx-2.3-mlx"
    model_dir.mkdir()
    assert derive_repo_id({"recipe": "ltx-2.3"}, model_dir, api=_api()).count("-mlx") == 1
