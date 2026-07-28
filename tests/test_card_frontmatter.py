"""base_model and license must not degrade when a card is regenerated.

Refreshing dgrauet/ernie-image-pe-mlx with the code as it stood would have
published `license: other` in place of apache-2.0 (the CLI default won
unconditionally) and `base_model: baidu/ERNIE-Image-Turbo/pe`, which is a
subfolder and does not resolve on the Hub. Both were caught by comparing the
regenerated card against the published one before pushing.
"""

from __future__ import annotations

import importlib

import pytest

from mlx_forge.metadata import RecipeMetadata, hub_repo_from_source, is_hub_repo_id
from mlx_forge.recipes import AVAILABLE_RECIPES
from mlx_forge.upload import generate_model_card

RECIPE_NAMES = sorted(AVAILABLE_RECIPES)


def _front_matter(card: str) -> dict[str, str]:
    block = card.split("---")[1]
    out = {}
    for line in block.splitlines():
        if ":" in line and not line.startswith(" ") and not line.startswith("-"):
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


class TestIsHubRepoId:
    @pytest.mark.parametrize(
        "value", ["Lightricks/LTX-2.3", "facebook/vjepa2-vitl-fpc64-256", "baidu/ERNIE-Image"]
    )
    def test_accepts_repo_ids(self, value):
        assert is_hub_repo_id(value)

    @pytest.mark.parametrize(
        "value",
        [
            "baidu/ERNIE-Image-Turbo/pe",  # a subfolder
            "facebookresearch/vjepa2 (app/vjepa_2_1)",  # a source tree
            "netflix-void",  # no owner
            "",
            None,
        ],
    )
    def test_rejects_everything_else(self, value):
        assert not is_hub_repo_id(value)


class TestHubRepoFromSource:
    """base_model names the remote repo, whatever variant lives inside it."""

    @pytest.mark.parametrize(
        "source,expected",
        [
            ("baidu/ERNIE-Image-Turbo/pe", "baidu/ERNIE-Image-Turbo"),
            ("Lightricks/LTX-2.3", "Lightricks/LTX-2.3"),
            ("netflix/void-model", "netflix/void-model"),
            ("org/repo/a/b/c", "org/repo"),
        ],
    )
    def test_drops_anything_inside_the_repo(self, source, expected):
        assert hub_repo_from_source(source) == expected

    @pytest.mark.parametrize(
        "source", ["facebookresearch/vjepa2 (app/vjepa_2_1)", "netflix-void", "", None]
    )
    def test_returns_none_when_no_repo_is_named(self, source):
        assert hub_repo_from_source(source) is None


class TestBaseModelResolution:
    def test_subfolder_source_resolves_to_its_repo(self, tmp_path):
        """No declaration needed: /pe is inside baidu/ERNIE-Image-Turbo."""
        card = generate_model_card(
            tmp_path,
            split_info={"source": "baidu/ERNIE-Image-Turbo/pe"},
            config={},
            repo_id="u/m",
        )
        assert _front_matter(card)["base_model"] == "baidu/ERNIE-Image-Turbo"

    def test_declared_base_model_wins_over_prose_source(self, tmp_path):
        card = generate_model_card(
            tmp_path,
            split_info={
                "source": "baidu/ERNIE-Image-Turbo/pe",
                "base_model": "baidu/ERNIE-Image-Turbo",
            },
            config={},
            repo_id="u/m",
        )
        assert _front_matter(card)["base_model"] == "baidu/ERNIE-Image-Turbo"

    def test_valid_source_is_used_when_nothing_is_declared(self, tmp_path):
        card = generate_model_card(
            tmp_path, split_info={"source": "Lightricks/LTX-2.3"}, config={}, repo_id="u/m"
        )
        assert _front_matter(card)["base_model"] == "Lightricks/LTX-2.3"

    def test_unresolvable_source_emits_no_base_model(self, tmp_path):
        """Better absent than pointing at something that 404s."""
        card = generate_model_card(
            tmp_path,
            split_info={"source": "facebookresearch/vjepa2 (app/vjepa_2_1)"},
            config={},
            repo_id="u/m",
        )
        assert "base_model" not in _front_matter(card)

    def test_explicit_argument_still_wins(self, tmp_path):
        card = generate_model_card(
            tmp_path,
            split_info={"source": "Lightricks/LTX-2.3"},
            config={},
            repo_id="u/m",
            base_model="someone/else",
        )
        assert _front_matter(card)["base_model"] == "someone/else"


class TestLicenseResolution:
    def test_declared_license_survives_a_refresh(self, tmp_path):
        card = generate_model_card(
            tmp_path, split_info={"license": "apache-2.0"}, config={}, repo_id="u/m"
        )
        assert _front_matter(card)["license"] == "apache-2.0"

    def test_falls_back_to_other(self, tmp_path):
        card = generate_model_card(tmp_path, split_info={}, config={}, repo_id="u/m")
        assert _front_matter(card)["license"] == "other"

    def test_explicit_argument_wins(self, tmp_path):
        card = generate_model_card(
            tmp_path,
            split_info={"license": "apache-2.0"},
            config={},
            repo_id="u/m",
            license_id="mit",
        )
        assert _front_matter(card)["license"] == "mit"


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_every_recipe_declares_a_usable_base_model(recipe_name: str):
    """Either source is a repo id, or base_model is declared, or neither is emitted."""
    md = importlib.import_module(AVAILABLE_RECIPES[recipe_name]).METADATA
    if md.base_model is not None:
        assert is_hub_repo_id(md.base_model), f"{recipe_name}: base_model must resolve on the Hub"


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_declared_metadata_reaches_split_model(recipe_name: str):
    md = importlib.import_module(AVAILABLE_RECIPES[recipe_name]).METADATA
    fields = md.as_split_fields()
    assert fields["source"] == md.source
    if md.license:
        assert fields["license"] == md.license
    if md.base_model:
        assert fields["base_model"] == md.base_model


def test_metadata_without_optional_fields_stays_minimal():
    assert RecipeMetadata(source="a/b").as_split_fields() == {"source": "a/b"}
