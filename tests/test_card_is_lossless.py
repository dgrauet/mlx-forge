"""Regenerating a card must never lose content.

Every defect in this area had the same shape: a value that existed in the
published README and nowhere else, because it was typed once as a CLI flag.
Regenerating then republished the default. The invariant is therefore not
"the card is correct" but "regenerating loses nothing", checked against a
manifest that identifies its recipe.

`scripts/check_published_cards.py` applies the same criterion to the real Hub
repos; it needs network, so it is not a test.
"""

from __future__ import annotations

import importlib
import re

import pytest

from mlx_forge.recipes import AVAILABLE_RECIPES
from mlx_forge.upload import generate_model_card

RECIPE_NAMES = sorted(AVAILABLE_RECIPES)


def _sections(card: str) -> set[str]:
    return set(re.findall(r"^## (.+)$", card, re.M))


def _render(metadata, repo_id: str, tmp_path, **extra):
    split_info = {**metadata.as_split_fields(), **extra}
    return generate_model_card(
        tmp_path,
        split_info=split_info,
        config={},
        repo_id=repo_id,
        file_listing={"model.safetensors": 1024},
        usage_url=split_info.get("usage_url"),
        links=split_info.get("links"),
        cli_snippet=split_info.get("cli_snippet"),
    )


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_regenerating_is_idempotent(recipe_name: str, tmp_path):
    """The same declaration must render the same card, every time."""
    metadata = importlib.import_module(AVAILABLE_RECIPES[recipe_name]).METADATA
    first = _render(metadata, "u/m", tmp_path)
    assert first == _render(metadata, "u/m", tmp_path)


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_declared_sections_all_render(recipe_name: str, tmp_path):
    """A declared usage_url or links must reach the card, not sit unused."""
    metadata = importlib.import_module(AVAILABLE_RECIPES[recipe_name]).METADATA
    card = _render(metadata, "u/m", tmp_path)

    if metadata.usage_url:
        assert "Usage" in _sections(card)
        assert metadata.usage_url in card
    if metadata.links:
        assert "Related Projects" in _sections(card)
        for link in metadata.links:
            assert link.split(": ", 1)[1] in card


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_nothing_declared_is_dropped(recipe_name: str, tmp_path):
    """Every declared value must be findable in the rendered card."""
    metadata = importlib.import_module(AVAILABLE_RECIPES[recipe_name]).METADATA
    card = _render(metadata, "u/m", tmp_path)

    for value in (metadata.license, metadata.usage_url, metadata.usage_note):
        if value:
            assert value in card, f"{recipe_name}: declared {value!r} is missing from the card"


class TestSnippetTemplating:
    def test_repo_id_is_substituted(self, tmp_path):
        from mlx_forge.recipes import ernie_image_pe as pe

        card = _render(pe.METADATA, "dgrauet/ernie-image-pe-mlx-q4", tmp_path)
        assert "--pe-repo-id dgrauet/ernie-image-pe-mlx-q4" in card
        assert "{repo_id}" not in card, "the placeholder must never reach the card"

    def test_one_declaration_serves_every_quantization(self, tmp_path):
        from mlx_forge.recipes import ernie_image_pe as pe

        for repo in ("dgrauet/ernie-image-pe-mlx", "dgrauet/ernie-image-pe-mlx-q4"):
            assert f"--pe-repo-id {repo}" in _render(pe.METADATA, repo, tmp_path)

    def test_a_snippet_without_placeholder_is_untouched(self, tmp_path):
        from mlx_forge.metadata import RecipeMetadata

        metadata = RecipeMetadata(name="r", source="a/b", cli_snippet="run --thing")
        assert "run --thing" in _render(metadata, "u/m", tmp_path)


class TestKnownSources:
    def test_a_variant_source_identifies_the_recipe(self):
        from mlx_forge.recipes import ernie_image, resolve_recipe_metadata

        for source in ernie_image.METADATA.known_sources:
            resolved = resolve_recipe_metadata({"source": source})
            assert resolved is not None
            assert resolved.name == "ernie-image"

    def test_recipes_without_variants_declare_none(self):
        from mlx_forge.recipes import void_model

        assert void_model.METADATA.known_sources == ()
