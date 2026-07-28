"""A converted directory must be traceable back to the recipe that made it.

Metadata is declared in the recipe and travels through split_model.json at
convert time. A directory converted before a field existed carries a manifest
without it, so `upload` would publish the default — a license reverting to
"other" — even though the correct value sits in the code. The binding closes
that gap.
"""

from __future__ import annotations

import importlib
import json

import pytest

from mlx_forge.recipes import AVAILABLE_RECIPES, resolve_recipe_metadata
from mlx_forge.upload import backfill_from_recipe

RECIPE_NAMES = sorted(AVAILABLE_RECIPES)


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_declared_name_matches_the_registry_key(recipe_name: str):
    """Otherwise the `recipe` key written into split_model.json resolves to nothing."""
    metadata = importlib.import_module(AVAILABLE_RECIPES[recipe_name]).METADATA
    assert metadata.name == recipe_name


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_resolves_by_the_recipe_key(recipe_name: str):
    resolved = resolve_recipe_metadata({"recipe": recipe_name})
    assert resolved is not None
    assert resolved.name == recipe_name


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_resolves_by_source_for_older_directories(recipe_name: str):
    """Manifests written before the `recipe` key still carry a unique source."""
    metadata = importlib.import_module(AVAILABLE_RECIPES[recipe_name]).METADATA
    resolved = resolve_recipe_metadata({"source": metadata.source})
    assert resolved is not None
    assert resolved.name == recipe_name


class TestResolutionLimits:
    def test_unknown_source_resolves_to_nothing(self):
        assert resolve_recipe_metadata({"source": "someone/unknown"}) is None

    def test_empty_manifest_resolves_to_nothing(self):
        assert resolve_recipe_metadata({}) is None

    def test_unknown_recipe_key_falls_back_to_source(self):
        resolved = resolve_recipe_metadata(
            {"recipe": "deleted-recipe", "source": "netflix/void-model"}
        )
        assert resolved is not None
        assert resolved.name == "void-model"


class TestBackfill:
    def test_recovers_a_license_the_manifest_predates(self, tmp_path):
        """The ernie-image-pe case: source is there, license is not."""
        stale = {"format": "split", "source": "baidu/ERNIE-Image-Turbo/pe"}
        (tmp_path / "split_model.json").write_text(json.dumps(stale))

        merged = backfill_from_recipe(tmp_path, stale)

        assert merged["license"] == "apache-2.0"
        assert merged["recipe"] == "ernie-image-pe"
        assert merged["format"] == "split", "existing keys must survive"

    def test_persists_so_the_next_refresh_needs_no_lookup(self, tmp_path):
        stale = {"source": "netflix/void-model"}
        (tmp_path / "split_model.json").write_text(json.dumps(stale))

        backfill_from_recipe(tmp_path, stale)

        stored = json.loads((tmp_path / "split_model.json").read_text())
        assert stored["recipe"] == "void-model"
        assert stored["license"] == "apache-2.0"

    def test_manifest_values_win_over_the_declaration(self, tmp_path):
        """A deliberate per-model override must not be reverted."""
        manifest = {"source": "netflix/void-model", "license": "mit"}
        (tmp_path / "split_model.json").write_text(json.dumps(manifest))

        assert backfill_from_recipe(tmp_path, manifest)["license"] == "mit"

    def test_unidentifiable_directory_is_left_alone(self, tmp_path):
        manifest = {"source": "someone/unknown"}
        assert backfill_from_recipe(tmp_path, manifest) == manifest
        assert not (tmp_path / "split_model.json").exists()

    def test_nothing_written_when_already_complete(self, tmp_path):
        from mlx_forge.recipes import void_model

        complete = {**void_model.METADATA.as_split_fields()}
        assert backfill_from_recipe(tmp_path, complete) == complete
        assert not (tmp_path / "split_model.json").exists()
