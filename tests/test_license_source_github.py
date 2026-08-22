"""Licence copies fetched from somewhere other than the upstream Hub repo.

LTX-2.5 publishes no LICENSE file on the Hub; the agreement its weights carry
lives in the Lightricks/LTX-2 GitHub repository. The provenance record
generalises without changing shape — a GitHub repo has an id and a commit as a
Hub repo has an id and a revision — so only the fetching learns a second scheme.
"""

from mlx_forge.metadata import LICENSE_KEYS, SPLIT_MODEL_KEYS, RecipeMetadata


class TestDeclaration:
    def test_license_source_is_carried_into_the_manifest(self):
        meta = RecipeMetadata(
            name="ltx-2.5",
            source="Lightricks/LTX-2.5",
            license_file="LICENSE",
            license_source="github:Lightricks/LTX-2/LICENSE",
        )
        assert meta.as_split_fields()["license_source"] == "github:Lightricks/LTX-2/LICENSE"

    def test_license_source_is_recipe_authoritative(self):
        # Same reasoning as the other licence keys: a licence is a fact about
        # the upstream model, not a property of one build, so correcting the
        # recipe must reach every already-published pack.
        assert "license_source" in LICENSE_KEYS
        assert "license_source" in SPLIT_MODEL_KEYS

    def test_gated_is_carried_when_true(self):
        meta = RecipeMetadata(name="r", source="a/b", gated=True)
        assert meta.as_split_fields()["gated"] is True

    def test_gated_is_omitted_when_false(self):
        # Ten published manifests have no `gated` key; adding a false one to
        # each on the next refresh would be noise in every card diff.
        meta = RecipeMetadata(name="r", source="a/b")
        assert "gated" not in meta.as_split_fields()

    def test_absent_license_source_changes_nothing(self):
        meta = RecipeMetadata(name="r", source="a/b", license_file="LICENSE")
        assert "license_source" not in meta.as_split_fields()
