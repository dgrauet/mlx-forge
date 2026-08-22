"""Licence copies fetched from somewhere other than the upstream Hub repo.

LTX-2.5 publishes no LICENSE file on the Hub; the agreement its weights carry
lives in the Lightricks/LTX-2 GitHub repository. The provenance record
generalises without changing shape — a GitHub repo has an id and a commit as a
Hub repo has an id and a revision — so only the fetching learns a second scheme.
"""

import pytest

from mlx_forge import convert as convert_mod
from mlx_forge.convert import ensure_license_file, parse_license_source
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


LICENCE_TEXT = b"LTX-2.x Community License Agreement\nLicense date: August 11, 2026\n"


class TestParseLicenseSource:
    def test_parses_a_github_spec(self):
        assert parse_license_source("github:Lightricks/LTX-2/LICENSE") == (
            "Lightricks/LTX-2",
            "LICENSE",
        )

    def test_parses_a_nested_path(self):
        assert parse_license_source("github:o/r/legal/LICENSE.txt") == ("o/r", "legal/LICENSE.txt")

    def test_returns_none_for_anything_else(self):
        assert parse_license_source("Lightricks/LTX-2.5") is None
        assert parse_license_source("") is None

    def test_rejects_a_truncated_spec(self):
        # "github:owner/repo" names no file; silently treating it as a Hub repo
        # would fetch the wrong thing rather than say the declaration is wrong.
        with pytest.raises(ValueError, match="github:"):
            parse_license_source("github:Lightricks/LTX-2")


class TestFetchFromGithub:
    def test_writes_the_file_and_records_github_provenance(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            convert_mod,
            "fetch_github_license",
            lambda repo, path: (LICENCE_TEXT, "abc123"),
        )
        info: dict = {
            "source": "Lightricks/LTX-2.5",
            "license_file": ["LICENSE"],
            "license_source": "github:Lightricks/LTX-2/LICENSE",
        }

        written = ensure_license_file(tmp_path, info)

        assert written == [tmp_path / "LICENSE"]
        assert (tmp_path / "LICENSE").read_bytes() == LICENCE_TEXT
        record = info["license_provenance"]["LICENSE"]
        assert record["repo"] == "github:Lightricks/LTX-2"
        assert record["revision"] == "abc123"

    def test_a_second_run_makes_no_network_call(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            convert_mod,
            "fetch_github_license",
            lambda repo, path: (LICENCE_TEXT, "abc123"),
        )
        info = {
            "source": "Lightricks/LTX-2.5",
            "license_file": ["LICENSE"],
            "license_source": "github:Lightricks/LTX-2/LICENSE",
        }
        ensure_license_file(tmp_path, info)

        def explode(repo, path):
            raise AssertionError("second run must not fetch")

        monkeypatch.setattr(convert_mod, "fetch_github_license", explode)
        assert ensure_license_file(tmp_path, info) == [tmp_path / "LICENSE"]

    def test_refuses_a_copy_that_drifted_from_its_record(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            convert_mod,
            "fetch_github_license",
            lambda repo, path: (LICENCE_TEXT, "abc123"),
        )
        info = {
            "source": "Lightricks/LTX-2.5",
            "license_file": ["LICENSE"],
            "license_source": "github:Lightricks/LTX-2/LICENSE",
        }
        ensure_license_file(tmp_path, info)
        (tmp_path / "LICENSE").write_bytes(b"something else entirely\n")

        with pytest.raises(SystemExit, match="undocumented source"):
            ensure_license_file(tmp_path, info)

    def test_a_hub_source_is_untouched_by_the_new_branch(self, tmp_path, monkeypatch):
        # The ten existing recipes declare no license_source; their path must
        # not acquire a GitHub detour.
        calls = []
        monkeypatch.setattr(
            convert_mod,
            "fetch_github_license",
            lambda repo, path: calls.append(repo) or (b"", None),
        )
        info = {"source": "a/b", "license_file": ["LICENSE"]}
        (tmp_path / "LICENSE").write_bytes(LICENCE_TEXT)
        ensure_license_file(tmp_path, info, strict=False)
        assert calls == []

    def test_a_mismatch_against_an_unrecorded_local_copy_leaves_no_scratch_file(
        self, tmp_path, monkeypatch
    ):
        # The "older pack, vouch against upstream" case: a local LICENSE
        # exists with no recorded provenance, and the freshly fetched GitHub
        # copy does not match it. The refusal must not leave the `.fetched`
        # scratch file behind in the model directory.
        monkeypatch.setattr(
            convert_mod,
            "fetch_github_license",
            lambda repo, path: (LICENCE_TEXT, "abc123"),
        )
        info = {
            "source": "Lightricks/LTX-2.5",
            "license_file": ["LICENSE"],
            "license_source": "github:Lightricks/LTX-2/LICENSE",
        }
        (tmp_path / "LICENSE").write_bytes(b"something else entirely\n")

        with pytest.raises(SystemExit, match="differs"):
            ensure_license_file(tmp_path, info)

        assert list(tmp_path.glob(".*.fetched")) == []
