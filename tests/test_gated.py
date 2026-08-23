"""Gating is declared by the recipe, reported by upload, and never a side effect.

Changing who may download a published repo is an outward-facing act. The
recipe says what the gating should be; upload says whether it matches; only an
explicit --set-gated changes it.
"""

from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from mlx_forge import convert as convert_mod
from mlx_forge.convert import download_hf_files
from mlx_forge.upload import apply_gating, gating_mismatch, repo_exists, resolve_gating

if TYPE_CHECKING:
    from huggingface_hub import HfApi


class FakeInfo:
    def __init__(self, gated):
        self.gated = gated


class FakeApi:
    """Records calls in order, so ordering guarantees (gate-before-upload) can be asserted."""

    def __init__(self, gated=False, exists=True):
        self._gated = gated
        self._exists = exists
        self.settings_calls = []
        self.create_repo_calls = []
        self.call_order: list[str] = []

    def model_info(self, repo_id, **kwargs):
        if not self._exists:
            response = Mock()
            response.status_code = 404
            raise RepositoryNotFoundError(f"{repo_id} not found", response=response)
        return FakeInfo(self._gated)

    def create_repo(self, repo_id, *, exist_ok=True, private=False, **kwargs):
        self._exists = True
        self.create_repo_calls.append(repo_id)
        self.call_order.append("create_repo")
        return f"https://huggingface.co/{repo_id}"

    def update_repo_settings(self, repo_id, *, gated=None, **kwargs):
        self.settings_calls.append((repo_id, gated))
        self.call_order.append("update_repo_settings")

    def upload_file(self, **kwargs):
        self.call_order.append("upload_file")

    def upload_folder(self, **kwargs):
        self.call_order.append("upload_folder")


def api(gated=False, exists=True):
    return cast("HfApi", FakeApi(gated, exists=exists))


class TestGatingMismatch:
    def test_reports_an_open_repo_that_should_be_gated(self):
        message = gating_mismatch("me/ltx-2.5-mlx", {"gated": True}, api(gated=False))
        assert message is not None
        assert "--set-gated" in message

    def test_silent_when_the_repo_already_matches(self):
        assert gating_mismatch("me/ltx-2.5-mlx", {"gated": True}, api(gated="auto")) is None

    def test_silent_when_nothing_is_declared(self):
        # The ten existing recipes declare no gating and must produce no noise.
        assert gating_mismatch("me/other-mlx", {}, api(gated=False)) is None

    def test_does_not_report_an_undeclared_repo_that_is_gated(self):
        # An operator may have gated a repo by hand. Undoing that is not this
        # tool's business, and reporting it invites exactly that.
        assert gating_mismatch("me/other-mlx", {}, api(gated="manual")) is None

    def test_reporting_never_changes_the_repo(self):
        fake = FakeApi(gated=False)
        gating_mismatch("me/ltx-2.5-mlx", {"gated": True}, cast("HfApi", fake))
        assert fake.settings_calls == []


class TestApplyGating:
    def test_sets_auto_gating_when_declared(self):
        fake = FakeApi(gated=False)
        apply_gating("me/ltx-2.5-mlx", {"gated": True}, cast("HfApi", fake))
        assert fake.settings_calls == [("me/ltx-2.5-mlx", "auto")]

    def test_refuses_when_nothing_is_declared(self):
        fake = FakeApi(gated=False)
        with pytest.raises(SystemExit, match="declares no gating"):
            apply_gating("me/other-mlx", {}, cast("HfApi", fake))
        assert fake.settings_calls == []


class TestGatedDownload:
    def test_says_how_to_get_access(self, tmp_path, monkeypatch, capsys):
        def refuse(**kwargs):
            mock_response = Mock()
            mock_response.status_code = 403
            raise GatedRepoError("Repository is gated", response=mock_response)

        monkeypatch.setattr(convert_mod, "hf_hub_download", refuse)

        with pytest.raises(SystemExit):
            download_hf_files("Lightricks/LTX-2.5", ["README.md"], tmp_path)

        out = capsys.readouterr().out
        assert "huggingface.co/Lightricks/LTX-2.5" in out
        assert "hf auth login" in out

    def test_the_branch_precedes_repository_not_found(self, tmp_path, monkeypatch, capsys):
        # GatedRepoError subclasses RepositoryNotFoundError. If the generic
        # branch is reached first this message never appears.
        def refuse(**kwargs):
            mock_response = Mock()
            mock_response.status_code = 403
            raise GatedRepoError("Repository is gated", response=mock_response)

        monkeypatch.setattr(convert_mod, "hf_hub_download", refuse)
        with pytest.raises(SystemExit):
            download_hf_files("Lightricks/LTX-2.5", ["README.md"], tmp_path)
        assert "not found" not in capsys.readouterr().out


class TestRepoExists:
    def test_true_for_a_repo_that_can_be_read(self):
        assert repo_exists("me/ltx-2.5-mlx", api(exists=True)) is True

    def test_false_for_a_missing_repo(self):
        assert repo_exists("me/ltx-2.5-mlx", api(exists=False)) is False

    def test_a_non_missing_error_is_not_read_as_missing(self):
        # A network or permission problem is not "the repo doesn't exist" —
        # conflating the two would let a caller create/gate a repo that is
        # already there and simply could not be read just now.
        class FlakyApi:
            def model_info(self, repo_id, **kwargs):
                raise ConnectionError("network is down")

        with pytest.raises(ConnectionError):
            repo_exists("me/ltx-2.5-mlx", cast("HfApi", FlakyApi()))


class TestResolveGatingFirstUpload:
    """The repo does not exist yet — the case the defect was in."""

    def test_refuses_a_gated_recipe_without_set_gated(self, capsys):
        fake = FakeApi(exists=False)
        with pytest.raises(SystemExit, match="--set-gated"):
            resolve_gating(
                "me/ltx-2.5-mlx",
                {"gated": True},
                cast("HfApi", fake),
                set_gated=False,
                dry_run=False,
            )
        # Nothing was created and nothing was gated.
        assert fake.create_repo_calls == []
        assert fake.settings_calls == []

    def test_refusal_message_is_actionable(self):
        fake = FakeApi(exists=False)
        with pytest.raises(SystemExit) as excinfo:
            resolve_gating(
                "me/ltx-2.5-mlx",
                {"gated": True},
                cast("HfApi", fake),
                set_gated=False,
                dry_run=False,
            )
        message = str(excinfo.value)
        assert "me/ltx-2.5-mlx" in message
        assert "does not exist" in message
        assert "gated" in message
        assert "--set-gated" in message

    def test_set_gated_creates_then_gates_before_any_upload(self):
        fake = FakeApi(exists=False)
        resolve_gating(
            "me/ltx-2.5-mlx",
            {"gated": True},
            cast("HfApi", fake),
            set_gated=True,
            dry_run=False,
        )
        assert fake.create_repo_calls == ["me/ltx-2.5-mlx"]
        assert fake.settings_calls == [("me/ltx-2.5-mlx", "auto")]
        # The safety property: gating happens before any file could land, so
        # simulate the upload that follows and confirm it comes after.
        fake.upload_folder(repo_id="me/ltx-2.5-mlx")
        assert fake.call_order == ["create_repo", "update_repo_settings", "upload_folder"]

    def test_a_recipe_that_declares_no_gating_is_unaffected(self):
        # The ten existing recipes declare no gating and must keep working on
        # a first upload with no flag at all.
        fake = FakeApi(exists=False)
        resolve_gating(
            "me/other-mlx",
            {},
            cast("HfApi", fake),
            set_gated=False,
            dry_run=False,
        )
        assert fake.create_repo_calls == []
        assert fake.settings_calls == []

    def test_dry_run_refuses_the_same_way_a_real_run_would(self):
        # A dry run must not offer false confidence about a path that would
        # actually be refused (or crash, before this fix).
        fake = FakeApi(exists=False)
        with pytest.raises(SystemExit, match="--set-gated"):
            resolve_gating(
                "me/ltx-2.5-mlx",
                {"gated": True},
                cast("HfApi", fake),
                set_gated=False,
                dry_run=True,
            )
        assert fake.create_repo_calls == []

    def test_dry_run_with_set_gated_describes_create_then_gate(self, capsys):
        fake = FakeApi(exists=False)
        resolve_gating(
            "me/ltx-2.5-mlx",
            {"gated": True},
            cast("HfApi", fake),
            set_gated=True,
            dry_run=True,
        )
        out = capsys.readouterr().out
        assert "create" in out
        assert "gated=auto" in out
        # Nothing actually happened.
        assert fake.create_repo_calls == []
        assert fake.settings_calls == []


class TestResolveGatingExistingRepo:
    """An existing repo keeps today's behaviour: report, never touch, without the flag."""

    def test_reports_a_mismatch_and_changes_nothing(self, capsys):
        fake = FakeApi(gated=False, exists=True)
        resolve_gating(
            "me/ltx-2.5-mlx",
            {"gated": True},
            cast("HfApi", fake),
            set_gated=False,
            dry_run=False,
        )
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "--set-gated" in out
        assert fake.create_repo_calls == []
        assert fake.settings_calls == []

    def test_set_gated_applies_without_creating_a_repo(self):
        fake = FakeApi(gated=False, exists=True)
        resolve_gating(
            "me/ltx-2.5-mlx",
            {"gated": True},
            cast("HfApi", fake),
            set_gated=True,
            dry_run=False,
        )
        assert fake.create_repo_calls == []
        assert fake.settings_calls == [("me/ltx-2.5-mlx", "auto")]

    def test_dry_run_set_gated_reports_without_changing(self, capsys):
        fake = FakeApi(gated=False, exists=True)
        resolve_gating(
            "me/ltx-2.5-mlx",
            {"gated": True},
            cast("HfApi", fake),
            set_gated=True,
            dry_run=True,
        )
        out = capsys.readouterr().out
        assert "would set" in out
        assert fake.settings_calls == []

    def test_no_declaration_produces_no_noise(self, capsys):
        fake = FakeApi(gated=False, exists=True)
        resolve_gating(
            "me/other-mlx",
            {},
            cast("HfApi", fake),
            set_gated=False,
            dry_run=False,
        )
        assert capsys.readouterr().out == ""
        assert fake.create_repo_calls == []
        assert fake.settings_calls == []
