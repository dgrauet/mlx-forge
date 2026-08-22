"""Gating is declared by the recipe, reported by upload, and never a side effect.

Changing who may download a published repo is an outward-facing act. The
recipe says what the gating should be; upload says whether it matches; only an
explicit --set-gated changes it.
"""

from typing import TYPE_CHECKING, cast
from unittest.mock import Mock

import pytest
from huggingface_hub.errors import GatedRepoError

from mlx_forge import convert as convert_mod
from mlx_forge.convert import download_hf_files
from mlx_forge.upload import apply_gating, gating_mismatch

if TYPE_CHECKING:
    from huggingface_hub import HfApi


class FakeInfo:
    def __init__(self, gated):
        self.gated = gated


class FakeApi:
    def __init__(self, gated=False):
        self._gated = gated
        self.settings_calls = []

    def model_info(self, repo_id, **kwargs):
        return FakeInfo(self._gated)

    def update_repo_settings(self, repo_id, *, gated=None, **kwargs):
        self.settings_calls.append((repo_id, gated))


def api(gated=False):
    return cast("HfApi", FakeApi(gated))


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
