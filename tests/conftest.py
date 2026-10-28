"""Suite-wide fixtures."""

from __future__ import annotations

import pytest

from mlx_forge import convert


class UnexpectedNetworkAccessError(ConnectionError):
    """A test reached the real Hub or GitHub instead of a stub."""


@pytest.fixture(autouse=True)
def _no_real_downloads(monkeypatch):
    """Run the suite as if offline: no test may fetch from the Hub or GitHub.

    The suite once depended on upstream state without knowing it: when
    Lightricks deleted LTX-2.3's Hub LICENSE (2026-10-22), four upload tests
    started failing on an unrelated Dependabot PR. A ConnectionError, so code
    paths that tolerate a failed fetch (a non-strict licence copy during
    convert) behave exactly as they would offline, and the ones that require it
    fail deterministically instead of depending on upstream. Tests that need a
    download stub it themselves; their monkeypatch wins.
    """

    def refuse(*args, **kwargs):
        raise UnexpectedNetworkAccessError(
            f"test reached the network ({args or ''} {kwargs or ''}); stub the download"
        )

    monkeypatch.setattr(convert, "hf_hub_download", refuse)
    monkeypatch.setattr(convert, "fetch_github_license", refuse)


@pytest.fixture
def stub_github_licence(monkeypatch):
    """A canned GitHub licence fetch, for tests driving an LTX-2.3 upload.

    LTX-2.3 ships the LTX-2 agreement from github.com/Lightricks/LTX-2; a card
    refresh or upload must pass a copy on, so these tests need one to exist.
    """
    monkeypatch.setattr(
        convert,
        "fetch_github_license",
        lambda repo, path: (b"LTX-2 Community License Agreement\n", "0" * 40),
    )
