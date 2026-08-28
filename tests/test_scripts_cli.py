"""The maintenance scripts parse their arguments instead of scanning sys.argv.

A hand-rolled `sys.argv.index("--author")` scan ignored unknown flags and hit
the network before saying so; `--help` printed nothing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
NAMES = [
    "check_card_completeness",
    "check_license_provenance",
    "check_published_cards",
    "check_published_links",
]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", NAMES)
def test_author_is_parsed(name):
    module = _load(name)
    assert module.parse_args(["--author", "someone"]).author == "someone"
    assert module.parse_args([]).author == "dgrauet"


@pytest.mark.parametrize("name", NAMES)
def test_unknown_flag_is_rejected(name):
    module = _load(name)
    with pytest.raises(SystemExit) as exc_info:
        module.parse_args(["--bogus"])
    assert exc_info.value.code == 2


def test_verbose_and_no_links_survive():
    assert _load("check_published_cards").parse_args(["-v"]).verbose is True
    assert _load("check_published_links").parse_args(["--no-links"]).check_links is False
