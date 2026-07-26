"""Keep the docs honest about what the code actually supports.

The README's model table drifted to 6 of 11 recipes before anyone noticed, and
removing a recipe left dangling links behind. Both are mechanical to check, so
they are checked here rather than by eye.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from mlx_forge.recipes import AVAILABLE_RECIPES

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

#: A row of the "Supported Models" table: | [Name](url) (blurb) | `recipe` | Status |
_TABLE_ROW = re.compile(r"^\| \[.*?\| `([^`]+)` \| (\w+) \|$", re.M)

#: Relative markdown links, excluding external URLs and pure anchors.
_RELATIVE_LINK = re.compile(r"\]\((?!https?:|#)([^)#]+)")


def _markdown_files() -> list[Path]:
    return [README, REPO_ROOT / "CLAUDE.md", *sorted((REPO_ROOT / "docs").rglob("*.md"))]


class TestReadmeModelTable:
    def _rows(self) -> dict[str, str]:
        return {recipe: status for recipe, status in _TABLE_ROW.findall(README.read_text())}

    def test_every_recipe_is_listed(self):
        assert set(self._rows()) >= set(AVAILABLE_RECIPES), (
            "a recipe exists but the README does not mention it"
        )

    def test_no_recipe_is_listed_that_does_not_exist(self):
        assert set(self._rows()) <= set(AVAILABLE_RECIPES), (
            "the README advertises a recipe the CLI cannot run"
        )

    def test_each_recipe_appears_once(self):
        listed = [r for r, _ in _TABLE_ROW.findall(README.read_text())]
        assert len(listed) == len(set(listed))

    # Row order is deliberately NOT asserted: the README leads with the flagship
    # model, which need not be whatever comes first in AVAILABLE_RECIPES.


@pytest.mark.parametrize("path", _markdown_files(), ids=lambda p: p.name)
def test_relative_links_resolve(path: Path):
    broken = [t for t in _RELATIVE_LINK.findall(path.read_text()) if not (path.parent / t).exists()]
    assert broken == []


def test_recipe_module_paths_are_importable():
    """AVAILABLE_RECIPES values must be real modules, not stale dotted paths."""
    import importlib

    for name, module_path in AVAILABLE_RECIPES.items():
        assert importlib.import_module(module_path), name
