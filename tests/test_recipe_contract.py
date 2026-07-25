"""Contract tests run against every registered recipe.

Recipes are dispatched by duck-typing in cli.py, so a recipe that forgets to
implement one of the required functions used to surface as a raw AttributeError
traceback at dispatch time (six recipes shipped without `split`). These tests
cover the whole registry so a new recipe cannot regress the contract.
"""

from __future__ import annotations

import argparse
import importlib
from unittest.mock import patch

import pytest

from mlx_forge.cli import _require_recipe_command, main
from mlx_forge.recipes import AVAILABLE_RECIPES, COMMAND_REQUIREMENTS, missing_recipe_attrs

RECIPE_NAMES = sorted(AVAILABLE_RECIPES)
COMMANDS = sorted(COMMAND_REQUIREMENTS)


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
@pytest.mark.parametrize("command", COMMANDS)
def test_recipe_implements_command(recipe_name: str, command: str):
    """Every recipe exposes the arg-registration and entry-point for every command."""
    module = importlib.import_module(AVAILABLE_RECIPES[recipe_name])
    assert missing_recipe_attrs(module, command) == []


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
@pytest.mark.parametrize("command", COMMANDS)
def test_recipe_args_register_on_a_parser(recipe_name: str, command: str):
    """add_*_args must accept a bare parser without blowing up."""
    module = importlib.import_module(AVAILABLE_RECIPES[recipe_name])
    getattr(module, COMMAND_REQUIREMENTS[command][0])(argparse.ArgumentParser())


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_split_dispatch_never_raises_attribute_error(recipe_name: str, tmp_path):
    """`mlx-forge split <recipe> <dir>` reports cleanly instead of crashing.

    A recipe with nothing to split prints a message; one that needs a unified
    file it cannot find exits 1. Neither may raise AttributeError.
    """
    argv = ["mlx-forge", "split", recipe_name, str(tmp_path / "missing")]
    with patch("sys.argv", argv):
        try:
            main()
        except SystemExit as exc:
            assert exc.code in (0, 1)


class TestRequireRecipeCommand:
    def test_missing_attr_exits_with_guidance(self, capsys):
        stub = argparse.Namespace()  # a "recipe" implementing nothing
        with pytest.raises(SystemExit) as exc_info:
            _require_recipe_command(stub, "split", "stub-recipe")
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "stub-recipe" in out
        assert "split" in out

    def test_complete_recipe_passes(self):
        module = importlib.import_module(AVAILABLE_RECIPES["ltx-2.3"])
        _require_recipe_command(module, "split", "ltx-2.3")
