"""`mlx-forge validate` must fail loudly, for every recipe.

ernie-image and ernie-image-pe printed "✗ 5 checks failed" and then exited 0,
because their validate() called result.summary() without the
`if not result.passed: raise SystemExit(1)` the other eight recipes have. Any
script or CI gating on the exit code read a broken conversion as valid.

Checked across the whole registry so the next recipe cannot repeat it.
"""

from __future__ import annotations

import argparse
import importlib

import pytest

from mlx_forge.recipes import AVAILABLE_RECIPES

RECIPE_NAMES = sorted(AVAILABLE_RECIPES)


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_validate_exits_nonzero_on_an_empty_model_dir(recipe_name: str, tmp_path, capsys):
    """An empty directory passes no recipe's checks, so none may exit 0."""
    module = importlib.import_module(AVAILABLE_RECIPES[recipe_name])

    with pytest.raises(SystemExit) as exc_info:
        module.validate(argparse.Namespace(model_dir=str(tmp_path)))

    assert exc_info.value.code == 1, (
        f"{recipe_name}: validate reported failures but exited "
        f"{exc_info.value.code} — callers cannot detect a bad conversion"
    )


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_validate_exits_nonzero_on_a_missing_model_dir(recipe_name: str, tmp_path, capsys):
    module = importlib.import_module(AVAILABLE_RECIPES[recipe_name])

    with pytest.raises(SystemExit) as exc_info:
        module.validate(argparse.Namespace(model_dir=str(tmp_path / "does-not-exist")))

    assert exc_info.value.code == 1
    # A missing directory must be named as such. Without the guard the run
    # still exits 1, but only after a cascade of confusing check failures —
    # so asserting the exit code alone does not cover this.
    assert "does not exist" in capsys.readouterr().out


@pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
def test_validate_reports_the_failures_it_found(recipe_name: str, tmp_path, capsys):
    """Exiting 1 silently would be no better — the report must name what failed."""
    module = importlib.import_module(AVAILABLE_RECIPES[recipe_name])

    with pytest.raises(SystemExit):
        module.validate(argparse.Namespace(model_dir=str(tmp_path)))

    assert "checks failed" in capsys.readouterr().out
