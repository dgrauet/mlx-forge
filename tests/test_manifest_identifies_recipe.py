"""Every manifest a recipe writes must name the recipe that wrote it.

`upload` binds a converted directory back to its declaration through
split_model.json — that is what supplies the licence, the links and the usage
section when a card is refreshed months later. A manifest missing the
declaration is not a cosmetic gap: `resolve_recipe_metadata` returns None, and
the regenerated card comes out stripped of everything the recipe knew.

Measured on the Hub: five published repos (vjepa-2.0, vjepa-2.1, void-model in
three builds) carry such a manifest and lose 7 to 33 lines on a refresh. The
vjepa-2.1 recipe still wrote one, so reconverting it would have reproduced the
same unusable manifest — hence a test rather than a one-line fix.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from mlx_forge.recipes import AVAILABLE_RECIPES


def _recipe_source(name: str) -> tuple[Path, ast.Module]:
    import importlib

    module = importlib.import_module(AVAILABLE_RECIPES[name])
    path = Path(inspect.getfile(module))
    return path, ast.parse(path.read_text())


def _dict_arg(call: ast.Call) -> ast.Dict | None:
    """The manifest literal a write_split_model call passes, if it is inline."""
    for node in [*call.args, *(kw.value for kw in call.keywords)]:
        if isinstance(node, ast.Dict):
            return node
    return None


def _names_in(node: ast.AST) -> set[str]:
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


@pytest.mark.parametrize("recipe", sorted(AVAILABLE_RECIPES))
def test_every_written_manifest_carries_the_declaration(recipe):
    """Each write_split_model call must splat as_split_fields() into its dict.

    Checked statically because running a conversion needs multi-GB checkpoints.
    The call sites are the contract; a recipe that assembles the manifest in a
    variable is resolved by following that variable's assignment.
    """
    path, tree = _recipe_source(recipe)

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "write_split_model"
    ]
    assert calls, f"{recipe} never writes a manifest"

    # A manifest built in a local then passed by name: find that assignment.
    assignments: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            assignments.setdefault(node.targets[0].id, node.value)
        # `split_info: dict = {...}` — the annotation makes .value optional to
        # the type checker, though a bare declaration cannot be a manifest.
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                assignments.setdefault(node.target.id, node.value)

    for call in calls:
        literal = _dict_arg(call)
        if literal is not None:
            source = literal
        else:
            named = [a for a in call.args if isinstance(a, ast.Name)]
            assert named, f"{path.name}:{call.lineno} passes a manifest this test cannot follow"
            source = assignments.get(named[-1].id)
            assert source is not None, (
                f"{path.name}:{call.lineno} passes {named[-1].id}, never assigned"
            )

        assert "as_split_fields" in _names_in(source), (
            f"{path.name}:{call.lineno} writes a manifest without "
            "**METADATA.as_split_fields(); a card refreshed from it loses the "
            "recipe's licence, links and usage"
        )


@pytest.mark.parametrize("recipe", sorted(AVAILABLE_RECIPES))
def test_a_recipe_declaration_round_trips_through_the_manifest(recipe):
    """What as_split_fields() writes must be enough to find the recipe again."""
    import importlib

    from mlx_forge.recipes import resolve_recipe_metadata

    metadata = importlib.import_module(AVAILABLE_RECIPES[recipe]).METADATA
    resolved = resolve_recipe_metadata(metadata.as_split_fields())

    assert resolved is not None, f"{recipe}'s own manifest does not identify it"
    assert resolved.name == metadata.name


def test_hunyuan3d_manifest_uses_the_flat_quantization_keys(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import mlx_forge.convert as convert
    from mlx_forge.recipes.hunyuan3d_21 import _write_config_files

    monkeypatch.setattr(convert, "ensure_license_file", lambda *a, **k: [])
    args = SimpleNamespace(stage="shape", quantize=True, bits=4, group_size=32)
    _write_config_files(tmp_path, {"components": ["dit"]}, ["dit"], args, "dit")
    manifest = json.loads((tmp_path / "split_model.json").read_text())
    assert manifest["quantized"] is True
    assert manifest["quantization_bits"] == 4
    assert manifest["quantization_group_size"] == 32
    assert "quantization" not in manifest
