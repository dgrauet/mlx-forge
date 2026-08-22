"""Model-specific conversion recipes.

Each recipe defines:
- Key classification (which component a weight belongs to)
- Key sanitization (PyTorch names -> MLX names)
- Conv transposition rules
- Config extraction
- Validation checks

Recipes are plain modules dispatched by name from cli.py. COMMAND_REQUIREMENTS
below is the contract each one must satisfy; `missing_recipe_attrs` turns a
violation into an actionable message instead of an AttributeError traceback.
"""

from __future__ import annotations

import argparse
from typing import Protocol

from ..metadata import RecipeMetadata

AVAILABLE_RECIPES = {
    "ideogram-4": "mlx_forge.recipes.ideogram_4",
    "ltx-2.3": "mlx_forge.recipes.ltx_23",
    "ltx-2.5": "mlx_forge.recipes.ltx_25",
    "matrix-game-3.0": "mlx_forge.recipes.matrix_game_3_0",
    "cogvideox-fun-v1.5-5b-inp": "mlx_forge.recipes.cogvideox_fun_v1_5_5b_inp",
    "void-model": "mlx_forge.recipes.void_model",
    "hunyuan3d-2.1": "mlx_forge.recipes.hunyuan3d_21",
    "ernie-image": "mlx_forge.recipes.ernie_image",
    "ernie-image-pe": "mlx_forge.recipes.ernie_image_pe",
    "vjepa-2.1-vitl": "mlx_forge.recipes.vjepa_2_1_vitl",
    "vjepa-2.0-vitl": "mlx_forge.recipes.vjepa_2_0_vitl",
}

# command -> (arg-registration function, entry point) every recipe must expose.
COMMAND_REQUIREMENTS: dict[str, tuple[str, str]] = {
    "convert": ("add_convert_args", "convert"),
    "validate": ("add_validate_args", "validate"),
    "split": ("add_split_args", "split"),
}


class RecipeModule(Protocol):
    """Structural type of a recipe module.

    A recipe whose model needs no splitting still implements `split` — as a
    no-op that says so — rather than omitting it.
    """

    def add_convert_args(self, parser: argparse.ArgumentParser) -> None: ...
    def convert(self, args: argparse.Namespace) -> None: ...
    def add_validate_args(self, parser: argparse.ArgumentParser) -> None: ...
    def validate(self, args: argparse.Namespace) -> None: ...
    def add_split_args(self, parser: argparse.ArgumentParser) -> None: ...
    def split(self, args: argparse.Namespace) -> None: ...


def missing_recipe_attrs(module: object, command: str) -> list[str]:
    """Return the attributes `module` lacks to serve `command` (empty if complete)."""
    return [attr for attr in COMMAND_REQUIREMENTS[command] if not hasattr(module, attr)]


def resolve_recipe_metadata(split_info: dict) -> RecipeMetadata | None:
    """Find the declaration behind a converted directory.

    `convert` writes the recipe name into split_model.json, so a directory
    produced today says which recipe made it. Directories converted before that
    key existed are matched on `source` instead, which is unique across recipes
    — that is what lets an old model_dir still pick up a license or a
    base_model declared since.

    Returns None when neither works (e.g. ernie-image converted with
    --variant sft, whose source is the SFT repo rather than the recipe's
    declared one); callers then fall back to whatever the manifest holds.
    """
    import importlib

    name = split_info.get("recipe")
    if name in AVAILABLE_RECIPES:
        return importlib.import_module(AVAILABLE_RECIPES[name]).METADATA

    source = split_info.get("source")
    if source:
        for module_path in AVAILABLE_RECIPES.values():
            metadata = getattr(importlib.import_module(module_path), "METADATA", None)
            if metadata is None:
                continue
            if source == metadata.source or source in metadata.known_sources:
                return metadata
    return None
