"""MLX Forge CLI — Convert, quantize, split, and validate ML models for Apple MLX.

Usage:
    mlx-forge convert ltx23 [--quantize --bits 8]
    mlx-forge validate ltx23 <model_dir> [--source <checkpoint>]
    mlx-forge split ltx23 <model_dir>
    mlx-forge quantize <input.safetensors> [--bits 8 --group-size 64]
"""

from __future__ import annotations

import argparse
import importlib
import sys

from . import __version__
from .recipes import AVAILABLE_RECIPES


def _get_recipe(name: str):
    """Import and return a recipe module by name."""
    if name not in AVAILABLE_RECIPES:
        print(f"Unknown recipe: {name}")
        print(f"Available recipes: {', '.join(AVAILABLE_RECIPES)}")
        sys.exit(1)
    return importlib.import_module(AVAILABLE_RECIPES[name])


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlx-forge",
        description="Convert, quantize, split, and validate ML models for Apple MLX",
    )
    parser.add_argument("--version", action="version", version=f"mlx-forge {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- convert ---
    convert_parser = subparsers.add_parser("convert", help="Convert a model to MLX format")
    convert_parser.add_argument("recipe", choices=list(AVAILABLE_RECIPES), help="Model recipe")
    # Recipe-specific args are added dynamically after parsing the recipe name

    # --- validate ---
    validate_parser = subparsers.add_parser("validate", help="Validate a converted model")
    validate_parser.add_argument("recipe", choices=list(AVAILABLE_RECIPES), help="Model recipe")

    # --- split ---
    split_parser = subparsers.add_parser("split", help="Split a unified model into components")
    split_parser.add_argument("recipe", choices=list(AVAILABLE_RECIPES), help="Model recipe")

    # --- quantize (generic) ---
    quantize_parser = subparsers.add_parser("quantize", help="Quantize a safetensors file")
    quantize_parser.add_argument("input", type=str, help="Input .safetensors file")
    quantize_parser.add_argument("--output", type=str, default=None, help="Output file (default: overwrite)")
    quantize_parser.add_argument("--bits", type=int, default=8, choices=[4, 8], help="Bits (default: 8)")
    quantize_parser.add_argument("--group-size", type=int, default=64, help="Group size (default: 64)")
    quantize_parser.add_argument("--prefix", type=str, default=None,
                                 help="Only quantize keys starting with this prefix")

    # Two-pass parsing: first get the command and recipe, then add recipe-specific args
    args, remaining = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "quantize":
        _run_generic_quantize(args)
        return

    if args.command in ("convert", "validate", "split"):
        recipe = _get_recipe(args.recipe)

        # Create a new parser for the recipe-specific args
        recipe_parser = argparse.ArgumentParser(prog=f"mlx-forge {args.command} {args.recipe}")

        if args.command == "convert":
            recipe.add_convert_args(recipe_parser)
            recipe_args = recipe_parser.parse_args(remaining)
            recipe.convert(recipe_args)
        elif args.command == "validate":
            recipe.add_validate_args(recipe_parser)
            recipe_args = recipe_parser.parse_args(remaining)
            recipe.validate(recipe_args)
        elif args.command == "split":
            recipe.add_split_args(recipe_parser)
            recipe_args = recipe_parser.parse_args(remaining)
            recipe.split(recipe_args)
        return

    parser.print_help()


def _run_generic_quantize(args) -> None:
    """Run generic quantization on a safetensors file."""
    from pathlib import Path
    from .quantize import quantize_file, default_should_quantize

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    output_path = Path(args.output) if args.output else None

    if args.prefix:
        prefix = args.prefix

        def should_quantize(key: str, weight):
            return key.startswith(prefix) and default_should_quantize(key, weight)
    else:
        should_quantize = default_should_quantize

    config_path = (output_path or input_path).parent / "quantize_config.json"
    quantize_file(
        input_path, output_path,
        bits=args.bits, group_size=args.group_size,
        should_quantize=should_quantize,
        config_path=config_path,
    )


if __name__ == "__main__":
    main()
