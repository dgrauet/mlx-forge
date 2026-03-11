"""MLX Forge CLI — Convert, quantize, split, validate, and upload ML models for Apple MLX.

Usage:
    mlx-forge convert ltx-2.3 [--quantize --bits 8]
    mlx-forge validate ltx-2.3 <model_dir> [--source <checkpoint>]
    mlx-forge split ltx-2.3 <model_dir>
    mlx-forge quantize <input.safetensors> [--bits 8 --group-size 64]
    mlx-forge upload <model_dir> [--collection "MLX Forge Models"]
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
        description=("Convert, quantize, split, validate, and upload ML models for Apple MLX"),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mlx-forge {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- convert ---
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a model to MLX format",
    )
    convert_parser.add_argument(
        "recipe",
        choices=list(AVAILABLE_RECIPES),
        help="Model recipe",
    )

    # --- validate ---
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a converted model",
    )
    validate_parser.add_argument(
        "recipe",
        choices=list(AVAILABLE_RECIPES),
        help="Model recipe",
    )

    # --- split ---
    split_parser = subparsers.add_parser(
        "split",
        help="Split a unified model into components",
    )
    split_parser.add_argument(
        "recipe",
        choices=list(AVAILABLE_RECIPES),
        help="Model recipe",
    )

    # --- quantize (generic) ---
    quantize_parser = subparsers.add_parser(
        "quantize",
        help="Quantize a safetensors file",
    )
    quantize_parser.add_argument(
        "input",
        type=str,
        help="Input .safetensors file",
    )
    quantize_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default: overwrite)",
    )
    quantize_parser.add_argument(
        "--bits",
        type=int,
        default=8,
        choices=[4, 8],
        help="Bits (default: 8)",
    )
    quantize_parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Group size (default: 64)",
    )
    quantize_parser.add_argument(
        "--key-prefix",
        type=str,
        default=None,
        help="Only quantize weight keys starting with this prefix",
    )

    # --- upload (generic) ---
    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload a converted model to HuggingFace Hub",
    )
    upload_parser.add_argument(
        "model_dir",
        type=str,
        help="Path to converted model directory",
    )
    upload_parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HF repo ID (default: auto-derived from metadata)",
    )
    upload_parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="HF namespace/org (default: authenticated user)",
    )
    upload_parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection title to add the model to",
    )
    upload_parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload MLX model via mlx-forge",
        help="Commit message for the upload",
    )
    upload_parser.add_argument(
        "--license",
        type=str,
        default="other",
        help="License for model card (default: other)",
    )
    upload_parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model HF ID for model card",
    )
    upload_parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repo",
    )

    # Two-pass parsing: first get the command and recipe, then add recipe-specific args
    args, remaining = parser.parse_known_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "quantize":
        _run_generic_quantize(args)
        return

    if args.command == "upload":
        _run_upload(args)
        return

    if args.command in ("convert", "validate", "split"):
        recipe = _get_recipe(args.recipe)

        # Create a new parser for the recipe-specific args
        recipe_parser = argparse.ArgumentParser(
            prog=f"mlx-forge {args.command} {args.recipe}",
        )

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

    from .quantize import default_should_quantize, quantize_file

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    output_path = Path(args.output) if args.output else None

    if args.key_prefix:
        key_prefix = args.key_prefix

        def should_quantize(key: str, weight):
            return key.startswith(key_prefix) and default_should_quantize(key, weight)
    else:
        should_quantize = default_should_quantize

    config_path = (output_path or input_path).parent / "quantize_config.json"
    quantize_file(
        input_path,
        output_path,
        bits=args.bits,
        group_size=args.group_size,
        should_quantize=should_quantize,
        config_path=config_path,
    )


def _run_upload(args) -> None:
    """Upload a converted model directory to HuggingFace Hub."""
    from pathlib import Path

    from huggingface_hub import HfApi

    from .upload import (
        derive_repo_id,
        generate_model_card,
        load_model_metadata,
        upload_model,
    )

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: {model_dir} not found")
        sys.exit(1)

    safetensor_files = list(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        print(f"ERROR: No .safetensors files found in {model_dir}")
        print("Run conversion and/or splitting before uploading.")
        sys.exit(1)

    api = HfApi()
    split_info, config = load_model_metadata(model_dir)

    # Derive or use explicit repo ID
    if args.repo_id:
        repo_id = args.repo_id
    else:
        if not split_info:
            print("ERROR: No split_model.json found — use --repo-id")
            sys.exit(1)
        repo_id = derive_repo_id(
            split_info,
            model_dir,
            api=api,
            namespace=args.namespace,
        )

    print(f"Repo ID: {repo_id}")

    # Generate and write model card
    card_content = generate_model_card(
        model_dir,
        split_info=split_info,
        config=config,
        repo_id=repo_id,
        base_model=args.base_model,
        license_id=args.license,
    )
    readme_path = model_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(card_content)
    print(f"Generated model card: {readme_path}")

    # Upload
    url = upload_model(
        model_dir,
        api=api,
        repo_id=repo_id,
        commit_message=args.commit_message,
        private=args.private,
        collection_title=args.collection,
    )
    print(f"\nDone! {url}")


if __name__ == "__main__":
    main()
