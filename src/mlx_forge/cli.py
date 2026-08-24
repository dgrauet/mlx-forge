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
from .recipes import AVAILABLE_RECIPES, missing_recipe_attrs


def _get_recipe(name: str):
    """Import and return a recipe module by name."""
    if name not in AVAILABLE_RECIPES:
        print(f"Unknown recipe: {name}")
        print(f"Available recipes: {', '.join(AVAILABLE_RECIPES)}")
        sys.exit(1)
    return importlib.import_module(AVAILABLE_RECIPES[name])


def _require_recipe_command(recipe, command: str, recipe_name: str) -> None:
    """Abort with an actionable message when a recipe cannot serve `command`."""
    missing = missing_recipe_attrs(recipe, command)
    if missing:
        print(f"ERROR: recipe '{recipe_name}' does not implement '{command}'.")
        print(f"Missing: {', '.join(missing)}")
        print("This is a recipe bug — please report it or add the missing function.")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser.

    Returns:
        Configured ArgumentParser with all subparsers registered.
    """
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
        default=None,
        help=(
            "SPDX license for the model card. Defaults to the one the recipe "
            "declares, then to 'other'. Passing it explicitly overrides both."
        ),
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
    upload_parser.add_argument(
        "--usage-url",
        type=str,
        default=None,
        help="URL to an inference project that uses these weights (added to model card)",
    )
    upload_parser.add_argument(
        "--link",
        action="append",
        default=None,
        metavar="'Label: URL'",
        help=(
            "Extra related-project link for the model card, ADDED to the ones "
            "the recipe declares (repeatable, format: 'Label: URL'). Use it for "
            "what is true of this repo alone, typically its sibling builds "
            "('q8 variant: https://huggingface.co/...'); a link true of every "
            "build belongs in the recipe. Recorded in split_model.json so a "
            "later --card-only refresh keeps it."
        ),
    )
    upload_parser.add_argument(
        "--note",
        type=str,
        default=None,
        help=(
            "Paragraph shown under the model's opening lines, for what is "
            "measured on this build alone — a memory footprint, a quality "
            "figure. Recorded in split_model.json so a later --card-only "
            "refresh keeps it. State only what you have actually measured: it "
            "is published verbatim."
        ),
    )
    upload_parser.add_argument(
        "--cli-snippet",
        type=str,
        default=None,
        help=(
            "Bash snippet to embed in the model card's Usage section as a code block. "
            "Published verbatim and now persisted in split_model.json, so it is "
            "reused by every later --card-only refresh: only reference a package or "
            "command that actually exists."
        ),
    )
    upload_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Render the card and show what would change on the remote, without "
            "writing or uploading anything. Use before refreshing a published card."
        ),
    )
    upload_parser.add_argument(
        "--set-gated",
        action="store_true",
        help="Apply the gating the recipe declares to the target repo (outward-facing; "
        "without it a mismatch is only reported)",
    )
    mode_group = upload_parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--card-only",
        action="store_true",
        help=(
            "Push ONLY the regenerated model card (README.md). Skips the weights upload "
            "entirely — use this to refresh card content without re-hashing safetensors."
        ),
    )
    mode_group.add_argument(
        "--add-only",
        action="store_true",
        help=(
            "Delta upload: skip files whose names already exist on the remote repo. "
            "Useful after `convert --skip-shared` to push only the new variant. "
            "Refuses to run if the repo doesn't exist."
        ),
    )

    return parser


def main() -> None:
    parser = build_parser()

    # Two-pass parsing means argparse consumes -h/--help during the first pass,
    # prints the generic recipe-chooser and exits — so the recipe's own flags
    # (--variant, --skip-shared, --config-only, ...) were undiscoverable from
    # the CLI. When both the command and the recipe are named, route help to
    # the recipe parser instead.
    argv = sys.argv[1:]
    if (
        len(argv) >= 2
        and argv[0] in ("convert", "validate", "split")
        and argv[1] in AVAILABLE_RECIPES
        and any(a in ("-h", "--help") for a in argv[2:])
    ):
        recipe = _get_recipe(argv[1])
        _require_recipe_command(recipe, argv[0], argv[1])
        recipe_parser = argparse.ArgumentParser(prog=f"mlx-forge {argv[0]} {argv[1]}")
        getattr(recipe, f"add_{argv[0]}_args")(recipe_parser)
        recipe_parser.parse_args(["--help"])  # prints the recipe's help and exits

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
        _require_recipe_command(recipe, args.command, args.recipe)

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


def _remote_variants(api, repo_id: str) -> tuple[list[str] | None, list[str] | None]:
    """Transformer variants and LoRAs as they exist on the remote.

    A delta upload adds files this directory never had, so a refresh must read
    the repo rather than the manifest. Returns (None, None) when the repo
    cannot be queried, letting the card fall back to split_model.json.
    """
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    try:
        info = api.model_info(repo_id)
        remote_files = [s.rfilename for s in (info.siblings or [])]
    except (RepositoryNotFoundError, HfHubHTTPError, OSError, ConnectionError):
        return None, None

    if not remote_files:
        return None, None

    variants = sorted(
        v
        for f in remote_files
        if f.startswith("transformer-") and f.endswith(".safetensors")
        for v in [f.removeprefix("transformer-").removesuffix(".safetensors")]
        if v
    )
    loras = sorted(f for f in remote_files if "lora" in f and f.endswith(".safetensors"))
    print(f"Detected variants on remote: {', '.join(variants) or '(none)'}")
    print(f"Detected LoRAs on remote: {', '.join(loras) or '(none)'}")
    return variants, loras


def _card_file_listing(api, repo_id: str, model_dir, *, card_only: bool = False) -> dict[str, int]:
    """What the repo will contain: the remote listing plus what is about to go up.

    The card's other derived section (transformer_variants) already reads the
    remote, so deriving the file list from the local directory alone produced
    cards that contradicted themselves after a delta upload.

    In --card-only mode nothing local is uploaded, so the remote is the whole
    truth: a local directory holding a different build would otherwise publish
    sizes that do not match the repo.
    """
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    from .upload import iter_model_files

    listing: dict[str, int] = {}
    try:
        info = api.model_info(repo_id, files_metadata=True)
        for sibling in info.siblings or []:
            if sibling.size is not None:
                listing[sibling.rfilename] = sibling.size
    except (RepositoryNotFoundError, HfHubHTTPError, OSError, ConnectionError):
        pass  # new repo, or offline: the local directory is the whole story

    if card_only and listing:
        # ...except the files this mode does push. A refresh carries the licence
        # copy up alongside the card, and a card whose Files section omits it
        # would describe the repo as it was a second before.
        from pathlib import Path

        from .metadata import license_files
        from .upload import load_model_metadata

        declared = load_model_metadata(model_dir)[0].get("license_file")
        for name in (Path(f).name for f in license_files(declared)):
            local = model_dir / name
            if local.exists():
                listing[name] = local.stat().st_size
        return listing

    for p in iter_model_files(model_dir):
        listing[p.relative_to(model_dir).as_posix()] = p.stat().st_size
    return listing


def _show_card_diff(api, repo_id: str, card: str, model_dir, *, card_only: bool) -> None:
    """Print what a real run would change on the remote, and stop there."""
    import difflib
    import re

    from huggingface_hub import hf_hub_download

    try:
        published = open(hf_hub_download(repo_id, "README.md")).read()
        etat = f"against the card published at {repo_id}"
    except Exception:
        published = ""
        etat = f"{repo_id} has no card yet — everything below is new"

    print(f"\n{'=' * 60}")
    print(f"DRY RUN — nothing is written or uploaded ({etat})")
    print("=" * 60)

    diff = list(
        difflib.unified_diff(
            published.splitlines(), card.splitlines(), "published", "regenerated", lineterm=""
        )
    )
    if not diff:
        print("\nThe card is already up to date — a real run would change nothing.")
    else:
        print()
        for line in diff:
            print(line)

    def entrees(prefixe, entete):
        return [
            line[1:]
            for line in diff
            if line.startswith(prefixe) and not line.startswith(entete) and line[1:].strip()
        ]

    # A file whose size changed shows as one removal plus one addition. Pairing
    # them on the entry name keeps the warning meaningful: it must fire on
    # content that disappears, not on a size that moved.
    def cle(line: str) -> str:
        if line.startswith("- `"):
            # A file entry: pair on the backticked name, so a changed size is a
            # change and not a loss. Splitting on "` (" failed when the
            # published entry carried no size at all — hand-written cards list
            # `config.json` bare — leaving the two keys one backtick apart and
            # reporting a file that is still there as gone.
            return line.split("`")[1]
        if line.startswith("- **") and ":**" in line:  # a metadata line: pair on the label
            return line.split(":**", 1)[0]
        # A front-matter field: pair on the key, so correcting a value reads as a
        # change and not a loss. `license: other` -> `license: apache-2.0` was
        # counted as content disappearing, and a warning that cries wolf on every
        # corrected value is one nobody reads when it means it.
        if re.match(r"^[A-Za-z_][\w-]*: ", line):
            return line.split(":", 1)[0]
        return line

    ajoutees = {cle(line) for line in entrees("+", "+++")}
    perdues = [line for line in entrees("-", "---") if cle(line) not in ajoutees]
    print(f"\n{'-' * 60}")
    if perdues:
        print(f"WARNING: {len(perdues)} line(s) would be REMOVED from the published card.")
        print("Content that exists only in the published README is lost by regenerating.")
        print("Declare it in the recipe, or pass the matching flag, before pushing.")
    else:
        print("No content would be lost.")

    print("\nA real run would push:")
    if card_only:
        # Asked of the same function the real run obeys, so this cannot drift.
        from .upload import card_only_files

        for name in card_only_files(model_dir):
            print(f"  {name}")
    else:
        print("  README.md")
        print(f"  every file in {model_dir} (see the Files section above)")


def _run_upload(args) -> None:
    """Upload a converted model directory to HuggingFace Hub."""
    from pathlib import Path

    from huggingface_hub import HfApi

    from .convert import ensure_license_file, write_split_model
    from .upload import (
        backfill_from_recipe,
        backfill_quantization,
        card_links,
        derive_repo_id,
        generate_model_card,
        load_model_metadata,
        persist_card_metadata,
        resolve_gating,
        upload_model,
    )

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: {model_dir} not found")
        sys.exit(1)

    # --card-only pushes the README and the manifest, never the weights, so it
    # must work from a directory that holds only metadata.
    if not args.card_only and not list(model_dir.glob("*.safetensors")):
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

    # A directory converted before a metadata field existed carries a manifest
    # without it; recover it from the recipe rather than publishing the default.
    split_info = backfill_from_recipe(model_dir, split_info, dry_run=args.dry_run)

    # Same idea for the build's own facts: the quantizer recorded them in
    # quantize_config.json, and a manifest written before quantizing never did.
    split_info = backfill_quantization(model_dir, split_info, dry_run=args.dry_run)

    # Gating is declared by the recipe, never applied as a side effect of an
    # unrelated upload: only --set-gated changes what may download the repo.
    # On a first upload this also creates the repo (ahead of upload_model's
    # own exist_ok create_repo) when --set-gated is passed, so a gated build
    # is never briefly public between repo creation and gating — see
    # resolve_gating's docstring for the full reasoning.
    resolve_gating(
        repo_id,
        split_info,
        api,
        set_gated=args.set_gated,
        dry_run=args.dry_run,
        private=args.private,
    )

    # Publishing a derivative of a community-licensed model obliges us to hand
    # the recipient a copy of the agreement, so this runs before the file
    # listing is built: the licence is part of what the card advertises, and a
    # missing one stops the upload rather than being reported afterwards.
    vouched_before = split_info.get("license_provenance")
    ensure_license_file(model_dir, split_info, dry_run=args.dry_run)
    # ensure_license_file records the copy's origin into split_info but does not
    # write; persist it here, or the next run would have nothing to check the
    # file against and would go back to trusting whatever sits at that name.
    if split_info.get("license_provenance") != vouched_before and not args.dry_run:
        write_split_model(model_dir, split_info)
        print("Recorded licence provenance in split_model.json")

    # Record anything the operator supplied, so a later --card-only refresh
    # does not silently drop it.
    split_info = persist_card_metadata(
        model_dir,
        split_info,
        usage_url=args.usage_url,
        links=args.link,
        cli_snippet=args.cli_snippet,
        note=args.note,
        dry_run=args.dry_run,
    )

    # Describe the repo as it will be, not just the local directory: after a
    # delta upload the remote holds files this directory never had.
    file_listing = _card_file_listing(api, repo_id, model_dir, card_only=args.card_only)

    # A refresh describes the repo as it is, including files a delta upload
    # added that this directory never held.
    variants, loras = _remote_variants(api, repo_id) if args.card_only else (None, None)

    # Generate and write model card
    card_content = generate_model_card(
        model_dir,
        split_info=split_info,
        config=config,
        repo_id=repo_id,
        base_model=args.base_model,
        license_id=args.license,
        usage_url=args.usage_url or split_info.get("usage_url"),
        # The recipe's links plus this repo's own; --link adds, never replaces.
        links=card_links(split_info, args.link),
        cli_snippet=args.cli_snippet or split_info.get("cli_snippet"),
        file_listing=file_listing,
        transformer_variants=variants,
        lora_files=loras,
    )
    if args.dry_run:
        _show_card_diff(api, repo_id, card_content, model_dir, card_only=args.card_only)
        return

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
        card_only=args.card_only,
        add_only=args.add_only,
    )
    print(f"\nDone! {url}")


if __name__ == "__main__":
    main()
