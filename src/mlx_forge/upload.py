"""Upload a converted MLX model directory to HuggingFace Hub.

Creates a repo with auto-derived naming, generates a model card,
uploads model files, and optionally adds to a collection.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Enable high-performance mode for hf-xet (saturates network bandwidth)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from .convert import SPLIT_MODEL_FILENAME, write_split_model
from .metadata import hub_repo_from_source, license_files
from .quantize import format_bytes, read_quantize_config


def load_model_metadata(model_dir: Path) -> tuple[dict, dict]:
    """Load split_model.json and config.json from a model directory.

    Args:
        model_dir: Path to converted model directory.

    Returns:
        Tuple of (split_info, config) dicts. Missing files yield empty dicts.
    """
    split_info: dict = {}
    split_path = model_dir / "split_model.json"
    if split_path.exists():
        with open(split_path) as f:
            split_info = json.load(f)

    config: dict = {}
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    return split_info, config


def derive_repo_id(
    split_info: dict, model_dir: Path, *, api: HfApi, namespace: str | None = None
) -> str:
    """Derive a HuggingFace repo ID from model metadata.

    Pattern: {namespace}/{model}-mlx[-q{bits}]

    Args:
        split_info: Parsed split_model.json contents.
        model_dir: Path to converted model directory (fallback for model name).
        api: HfApi instance (used for whoami if namespace is None).
        namespace: HF namespace/org (default: authenticated user).

    Returns:
        Repo ID string like "user/ltx-2.3-mlx-q8".
    """
    # The recipe declaration names the build, and is what the published repo
    # names encode. `source` is prose: its last segment can be a subfolder, so
    # deriving from it turned "baidu/ERNIE-Image-Turbo/pe" into "pe-mlx" — a
    # junk repo — and truncating to the parent repo would be worse still, since
    # that is the sibling model's real repository.
    source = split_info.get("source", "")
    recipe = split_info.get("recipe")
    variant = split_info.get("variant")
    if recipe:
        model_name = "-".join(part for part in (recipe, variant) if part).lower()
    elif re.search(r"-mlx(?:-q\d+)?$", model_dir.name):
        # A manifest predating the `recipe` key: the directory the operator
        # named is a better signal than a source that may point inside a repo.
        model_name = model_dir.name.lower()
    elif "/" in source:
        model_name = source.split("/")[-1].lower()
    else:
        model_name = source.lower() or model_dir.name

    # Converted model dirs are named "<model>-mlx[-q{bits}]". Strip that suffix
    # to recover the base name (so rebuilding repo_name below doesn't double it,
    # e.g. vjepa-2.0-vitl-mlx -> vjepa-2.0-vitl-mlx-mlx), but REMEMBER any bits encoded
    # in the dir name: some recipes (e.g. vjepa2) record quantization only in the
    # dir name + a separate quantize_config.json, not in split_model.json. No-op
    # for source-derived names (they carry no -mlx suffix).
    # Read the bits off the DIRECTORY, which is where they are encoded, rather
    # than off model_name — which no longer carries the suffix when the recipe
    # named it. Some recipes (vjepa2) record quantization nowhere else.
    dir_bits: int | None = None
    dir_match = re.search(r"-mlx-q(\d+)$", model_dir.name)
    if dir_match:
        dir_bits = int(dir_match.group(1))

    suffix_match = re.search(r"-mlx(?:-q(\d+))?$", model_name)
    if suffix_match:
        model_name = model_name[: suffix_match.start()]

    quantized = split_info.get("quantized", False)
    bits = split_info.get("quantization_bits") if quantized else None
    if bits is None:
        bits = dir_bits  # fall back to bits encoded in the dir name

    if namespace is None:
        try:
            user_info = api.whoami()
        except Exception as e:
            raise SystemExit(
                "Could not resolve HF namespace. "
                "Run `huggingface-cli login` or set HF_TOKEN, "
                "or pass --namespace explicitly."
            ) from e
        namespace = user_info["name"]

    parts = [model_name, "mlx"]
    if bits:
        parts.append(f"q{bits}")

    repo_name = "-".join(parts)
    return f"{namespace}/{repo_name}"


#: Not part of the model: the card itself, and files the Hub creates. The local
#: listing never contained these, but the remote one does, so filter here too.
_CARD_EXCLUDED = frozenset({"README.md", ".gitattributes", ".gitignore"})


def _is_plumbing(name: str) -> bool:
    """Whether a repo path is Hub/Git plumbing rather than model content."""
    return name in _CARD_EXCLUDED or name.split("/")[-1].startswith(".")


def generate_model_card(
    model_dir: Path,
    *,
    split_info: dict,
    config: dict,
    repo_id: str,
    base_model: str | None = None,
    license_id: str | None = None,
    usage_url: str | None = None,
    links: list[str] | None = None,
    cli_snippet: str | None = None,
    transformer_variants: list[str] | None = None,
    lora_files: list[str] | None = None,
    file_listing: dict[str, int] | None = None,
) -> str:
    """Render the model card from ``templates/model-card.md.j2``.

    Args:
        model_dir: Path to converted model directory.
        split_info: Parsed split_model.json contents.
        config: Parsed config.json contents.
        repo_id: Target HF repo ID (used in card title).
        base_model: Base model HF ID (default: read from split_info).
        license_id: SPDX identifier. None falls back to the recipe's declared
            license, then to "other" — passing the CLI default unconditionally
            used to downgrade apache-2.0/mit cards on every refresh.
        usage_url: Optional URL to an inference project that uses these weights.
        links: Optional list of related project links in "Label: URL" format.
        cli_snippet: Optional bash snippet to include in the Usage section.
        transformer_variants: Override for transformer variant list (default: read from split_info).
        lora_files: Optional list of LoRA file names to include in the card.

    The `license_name`/`license_link` front-matter fields and the License
    section come from the manifest only: "other" is the SPDX escape hatch and
    identifies nothing on its own, so a card claiming it without naming the
    agreement tells a recipient neither what the terms are nor where to read
    them. They mirror what the upstream repo declares.

    Returns:
        Model card content as a string.
    """
    from importlib.resources import files

    from jinja2 import Environment

    # `source` is prose and is not always a Hub repo id — ".../pe" is a
    # subfolder, "facebookresearch/vjepa2 (app/vjepa_2_1)" a source tree. A
    # front-matter base_model that does not resolve is worse than none.
    source = split_info.get("source", "")
    if base_model is None:
        base_model = split_info.get("base_model") or hub_repo_from_source(source)
    if transformer_variants is None:
        transformer_variants = list(split_info.get("transformer_variants", []) or [])

    quantized = split_info.get("quantized", False)
    bits = split_info.get("quantization_bits")
    group_size = split_info.get("quantization_group_size")

    # quantize_config.json is written by the quantizer itself, so it is the
    # authority on width and group size. Some recipes write the manifest before
    # quantizing and never record either — void-model published three repos
    # that way — and the group size lives nowhere else regardless.
    qconfig = read_quantize_config(model_dir) or {}
    if qconfig:
        quantized = True
        bits = bits or qconfig.get("bits")
        group_size = group_size or qconfig.get("group_size")

    model_version = config.get("model_version")

    # Build the Files section from what the upload actually publishes — the
    # same iter_model_files() the upload uses, so the card cannot advertise a
    # different set. It listed only top-level *.safetensors/*.json before, which
    # omitted the tokenizer files (published since the upload stopped filtering
    # by suffix) and everything in a subdirectory.
    #
    # README.md is excluded: it is written after this runs, so listing it would
    # make a first upload and a later --card-only refresh produce different
    # cards for the same model.
    if file_listing is None:
        file_listing = {}
        if model_dir.exists():
            file_listing = {
                p.relative_to(model_dir).as_posix(): p.stat().st_size
                for p in iter_model_files(model_dir)
            }
    model_files = [
        type("F", (), {"name": name, "size_str": format_bytes(size)})()
        for name, size in sorted(file_listing.items())
        if not _is_plumbing(name)
    ]

    template_text = files("mlx_forge.templates").joinpath("model-card.md.j2").read_text()
    env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    template = env.from_string(template_text)

    return template.render(
        repo_id=repo_id,
        base_model=base_model,
        license_id=license_id or split_info.get("license") or "other",
        license_name=split_info.get("license_name"),
        license_link=split_info.get("license_link"),
        # The card points at the local copies, whose names are the upstream
        # basenames — `license_file` may name a path inside the upstream repo.
        license_files=[Path(f).name for f in license_files(split_info.get("license_file"))],
        transformer_variants=transformer_variants,
        lora_files=lora_files or [],
        model_version=model_version,
        quantized=quantized,
        bits=bits,
        group_size=group_size,
        # Present only when the recipe says what its --quantize touches; that
        # declaration is what opts a model into the fuller quantized card.
        quantization_scope=split_info.get("quantization_scope"),
        quantized_from=unquantized_repo(repo_id, bits),
        quantize_command=quantize_command(split_info, bits),
        # build_note, not `notes`: six published manifests already use `notes`
        # for a {component: explanation} table, and rendering one as prose
        # crashed the card. Guarded as well, since the two keys will coexist.
        build_note=(bn if isinstance(bn := split_info.get("build_note"), str) else None),
        usage_url=usage_url,
        cli_snippet=(cli_snippet or "").format(repo_id=repo_id) or None,
        usage_note=split_info.get("usage_note"),
        links=links or [],
        model_files=model_files,
    )


#: Never uploaded: OS/editor droppings and caches that can appear in a model dir.
#: This is a deny-list on purpose. An allow-list of suffixes is what silently
#: dropped spiece.model and chat_template.jinja from published repos — a
#: conversion's output is complete by construction (copy_required_files fails
#: loudly on a missing file), so the upload must not second-guess which of
#: those files matter.
IGNORE_PATTERNS = [
    ".DS_Store",
    "**/.DS_Store",
    "__pycache__/**",
    "**/__pycache__/**",
    "*.pyc",
    ".git/**",
    "*.lock",
    "*.tmp",
]


def _is_ignored(relative_path: str) -> bool:
    """Whether a repo-relative path matches IGNORE_PATTERNS."""
    from fnmatch import fnmatch

    parts = relative_path.split("/")
    if ".DS_Store" in parts or "__pycache__" in parts or ".git" in parts:
        return True
    return any(fnmatch(relative_path, pattern) for pattern in IGNORE_PATTERNS)


def iter_model_files(model_dir: Path) -> list[Path]:
    """Every file of a converted model, recursively, minus junk.

    Single source of truth for what "the model" is, shared by the full upload
    and the --add-only delta path so the two modes cannot disagree.
    """
    return sorted(
        p
        for p in model_dir.rglob("*")
        if p.is_file() and not _is_ignored(p.relative_to(model_dir).as_posix())
    )


def backfill_from_recipe(model_dir: Path, split_info: dict) -> dict:
    """Fill in card metadata the manifest predates, from the recipe declaration.

    A directory converted before a field existed carries a manifest without it,
    so refreshing its card would publish the default instead of the declared
    value — a license reverting to "other", for instance. The recipe is
    identified by the `recipe` key, or by `source` for older directories.

    Absent keys are filled and otherwise the manifest wins, because it describes
    one converted directory — except for LICENSE_KEYS, where the recipe wins
    outright. A licence is a fact about the upstream model rather than a
    property of a build, so correcting our reading of it has to reach packs
    converted before the correction; fill-only-what-is-absent left
    matrix-game-3.0's stale `license: other` unreachable on the Hub. Operator
    flags still win over both at render time.

    Returns:
        The updated split_info (unchanged, and nothing written, if there is
        nothing to change or no recipe can be identified).
    """
    from .metadata import LICENSE_KEYS
    from .recipes import resolve_recipe_metadata

    metadata = resolve_recipe_metadata(split_info)
    if metadata is None:
        return split_info

    declared = metadata.as_split_fields()
    added = {k: v for k, v in declared.items() if k not in split_info}
    corrected = {
        k: v for k, v in declared.items() if k in LICENSE_KEYS and split_info.get(k, v) != v
    }
    # A licence field the recipe no longer declares is stale too: leaving a
    # license_name behind after a model turns out to be plain apache-2.0 would
    # keep naming an agreement that does not apply.
    dropped = [k for k in LICENSE_KEYS if k in split_info and k not in declared]
    if not (added or corrected or dropped):
        return split_info

    merged = {**split_info, **added, **corrected}
    for key in dropped:
        del merged[key]
    write_split_model(model_dir, merged)

    if added:
        print(f"Backfilled from the {metadata.name} recipe: {', '.join(sorted(added))}")
    for key in sorted(corrected):
        print(
            f"Corrected from the {metadata.name} recipe: "
            f"{key} {split_info[key]!r} -> {merged[key]!r}"
        )
    if dropped:
        print(f"Dropped (no longer declared by {metadata.name}): {', '.join(sorted(dropped))}")
    return merged


def unquantized_repo(repo_id: str, bits: int | None) -> str | None:
    """The bf16 repo a quantized build came from, or None if it is not one.

    Derived from the house naming convention (`default_output_dir`), not
    declared: "dgrauet/void-model-mlx-q8" is quantized from
    "dgrauet/void-model-mlx". A declaration would be one more thing to keep in
    step with a name the tool already controls.
    """
    if not bits:
        return None
    stripped = re.sub(rf"-q{bits}$", "", repo_id)
    return stripped if stripped != repo_id else None


def quantize_command(split_info: dict, bits: int | None) -> str | None:
    """The `mlx-forge convert` line that produced a quantized build."""
    recipe = split_info.get("recipe")
    if not recipe or not bits:
        return None
    variant = split_info.get("variant")
    parts = ["mlx-forge convert", recipe]
    if variant:
        parts.append(f"--variant {variant}")
    parts.append(f"--quantize --bits {bits}")
    return " ".join(parts)


def sibling_links(split_info: dict, supplied: list[str] | None) -> list[str]:
    """The operator's links that the recipe does not already declare.

    Keeps the manifest holding only what is specific to this repo — its
    "q8 variant: ..." siblings — rather than a frozen copy of the declaration.
    Order is preserved and duplicates dropped, so passing a declared link again
    is a no-op instead of printing it twice.
    """
    declared = list(split_info.get("links") or [])
    seen = set(declared)
    out = []
    for link in supplied or []:
        if link not in seen:
            seen.add(link)
            out.append(link)
    return out


def card_links(split_info: dict, supplied: list[str] | None = None) -> list[str]:
    """Every link the card shows: the recipe's, then this repo's own.

    The declaration comes first so the common links read the same across a
    model's bf16/q8/q4 repos, with the siblings appended.
    """
    links = list(split_info.get("links") or [])
    for link in [*(split_info.get("extra_links") or []), *(supplied or [])]:
        if link not in links:
            links.append(link)
    return links


def persist_card_metadata(
    model_dir: Path,
    split_info: dict,
    *,
    usage_url: str | None,
    links: list[str] | None,
    cli_snippet: str | None,
    note: str | None = None,
) -> dict:
    """Store operator-supplied card metadata so a later refresh keeps it.

    `--card-only` is documented as idempotent, but a flag typed once lives only
    in that invocation: refreshing without re-typing --cli-snippet republished
    the card with its Usage section gone. Anything the operator passes is
    written back into split_model.json, which the upload then carries.

    Returns:
        The updated split_info (unchanged, and nothing written, if no metadata
        was supplied).
    """
    # Links are stored apart from the recipe's own, under `extra_links`. A repo
    # needs the declared links PLUS its siblings ("q8 variant: ..."), which are
    # true of that repo and of no other. Folding them into `links` would freeze
    # a copy of the declaration in the manifest, so a link later added to the
    # recipe would never reach the repos that once passed --link.
    supplied = {
        "usage_url": usage_url,
        "extra_links": sibling_links(split_info, links),
        "cli_snippet": cli_snippet,
        "build_note": note,
    }
    new = {k: v for k, v in supplied.items() if v and split_info.get(k) != v}
    if not new:
        return split_info

    merged = {**split_info, **new}
    write_split_model(model_dir, merged)
    print(f"Recorded in split_model.json: {', '.join(sorted(new))}")
    return merged


def card_only_files(model_dir: Path) -> list[str]:
    """Repo-relative names a `--card-only` refresh pushes, README first.

    One function so `--dry-run` cannot advertise a different set than the real
    run performs: the two used to state the list independently, and the dry run
    kept claiming two files after the licence copy became a third.

    Beyond the card: split_model.json, because metadata the operator supplied
    would otherwise stay local and be lost by the next refresh; and the licence
    copies, because this mode is how an already-published repo is brought up to
    date, and a card linking to `./LICENSE` must not land in a repo without one.

    README.md is always first and always listed: `--dry-run` asks before it is
    written, and the upload writes it before pushing.
    """
    declared = load_model_metadata(model_dir)[0].get("license_file")
    companions = [SPLIT_MODEL_FILENAME, *(Path(f).name for f in license_files(declared))]
    return ["README.md", *(n for n in companions if (model_dir / n).exists())]


def upload_model(
    model_dir: Path,
    *,
    api: HfApi,
    repo_id: str,
    commit_message: str = "Upload MLX model via mlx-forge",
    private: bool = False,
    collection_title: str | None = None,
    card_only: bool = False,
    add_only: bool = False,
) -> str:
    """Upload a model directory to HuggingFace Hub.

    Args:
        model_dir: Path to converted model directory.
        api: HfApi instance.
        repo_id: HF repo ID (e.g. "user/ltx-2.3-mlx-distilled-q8").
        commit_message: Commit message for the upload.
        private: Whether to create a private repo.
        collection_title: If set, create/add to this collection.
        card_only: If True, push only the model card (README.md).
        add_only: If True, skip files already present on the remote repo
            (delta upload). Refuses to run if the repo does not exist yet.

    Returns:
        The repo URL.
    """
    if add_only:
        # Verify the repo exists (refuse if not — this mode is incremental)
        try:
            info = api.model_info(repo_id)
        except RepositoryNotFoundError:
            print(
                f"ERROR: --add-only refuses to run on non-existent repo '{repo_id}'. "
                "Use a normal upload to create the repo first."
            )
            raise SystemExit(1)
        except (HfHubHTTPError, OSError, ConnectionError) as e:
            print(f"ERROR: Could not query repo '{repo_id}': {e}")
            raise SystemExit(1)

        remote = {s.rfilename for s in (info.siblings or [])}
        if not model_dir.is_dir():
            print(f"ERROR: model directory does not exist: {model_dir}")
            raise SystemExit(1)
        # Compare on the repo-relative path, not the basename: a nested
        # google/umt5-xxl/spiece.model must neither be masked by a root-level
        # spiece.model on the remote nor be re-uploaded when already there.
        candidates = [(p, p.relative_to(model_dir).as_posix()) for p in iter_model_files(model_dir)]
        new_files = [(p, rel) for p, rel in candidates if rel not in remote]

        if not new_files:
            print(f"Nothing to upload (all {len(candidates)} files already on remote)")
            return f"https://huggingface.co/{repo_id}"

        skipped = [rel for _, rel in candidates if rel in remote]
        if skipped:
            print(f"Skipped (on remote): {', '.join(skipped)}")

        for p, rel in new_files:
            msg = f"{commit_message}: {rel}" if len(new_files) > 1 else commit_message
            print(f"Uploading: {rel}")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=rel,
                repo_id=repo_id,
                commit_message=msg,
            )
        return f"https://huggingface.co/{repo_id}"

    # Create repo
    print(f"Creating repo: {repo_id}")
    try:
        repo_url = api.create_repo(repo_id=repo_id, exist_ok=True, private=private)
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", None)
        if status == 403:
            print(
                "ERROR: Permission denied. Your HuggingFace token needs 'write' permission.\n"
                "Generate a new token at https://huggingface.co/settings/tokens"
            )
        elif status == 401:
            print("ERROR: Authentication failed. Run 'huggingface-cli login' or set HF_TOKEN.")
        else:
            print(f"ERROR: Failed to create repo '{repo_id}': {e}")
        raise SystemExit(1)
    except (OSError, ConnectionError) as e:
        print(f"ERROR: Network error creating repo: {e}")
        raise SystemExit(1)

    # Upload files. In --card-only mode we push ONLY the model card — this
    # avoids re-hashing multi-GB safetensors when the weights are unchanged
    # and only the README needs refreshing (e.g. appending a CLI example).
    try:
        if card_only:
            # Push the card as generated by the caller. This used to regenerate
            # it here, from the manifest alone and without the file listing,
            # links or license the caller had just assembled — so the card that
            # went up was not the one --dry-run had shown, and refreshing
            # dgrauet/matrix-game-3.0-mlx dropped its Related Projects section.
            readme_path = model_dir / "README.md"
            if not readme_path.exists():
                print(f"ERROR: {readme_path} not found — nothing to push")
                raise SystemExit(1)

            print(f"Uploading {readme_path.name} -> {repo_id}...")
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message=commit_message,
            )
            for name in card_only_files(model_dir)[1:]:
                path = model_dir / name
                print(f"Uploading {name} -> {repo_id}...")
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=name,
                    repo_id=repo_id,
                    commit_message=commit_message,
                )
        else:
            # The upload is a deny-list, so anything left in the directory goes
            # up. Show what that is before pushing — a stray source checkpoint
            # is now a visible multi-GB line rather than a silent skip.
            files = iter_model_files(model_dir)
            total = sum(p.stat().st_size for p in files)
            print(f"Uploading {model_dir} -> {repo_id}...")
            print(f"  {len(files)} files, {format_bytes(total)}:")
            for p in files:
                rel = p.relative_to(model_dir).as_posix()
                print(f"    {rel} ({format_bytes(p.stat().st_size)})")
            api.upload_folder(
                repo_id=repo_id,
                folder_path=str(model_dir),
                ignore_patterns=IGNORE_PATTERNS,
                commit_message=commit_message,
            )
    except HfHubHTTPError as e:
        print(f"ERROR: Upload failed: {e}")
        raise SystemExit(1)
    except (OSError, ConnectionError) as e:
        print(f"ERROR: Network error during upload: {e}")
        raise SystemExit(1)

    url = str(repo_url)
    print(f"Uploaded: {url}")

    # Collection operations (non-critical)
    if collection_title:
        print(f"Adding to collection: {collection_title}")
        try:
            coll = api.create_collection(title=collection_title, exists_ok=True)
            api.add_collection_item(
                collection_slug=coll.slug,
                item_id=repo_id,
                item_type="model",
                exists_ok=True,
            )
            print(f"Collection: https://huggingface.co/collections/{coll.slug}")
        except Exception as e:
            print(f"WARNING: Could not add to collection '{collection_title}': {e}")

    return url
