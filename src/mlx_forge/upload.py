"""Upload a converted MLX model directory to HuggingFace Hub.

Creates a repo with auto-derived naming, generates a model card,
uploads model files, and optionally adds to a collection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Enable high-performance mode for hf-xet (saturates network bandwidth)
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from .quantize import format_bytes


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
    # Extract model name from source (e.g. "Lightricks/LTX-2.3" -> "ltx-2.3")
    source = split_info.get("source", "")
    if "/" in source:
        model_name = source.split("/")[-1].lower()
    else:
        model_name = source.lower() or model_dir.name

    quantized = split_info.get("quantized", False)
    bits = split_info.get("quantization_bits")

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
    if quantized and bits:
        parts.append(f"q{bits}")

    repo_name = "-".join(parts)
    return f"{namespace}/{repo_name}"


def generate_model_card(
    model_dir: Path,
    *,
    split_info: dict,
    config: dict,
    repo_id: str,
    base_model: str | None = None,
    license_id: str = "other",
    usage_url: str | None = None,
    links: list[str] | None = None,
    cli_snippet: str | None = None,
    transformer_variants: list[str] | None = None,
    lora_files: list[str] | None = None,
) -> str:
    """Render the model card from ``templates/model-card.md.j2``.

    Args:
        model_dir: Path to converted model directory.
        split_info: Parsed split_model.json contents.
        config: Parsed config.json contents.
        repo_id: Target HF repo ID (used in card title).
        base_model: Base model HF ID (default: read from split_info).
        license_id: SPDX license identifier.
        usage_url: Optional URL to an inference project that uses these weights.
        links: Optional list of related project links in "Label: URL" format.
        cli_snippet: Optional bash snippet to include in the Usage section.
        transformer_variants: Override for transformer variant list (default: read from split_info).
        lora_files: Optional list of LoRA file names to include in the card.

    Returns:
        Model card content as a string.
    """
    from importlib.resources import files

    from jinja2 import Environment

    source = split_info.get("source", "")
    if base_model is None:
        base_model = source or None
    if transformer_variants is None:
        transformer_variants = list(split_info.get("transformer_variants", []) or [])

    quantized = split_info.get("quantized", False)
    bits = split_info.get("quantization_bits")
    model_version = config.get("model_version")

    # Build file listing from local dir (only files that exist)
    model_files = []
    if model_dir.exists():
        for p in sorted(model_dir.iterdir()):
            if p.is_file() and p.suffix in (".safetensors", ".json"):
                model_files.append(
                    type("F", (), {"name": p.name, "size_str": format_bytes(p.stat().st_size)})()
                )

    template_text = files("mlx_forge.templates").joinpath("model-card.md.j2").read_text()
    env = Environment(trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True)
    template = env.from_string(template_text)

    return template.render(
        repo_id=repo_id,
        base_model=base_model,
        license_id=license_id,
        transformer_variants=transformer_variants,
        lora_files=lora_files or [],
        model_version=model_version,
        quantized=quantized,
        bits=bits,
        usage_url=usage_url,
        cli_snippet=cli_snippet,
        links=links or [],
        model_files=model_files,
    )


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

        remote = {s.rfilename for s in info.siblings}
        if not model_dir.is_dir():
            print(f"ERROR: model directory does not exist: {model_dir}")
            raise SystemExit(1)
        candidates = sorted(
            p
            for p in model_dir.iterdir()
            if p.is_file() and (p.suffix in (".safetensors", ".json") or p.name == "README.md")
        )
        new_files = [p for p in candidates if p.name not in remote]

        if not new_files:
            print(f"Nothing to upload (all {len(candidates)} files already on remote)")
            return f"https://huggingface.co/{repo_id}"

        skipped = [p.name for p in candidates if p.name in remote]
        if skipped:
            print(f"Skipped (on remote): {', '.join(skipped)}")

        for p in new_files:
            msg = f"{commit_message}: {p.name}" if len(new_files) > 1 else commit_message
            print(f"Uploading: {p.name}")
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=p.name,
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
            # Derive transformer variants and LoRA files from remote (idempotent refresh)
            try:
                info = api.model_info(repo_id)
                remote_files = [s.rfilename for s in info.siblings]
            except (HfHubHTTPError, OSError, ConnectionError):
                remote_files = []  # fall through with local data only

            if remote_files:
                transformer_variants = sorted(
                    v
                    for f in remote_files
                    if f.startswith("transformer-") and f.endswith(".safetensors")
                    for v in [f.removeprefix("transformer-").removesuffix(".safetensors")]
                    if v
                )
                lora_files = sorted(
                    f for f in remote_files if "lora" in f and f.endswith(".safetensors")
                )
                print(f"Detected variants on remote: {', '.join(transformer_variants) or '(none)'}")
                print(f"Detected LoRAs on remote: {', '.join(lora_files) or '(none)'}")
            else:
                transformer_variants = None  # generate_model_card falls back to split_info
                lora_files = None

            # Regenerate README with remote-derived lists
            split_info, config_data = load_model_metadata(model_dir)
            readme_text = generate_model_card(
                model_dir,
                split_info=split_info,
                config=config_data,
                repo_id=repo_id,
                transformer_variants=transformer_variants,
                lora_files=lora_files,
            )
            readme_path = model_dir / "README.md"
            readme_path.write_text(readme_text)

            print(f"Uploading {readme_path.name} -> {repo_id}...")
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=repo_id,
                commit_message=commit_message,
            )
        else:
            print(f"Uploading {model_dir} -> {repo_id}...")
            api.upload_folder(
                repo_id=repo_id,
                folder_path=str(model_dir),
                allow_patterns=["*.safetensors", "*.json", "README.md"],
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
