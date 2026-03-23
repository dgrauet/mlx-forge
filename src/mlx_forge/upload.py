"""Upload a converted MLX model directory to HuggingFace Hub.

Creates a repo with auto-derived naming, generates a model card,
uploads model files, and optionally adds to a collection.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Enable hf_transfer for faster uploads when available
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

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
) -> str:
    """Generate a HuggingFace model card with YAML frontmatter.

    Args:
        model_dir: Path to converted model directory.
        split_info: Parsed split_model.json contents.
        config: Parsed config.json contents.
        repo_id: Target HF repo ID (used in card title).
        base_model: Base model HF ID (default: read from split_info).
        license_id: SPDX license identifier.

    Returns:
        Model card content as a string.
    """
    source = split_info.get("source", "")
    transformer_variants = split_info.get("transformer_variants", [])
    quantized = split_info.get("quantized", False)
    bits = split_info.get("quantization_bits")
    model_version = config.get("model_version")

    if base_model is None:
        base_model = source or None

    # YAML frontmatter
    lines = ["---"]
    lines.append("library_name: mlx")
    lines.append(f"license: {license_id}")
    if base_model:
        lines.append(f"base_model: {base_model}")
    lines.append("tags:")
    for tag in ["mlx", "mlx-forge", "apple-silicon", "safetensors"]:
        lines.append(f"  - {tag}")
    lines.append("---")
    lines.append("")

    # Body
    lines.append(f"# {repo_id}")
    lines.append("")
    if base_model:
        lines.append(
            f"MLX format conversion of [{base_model}](https://huggingface.co/{base_model})."
        )
    else:
        lines.append("MLX format model.")
    lines.append("")
    lines.append("Converted with [mlx-forge](https://github.com/dgrauet/mlx-forge).")
    lines.append("")

    # Details
    details = []
    if transformer_variants:
        details.append(f"- **Transformer variants:** {', '.join(transformer_variants)}")
    if model_version:
        details.append(f"- **Model version:** {model_version}")
    if quantized and bits:
        details.append(f"- **Quantization:** int{bits}")
    if details:
        lines.extend(details)
        lines.append("")

    # File listing
    model_files = sorted(
        p for p in model_dir.iterdir() if p.is_file() and p.suffix in (".safetensors", ".json")
    )
    if model_files:
        lines.append("## Files")
        lines.append("")
        for p in model_files:
            lines.append(f"- `{p.name}` ({format_bytes(p.stat().st_size)})")
        lines.append("")

    return "\n".join(lines)


def upload_model(
    model_dir: Path,
    *,
    api: HfApi,
    repo_id: str,
    commit_message: str = "Upload MLX model via mlx-forge",
    private: bool = False,
    collection_title: str | None = None,
) -> str:
    """Upload a model directory to HuggingFace Hub.

    Args:
        model_dir: Path to converted model directory.
        api: HfApi instance.
        repo_id: HF repo ID (e.g. "user/ltx-2.3-mlx-distilled-q8").
        commit_message: Commit message for the upload.
        private: Whether to create a private repo.
        collection_title: If set, create/add to this collection.

    Returns:
        The repo URL.
    """
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

    # Upload files
    print(f"Uploading {model_dir} -> {repo_id}...")
    try:
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
