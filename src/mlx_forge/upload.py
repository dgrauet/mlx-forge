"""Upload a converted MLX model directory to HuggingFace Hub.

Creates a repo with auto-derived naming, generates a model card,
uploads model files, and optionally adds to a collection.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi


def derive_repo_id(model_dir: Path, *, namespace: str | None = None) -> str:
    """Derive a HuggingFace repo ID from model metadata.

    Reads split_model.json to extract source and variant info.
    Pattern: {namespace}/{model}-mlx-{variant}[-q{bits}]

    Args:
        model_dir: Path to converted model directory.
        namespace: HF namespace/org (default: authenticated user).

    Returns:
        Repo ID string like "user/ltx-2.3-mlx-distilled-q8".
    """
    split_info_path = model_dir / "split_model.json"
    if not split_info_path.exists():
        raise FileNotFoundError(
            f"No split_model.json in {model_dir} — use --repo-id to specify manually"
        )

    with open(split_info_path) as f:
        split_info = json.load(f)

    # Extract model name from source (e.g. "Lightricks/LTX-2.3" -> "ltx-2.3")
    source = split_info.get("source", "")
    if "/" in source:
        model_name = source.split("/")[-1].lower()
    else:
        model_name = source.lower() or model_dir.name

    variant = split_info.get("variant", "")
    quantized = split_info.get("quantized", False)
    bits = split_info.get("quantization_bits")

    if namespace is None:
        api = HfApi()
        user_info = api.whoami()
        namespace = user_info["name"]

    parts = [model_name, "mlx"]
    if variant:
        parts.append(variant)
    if quantized and bits:
        parts.append(f"q{bits}")

    repo_name = "-".join(parts)
    return f"{namespace}/{repo_name}"


def generate_model_card(
    model_dir: Path,
    *,
    repo_id: str,
    base_model: str | None = None,
    license_id: str = "other",
) -> str:
    """Generate a HuggingFace model card with YAML frontmatter.

    Args:
        model_dir: Path to converted model directory.
        repo_id: Target HF repo ID (used in card title).
        base_model: Base model HF ID (default: read from split_model.json).
        license_id: SPDX license identifier.

    Returns:
        Model card content as a string.
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

    source = split_info.get("source", "")
    variant = split_info.get("variant", "")
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
            f"MLX format conversion of"
            f" [{base_model}](https://huggingface.co/{base_model})."
        )
    else:
        lines.append("MLX format model.")
    lines.append("")
    lines.append(
        "Converted with [mlx-forge](https://github.com/dgrauet/mlx-forge)."
    )
    lines.append("")

    # Details
    details = []
    if variant:
        details.append(f"- **Variant:** {variant}")
    if model_version:
        details.append(f"- **Model version:** {model_version}")
    if quantized and bits:
        details.append(f"- **Quantization:** int{bits}")
    if details:
        lines.extend(details)
        lines.append("")

    # File listing
    model_files = sorted(
        p for p in model_dir.iterdir()
        if p.is_file() and p.suffix in (".safetensors", ".json")
    )
    if model_files:
        lines.append("## Files")
        lines.append("")
        for p in model_files:
            size_mb = p.stat().st_size / (1024 * 1024)
            lines.append(f"- `{p.name}` ({size_mb:.1f} MB)")
        lines.append("")

    return "\n".join(lines)


def upload_model(
    model_dir: Path,
    *,
    repo_id: str,
    commit_message: str = "Upload MLX model via mlx-forge",
    private: bool = False,
    collection_title: str | None = None,
) -> str:
    """Upload a model directory to HuggingFace Hub.

    Args:
        model_dir: Path to converted model directory.
        repo_id: HF repo ID (e.g. "user/ltx-2.3-mlx-distilled-q8").
        commit_message: Commit message for the upload.
        private: Whether to create a private repo.
        collection_title: If set, create/add to this collection.

    Returns:
        The repo URL.
    """
    api = HfApi()

    print(f"Creating repo: {repo_id}")
    repo_url = api.create_repo(repo_id=repo_id, exist_ok=True, private=private)

    print(f"Uploading {model_dir} -> {repo_id}...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(model_dir),
        allow_patterns=["*.safetensors", "*.json", "README.md"],
        commit_message=commit_message,
    )

    url = str(repo_url)
    print(f"Uploaded: {url}")

    if collection_title:
        print(f"Adding to collection: {collection_title}")
        coll = api.create_collection(title=collection_title, exists_ok=True)
        api.add_collection_item(
            collection_slug=coll.slug,
            item_id=repo_id,
            item_type="model",
            exists_ok=True,
        )
        print(f"Collection: https://huggingface.co/collections/{coll.slug}")

    return url
